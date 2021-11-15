from absl import app
import jax
from jax._src.numpy.lax_numpy import argsort, interp
import jax.numpy as jnp
from jaxopt import implicit_diff
from jaxopt import linear_solve
from jaxopt import OptaxSolver, GradientDescent
from matplotlib.pyplot import vlines
import optax
from sklearn import datasets
from sklearn import model_selection
from sklearn import preprocessing
import matplotlib.pylab as plt
import numpy as np
import jax.scipy as jsp
import tqdm

def interpolate(i0,i1,lmbda):
  return (1-lmbda)**2 * i0 + lmbda**2 * i1

lmbda_init = 0.1
lmbda_gt = 0.9
h,w = 100,100
dw = 3
key1 = jax.random.PRNGKey(42)
key2 = jax.random.PRNGKey(43)

# inpt0,inpt1 = jnp.concatenate((jnp.ones((w,h//2)),jnp.zeros((w,h//2)))),jnp.concatenate((jnp.zeros((w,h//2)),jnp.ones((w,h//2))))
inpt0,inpt1 = jax.random.uniform(key1,(h,w)),jax.random.uniform(key2,(h,w))
window_gt = jnp.array([[0,0,0],[0,lmbda_gt,0],[0,0,0]])
image_gt = interpolate(inpt0,inpt1,lmbda_gt)
init_inpt = jnp.zeros_like(inpt0)
init_window = jnp.array([[0,0,0],[0,lmbda_init,0],[0,0,0]])


@jax.jit
def stencil_residual(image, window, inpt0,inpt1):
  """Objective function."""
  avg_weight = 0.5 ** 0.5 * 1 / len(inpt0.reshape(-1)) ** 0.5
  r1 = (image - inpt0).reshape(h,w)
  conv_image = jsp.signal.convolve((image - inpt1).reshape(h,w), window.reshape(dw,dw), mode='same')
  out = jnp.concatenate(( r1.reshape(-1), conv_image.reshape(-1)),axis=0)
  return avg_weight * out


@jax.jit
def screen_poisson_objective(image, window, inpt0,inpt1):
  """Objective function."""
  return (stencil_residual(image, window, inpt0,inpt1) ** 2).sum()


@implicit_diff.custom_root(jax.grad(screen_poisson_objective))
def screen_poisson_solver(init_image,window, inpt0,inpt1):
    f = lambda u:stencil_residual(u,window,inpt0,inpt1)
    def matvec(u):
        jtd = jax.jvp(f,(init_image,),(u,))[1]
        return jax.vjp(f,init_image)[1](jtd)[0]
    def jtf(x):
      return jax.vjp(f,x)[1](f(x))[0]

    #gauss newton loop
    gn_iters = 3
    x = init_image
    for _ in range(gn_iters):
        x += linear_solve.solve_cg(matvec=matvec,
                                b=-jtf(x),
                                init=x,
                                maxiter=100)
    return x

# x = screen_poisson_solver(init_inpt, init_window, inpt)
# print('tst')
@jax.jit
def outer_objective(window, init_inner, data):
    """Validation loss."""
    inpt0,inpt1, gt = data
    # We use the bijective mapping l2reg = jnp.exp(theta)
    # both to optimize in log-space and to ensure positivity.
    f = lambda u: screen_poisson_solver(init_inner, u, inpt0,inpt1)
    f_v = (f(window) - gt) ** 2
    # print('outer_obj ',f(window))
    return f_v.mean()

def ndarray2texmatrix(a):
  txt = ''
  for i in range(a.shape[0]):
    for j in range(a.shape[1]):
      # if (j <a.shape[1]-1):
      txt += '%.04f' % a[i][j] + ' '
      # elif(i < a.shape[0]-1):
    txt += '\n'
  return txt
  # return '\\begin{pmatrix} ' + txt + ' \\end{pmatrix}'

def hyper_optimization():
    im_gt = screen_poisson_solver(init_inpt, window_gt, inpt0,inpt1)
    data = [inpt0,inpt1, im_gt]
    count = 20
    # lmbdas = jnp.linspace(lmbda_init,lmbda_gt,count)
    # valid_loss = jnp.array([outer_objective(lmbdas[i], init_inpt, data) for i in range(count)])
    window = init_window
    lr = .1
    import cv2
    for i in tqdm.trange(2000):
      g = jax.grad(outer_objective,argnums=0)(window, init_inpt, data)
      loss = outer_objective(window, init_inpt, data)
      # im_pred = screen_poisson_solver(init_inpt, window, inpt)

      # plt.figure(figsize=(24,7))
      # plt.subplot(1,5,1)
      # # plt.plot(lmbdas,valid_loss,'r')
      # # plt.scatter(lmbda_init,valid_loss[0],color='r')
      # # plt.scatter(lmbda_gt,valid_loss[-1],color='g')
      # # plt.arrow(window,loss,1,g[0],color='b')
      # plt.rcParams.update({"text.usetex": True,"font.family": "sans-serif","font.sans-serif": ["Helvetica"]})
      # text = '$\\nabla_K x^*(K)$ \n' + ndarray2texmatrix(g)+ '\n K pred \n'+ndarray2texmatrix(window) + '\n K gt \n' + ndarray2texmatrix(window_gt)
      # plt.title('Gradients')
      # # text = '$\\begin{pmatrix} 0 & 0 & 0 \\\\ 0 & 0 & 0 \\\\ 0 & 0 & 0 \\\\ \\end{pmatrix}'
      # plt.text(0,0,text)
      # plt.subplot(1,5,2)
      # plt.imshow(im_pred)
      # plt.title('Predicted')
      # plt.subplot(1,5,3)
      # plt.imshow(im_gt)
      # plt.title('Ground truth')
      # plt.subplot(1,5,4)
      # plt.imshow(inpt[0])
      # plt.title('Input 1')
      # plt.subplot(1,5,5)
      # plt.imshow(inpt[1])
      # plt.title('Input 2')
      # plt.savefig('out/plt_%04d.png'%i)

      print('g ',g)
      print('window pred ', window)
      print('window gt ', window_gt)
      print('loss ', loss)
      window -= lr * g


hyper_optimization()
