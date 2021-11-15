from absl import app
import jax
from jax._src.numpy.lax_numpy import argsort, interp, zeros_like
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
import cvgutils.Viz as cvgviz

lmbda_init = 0.1
lmbda_gt = 0.9
h,w = 100,100
dw = 3
key1 = jax.random.PRNGKey(42)
key2 = jax.random.PRNGKey(43)

inpt = jax.random.uniform(key1,(h,w))
window_gt = jnp.array([[0,0,0],[-1,1,0],[0,0,0]])
image_gt = jsp.signal.convolve(inpt, window_gt, mode='same')
init_inpt = jnp.zeros_like(inpt)
init_window = jax.random.uniform(key1,(dw,dw))#*0.5 + 0.5*window_gt
logger = cvgviz.logger('./logger','tb','autodiff','autodiff')

@jax.jit
def stencil_residual(image, window, data):
  """Objective function."""
  avg_weight = 0.5 ** 0.5 * 1 / len(data.reshape(-1)) ** 0.5
  r1 =  image - data
  conv_image = jsp.signal.convolve(image.reshape(h,w), window.reshape(dw,dw) , mode='same')
  out = jnp.concatenate(( r1.reshape(-1), conv_image.reshape(-1)),axis=0)
  return avg_weight * out


@jax.jit
def screen_poisson_objective(image, window, data):
  """Objective function."""
  return (stencil_residual(image, window, data) ** 2).sum()


@implicit_diff.custom_root(jax.grad(screen_poisson_objective))
def screen_poisson_solver(init_image,window, data):
    f = lambda u:stencil_residual(u,window,data)
    loss = lambda u:screen_poisson_objective(u,window,data)
    def matvec(u):
        jtd = jax.jvp(f,(init_image,),(u,))[1]
        return jax.vjp(f,init_image)[1](jtd)[0]
    def jtf(x):
      return jax.vjp(f,x)[1](f(x))[0]

    #gauss newton loop
    gn_iters = 3
    x = init_image
    for i in range(gn_iters):
        x += linear_solve.solve_cg(matvec=matvec,
                                b=-jtf(x),
                                init=x,
                                maxiter=100)
        logger.addScalar(np.float32(loss(x)),'loss_GN_%04i_%04i'%(logger.step,i))
    return x

# @jax.jit
def outer_objective(window, init_inner, data):
    """Validation loss."""
    inpt, gt = data
    f = lambda u: screen_poisson_solver(init_inner, u, inpt)
    f_v = (f(window) - gt) ** 2
    return f_v.mean()

def fd(window, init_inner, data,delta):
  grad = jnp.zeros_like(window.reshape(-1))
  for i in range(window.reshape(-1).shape[0]):
    winf = window.reshape(-1)
    winb = window.reshape(-1)
    winf = winf.at[i].add(delta/2)
    winb = winb.at[i].add(-delta/2)
    f = outer_objective(winf.reshape(dw,dw), init_inner, data)
    b = outer_objective(winb.reshape(dw,dw), init_inner, data)
    grad = grad.at[i].set((f - b) / delta)
  return grad.reshape(dw,dw)

def hyper_optimization():
    
    # im_gt = jsp.signal.convolve(inpt, window_gt, mode='same')
    im_gt = screen_poisson_solver(init_inpt, window_gt, inpt)
    data = [inpt, im_gt]
    count = 20
    # lmbdas = jnp.linspace(lmbda_init,lmbda_gt,count)
    # valid_loss = jnp.array([outer_objective(lmbdas[i], init_inpt, data) for i in range(count)])
    window = init_window
    lr = 0.99
    #compare to fd
    # fd_grad = fd(window, init_inpt, data,0.00001)
    # g_grad = jax.grad(outer_objective,argnums=0)(window, init_inpt, data)
    # print('tst')
    # f = lambda u:outer_objective(u, init_inpt, data)
    
    # solver = OptaxSolver(fun=f, opt=optax.adam(3e-4), implicit_diff=True)
    # initial, _ =solver.init(init_params=init_window)
    # result, _ = solver.run(init_params = initial)
    # print('result ',result)

    for i in tqdm.trange(2000):
      g = jax.grad(outer_objective,argnums=0)(window, init_inpt, data)
      loss = outer_objective(window, init_inpt, data)
      logger.addScalar(loss,'loss_GD')
      logger.takeStep()
      # plt.figure(figsize=(24,7))
      # plt.subplot(1,4,1)
      # plt.plot(lmbdas,valid_loss,'r')
      # plt.scatter(lmbda_init,valid_loss[0],color='r')
      # plt.scatter(lmbda_gt,valid_loss[-1],color='g')
      # plt.arrow(window,loss,1,g[0],color='b')
      # plt.subplot(1,4,2)
      # plt.imshow(im_gt)
      # plt.subplot(1,4,3)
      # plt.imshow(inpt[0])
      # plt.subplot(1,4,4)
      # plt.imshow(inpt[1])
      # plt.savefig('out/plt_%04d.png'%i)

      print('g \n',g)
      print('window pred \n', window)
      print('window gt \n', window_gt, ' loss ', loss)
      window -= lr * g


hyper_optimization()
