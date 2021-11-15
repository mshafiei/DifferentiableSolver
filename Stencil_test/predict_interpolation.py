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
  return (1-lmbda) * i0 + lmbda * i1

lmbda_init = 0.1
lmbda_gt = 0.9
h,w = 100,100
dw = 1
key1 = jax.random.PRNGKey(42)
key2 = jax.random.PRNGKey(43)

inpt = [jnp.concatenate((jnp.ones((w,h//2)),jnp.zeros((w,h//2)))),jnp.concatenate((jnp.zeros((w,h//2)),jnp.ones((w,h//2))))]
window_gt = jnp.array([lmbda_gt])
image_gt = interpolate(inpt[0],inpt[1],lmbda_gt)
init_inpt = jnp.zeros_like(inpt[0])
init_window = jnp.array([lmbda_init])


@jax.jit
def stencil_residual(image, window, data):
  """Objective function."""
  avg_weight = 0.5 ** 0.5 * 1 / len(data[0].reshape(-1)) ** 0.5
  r1 =  jsp.signal.convolve((image - data[0]).reshape(h,w), (1-window.reshape(dw,dw))**0.5, mode='same')
  conv_image = jsp.signal.convolve((image - data[1]).reshape(h,w), window.reshape(dw,dw)**0.5 , mode='same')
  out = jnp.concatenate(( r1.reshape(-1), conv_image.reshape(-1)),axis=0)
  return avg_weight * out


@jax.jit
def screen_poisson_objective(image, window, data):
  """Objective function."""
  return (stencil_residual(image, window, data) ** 2).sum()


@implicit_diff.custom_root(jax.grad(screen_poisson_objective))
def screen_poisson_solver(init_image,window, data):
    f = lambda u:stencil_residual(u,window,data)
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
    inpt, gt = data
    # We use the bijective mapping l2reg = jnp.exp(theta)
    # both to optimize in log-space and to ensure positivity.
    f = lambda u: screen_poisson_solver(init_inner, u, inpt)
    f_v = (f(window) - gt) ** 2
    # print('outer_obj ',f(window))
    return f_v.mean()

def hyper_optimization():
    im_gt = interpolate(inpt[0],inpt[1],lmbda_gt)
    data = [inpt, im_gt]
    count = 20
    lmbdas = jnp.linspace(lmbda_init,lmbda_gt,count)
    valid_loss = jnp.array([outer_objective(lmbdas[i], init_inpt, data) for i in range(count)])
    window = init_window
    lr = .5
    import cv2
    for i in tqdm.trange(2000):
      g = jax.grad(outer_objective,argnums=0)(window, init_inpt, data)
      loss = outer_objective(window, init_inpt, data)

      plt.figure(figsize=(24,7))
      plt.subplot(1,4,1)
      plt.plot(lmbdas,valid_loss,'r')
      plt.scatter(lmbda_init,valid_loss[0],color='r')
      plt.scatter(lmbda_gt,valid_loss[-1],color='g')
      plt.arrow(window,loss,1,g[0],color='b')
      plt.subplot(1,4,2)
      plt.imshow(im_gt)
      plt.subplot(1,4,3)
      plt.imshow(inpt[0])
      plt.subplot(1,4,4)
      plt.imshow(inpt[1])
      plt.savefig('out/plt_%04d.png'%i)

      print('g ',g,' window pred ', window,' window gt ', lmbda_gt, ' loss ', loss)
      window -= lr * g


hyper_optimization()
