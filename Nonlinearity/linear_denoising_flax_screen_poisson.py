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
import cvgutils.nn.jaxutils as jaxutils
from jax.experimental import stax
import cvgutils.Image as cvgim
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from flax import optim
@jax.jit
def stencil_residual(pp_image, hp_nn, data):
  dw, h, w, inpt = data
  """Objective function."""
  avg_weight = (1. / 2.) ** 0.5 *  (1. / pp_image.reshape(-1).shape[0] ** 0.5)
  r1 =  pp_image - inpt
  flag = True
  if(flag):
    dy = hp_nn * (pp_image[1:,:,:] - pp_image[:-1,:,:])
    dx = hp_nn * (pp_image[:,1:,:] - pp_image[:,:-1,:])
  out = jnp.concatenate(( r1.reshape(-1), dx.reshape(-1), dy.reshape(-1)),axis=0)
  return avg_weight * out


@jax.jit
def screen_poisson_objective(pp_image, hp_nn, data):
  """Objective function."""
  return (stencil_residual(pp_image, hp_nn, data) ** 2).sum()


@implicit_diff.custom_root(jax.grad(screen_poisson_objective))
def screen_poisson_solver(init_image,hp_nn, data):
    f = lambda pp_image:stencil_residual(pp_image,hp_nn,data)
    loss = lambda pp_image:screen_poisson_objective(pp_image,hp_nn,data)
    def matvec(pp_image):
        jtd = jax.jvp(f,(init_image,),(pp_image,))[1]
        return jax.vjp(f,init_image)[1](jtd)[0]
    def jtf(x):
      return jax.vjp(f,x)[1](f(x))[0]

    gn_iters = 3
    x = init_image
    for i in range(gn_iters):
        x += linear_solve.solve_cg(matvec=matvec,
                                b=-jtf(x),
                                init=x,
                                maxiter=100)

    return x

@jax.jit
def outer_objective(hp_nn, init_inner, data):
    """Validation loss."""
    _,_,_,_, gt = data
    f = lambda hp_nn: screen_poisson_solver(init_inner, hp_nn, data[:-1])
    f_v = ((f(hp_nn) - gt) ** 2).mean()
    return f_v

def hyper_optimization():
    dw = 3
    key4 = jax.random.PRNGKey(45)
    gt_image = cvgim.imread('~/Projects/cvgutils/tests/testimages/wood_texture.jpg')
    gt_image = cvgim.resize(gt_image,scale=0.10) * 2
    noise = jax.random.normal(key4,gt_image.shape) * 0.3
    noisy_image = jnp.clip(gt_image + noise,0,1)
    
    init_inpt = jnp.zeros_like(gt_image)
    im_gt = jnp.array(gt_image)
    h,w = gt_image.shape[0],gt_image.shape[1]

    logger = cvgviz.logger('./logger','tb','autodiff','autodiff_lambda')
    data = [dw,h,w,noisy_image, im_gt]
    params = 0.00001

    f = lambda hp_nn:outer_objective(hp_nn, init_inpt, data)
    lr = 0.1
    solver = OptaxSolver(fun=f, opt=optax.adam(lr), implicit_diff=True)
    state = solver.init_state(params)
    f_t = jax.jit(screen_poisson_solver)

    for i in tqdm.trange(10000):
      
      params, state = solver.update(params=params, state=state)
      loss = f(params)
      print('loss ',loss)
      logger.addScalar(loss,'loss_GD')
      if(i%10 == 0):
        output = f_t(init_inpt, params, data[:-1])
        imshow = jnp.concatenate((output,noisy_image,im_gt),axis=1)
        imshow = jnp.clip(imshow,0,1)
        logger.addImage(np.array(imshow).transpose(2,0,1),'Image')
      logger.takeStep()


hyper_optimization()
