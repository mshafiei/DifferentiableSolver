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
# import cvgutils.nn.jaxutils as jaxutils
from jax.experimental import stax
import cvgutils.Image as cvgim
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from flax import optim
import time

from flax import linen as nn
import jax.numpy as jnp
import jax
from flax.training import train_state
import optax                          # The Optax gradient processing and optimization library
import numpy as np                    # Ordinary NumPy
from jax.tree_util import tree_flatten, tree_unflatten
from jax.experimental import host_callback as hcb
logger = cvgviz.logger('./logger','tb','autodiff','autodiff_2convnet_3features_relu')

dw = 3
key4 = jax.random.PRNGKey(45)
gt_image = cvgim.imread('~/Projects/cvgutils/tests/testimages/wood_texture.jpg')[:32,:64,:] *2
noise = jax.random.normal(key4,gt_image.shape) * 0.3
noisy_image = jnp.clip(gt_image + noise,0,1)

init_inpt = jnp.zeros_like(gt_image)
# init_inpt = init_inpt.at[100,100,:].set(1)
im_gt = jnp.array(gt_image)
h,w = gt_image.shape[0],gt_image.shape[1]

data = [dw,h,w,noisy_image, im_gt]

class Conv3features(nn.Module):

  def setup(self):
    self.straight1       = nn.Conv(1,(1,1),strides=(1,1),use_bias=True)
    self.straight2       = nn.Conv(3,(3,3),strides=(1,1),use_bias=True)
    
  def __call__(self,x):
    # return nn.softplus(self.straight1(x))
    l1 = nn.softplus(self.straight1(x))
    return nn.softplus(self.straight2(l1))

@jax.jit
def stencil_residual(pp_image, hp_nn):
  _, _, _, inpt,_ = data
  """Objective function."""
  avg_weight = (1. / 2.) ** 0.5 *  (1. / pp_image.reshape(-1).shape[0] ** 0.5)
  r1 =  pp_image - inpt
  unet_out = Conv3features().apply({'params': hp_nn}, pp_image)
    
  out = jnp.concatenate(( r1.reshape(-1), unet_out.reshape(-1)),axis=0)
  return avg_weight * out


@jax.jit
def screen_poisson_objective(pp_image, hp_nn):
  """Objective function."""
  return (stencil_residual(pp_image, hp_nn) ** 2).sum()


# @implicit_diff.custom_root(jax.grad(screen_poisson_objective))
def screen_poisson_solver_unrolled(init_image,hp_nn):
    f = lambda pp_image:stencil_residual(pp_image,hp_nn)
    loss = lambda pp_image:screen_poisson_objective(pp_image,hp_nn)
    optim_cond = jax.grad(loss)

    gn_iters = 3
    x = init_image
    for i in range(gn_iters):
      def Ax(pp_image):
          jtd = jax.jvp(f,(x,),(pp_image,))[1]
          return jax.vjp(f,x)[1](jtd)[0]
      def jtf(x):
        return jax.vjp(f,x)[1](f(x))[0]
      d = linear_solve.solve_cg(matvec=Ax,
                              b=-jtf(x),
                              init=x,
                              maxiter=100)
      x += d
      hcb.id_print(optim_cond(x),name='optim_cond ')
    return x

@implicit_diff.custom_root(jax.grad(screen_poisson_objective))
def screen_poisson_solver_ID(init_image,hp_nn):
    f = lambda pp_image:stencil_residual(pp_image,hp_nn)
    loss = lambda pp_image:screen_poisson_objective(pp_image,hp_nn)
    optim_cond = jax.grad(loss)

    gn_iters = 10
    x = init_image
    for i in range(gn_iters):
      def Ax(pp_image):
          jtd = jax.jvp(f,(x,),(pp_image,))[1]
          return jax.vjp(f,x)[1](jtd)[0]
      def jtf(x):
        return jax.vjp(f,x)[1](f(x))[0]
      d = linear_solve.solve_cg(matvec=Ax,
                              b=-jtf(x),
                              init=x,
                              maxiter=100)
      x += d
    return x

@jax.jit
def x_star_id(hp_nn, init_inner):
  f_id = lambda hp_nn: screen_poisson_solver_ID(init_inner, hp_nn)
  return f_id(hp_nn).sum()

@jax.jit
def x_star_unrolled(hp_nn, init_inner):
  f_unrolled = lambda hp_nn: screen_poisson_solver_unrolled(init_inner, hp_nn)
  return f_unrolled(hp_nn).sum()


def implicit_diff(params):
  hyper_param, prime_param = params
  f = lambda hp_nn:x_star(hp_nn, *params[1:])
  dF = jax.grad(screen_poisson_objective)
  sol = screen_poisson_solver(prime_param,hyper_param)
  # dfv_dx = jax.grad(f)
  def jtjft_dx_dlmbda(u):
    _,vjpfun = jax.vjp(dF,(sol,hyper_param))
    dx_dx,_ = vjpfun(sol)
    return 

  def jft_dlambda():
    _,vjpfun = jax.vjp(dF,(sol,hyper_param))
    _,dx_dlambda = vjpfun(sol)
    return dx_dlambda

  d = linear_solve.solve_cg(matvec=jtjft_dx_dlmbda(),
                        b=-jft_dlambda(),
                        init=jnp.zeros_like(jft_dlambda()),
                        maxiter=100)
   

def check_with_unrolled(params):

  f_id = lambda hp_nn:x_star_id(hp_nn, *params[1:])
  f_unrolled = lambda hp_nn:x_star_unrolled(hp_nn, *params[1:])
  # grad_implicit = implicit_diff(params)
  _,grad_unrolled = jax.value_and_grad(f_unrolled)(params[0])#jax unrolled
  _,grad_id = jax.value_and_grad(f_id)(params[0])#jax unrolled
  squared_diff = jax.tree_multimap(lambda x, y: (x-y)**2, grad_unrolled,grad_id)
  diff =  jax.tree_util.tree_reduce(lambda val, elem: val+elem,squared_diff).sum()
  hcb.id_print(diff,name='diff ')
  return diff

def hyper_optimization():

  cnn = Conv3features()
  rng = jax.random.PRNGKey(1)
  testim = jax.random.uniform(rng,[1, h, w, 3])
  rng, init_rng = jax.random.split(rng)
  params = cnn.init(init_rng, testim)['params']

  check_with_unrolled([params,init_inpt])
  # check_with_unrolled([params,init_inpt])



    
hyper_optimization()
