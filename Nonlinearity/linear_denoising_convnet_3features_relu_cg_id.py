import coax
import re
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

from flax import linen as nn
import jax.numpy as jnp
import jax
from flax.training import train_state
import optax                          # The Optax gradient processing and optimization library
import numpy as np                    # Ordinary NumPy
import tensorflow as tf
from jax.tree_util import tree_flatten, tree_unflatten
from jax.experimental import host_callback as hcb
import pickle
import os

gn_iters = 30
nhierarchies = 1
scale = 2**nhierarchies
dw = 3
key4 = jax.random.PRNGKey(45)
gt_image = cvgim.imread('~/Projects/cvgutils/tests/testimages/wood_texture.jpg')[:256,:512,:] *2
# gt_image = cvgim.resize(gt_image,scale=0.10) * 2
logger = cvgviz.logger('./logger','tb','autodiff','autodiff_conv_softplus_2layer_cg_id')
noise = jax.random.normal(key4,gt_image.shape) * 0.3
noisy_image = jax.device_put(jnp.clip(gt_image + noise,0,1))

init_inpt = jax.device_put(jnp.zeros_like(gt_image))
im_gt = jax.device_put(jnp.array(gt_image))
h,w = gt_image.shape[0],gt_image.shape[1]

data = [dw,h,w,noisy_image, im_gt]

class Conv3features(nn.Module):

  def setup(self):
    self.straight1       = nn.Conv(3,(3,3),strides=(1,1),use_bias=True)
    self.straight2       = nn.Conv(3,(3,3),strides=(1,1),use_bias=True)
    
  def __call__(self,x):
    # return self.straight1(x)
    l1 = nn.softplus(self.straight1(x))
    return nn.softplus(self.straight2(l1))

@jax.jit
def stencil_residual(pp_image, hp_nn, data):
  _, _, _, inpt,_ = data
  """Objective function."""
  avg_weight = (1. / 2.) ** 0.5 *  (1. / pp_image.reshape(-1).shape[0] ** 0.5)
  r1 =  pp_image - inpt
  unet_out = Conv3features().apply({'params': hp_nn}, pp_image)
  out = jnp.concatenate(( r1.reshape(-1), unet_out.reshape(-1)),axis=0)
  return out * avg_weight


@jax.jit
def screen_poisson_objective(pp_image, hp_nn, data):
  """Objective function."""
  return (stencil_residual(pp_image, hp_nn, data) ** 2).sum()




@jax.jit
def linear_solver_obj(x,hp_nn,data):
  _, _, _, inpt,_ = data
  f = lambda pp_image:stencil_residual(pp_image,hp_nn,[*data[:-2],inpt,data[-1]])
  return ((f(x) + jax.vjp(f,x)[1](d)[0]) ** 2).sum()

  

@implicit_diff.custom_root(jax.grad(linear_solver_obj),has_aux=True)
def linear_solver_id(x,hp_nn,data):
  f = lambda pp_image:stencil_residual(pp_image,hp_nn,[*data[:-2],inpt,data[-1]])
  def Ax(d):
    jtd = jax.jvp(f,(x,),(d,))[1]
    return jax.vjp(f,x)[1](jtd)[0]
  def jtf(x):
    return jax.vjp(f,x)[1](f(x))[0]
  d = linear_solve.solve_cg(matvec=Ax,
                          b=-jtf(x),
                          init=x,
                          maxiter=100)
  return d, ((Ax(d) +jtf(x)) ** 2).sum()





# @implicit_diff.custom_root(jax.grad(screen_poisson_objective),has_aux=True)
@jax.jit
def nonlinear_solver(init_image,hp_nn,data):
  scale = 2**nhierarchies

  x = init_image
  # x = jax.image.resize(x,(x.shape[0]//2**(nhierarchies-1),x.shape[1]//2**(nhierarchies-1),x.shape[2]),"trilinear")

  # for k in range(nhierarchies-1,-1,-1):
  _, _, _, inpt,_ = data
  # inpt = jax.image.resize(inpt,(inpt.shape[0]//2**k,inpt.shape[1]//2**k,inpt.shape[2]),"trilinear")
  # f = lambda pp_image:stencil_residual(pp_image,hp_nn,[*data[:-2],inpt,data[-1]])
  loss = lambda pp_image:screen_poisson_objective(pp_image,hp_nn,[*data[:-2],inpt,data[-1]])
  optim_cond = lambda x: (jax.grad(loss)(x) ** 2).sum()
  def loop_body(args):
    x,count, gn_opt_err, gn_loss,linear_opt_err = args
    d, linea_opt = linear_solver_id(x,hp_nn,data)
    x += 0.3 * d

    # if(count <200):
    linear_opt_err = linear_opt_err.at[count.astype(int)].set(linea_opt)
    gn_opt_err = gn_opt_err.at[count.astype(int)].set(optim_cond(x))
    gn_loss = gn_loss.at[count.astype(int)].set(screen_poisson_objective(x,hp_nn,data))
    count += 1
    return (x,count, gn_opt_err, gn_loss,linear_opt_err)

  # x,count, gn_opt_err, gn_loss, linear_opt_err= jax.lax.while_loop(lambda x:optim_cond(x[0]) >= 1e-18,loop_body,(x,0.0,-jnp.ones(200),-jnp.ones(200),-jnp.ones(200))) 
  x,count, gn_opt_err, gn_loss,linear_opt_err = jax.lax.fori_loop(0,3,lambda i,val:loop_body(val),(x,0.0,-jnp.ones(200),-jnp.ones(200),-jnp.ones(200))) 
  # count, gn_opt_err, gn_loss = 0, [0], [0]
  # x += linear_solver_id(x,hp_nn,data)
  return x,(count, gn_opt_err, gn_loss)


@jax.jit
def outer_objective_id(hp_nn, init_inner,data):
    """Validation loss."""
    gt = data[-1]
    f = lambda hp_nn: nonlinear_solver(init_inner, hp_nn,data)
    x,aux = f(hp_nn)
    f_v = ((x - gt) ** 2).mean()
    return f_v.mean(),(x,*aux)

def hyper_optimization():
  import time
  start = time.time()

  # tf.config.experimental.set_visible_devices([], 'GPU')
  # f = lambda x: screen_poisson_objective(init_data,x,data)
  
  # rng = jax.random.PRNGKey(0)
  # rng, key = jax.random.split(rng)

  # init_data = jnp.ones((100,100))

  delta = 0.001
  rng = jax.random.PRNGKey(1)
  rng, init_rng = jax.random.split(rng)
  testim = jax.device_put(jax.random.uniform(rng,[1, h, w, 3]))
  params = Conv3features().init(init_rng, testim)['params']



  lr = 0.001
  
  # rng = jax.random.PRNGKey(0)
  
  f = lambda hp_nn:outer_objective_id(hp_nn, init_inpt,data)
  # x, params = coax.utils.load('params.coax')
  # v_loss = outer_objective_id(params, x,data)
  # df,_ = jax.grad(f,has_aux=True)(params)
  solver = OptaxSolver(fun=outer_objective_id, opt=optax.adam(lr), implicit_diff=True,has_aux=True)
  state = solver.init_state(params)
  x = init_inpt
  for i in tqdm.trange(10000):
    # delta = df(params)
    # delta = jax.tree_map(lambda x: -lr * x,delta)
    # params = solver._apply_updates(params,delta)
    params, state = solver.update(params, state,init_inner=x,data=data)
    x,count, gn_opt_err, gn_loss = state.aux
    
    # l = [linear_solver_id(init_inpt,params,data).mean() for i in range(50)]
    end = time.time()
    print('time: ',end - start)
    print('loss ',state.value, ' gn iteration count ', count)
    logger.addScalar(state.value,'loss_GD')
    
    # if(i==199):
    #   coax.utils.dump([x,params], 'params.coax')

    if(i%10 == 0):
      imshow = jnp.concatenate((x,noisy_image,im_gt),axis=1)
      imshow = jnp.clip(imshow,0,1)
      logger.addImage(np.array(imshow).transpose(2,0,1),'Image')
    logger.takeStep()
    
    if(i%10 == 0):
      for l in gn_opt_err:
        if(l == -1):
          break
        logger.addScalar(l,'gn_opt_err_%04d'%i)
        logger.takeStep()

      for l in gn_loss:
        if(l == -1): 
          break
        logger.addScalar(l,'gn_loss%04d'%i)
        logger.takeStep()

    # loss = f(params)
    # print('loss ',loss)


hyper_optimization()
