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
from jax.tree_util import tree_flatten, tree_unflatten
from jax.experimental import host_callback as hcb
import pickle


gn_iters = 30
nhierarchies = 1
scale = 2**nhierarchies
dw = 3
key4 = jax.random.PRNGKey(45)
gt_image = cvgim.imread('~/Projects/cvgutils/tests/testimages/wood_texture.jpg')[:32,:64,:] *2
# gt_image = cvgim.resize(gt_image,scale=0.10) * 2
noise = jax.random.normal(key4,gt_image.shape) * 0.3
noisy_image = jnp.clip(gt_image + noise,0,1)

# noisy_image = jnp.zeros_like(gt_image)
# noisy_image = noisy_image.at[100,100,:].set(1)
init_inpt = jnp.zeros_like(gt_image)
# init_inpt = init_inpt.at[100,100,:].set(1)
im_gt = jnp.array(gt_image)
h,w = gt_image.shape[0],gt_image.shape[1]

data = [dw,h,w,noisy_image, im_gt]

class Conv3features(nn.Module):

  def setup(self):
    self.straight1       = nn.Conv(3,(3,3),strides=(1,1),use_bias=True)
    self.straight2       = nn.Conv(3,(3,3),strides=(1,1),use_bias=True)
    
  def __call__(self,x):
    # return self.straight1(x)
    # return nn.softplus(self.straight1(x))
    l1 = nn.softplus(self.straight1(x))
    return nn.softplus(self.straight2(l1))

@jax.jit
def stencil_residual(pp_image, hp_nn, data):
  _, _, _, inpt,_ = data
  """Objective function."""
  avg_weight = (1. / 2.) ** 0.5 *  (1. / pp_image.reshape(-1).shape[0] ** 0.5)
  r1 =  pp_image - inpt
  flag = False
  if(flag):
    dy = pp_image[1:,:,:] - pp_image[:-1,:,:]
    dx = pp_image[:,1:,:] - pp_image[:,:-1,:]
  else:
    unet_out = Conv3features().apply({'params': hp_nn}, pp_image)
    
    # dy = jnp.concatenate((dy1,dy2,dy3),axis=-1)

  out = jnp.concatenate(( r1.reshape(-1), unet_out.reshape(-1)),axis=0)
  return avg_weight * out


@jax.jit
def screen_poisson_objective(pp_image, hp_nn, data):
  """Objective function."""
  return (stencil_residual(pp_image, hp_nn, data) ** 2).sum()


# @implicit_diff.custom_root(jax.grad(screen_poisson_objective))
def screen_poisson_solver_unrolled(init_image,hp_nn, data):

  scale = 2**nhierarchies
  #downsample x_0 k times
  #for i in range(k)
  #complete gn loop
  #upsample x_i
  x = init_image
  x = jax.image.resize(x,(x.shape[0]//2**(nhierarchies-1),x.shape[1]//2**(nhierarchies-1),x.shape[2]),"trilinear")
  for k in range(nhierarchies-1,-1,-1):
    _, _, _, inpt,_ = data
    inpt = jax.image.resize(inpt,(inpt.shape[0]//2**k,inpt.shape[1]//2**k,inpt.shape[2]),"trilinear")
    f = lambda pp_image:stencil_residual(pp_image,hp_nn,[*data[:-2],inpt,data[-1]])
    for _ in range(gn_iters):
      def Ax(pp_image):
          jtd = jax.jvp(f,(x,),(pp_image,))[1]
          return jax.vjp(f,x)[1](jtd)[0]
      def jtf(x):
        return jax.vjp(f,x)[1](f(x))[0]
      d = linear_solve.solve_cg(matvec=Ax,
                              b=-jtf(x),
                              init=x,
                              maxiter=100)
      # hcb.id_print(((Ax(d) + jtf(x)) ** 2).mean(),name='cg optimality unrolled ')
      x += d
    if(k >0):
      x = jax.image.resize(x,(x.shape[0]*2,x.shape[1]*2,x.shape[2]),"trilinear")

  loss = lambda pp_image:screen_poisson_objective(pp_image,hp_nn,[*data[:-2],inpt,data[-1]])
  optim_cond = jax.grad(loss)
  print('optimality cond unrolled ', (optim_cond(x) ** 2).mean())
  return x

@implicit_diff.custom_root(jax.grad(screen_poisson_objective))
def screen_poisson_solver_id(init_image,hp_nn,data):
  scale = 2**nhierarchies
  #downsample x_0 k times
  #for i in range(k)
  #complete gn loop
  #upsample x_i
  x = init_image
  x = jax.image.resize(x,(x.shape[0]//2**(nhierarchies-1),x.shape[1]//2**(nhierarchies-1),x.shape[2]),"trilinear")
  for k in range(nhierarchies-1,-1,-1):
    _, _, _, inpt,_ = data
    inpt = jax.image.resize(inpt,(inpt.shape[0]//2**k,inpt.shape[1]//2**k,inpt.shape[2]),"trilinear")
    f = lambda pp_image:stencil_residual(pp_image,hp_nn,[*data[:-2],inpt,data[-1]])
    loss = lambda pp_image:screen_poisson_objective(pp_image,hp_nn,[*data[:-2],inpt,data[-1]])
    optim_cond = jax.grad(loss)
    # for _ in range(gn_iters):
    while((optim_cond(x) ** 2).mean() >= 1e-18):
      def Ax(pp_image):
          jtd = jax.jvp(f,(x,),(pp_image,))[1]
          return jax.vjp(f,x)[1](jtd)[0]
      def jtf(x):
        return jax.vjp(f,x)[1](f(x))[0]
      d = linear_solve.solve_cg(matvec=Ax,
                              b=-jtf(x),
                              init=x,
                              maxiter=100)
      # hcb.id_print(((Ax(d) + jtf(x)) ** 2).mean(),name='cg optimality id ')
      x += d
    if(k >0):
      x = jax.image.resize(x,(x.shape[0]*2,x.shape[1]*2,x.shape[2]),"trilinear")
  print('optimality cond jax id ', (optim_cond(x) ** 2).mean())
  return x

@implicit_diff.custom_root(jax.grad(screen_poisson_objective))
def screen_poisson_hierarchical_solver_id(init_image,hp_nn,data):

  #downsample x_0 k times
  #for i in range(k)
  #complete gn loop
  #upsample x_i
  x = init_image
  x = jax.image.resize(x,(x.shape[0]//2**(nhierarchies-1),x.shape[1]//2**(nhierarchies-1),x.shape[2]),"trilinear")
  for k in range(nhierarchies-1,-1,-1):
    _, _, _, inpt,_ = data
    inpt = jax.image.resize(inpt,(inpt.shape[0]//2**k,inpt.shape[1]//2**k,inpt.shape[2]),"trilinear")
    f = lambda pp_image:stencil_residual(pp_image,hp_nn,[*data[:-2],inpt,data[-1]])
    loss = lambda pp_image:screen_poisson_objective(pp_image,hp_nn,[*data[:-2],inpt,data[-1]])
    optim_cond = jax.grad(loss)
    for _ in range(gn_iters):
      def Ax(pp_image):
          jtd = jax.jvp(f,(x,),(pp_image,))[1]
          return jax.vjp(f,x)[1](jtd)[0]
      def jtf(x):
        return jax.vjp(f,x)[1](f(x))[0]
      d = linear_solve.solve_cg(matvec=Ax,
                              b=-jtf(x),
                              init=x,
                              maxiter=100)
      # hcb.id_print(((Ax(d) + jtf(x)) ** 2).mean(),name='cg optimality id ')
      x += d
    if(k >0):
      x = jax.image.resize(x,(x.shape[0]*2,x.shape[1]*2,x.shape[2]),"trilinear")
  print('optimality cond ', (optim_cond(x) ** 2).mean())
  return x

# @jax.jit
def outer_objective_id(hp_nn, init_inner,data):
    """Validation loss."""
    gt = data[-1]
    f = lambda hp_nn: screen_poisson_solver_id(init_inner, hp_nn,data)
    loss = lambda pp_image:screen_poisson_objective(pp_image,hp_nn,data)
    # optim_cond = jax.grad(loss)
    x = f(hp_nn)
    # hcb.id_print((optim_cond(x) ** 2).mean(),name='df_t/d_x id ')
    f_v = ((x - gt) ** 2).mean()
    return f_v
    # return x.sum()
# 
# @jax.jit
def outer_objective_unrolled(hp_nn, init_inner,data):
    """Validation loss."""
    # gt = data[-1]
    f = lambda hp_nn: screen_poisson_solver_unrolled(init_inner, hp_nn,data)
    loss = lambda pp_image:screen_poisson_objective(pp_image,hp_nn,data)
    optim_cond = jax.grad(loss)
    x = f(hp_nn)
    hcb.id_print((optim_cond(x) ** 2).mean(),name='df_t/d_x unrolled ')
    # f_v = ((x - gt) ** 2).mean()
    return x.sum()

def implicit_diff(params):
  hyper_param, prime_param = params
  F = jax.grad(screen_poisson_objective)
  sol = screen_poisson_solver_unrolled(prime_param,hyper_param,data)
  F_v = lambda u: F(sol,u,data)
  F_x = lambda u: F(u,hyper_param,data)

  #x: n
  #f:n
  #l: m
  #dfdl: nxm
  #dxdl: nxm
  #dfdx:nxn

  def vmap_df_dx(u):
    def df_dx(u): 
      #u: nxm
      #jvp: nxn
      dims = list(range(len(sol.shape)))
      dimsp1 = list(range(1,1+len(sol.shape)))
      batched = u.reshape(*sol.shape,-1).transpose(dims[-1]+1,*dims)
      g_x = lambda x: jax.vjp(F_x,sol)[1](x)
      return jax.vmap(g_x)(batched)[0].transpose(*dimsp1,0).reshape(*u.shape)
      # return jvpfun


    return jax.tree_map(lambda x: df_dx(x), u)

  def jf_dlambda():
    a = jax.jacfwd(F_v)(hyper_param)
    return jax.tree_multimap(lambda x: -x, a)

  a = jf_dlambda()
  vmap_df_dx(a)
  zero_map = jax.tree_multimap(lambda x: x*0, jf_dlambda())
  d = linear_solve.solve_cg(matvec=vmap_df_dx,
                        b=jf_dlambda(),
                        init=zero_map,
                        maxiter=100)
  
  return d
   

def fd(hyper_params, init_inner, data,delta):
  f_unrolled = lambda hp_nn:screen_poisson_solver_unrolled (init_inner,hp_nn,data)
  from jax.tree_util import tree_flatten, tree_unflatten
  grad_flat, grad_tree = tree_flatten(hyper_params)
  for i in tqdm.trange(len(grad_flat)):
    value_flat, value_tree = tree_flatten(hyper_params)
    shape = value_flat[i].shape
    for j in tqdm.trange(value_flat[i].reshape(-1).shape[0]):
      vff = value_flat.copy()
      vfb = value_flat.copy()
      # vff[i] = vff[i].reshape(-1).at[j].set(vff[i].reshape(-1)[j] + delta/2)
      # vfb[i] = vfb[i].reshape(-1).at[j].set(vfb[i].reshape(-1)[j] - delta/2)
      vff[i].reshape(-1)[j] += delta/2
      vfb[i].reshape(-1)[j] -= delta/2
      vff[i] = vff[i].reshape(*shape)
      vfb[i] = vfb[i].reshape(*shape)
      vff_tree = tree_unflatten(value_tree, vff)
      vfb_tree = tree_unflatten(value_tree, vfb)
      ff = f_unrolled(vff_tree)
      fb = f_unrolled(vfb_tree)
      # grad_flat[i] = grad_flat[i].reshape(-1).at[j].set((ff - fb) / delta)
      grad_flat[i].reshape(-1)[j] = (ff - fb) / delta
    grad_flat[i] = grad_flat[i].reshape(*shape)
  grad_tree = tree_unflatten(grad_tree, grad_flat)
  return grad_tree

# @jax.jit
def check_with_unrolled(params,data):
  f_unrolled = lambda hp_nn:screen_poisson_solver_unrolled ( *params[1:],hp_nn,data)
  f_id = lambda hp_nn:screen_poisson_hierarchical_solver_id( *params[1:],hp_nn,data)
  # implicit_diff_val = implicit_diff(params)
  # # fd_grad = fd(params[0], *params[1:], data,0.01)
  grad_id = jax.jacobian(f_id)(params[0])
  # grad_unrolled = jax.jacobian(f_unrolled)(params[0])
  
  # # squared_diff_fd = jax.tree_multimap(lambda x, y: (x-y)**2, fd_grad,grad_unrolled)
  # squared_diff_jaxid_unrolled = jax.tree_multimap(lambda x, y: (x-y)**2, grad_id,grad_unrolled)
  # squared_diff_myid_unrolled = jax.tree_multimap(lambda x, y: (x-y)**2, implicit_diff_val,grad_unrolled)
  
  # # fd_sum = [i.sum() for i in jax.tree_flatten(squared_diff_fd)[0]]
  # jaxid_unrolled_sum = [i.sum() for i in jax.tree_flatten(squared_diff_jaxid_unrolled)[0]]
  # myid_unrolled_sum = [i.sum() for i in jax.tree_flatten(squared_diff_myid_unrolled)[0]]
  # # hcb.id_print(jnp.array(fd_sum).mean(),name='fd_diff')
  # hcb.id_print(jnp.array(jaxid_unrolled_sum).mean(),name='jax_diff')
  # hcb.id_print(jnp.array(myid_unrolled_sum).mean(),name='my_diff')
  # # return jax_diff

def hyper_optimization():
  # import pickle
  # # # # pickle.dump( params , open( 'weights_1x1.pkl' , 'wb' ) )
  # params = pickle.load( open( 'weights.pkl' , 'rb' ))
  # check_with_unrolled([params, init_inpt],data)

  delta = 0.001

  cnn = Conv3features()
  rng = jax.random.PRNGKey(1)
  testim = jax.random.uniform(rng,[1, h, w, 3])
  rng, init_rng = jax.random.split(rng)
  params = cnn.init(init_rng, testim)['params']

  rng = jax.random.PRNGKey(0)

  import time
  start_time = time.time()
  f = lambda hp_nn:outer_objective_id(hp_nn, init_inpt,data)
  end_time = time.time()
  # logger.addScalar(end_time - start_time,'compile_time')

  lr = 0.01
  solver = OptaxSolver(fun=outer_objective_id, opt=optax.adam(lr), implicit_diff=True)
  # optimality_func = lambda :jax.grad(screen_poisson_objective)
  # solver.optimality_fun = 
  state = solver.init_state(params)
  import pickle
  # result, _ = solver.run(init_params = state)
  # f_t = screen_poisson_solver
  for i in tqdm.trange(10000):
    # if (i%100==0):
    # pickle.dump( params , open( 'weights.pkl' , 'wb' ) )
    # weights = pickle.load( open( 'weights.pkl' , 'rb' ))
    # check_with_unrolled([params, init_inpt],data)

    # start_time = time.time()
    params, state = solver.update(params, state,init_inner=init_inpt,data=data)
    # solver.optimality_fun
    # optimality_err = solver.l2_optimality_error(params)
    # end_time = time.time()
    # logger.addScalar(end_time - start_time,'update_time')
    # params = optax.apply_updates(params, updates)
    loss = f(params)
    print('loss ',loss)
    # logger.addScalar(loss,'loss_GD')
    # if(i%10 == 0):
      # output = f_t(init_inpt, params)
      # imshow = jnp.concatenate((output,noisy_image,im_gt),axis=1)
      # imshow = jnp.clip(imshow,0,1)
      # logger.addImage(np.array(imshow).transpose(2,0,1),'Image')
    # logger.takeStep()

    
  #   print('loss ', loss)
  #   params = jax.tree_multimap(lambda x,dfx:  x - lr * dfx, params, grad)


hyper_optimization()
