import jax
import jax.numpy as jnp
from jaxopt import implicit_diff, OptaxSolver, linear_solve
import optax
import numpy as np
import tqdm
import cvgutils.Viz as cvgviz
import cvgutils.Image as cvgim
from flax import linen as nn
import tensorflow as tf
import timeit

################ inner loop model end ############################
class Conv3features(nn.Module):

  def setup(self):
    self.straight1       = nn.Conv(3,(3,3),strides=(1,1),use_bias=True)
    self.straight2       = nn.Conv(32,(3,3),strides=(1,1),use_bias=True)
    self.straight3       = nn.Conv(3,(3,3),strides=(1,1),use_bias=True)
    self.groupnorm1      = nn.GroupNorm(1)
    self.groupnorm2      = nn.GroupNorm(8)
    
  def __call__(self,x):
    l1 = self.groupnorm1(nn.softplus(self.straight1(x)))
    l2 = self.groupnorm2(nn.softplus(self.straight2(l1)))
    return nn.softplus(self.straight3(l2))

def stencil_residual(pp_image, hp_nn, data):
  smooth_image = pp_image['smooth_image']
  scalemap = pp_image['scalemap']
  ambient_image = data['ambient']
  flash_image = data['flash']

  l1 = data['lambda1']
  l2 = data['lambda2']
  l3 = data['lambda3']
  e2 = smooth_image - ambient_image
  eps = 0.0001
  dy_smooth = smooth_image[...,1:,:,:] - smooth_image[...,:-1,:,:]
  dx_smooth = smooth_image[...,:,1:,:] - smooth_image[...,:,:-1,:]
  # dx_ambient = ambient_image[...,1:,:] - ambient_image[...,:-1,:]
  # dy_ambient = ambient_image[...,:,1:] - ambient_image[...,:,:-1]
  dy_flash = flash_image[...,1:,:,:] - flash_image[...,:-1,:,:]
  dx_flash = flash_image[...,:,1:,:] - flash_image[...,:,:-1,:]
  s_x = dx_smooth / (dx_flash + eps)
  s_y = dy_smooth / (dy_flash + eps)
  e1_1 = jnp.abs(scalemap[...,:,1:,:] - s_x)
  e1_2 = jnp.abs(scalemap[...,1:,:,:] - s_y)
  e3_1 = dx_smooth
  e3_2 = dy_smooth

  #prim_parameter: scalemap, smooth image
  #scalemap term, fidelity term, smoothness term, DD regularizer
  """Objective function."""
  avg_weight = (1. / 3.) ** 0.5 *  (1. / smooth_image.reshape(-1).shape[0] ** 0.5)
  # unet_out = Conv3features().apply({'params': hp_nn}, pp_image)
  out = jnp.concatenate((l1 * e1_1.reshape(-1),l1 * e1_2.reshape(-1), l2 * e2.reshape(-1),l3 * e3_1.reshape(-1),l3 * e3_2.reshape(-1)),axis=0)
  return out * avg_weight
################ inner loop model end ############################

################ linear and nonlinear solvers begin ##############
def screen_poisson_objective(pp_image, hp_nn, data):
  """Objective function."""
  return (stencil_residual(pp_image, hp_nn, data) ** 2).sum()
def cg_optimality(d,x,hp_nn,data):
  f = lambda pp_image:stencil_residual(pp_image,hp_nn,data)
  def Ax(pp_image):
    jtd = jax.jvp(f,(x,),(pp_image,))[1]
    return jax.vjp(f,x)[1](jtd)[0]
  def jtf(x):
    return jax.vjp(f,x)[1](f(x))[0]
  cg = Ax(d) + jtf(x)
  return cg


@implicit_diff.custom_root(cg_optimality,has_aux=True)
def linear_solver_id(d,x,hp_nn,data):
  f = lambda pp_image:stencil_residual(pp_image,hp_nn,data)
  def Ax(pp_image):
    jtd = jax.jvp(f,(x,),(pp_image,))[1]
    return jax.vjp(f,x)[1](jtd)[0]
  def jtf(x):
    return jax.vjp(f,x)[1](f(x))[0]
  b = tree_scalar_prod(-1,jtf(x))
  d = linear_solve.solve_cg(matvec=Ax,
                          b=b,
                          init=d,
                          maxiter=100)
  axd = Ax(d)
  aux = tree_diff_squared_sum(axd,b)
  aux = tree_sum_red(aux)
  return d, aux

# @implicit_diff.custom_root(jax.grad(screen_poisson_objective),has_aux=True)
def nonlinear_solver_id(init_image,hp_nn,data):

  x = init_image
  loss = lambda pp_image:screen_poisson_objective(pp_image,hp_nn,data)
  optim_cond = lambda x: tree_sum_red(tree_squared(jax.grad(loss)(x)))
  def loop_body(args):
    x,count, gn_opt_err, gn_loss,linear_opt_err = args
    d, linea_opt = linear_solver_id(None,x,hp_nn,data)
    x = tree_sum(x,d)

    linear_opt_err = linear_opt_err.at[count.astype(int)].set(linea_opt)
    gn_opt_err = gn_opt_err.at[count.astype(int)].set(optim_cond(x))
    gn_loss = gn_loss.at[count.astype(int)].set(tree_sum_red(screen_poisson_objective(x,hp_nn,data)))
    count += 1
    return (x,count, gn_opt_err, gn_loss,linear_opt_err)

  loop_count = 3
  # x,count, gn_opt_err, gn_loss, linear_opt_err,linea_opt= jax.lax.while_loop(lambda x:optim_cond(x[0]) >= 1e-10,loop_body,(x,0.0,-jnp.ones(200),-jnp.ones(200),-jnp.ones(200))) 
  x,count, gn_opt_err, gn_loss,linear_opt_err = jax.lax.fori_loop(0,loop_count,lambda i,val:loop_body(val),(x,0.0,-jnp.ones(loop_count),-jnp.ones(loop_count),-jnp.ones(loop_count))) 
  return x,(count,gn_opt_err, gn_loss,linear_opt_err)
################ linear and nonlinear solvers end #################

################ outer model start ###############################
@jax.jit
def outer_objective_id(hp_nn,init_inner,data):
    """Validation loss."""
    gt = data['ambient']
    f = lambda hp_nn: nonlinear_solver_id(init_inner, hp_nn,data)
    x, aux = f(hp_nn)
    f_v = ((x['smooth_image'] - gt) ** 2).sum()
    return f_v,(x,*aux)

def testing():
  logger = cvgviz.logger('./logger','filesystem','autodiff','flash_no_flash')
  rng = jax.random.PRNGKey(1)
  rng, init_rng = jax.random.split(rng)
  flash = cvgim.imread('/home/mohammad/Projects/optimizer/DifferentiableSolver/testImages/Flash.jpg')
  noflash = cvgim.imread('/home/mohammad/Projects/optimizer/DifferentiableSolver/testImages/NoFlash.jpg')
  flash = flash[None,...]
  noflash = noflash[None,...]
  
  
  inpt = {'scalemap':noflash,'smooth_image':flash}
  params = Conv3features().init(init_rng, flash)['params']
  data = {'lambda1':1.,'lambda2':1.,'lambda3':100000.,'flash':flash,'ambient': noflash}
  valloss, (x,count, gn_opt_err, gn_loss, lin_opt) = outer_objective_id(params,inpt,data)
  imshow = jnp.concatenate((x['scalemap'],x['smooth_image'],flash,noflash),axis=2)
  imshow = jnp.concatenate(imshow,axis=1)
  imshow = jnp.clip(imshow,0,1)
  logger.addImage(np.array(imshow).transpose(2,0,1),'Image')
  logger.takeStep()
  print('hi')

def training():
  import time
  start = time.time()

  tf.config.experimental.set_visible_devices([], 'GPU')
  batch_size = 1
  key4 = jax.random.PRNGKey(45)
  gt_image = cvgim.imread('~/Projects/cvgutils/tests/testimages/wood_texture.jpg')
  gt_image = cvgim.resize(gt_image,scale=0.10) * 2
  gt_image = jnp.repeat(gt_image[None,...],batch_size,0)
  logger = cvgviz.logger('./logger','tb','autodiff','autodiff_conv_softplus_2layer_normalized')
  noise = jax.device_put(jax.random.normal(key4,gt_image.shape)) * 0.3
  noisy_image = jax.device_put(jnp.clip(gt_image + noise,0,1))
  init_inpt = jax.device_put(jax.random.normal(key4,gt_image.shape))
  im_gt = jax.device_put(jnp.array(gt_image))
  data = {'lambda1':1.,'lambda2':1.,'lambda3':1.,'flash':noisy_image,'ambient': im_gt}

  rng = jax.random.PRNGKey(1)
  rng, init_rng = jax.random.split(rng)
  testim = jax.device_put(jax.random.uniform(rng,gt_image.shape))
  params = Conv3features().init(init_rng, testim)['params']

  lr = 0.0001
  
  solver = OptaxSolver(fun=outer_objective_id, opt=optax.adam(lr),implicit_diff=True,has_aux=True)
  state = solver.init_state(params)
  for i in tqdm.trange(10000):
    params, state = solver.update(params, state,init_inner=init_inpt,data=data)
    x,count, gn_opt_err, gn_loss, lin_opt = state.aux
    
    end = time.time()
    print('time: ',end - start)
    print('loss ',state.value, ' gn iteration count ', count)
    logger.addScalar(state.value / jnp.prod(jnp.array(gt_image.shape)),'loss_GD')

    if(i%10 == 0):
      imshow = jnp.concatenate((x,noisy_image,im_gt),axis=2)
      imshow = jnp.concatenate(imshow,axis=1)
      imshow = jnp.clip(imshow,0,1)
      logger.addImage(np.array(imshow).transpose(2,0,1),'Image')
    logger.takeStep()
################ outer model end #################################

# training()
testing()