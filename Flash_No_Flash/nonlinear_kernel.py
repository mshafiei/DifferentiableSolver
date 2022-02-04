import argparse
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
import os
import deepfnf_utils.utils as ut
import deepfnf_utils.tf_utils as tfu
from deepfnf_utils.dataset import Dataset
import pickle
import time
import natsort
tf.compat.v1.disable_eager_execution()

################ inner loop model end ############################
class Conv3features(nn.Module):

  def setup(self):
    self.straight1       = nn.Conv(12,(3,3),strides=(1,1),use_bias=True)
    self.straight2       = nn.Conv(64,(3,3),strides=(1,1),use_bias=True)
    self.straight3       = nn.Conv(3,(3,3),strides=(1,1),use_bias=True)
    self.groupnorm1      = nn.GroupNorm(1)
    self.groupnorm2      = nn.GroupNorm(8)
    
  def __call__(self,x):
    l1 = self.groupnorm1(nn.softplus(self.straight1(x)))
    l2 = self.groupnorm2(nn.softplus(self.straight2(l1)))
    return nn.tanh(self.straight3(l2))

def stencil_residual(pp_image, hp_nn, data):
  
  """Objective function."""
  
  r1 =  pp_image - data['init']
  unet_out = Conv3features().apply({'params': hp_nn}, data['net_inpt'])
  r2 = pp_image - unet_out
  out = jnp.concatenate(( r1.reshape(-1), r2.reshape(-1)),axis=0)
  return out #* data['avg_weight']
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
  d = linear_solve.solve_cg(matvec=Ax,
                          b=-jtf(x),
                          init=d,
                          maxiter=100)
  aux = ((Ax(d) +jtf(x)) ** 2).sum()
  return d, aux

# @implicit_diff.custom_root(jax.grad(screen_poisson_objective),has_aux=True)
def nonlinear_solver_id(init_image,hp_nn,data):

  x = init_image
  loss = lambda pp_image:screen_poisson_objective(pp_image,hp_nn,data)
  optim_cond = lambda x: (jax.grad(loss)(x) ** 2).sum()
  def loop_body(args):
    x,count, gn_opt_err, gn_loss,linear_opt_err = args
    d, linea_opt = linear_solver_id(None,x,hp_nn,data)
    x += 1.0 * d

    linear_opt_err = linear_opt_err.at[count.astype(int)].set(linea_opt)
    gn_opt_err = gn_opt_err.at[count.astype(int)].set(optim_cond(x))
    gn_loss = gn_loss.at[count.astype(int)].set(screen_poisson_objective(x,hp_nn,data))
    count += 1
    return (x,count, gn_opt_err, gn_loss,linear_opt_err)

  loop_count = 3
  # x,count, gn_opt_err, gn_loss, linear_opt_err,linea_opt= jax.lax.while_loop(lambda x:optim_cond(x[0]) >= 1e-10,loop_body,(x,0.0,-jnp.ones(200),-jnp.ones(200),-jnp.ones(200))) 
  x,count, gn_opt_err, gn_loss,linear_opt_err = jax.lax.fori_loop(0,loop_count,lambda i,val:loop_body(val),(x,0.0,-jnp.ones(loop_count),-jnp.ones(loop_count),-jnp.ones(loop_count))) 
  # args = (x,jnp.array([0.0]),-jnp.ones(loop_count),-jnp.ones(loop_count),-jnp.ones(loop_count))
  # for _ in range(loop_count):
  #   args = loop_body(args)
  # x,count, gn_opt_err, gn_loss,linear_opt_err = args
  return x,(count,gn_opt_err, gn_loss,linear_opt_err)
################ linear and nonlinear solvers end #################

def direct_solver_image_interpolate(init_inner,hp_nn,data):
  out = Conv3features().apply({'params': hp_nn}, data['net_inpt'])
  out =  0.5 * (data['init'] + out)
  return out, ([0],[0],[0],[0])
################ outer model start ###############################
# @jax.jit
def outer_objective_id(hp_nn,init_inner,data):
  """Validation loss."""
  # f = lambda hp_nn: nonlinear_solver_id(init_inner, hp_nn,data)
  f = lambda hp_nn: direct_solver_image_interpolate(init_inner, hp_nn,data)
  x, aux = f(hp_nn)
  # gt = data['gt']
  # init = data['init']
  x = tfu.camera_to_rgb_jax(
      x/data['alpha'], data['color_matrix'], data['adapt_matrix'])
  gt = tfu.camera_to_rgb_jax(
      data['gt'],
      data['color_matrix'], data['adapt_matrix'])
  init = tfu.camera_to_rgb_jax(
      init_inner/data['alpha'],
      data['color_matrix'], data['adapt_matrix'])

  l2 = ((x - gt) ** 2).sum()
  psnr = tfu.get_psnr_jax(x,gt)
  return l2,(x,init,gt,psnr,*aux)

def read_data(example):
  alpha = example['alpha'][:, None, None, None]
  dimmed_ambient, _ = tfu.dim_image(
      example['ambient'], alpha=alpha)
  dimmed_warped_ambient, _ = tfu.dim_image(
      example['warped_ambient'], alpha=alpha)
  # Make the flash brighter by increasing the brightness of the
  # flash-only image.
  flash = example['flash_only'] * ut.FLASH_STRENGTH + dimmed_ambient
  warped_flash = example['warped_flash_only'] * \
      ut.FLASH_STRENGTH + dimmed_warped_ambient
  sig_read = example['sig_read'][:, None, None, None]
  sig_shot = example['sig_shot'][:, None, None, None]
  noisy_ambient, _, _ = tfu.add_read_shot_noise(
      dimmed_ambient, sig_read=sig_read, sig_shot=sig_shot)
  # noisy_flash, _, _ = tfu.add_read_shot_noise(
  #     warped_flash, sig_read=sig_read, sig_shot=sig_shot)
  # noisy = tf.concat([noisy_ambient, noisy_flash], axis=-1)
  # noise_std = tfu.estimate_std(noisy, sig_read, sig_shot)
  # return {
  # 'init':jnp.array(noisy_ambient.numpy()),
  # 'net_inpt':jnp.array(tf.concat([noisy, noise_std], axis=-1).numpy()),
  # 'gt':jnp.array(example['ambient'].numpy())}
  # avg_weight = tf.pow((1 / 2) *  (1 / tf.reduce_prod(tf.shape(noisy)[1:])),0.5)
  # res = {
  # 'init':         noisy_ambient,
  # 'net_inpt':     tf.concat([noisy, noise_std], axis=-1),
  # 'gt':           example['ambient'],
  # 'color_matrix': example['color_matrix'],
  # 'adapt_matrix': example['adapt_matrix'],
  # 'alpha':        alpha,
  # 'avg_weight':   avg_weight
  # }
  return noisy_ambient


def hyper_optimization(dataset,sess,opts):
  

  #prepare input array
  
  # tf.config.experimental.set_visible_devices([], 'GPU')
  logger = cvgviz.logger('./logger/Flash_No_Flash','tb','Flash_No_Flash','interpolation')


  with tf.device('/gpu:%d' % 0):
    example = dataset.batches[0] 
    example = read_data(example)
  dataset.swap_train(sess)

  rng = jax.random.PRNGKey(1)
  rng, init_rng = jax.random.split(rng)
  testim = jax.device_put(sess.run(example['net_inpt']))
  params = Conv3features().init(init_rng, testim)['params']

  lr = opts.lr 
  max_iter = opts.max_iter
  viz_freq = opts.viz_freq
  save_param_freq = opts.save_param_freq
  solver = OptaxSolver(fun=outer_objective_id, opt=optax.adam(lr),implicit_diff=True,has_aux=True)
  state = solver.init_state(params)
  out = logger.load_params()
  if(out is not None):
    params = out['params']
    state = out['state']
    start_idx = out['idx']
  else:
    start_idx = -1

  for i in tqdm.trange(int(start_idx)+1,int(max_iter)):
    start = time.time()
    with tf.device('/gpu:0'):
      mode = 'train'
      if(i % viz_freq == 0):
        dataset.swap_val(sess)
        mode = 'val'
      elif((i-1) % viz_freq == 0):
        dataset.swap_train(sess)
      data = sess.run(example)
      data = {k:jnp.array(v) for k,v in data.items()}

      if(i%viz_freq != 0):
        params, state = solver.update(params, state,init_inner=data['init'],data=data)
        x,init,gt, psnr,count, gn_opt_err, gn_loss, lin_opt = state.aux
        l2 = state.value
      else:
        l2, (x,init,gt, psnr,count, gn_opt_err, gn_loss, lin_opt) = outer_objective_id(jax.lax.stop_gradient(params),init_inner=data['init'],data=data)
    end = time.time()
    print('Iteration time: ',end - start)
    print(mode, ' l2 loss ',l2/ jnp.prod(jnp.array(data['gt'].shape)), ' psnr ', psnr)
    logger.addScalar(l2 / jnp.prod(jnp.array(data['gt'].shape)),'loss_GD',mode=mode)
    logger.addScalar(psnr,'PSNR',mode=mode)

    if(i%viz_freq == 0 or (i-1)%viz_freq == 0):
      imshow = jnp.concatenate((x,init,gt),axis=2)
      imshow = jnp.concatenate(imshow,axis=1)
      imshow = jnp.clip(imshow,0,1)
      logger.addImage(np.array(imshow).transpose(2,0,1),'Image',mode=mode)
    for lopt,gopt,gloss in zip(lin_opt,gn_opt_err,gn_loss):
      logger.addScalar(lopt,'linear_optimality',mode=mode)
      logger.addScalar(gopt,'gn_optimality',mode=mode)
      logger.addScalar(gloss,'gn_loss',mode=mode)
      logger.takeStep()
    if(i % save_param_freq == 0):
      logger.save_params(params,state,i)
################ outer model end #################################


def create_dataset(opts):
  
  TLIST = '/home/mohammad/Projects/optimizer/DifferentiableSolver/data/train.txt'
  VPATH = '/home/mohammad/Projects/optimizer/DifferentiableSolver/data/valset/'

  BSZ = opts.batch_size
  IMSZ = opts.image_size
  displacement = opts.displacement
  LR = 1e-4
  DROP = (1.1e6, 1.25e6) # Learning rate drop
  MAXITER = 1.5e6

  VALFREQ = 2e1
  SAVEFREQ = 5e4
  ngpus = 1

  
  sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
      allow_soft_placement=True))
  sess.run(tf.compat.v1.global_variables_initializer())
  with tf.device('/gpu:%d' % 0):
    dataset = Dataset(TLIST, VPATH, bsz=BSZ, psz=IMSZ,
                          ngpus=ngpus, nthreads=4 * ngpus,
                          jitter=displacement, min_scale=opts.min_scale, max_scale=opts.max_scale, theta=opts.max_rotate)

  dataset.init_handles(sess)
  dataset.swap_train(sess)
  return dataset,sess
         
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=1, help='Image count in a batch')
parser.add_argument('--image_size', default=448, help='Width and height of an image')
parser.add_argument('--displacement', default=0, help='Random shift in pixels')
parser.add_argument('--min_scale', default=1e-10, help='Random shift in pixels')
parser.add_argument('--max_scale', default=1e-10, help='Random shift in pixels')
parser.add_argument('--max_rotate', default=0, help='Maximum rotation')
parser.add_argument('--lr', default=1e-4, help='Maximum rotation')
parser.add_argument('--max_iter', default=1e6, help='Maximum rotation')
parser.add_argument('--viz_freq', default=20, help='Maximum rotation')
parser.add_argument('--save_param_freq', default=20, help='Maximum rotation')
opts = parser.parse_args()


dataset, sess = create_dataset(opts)
hyper_optimization(dataset,sess,opts)
