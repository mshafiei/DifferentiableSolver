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
from jax.tree_util import tree_map, tree_reduce, tree_multimap
import jax.scipy as jsp
eps=0.004
eps_phi = 1e-3
# eta = 0.01
eta2 = 0.001#eta**2
alpha = 0.9
tree_scalar_prod = lambda x,y: tree_map(lambda z:x*z,y)
tree_sum_red = lambda x: tree_reduce(lambda y,z: y+z, x)
tree_diff_squared_sum = lambda x,y: tree_multimap(lambda z,w: ((z-w) ** 2).sum(),x,y)
tree_sum = lambda x,y: tree_multimap(lambda z,w: z+w,x,y)
tree_squared = lambda x: tree_map(lambda y: (y ** 2).sum(),x)


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

def safe_division(nom,denom,eps):
  return nom / (sign(denom) * jnp.maximum(jnp.abs(denom), eps))


def gauss(im):
  return im
  # x = jnp.linspace(-3, 3, 7)
  # window = jsp.stats.norm.pdf(x) * jsp.stats.norm.pdf(x[:, None])
  # return jsp.signal.convolve(im, window[None,:,:,None].repeat(3,2), mode='same')

def dx(x):
  return gauss(x)[...,:,1:,:] - gauss(x)[...,:,:-1,:]

def dy(x):
  return gauss(x)[...,1:,:,:] - gauss(x)[...,:-1,:,:]

def sign(x):
  return jnp.sign(x) + jnp.where(x==0,1.,0.)

def compute_weights(pp_image, data):
  flash = data['flash_image']
  ambient = data['ambient_image']

  eps = 0.004
  smooth_image = jax.lax.stop_gradient(pp_image['smooth_image'])
  scalemap = jax.lax.stop_gradient(pp_image['scalemap_image'])

  dy_flash, dx_flash,dy_ambient, dx_ambient, dy_smooth, dx_smooth, dy_scalemap, dx_scalemap = dy(flash), dx(flash), dy(ambient), dx(ambient), dy(smooth_image), dx(smooth_image), dy(scalemap), dx(scalemap)
  p_x = safe_division(1.,jnp.abs(dx_flash), eps)
  p_y = safe_division(1.,jnp.abs(dy_flash), eps)
  
  aux_x = dx_ambient * p_x 
  aux_y = dy_ambient * p_y 
  data['weight1_x'] = jnp.abs(scalemap[...,:,1:,:] - dx_smooth * p_x)
  data['weight1_y'] = jnp.abs(scalemap[...,1:,:,:] - dy_smooth * p_y)

  data['weight2'] = jnp.abs(smooth_image - ambient)
  data['aux'] = jnp.zeros_like(ambient)
  data['aux'] = data['aux'].at[...,1:,:,:].add(aux_y)
  data['aux'] = data['aux'].at[...,:,1:,:].add(aux_x)
  data['aux'] /= 2.
  return data


def stencil_residual(pp_image, hp_nn, data):
  ambient = data['ambient_image']
  flash = data['flash_image']
  l1 = data['lambda1'] ** 0.5
  l2 = data['lambda2'] ** 0.5
  l3 = data['lambda3'] ** 0.5
  w2 = data['weight2'] ** 0.5
  w1x = data['weight1_x'] ** 0.5
  w1y = data['weight1_y'] ** 0.5



  eps = 0.004
  smooth_image = pp_image['smooth_image']
  scalemap = pp_image['scalemap_image']
  """Objective function."""
  avg_weight = (1. / 4.) ** 0.5 *  (1. / smooth_image.reshape(-1).shape[0] ** 0.5)
  r1 =  smooth_image - ambient
  
  dy_flash, dx_flash,dy_ambient, dx_ambient, dy_smooth, dx_smooth, dy_scalemap, dx_scalemap = dy(flash), dx(flash), dy(ambient), dx(ambient), dy(smooth_image), dx(smooth_image), dy(scalemap), dx(scalemap)
  p_x = safe_division(1.,jnp.abs(dx_flash), eps)
  p_y = safe_division(1.,jnp.abs(dy_flash), eps)
  e1_x = (scalemap[...,:,1:,:] - p_x * dx_smooth)
  e1_y = (scalemap[...,1:,:,:] - p_y * dy_smooth)
  
  # mu1,mu2,v1x,v1y,v2x,v2y,sx,sy
  eta = 0.001
  eta2 = eta**2
  g2 = jnp.zeros_like(flash)
  g2 = g2.at[...,:,1:,:].add(dx_flash **2)
  g2 = g2.at[...,1:,:,:].add(dy_flash **2)
  g1 = g2 ** 0.5
  mu1 = eta2 / (g2+2*eta2)
  mu2 = (g2 + eta2) / (g2 + 2*eta2)
  v1x = dx_flash / (g1[...,:,1:,:] + eps)
  v1y = dy_flash / (g1[...,1:,:,:] + eps)
  v2x = -dy_flash / (g1[...,1:,:,:] + eps)
  v2y = dx_flash / (g1[...,:,1:,:] + eps)

  e3_1 = mu1[...,:,1:,:]**0.5 * (v1x * dx_scalemap)
  e3_2 = mu1[...,1:,:,:]**0.5 * (v1y * dy_scalemap)
  e3_3 = mu2[...,1:,1:,:]**0.5 * (v2x[...,:,1:,:] * dx_scalemap[...,1:,:,:])
  e3_4 = mu2[...,1:,1:,:]**0.5 * (v2y[...,1:,:,:] * dy_scalemap[...,:,1:,:])

  # unet_out = Conv3features().apply({'params': hp_nn}, pp_image)
  out = jnp.concatenate((
                         (l1 * w1x * e1_x).reshape(-1),
                         (l1 * w1y * e1_y).reshape(-1),
                         (l2 * w2 * r1).reshape(-1),
                         (l3 * e3_1).reshape(-1),
                         (l3 * e3_2).reshape(-1),
                         (l3 * e3_3).reshape(-1),
                         (l3 * e3_4).reshape(-1)
                         ),
                         axis=0)

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
  def M(x):
    smooth_image = x['smooth_image']
    scalemap = x['scalemap_image']
    flash = data['flash_image']
    eps = 0.004
    dy_flash = dy(flash)
    dx_flash = dx(flash)
    # # dy_ambient = jnp.zeros_like(ambient)
    # # dx_ambient = jnp.zeros_like(ambient)
    # dy_smooth = jnp.zeros_like(smooth_image)
    # dx_smooth = jnp.zeros_like(smooth_image)
    # dy_scalemap = jnp.zeros_like(scalemap)
    # dx_scalemap = jnp.zeros_like(scalemap)

    # dy_flash = dy_flash.at[...,1:,:,:].set(flash[...,1:,:,:] - flash[...,:-1,:,:])
    # dx_flash = dx_flash.at[...,:,1:,:].set(flash[...,:,1:,:] - flash[...,:,:-1,:])
    # # dy_ambient = dy_ambient.at[...,1:,:,:].set(ambient[...,1:,:,:] - ambient[...,:-1,:,:])
    # # dx_ambient = dx_ambient.at[...,:,1:,:].set(ambient[...,:,1:,:] - ambient[...,:,:-1,:])
    # dy_smooth = dy_smooth.at[...,1:,:,:].set(smooth_image[...,1:,:,:] - smooth_image[...,:-1,:,:])
    # dx_smooth = dx_smooth.at[...,:,1:,:].set(smooth_image[...,:,1:,:] - smooth_image[...,:,:-1,:])
    # s_x = dx_smooth / jnp.maximum(jnp.sign(dx_flash) * jnp.abs(dx_flash), eps)
    # s_y = dy_smooth / jnp.maximum(jnp.sign(dy_flash) * jnp.abs(dy_flash), eps)
    # e1_x = (scalemap - s_x)
    # e1_y = (scalemap - s_y)
    # dy_scalemap = dy_scalemap.at[...,1:,:,:].set(scalemap[...,1:,:,:] - scalemap[...,:-1,:,:])
    # dx_scalemap = dx_scalemap.at[...,:,1:,:].set(scalemap[...,:,1:,:] - scalemap[...,:,:-1,:])
    p_x = 1. / jnp.maximum(jnp.sign(dx_flash) * jnp.abs(dx_flash), eps)
    p_y = 1. / jnp.maximum(jnp.sign(dy_flash) * jnp.abs(dy_flash), eps)
    
    # wps_x = jnp.zeros_like(smooth_image)
    # wps_y = jnp.zeros_like(smooth_image)
    wps_x = data['weight1_x'] * p_x**2
    wps_y = data['weight1_y'] * p_y**2

    # pre_smooth_image = jnp.zeros_like(smooth_image)
    # pre_smooth_image = data['lambda2'] * data['weight2']

    # pre_smooth_image = pre_smooth_image.at[...,:,1:,:].add(data['lambda1'] *  wps_x)
    # pre_smooth_image = pre_smooth_image.at[...,:,1:-1,:].add(data['lambda1'] * wps_x[...,:,1:,:])
    # pre_smooth_image = pre_smooth_image.at[...,1:,:,:].add(data['lambda1'] *  wps_y)
    # pre_smooth_image = pre_smooth_image.at[...,1:-1,:,:].add(data['lambda1'] * wps_y[...,1:,:,:])
    # # denom = jnp.sign(pre_smooth_image) * jnp.maximum(jnp.abs(pre_smooth_image),0.001)
    pre_smooth_image = data['lambda1'] +data['lambda2'] 
    pre_smooth_image = 1.#safe_division(1.,pre_smooth_image,eps)

    pre_scalemap = jnp.zeros_like(scalemap)
    # pre_scalemap = pre_scalemap.at[...,:,1:,:].add(data['lambda1'] * data['weight1_x'])
    # pre_scalemap = pre_scalemap.at[...,1:,:,:].add(data['lambda1'] * data['weight1_y'])
    # pre_scalemap = pre_scalemap.at[...,:,1:,:].add(data['lambda3'])
    # pre_scalemap = pre_scalemap.at[...,:,:-1,:].add( data['lambda3'])
    # pre_scalemap = pre_scalemap.at[...,1:,:,:].add( data['lambda3'])
    # pre_scalemap = pre_scalemap.at[...,:-1,:,:].add( data['lambda3'])
    # denom = jnp.sign(pre_scalemap) * jnp.maximum(jnp.abs(pre_scalemap),0.001)
    pre_scalemap = data['lambda1'] + data['lambda3']
    pre_scalemap = 1.#safe_division(1.,pre_scalemap,eps)
    return {'smooth_image':pre_smooth_image * smooth_image,'scalemap_image':pre_scalemap*scalemap}

  def Ax(pp_image):
    jtd = jax.jvp(f,(x,),(pp_image,))[1]
    return jax.vjp(f,x)[1](jtd)[0]
  def jtf(x):
    return jax.vjp(f,x)[1](f(x))[0]
  jtx = jtf(x)
  
  d = linear_solve.solve_cg(matvec=Ax,
                          b=tree_map(lambda z:-1*z,jtx),
                          init=d,M=M,
                          maxiter=100)
  aux = tree_reduce(lambda x,y:x+y,tree_multimap(lambda x,y:((x+y)**2).sum(),Ax(d),jtx))
  return d, aux

# @implicit_diff.custom_root(jax.grad(screen_poisson_objective),has_aux=True)
def nonlinear_solver_id(init_image,hp_nn,data):

  x = init_image
  loss = lambda pp_image:screen_poisson_objective(pp_image,hp_nn,data)
  
  optim_cond = lambda w: tree_reduce(lambda x,y:x+y,tree_map(lambda x: (x**2).sum(),jax.grad(loss)(w)))
  def loop_body(args):
    x,count, gn_opt_err, gn_loss,linear_opt_err,data = args
    data = compute_weights(x, data)
    d, linea_opt = linear_solver_id(None,x,hp_nn,data)
    x = tree_multimap(lambda x,y:x+0.4*y,x,d)

    linear_opt_err = linear_opt_err.at[count.astype(int)].set(linea_opt)
    gn_opt_err = gn_opt_err.at[count.astype(int)].set(optim_cond(x))
    gn_loss = gn_loss.at[count.astype(int)].set(screen_poisson_objective(x,hp_nn,data))
    count += 1
    return (x,count, gn_opt_err, gn_loss,linear_opt_err,data)

  loop_count = 20
  # x,count, gn_opt_err, gn_loss, linear_opt_err,linea_opt= jax.lax.while_loop(lambda x:optim_cond(x[0]) >= 1e-10,loop_body,(x,0.0,-jnp.ones(200),-jnp.ones(200),-jnp.ones(200))) 
  x,count, gn_opt_err, gn_loss,linear_opt_err,_ = jax.lax.fori_loop(0,loop_count,lambda i,val:loop_body(val),(x,jnp.zeros(1),-jnp.ones(loop_count),-jnp.ones(loop_count),-jnp.ones(loop_count),data)) 
  return x,{'gn_opt_err':gn_opt_err, 'gn_loss':gn_loss,'linear_opt_err':linear_opt_err}
################ linear and nonlinear solvers end #################

def resize_params(params,sizes):
  ret = {}
  for key, val in params.items():
    if('_image' in key):
        ret[key] = jax.image.resize(val,sizes,'trilinear')
    else:
      ret[key] = val
  return ret


def hierarchical_id(init_inner,hp_nn,data):
  # n_hierarchies = data['n_hierarchies']
  def hierarchy(i,i_x,data,sizes,auxs):
    i_data = resize_params(data,sizes)
    i_x = resize_params(i_x,sizes)
    i_data = compute_weights(i_x, i_data)
    i_x,log_dict = nonlinear_solver_id(i_x,hp_nn,i_data)
    auxs['aux_%02i'%i] = log_dict
    return i_x,auxs

  i_x, auxs = [init_inner, {}]
  def size_scale(sizes,i):
    return [sizes[0], sizes[1]//2**i, sizes[2]//2**i, sizes[3]]
  # i_x, auxs, i_data = hierarchy(3,args)
  # i_x, auxs = hierarchy(0,i_x, data,size_scale(data['flash_image'].shape,3),auxs)
  # i_x, auxs = hierarchy(1,i_x, data,size_scale(data['flash_image'].shape,2),auxs)
  # i_x, auxs = hierarchy(2,i_x, data,size_scale(data['flash_image'].shape,1),auxs)
  i_x, auxs = hierarchy(3,i_x, data,size_scale(data['flash_image'].shape,0),auxs)
    
  return i_x, auxs

################ outer model start ###############################
@jax.jit
def outer_objective_id(hp_nn,init_inner,data):
    """Validation loss."""
    gt = data['ambient_image']
    f = lambda hp_nn: hierarchical_id(init_inner, hp_nn,data)
    x, aux = f(hp_nn)
    f_v = ((x['smooth_image']-gt) ** 2).sum()
    # f_v = ((x - gt) ** 2).sum()
    return f_v,(x,aux)


def testing():
  logger = cvgviz.logger('./logger','filesystem','autodiff','flash_no_flash')
  rng = jax.random.PRNGKey(1)
  rng, init_rng = jax.random.split(rng)
  flash = cvgim.imread('/home/mohammad/Projects/optimizer/DifferentiableSolver/flash_no_flash/cave_flash_1.png')
  noflash = cvgim.imread('/home/mohammad/Projects/optimizer/DifferentiableSolver/flash_no_flash/cave_noflash_1.png')
  flash = flash[None,...]
  noflash = noflash[None,...]
  smooth_image_init = jax.device_put(jax.random.uniform(rng,noflash.shape))
  scalemap_init = jax.device_put(jax.random.uniform(rng,noflash.shape))
  
  # lambda = 3 closeness
  # beta = 0.5 smoothness

  inpt = {'scalemap_image':scalemap_init,'smooth_image':smooth_image_init}
  data = {'lambda1':1,'lambda2':3,'lambda3':0.5,'ambient_image':noflash, 'flash_image':flash}
  # inpt = {'smooth_image':noflash}
  params = Conv3features().init(init_rng, noflash)['params']
  data = compute_weights(inpt, data)
  # data = {'lambda1':1.,'lambda2':1.,'lambda3':100000.,'flash':flash,'ambient': noflash}
  valloss, (x,aux) = outer_objective_id(params,inpt,data)
  # [logger.addScalar(i ,'loss_GN') for i in gn_loss]
  # [logger.addScalar(i ,'optimality_GN') for i in gn_opt_err]
  # [logger.addScalar(i ,'optimality_CG') for i in lin_opt]
  s_min, s_max = x['scalemap_image'].min(),x['scalemap_image'].max()
  s_normalized = (x['scalemap_image'] - s_min) / (s_max - s_min)
  s_normalized = jnp.clip(s_normalized ** (2.2 * 2)*6,0,1)
  imshow = jnp.concatenate((s_normalized,x['smooth_image'],noflash,flash),axis=2)
  imshow = jnp.concatenate(imshow,axis=1)
  imshow = jnp.clip(imshow,0,1)
  logger.addImage(np.array(imshow).transpose(2,0,1),'Image')
  logger.takeStep()
  print(aux)
  print('hi')

def hyper_optimization():
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
  data = {'inpt':noisy_image,'gt': im_gt}

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

# hyper_optimization()
testing()