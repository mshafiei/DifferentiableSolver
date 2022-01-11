import jax
from jax import tree_util
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
import json
from jax.lax import stop_gradient as nograd

eps=0.004
eps_phi = 1e-3
# eta = 0.01
eta2 = 0.001#eta**2
alpha = 0.9

def printDict(d):
  print(json.dumps(d,sort_keys=True, indent=4))

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

def gauss(im):
  return im

def dx(x):
  y = jnp.pad(x,((0,0),(0,0),(1,0),(0,0)),mode='symmetric')
  return gauss(y)[...,:,1:,:] - gauss(y)[...,:,:-1,:]

def dy(x):
  y = jnp.pad(x,((0,0),(1,0),(0,0),(0,0)),mode='symmetric')
  return gauss(y)[...,1:,:,:] - gauss(y)[...,:-1,:,:]

def dxt(x):
  x = jnp.pad(x,((0,0),(0,0),(0,1),(0,0)),mode='symmetric')
  return gauss(x)[...,:,:-1,:] - gauss(x)[...,:,1:,:]

def dyt(x):
  x = jnp.pad(x,((0,0),(0,1),(0,0),(0,0)),mode='symmetric')
  return gauss(x)[...,:-1,:,:] - gauss(x)[...,1:,:,:]

def sign(x):
  return jnp.sign(x) + jnp.where(x==0,1.,0.)

def safe_division(nom,denom,eps):
  return nom / (sign(denom) * jnp.maximum(jnp.abs(denom), eps))

def phi(nom,denom,eps):
  return nom / (jnp.abs(denom) ** (2 - alpha) + eps)

# def compute_weights(pp_image, data):
#   flash = data['flash_image']
#   ambient = data['ambient_image']
#   smooth_image = jax.lax.stop_gradient(pp_image['smooth_image'])
#   scalemap_image = jax.lax.stop_gradient(pp_image['scalemap_image'])
  

#   dy_flash, dx_flash, i_y, i_x = dy(flash), dx(flash), dy(smooth_image), dx(smooth_image)
#   data['weight1_inv_x'] = jnp.abs(scalemap_image[:,:,1:,:] - safe_division(i_x, dx_flash,eps))
#   data['weight1_inv_y'] = jnp.abs(scalemap_image[:,1:,:,:] - safe_division(i_y, dy_flash,eps))
#   data['weight2_inv'] =   jnp.abs(smooth_image - ambient)

#   return data


def stencil_residual(pp_image, hp_nn, data):
  e = compute_terms(pp_image, hp_nn, data)
  return jnp.concatenate(e, axis=0)

def compute_terms(pp_image, hp_nn, data):
  global gmask
  I0 = data['ambient_image']
  G = data['flash_image']
  l1 = data['lambda1'] ** 0.5
  l2 = data['lambda2'] ** 0.5
  l3 = data['lambda3'] ** 0.5
  # w2 = data['weight2_inv'] ** 0.5
  # w1x = data['weight1_inv_x'] ** 0.5
  # w1y = data['weight1_inv_y'] ** 0.5
  optimize_smooth = jax.lax.stop_gradient(data['optimize_smooth'])
  
  Gy, Gx = dy(G), dx(G)
  Gx2y2 = Gx**2 + Gy**2
  Gx1y1 = Gx2y2 ** 0.5
  if(optimize_smooth):
    I = pp_image['smooth_image']
    S = jax.lax.stop_gradient(pp_image['scalemap_image'])
    
    S = S.at[gmask].set(0)
  else:
    I = jax.lax.stop_gradient(pp_image['smooth_image'])
    S = pp_image['scalemap_image']
    

  Sx,Sy,Iy, Ix = dx(S),dy(S), dy(I), dx(I)
  sig1, sig2, Vx,Vy = eta2 / (Gx2y2 + 2*eta2), (Gx2y2 + eta2) / (Gx2y2 + 2*eta2), safe_division(Gx,Gx1y1,eps), safe_division(Gy,Gx1y1,eps)
  
  """ weights """
  denom = jnp.abs(nograd(S) - safe_division(nograd(Ix),Gx,eps))
  w1x = l1 * safe_division(1.,denom,eps) ** 0.5
  denom = jnp.abs(nograd(S) - safe_division(nograd(Iy),Gy,eps))
  w1y = l1 * safe_division(1.,denom,eps) ** 0.5
  denom = jnp.abs(nograd(I) - I0)
  w2 = l2 * safe_division(1.,denom,eps) ** 0.5
  """Objective function."""
  e1_x = w1x * (S - safe_division(Ix,Gx,eps))
  e1_y = w1y * (S - safe_division(Iy,Gy,eps))
  e2   = w2  * (I - I0)
  #eq. 13
  e3_1 = l3 * sig1 ** 0.5 * (Vx * Sx + Vy * Sy)
  e3_2 = l3 * sig2 ** 0.5 * (Vy * Sx - Vx * Sy)
  e1_term = jnp.concatenate((phi(e1_x,w1x,eps_phi).reshape(-1),
                             phi(e1_y,w1y,eps_phi).reshape(-1)),axis=0)
  e2_term = phi(e2,w2,eps_phi).reshape(-1)
  e3_term = jnp.concatenate((e3_1.reshape(-1),e3_2.reshape(-1)),axis=0)
  
  e = (e1_term, e2_term,e3_term)
  avg_weight = (1. / (nograd(I).reshape(-1).shape[0]) ** 0.5)
  e = [avg_weight * i for i in e]
  return e
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
    scalemap_image = x['scalemap_image']
    flash = data['flash_image'] 
    dy_flash, dx_flash = dy(flash), dx(flash)
    # pre_smooth_image = jnp.zeros_like(smooth_image)
    # pre_smooth_image = 1.
    p_x = safe_division(1.,dx_flash,eps)
    p_y = safe_division(1.,dy_flash,eps)
    
    pre_smooth_image = data['lambda2'] *  phi(1.,data['weight2_inv'],eps_phi)
    pre_smooth_image = pre_smooth_image.at[:,:,1:,:].add(data['lambda1'] *  p_x**2. * phi(1.,data['weight1_inv_x'],eps_phi)**2.)
    pre_smooth_image = pre_smooth_image.at[:,:,1:-1,:].add(data['lambda1'] *  p_x[:,:,1:,:] **2. * phi(1.,data['weight1_inv_x'][:,:,1:,:],eps_phi)**2.)
    pre_smooth_image = pre_smooth_image.at[:,1:,:,:].add(data['lambda1'] *  p_y ** 2. * phi(1.,data['weight1_inv_y'],eps_phi)**2.)
    pre_smooth_image = pre_smooth_image.at[:,1:-1,:,:].add(data['lambda1'] *  p_y[:,1:,:,:] ** 2. * phi(1.,data['weight1_inv_y'][:,1:,:,:],eps_phi)**2.)
    pre_smooth_image = safe_division(1.,pre_smooth_image,eps_phi)
    
    pre_scalemap_image = jnp.zeros_like(smooth_image) + data['lambda3']
    pre_scalemap_image = pre_scalemap_image.at[:,:,1:,:].add(data['lambda1'] *  phi(1.,data['weight1_inv_x'],eps_phi))
    pre_scalemap_image = pre_scalemap_image.at[:,1:,:,:].add(data['lambda1'] *  phi(1.,data['weight1_inv_y'],eps_phi))
    # pre_scalemap_image = 1.
    pre_scalemap_image = safe_division(1.,pre_scalemap_image,eps_phi)
    return {'smooth_image':pre_smooth_image * smooth_image,'scalemap_image':pre_scalemap_image * scalemap_image}

  def Ax(pp_image):
    jtd = jax.jvp(f,(x,),(pp_image,))[1]
    return jax.vjp(f,x)[1](jtd)[0]
  def jtf(x):
    return jax.vjp(f,x)[1](f(x))[0]
  jtx = jtf(x)
  
  d = linear_solve.solve_cg(matvec=Ax,
                          b=tree_map(lambda z:-1*z,jtx),
                          init=d,#M=M,
                          maxiter=100)
  aux = tree_reduce(lambda x,y:x+y,tree_multimap(lambda x,y:((x+y)**2).sum(),Ax(d),jtx))
  return d, aux

# @implicit_diff.custom_root(jax.grad(screen_poisson_objective),has_aux=True)
def nonlinear_solver_id(init_image,hp_nn,data):

  x = init_image
  loss = lambda pp_image:screen_poisson_objective(pp_image,hp_nn,data)
  
  optim_cond = lambda w: jax.tree_util.tree_map(lambda x:(x**2).sum(),jax.grad(loss)(w))
  def loop_body(iter,args):
    x,count, gn_opt_err, gn_loss1,gn_loss2,gn_loss,linear_opt_err,data = args['x'],args['count'],args['gn_opt_err'],args['gn_loss1'],args['gn_loss2'],args['gn_loss'],args['linear_opt_err'],args['data']
    data['optimize_smooth'] = iter % 2 == 0
    gn_lr = data['gn_lr']
    # data = compute_weights(x, data)
    d, linea_opt = linear_solver_id(None,x,hp_nn,data)
    # print('smooth ',d['smooth_image'].mean(),' scalemap ',d['scalemap_image'].mean())
    x = tree_multimap(lambda x,y:x+gn_lr *y,x, d)
    e = compute_terms(x,hp_nn,data)
    loss_i = [(i ** 2).sum() for i in e]
    print('loss ',e)
    linear_opt_err[count] = linea_opt
    gn_opt_err[count] = optim_cond(x)
    gn_loss1[count] = loss_i[0]
    gn_loss2[count] = loss_i[1]
    gn_loss[count] = loss(x)
    count += 1
    return {'x':x, 'count':count, 'gn_opt_err':gn_opt_err, 'gn_loss1':gn_loss1, 'gn_loss2':gn_loss2,'gn_loss':gn_loss,'linear_opt_err':linear_opt_err,'data':data}

  loop_count = 20
  # x,count, gn_opt_err, gn_loss, linear_opt_err,linea_opt= jax.lax.while_loop(lambda x:optim_cond(x[0]) >= 1e-10,loop_body,(x,0.0,-jnp.ones(200),-jnp.ones(200),-jnp.ones(200))) 
  aux = {'x':x,'count':0, 'gn_opt_err':[0] *loop_count, 'gn_loss1':[0] *loop_count, 'gn_loss2':[0] *loop_count, 'gn_loss':[0] *loop_count,'linear_opt_err':[0] *loop_count,'data':data}
  # aux = jax.lax.fori_loop(0,loop_count,lambda i,val:loop_body(val),aux)
  for i in tqdm.tqdm(range(loop_count)):
    aux = loop_body(i,aux)
  x = aux['x']
  del aux['x']
  del aux['data']
  return x, aux

def resize_params(params,sizes):
  ret = {}
  for key, val in params.items():
    if('_image' in key):
        ret[key] = jax.image.resize(val,sizes,'trilinear')
    else:
      ret[key] = val
  return ret


def hierarchical_id(hp_nn,init_inner,data):
  # n_hierarchies = data['n_hierarchies']
  def hierarchy(i,i_x,data,sizes,auxs):
    i_data = resize_params(data,sizes)
    i_x = resize_params(i_x,sizes)
    
    # i_data = compute_weights(i_x, i_data)
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


################ hierarchical, linear and nonlinear solvers end #################


################ outer model start ###############################
# @jax.jit
def outer_objective_id(hp_nn,init_inner,data):
    """Validation loss."""
    gt = data['ambient_image']
    f = lambda hp_nn: hierarchical_id(hp_nn, init_inner,data)
    x, aux = f(hp_nn)
    f_v = ((x['smooth_image']-gt) ** 2).sum()
    return f_v,(x,aux)

gmask = []

def testing():
  global gmask

  logger = cvgviz.logger('./logger','filesystem','autodiff','flash_no_flash')
  rng = jax.random.PRNGKey(1)
  rng, init_rng = jax.random.split(rng)
  flash = cvgim.imread('/home/mohammad/Projects/optimizer/DifferentiableSolver/flash_no_flash/cave_flash_1.png')# ** 2.2
  noflash = cvgim.imread('/home/mohammad/Projects/optimizer/DifferentiableSolver/flash_no_flash/cave_noflash_1.png')#** 2.2
  flash = flash[None,...]
  noflash = noflash[None,...]
  smooth_image_init = jax.device_put(jax.random.uniform(rng,noflash.shape))
  scalemap_init = jax.device_put(jax.random.uniform(rng,noflash.shape))
  
  Gy, Gx = dy(flash), dx(flash)
  Gx2y2 = Gx**2 + Gy**2
  Gx1y1 = Gx2y2 ** 0.5
  gmask = jnp.where(Gx1y1 < 0.005)

  inpt = {'smooth_image':smooth_image_init,'scalemap_image':scalemap_init}
  data = {'lambda1':1.0,'lambda2':3.0,'lambda3':0.0,'ambient_image':noflash, 'flash_image':flash,'gn_lr':1.0}
  params = Conv3features().init(init_rng, noflash)['params']
  
  valloss, (x,aux) = outer_objective_id(params,inpt,data)
  scalemap = (x['scalemap_image'] - x['scalemap_image'].min()) / (x['scalemap_image'].max() - x['scalemap_image'].min())
  scalemap = x['scalemap_image']

  # print('scalemap diff ',(sx**2).sum())
  imshow = jnp.concatenate((scalemap,x['smooth_image'],noflash,flash),axis=2)
  imshow = jnp.concatenate(imshow,axis=1)
  imshow = jnp.clip(imshow,0,1)# ** (1./2.2)
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