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
eps=0.004
eps_phi = 1e-4

################ inner loop model end ############################

def gauss(im):
  return im

def dx(x):
  x = jnp.pad(x,((0,0),(0,0),(1,0),(0,0)),mode='reflect')
  return gauss(x)[...,:,1:,:] - gauss(x)[...,:,:-1,:]

def dy(x):
  x = jnp.pad(x,((0,0),(1,0),(0,0),(0,0)),mode='reflect')
  return gauss(x)[...,1:,:,:] - gauss(x)[...,:-1,:,:]

def sign(x):
  return jnp.sign(x) + jnp.where(x==0,1.,0.)

def compute_weights(pp_image, data):
  flash = data['flash_image']
  ambient = data['ambient_image']
  smooth_image = jax.lax.stop_gradient(pp_image['smooth_image'])
  scalemap_image = jax.lax.stop_gradient(pp_image['scalemap_image'])
  

  dy_flash, dx_flash, i_y, i_x = dy(flash), dx(flash), dy(smooth_image), dx(smooth_image)
  data['weight1_inv_x'] = jnp.abs(scalemap_image - safe_division(i_x, dx_flash,eps))
  data['weight1_inv_y'] = jnp.abs(scalemap_image - safe_division(i_y, dy_flash,eps))
  data['weight2_inv']   =   jnp.abs(smooth_image - ambient)

  # data['weight1_inv_x_norm'] 
  # data['weight1_inv_y'] 
  # data['weight2_inv']   


  return data

def safe_division(nom,denom,eps):
  return nom / (sign(denom) * jnp.maximum(jnp.abs(denom), eps))

def phi(nom,denom,eps):
  return nom / (jnp.abs(denom)+ eps)

def stencil_residual(pp_image, hp_nn, data):
  e = compute_terms(pp_image, hp_nn, data)
  return jnp.concatenate(e, axis=0)

def compute_terms(pp_image, hp_nn, data):
  ambient = data['ambient_image']
  flash = data['flash_image']
  l1 = data['lambda1'] ** 0.5
  l2 = data['lambda2'] ** 0.5
  l3 = data['lambda3'] ** 0.5
  w2 = data['weight2_inv'] ** 0.5
  w1x = data['weight1_inv_x'] ** 0.5
  w1y = data['weight1_inv_y'] ** 0.5
  eps_phi = 0.001
  optimize_smooth = jax.lax.stop_gradient(data['optimize_smooth'])
  
  eta = 0.001
  if(optimize_smooth):
    smooth_image = pp_image['smooth_image']
    scalemap_image = jax.lax.stop_gradient(pp_image['scalemap_image'])
  else:
    smooth_image = jax.lax.stop_gradient(pp_image['smooth_image'])
    scalemap_image = pp_image['scalemap_image']

  # smooth_image = pp_image['smooth_image']
  # scalemap_image = pp_image['scalemap_image']
  
  dy_flash, dx_flash,dx_scalemap,dy_scalemap,i_y, i_x = dy(flash), dx(flash),dx(scalemap_image),dy(scalemap_image), dy(smooth_image), dx(smooth_image)

  eta2 = eta**2
  g2 = dx_flash **2 + dy_flash **2
  g1 = g2 ** 0.5
  mu1 = eta2 / (g2+2*eta2)
  mu2 = (g2 + eta2) / (g2 + 2*eta2)
  v1x = dx_flash / (g1 + eps)
  v1y = dy_flash / (g1 + eps)
  v2x = -dy_flash / (g1 + eps)
  v2y = dx_flash / (g1 + eps)

  e3_1_x  = l3 * mu1**0.5 * v1x * dx_scalemap
  e3_1_y  = l3 * mu1**0.5 * v1y * dy_scalemap
  # e3_1_xy = mu1 * 2 * v1y * dy_scalemap * v1x * dx_scalemap
  # e3_1_xy = l3 * phi(e3_1_xy**2,jnp.sqrt(jnp.abs(e3_1_xy)),eps_phi)
  e3_2_x  = l3 * mu2**0.5 * v2x * dx_scalemap
  e3_2_y  = l3 * mu2**0.5 * v2y * dy_scalemap
  # e3_2_xy = mu2 * 2 * v2x * dx_scalemap * v2y * dy_scalemap
  # e3_2_xy = l3 * phi(e3_2_xy,jnp.sqrt(jnp.abs(e3_2_xy)),eps_phi)

  e3_1_x = e3_1_x.reshape(-1)
  e3_1_y = e3_1_y.reshape(-1) 
  # e3_1_xy = e3_1_xy.reshape(-1)
  e3_2_x = e3_2_x.reshape(-1) 
  e3_2_y = e3_2_y.reshape(-1) 
  # e3_2_xy = e3_2_xy.reshape(-1)

  """Objective function."""
  e2 =  smooth_image - ambient

  e1_x = scalemap_image- safe_division(i_x,dx_flash,eps)
  e1_y = scalemap_image - safe_division(i_y,dy_flash,eps)
  e1_term = jnp.concatenate((l1 * phi(e1_x,w1x,eps_phi).reshape(-1),
          l1 * phi(e1_y,w1y,eps_phi).reshape(-1)),axis=0)
  # e1_term = l1 * phi(e1_x,w1x,eps_phi).reshape(-1)
  e2_term = l2 * phi(e2,w2,eps_phi).reshape(-1)
  
  e = (e1_term, e2_term, e3_1_x, e3_1_y, e3_2_x, e3_2_y)
  avg_weight = (1. / len(e)) ** 0.5 *  (1. / (smooth_image.reshape(-1).shape[0]) ** 0.5)
  e = [avg_weight * i for i in e]
  return e
  #  (etmp).reshape(-1),
  #  (l3 * e3_1).reshape(-1),
  #  (l3 * e3_2).reshape(-1)),
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
    pre_smooth_image += data['lambda1'] *  p_x**2. * phi(1.,data['weight1_inv_x'],eps_phi)**2.
    pre_smooth_image += data['lambda1'] *  p_x **2. * phi(1.,data['weight1_inv_x'],eps_phi)**2.
    pre_smooth_image += data['lambda1'] *  p_y ** 2. * phi(1.,data['weight1_inv_y'],eps_phi)**2.
    pre_smooth_image += data['lambda1'] *  p_y ** 2. * phi(1.,data['weight1_inv_y'],eps_phi)**2.
    pre_smooth_image = safe_division(1.,pre_smooth_image,eps_phi)
    
    pre_scalemap_image = jnp.zeros_like(smooth_image) + data['lambda3']
    pre_scalemap_image = data['lambda1'] *  phi(1.,data['weight1_inv_x'],eps_phi)
    pre_scalemap_image = data['lambda1'] *  phi(1.,data['weight1_inv_y'],eps_phi)
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
                          init=d,M=M,
                          maxiter=100)
  aux = tree_multimap(lambda x,y:((x+y)**2).mean(),Ax(d),jtx)
  return d, aux

# @implicit_diff.custom_root(jax.grad(screen_poisson_objective),has_aux=True)
def nonlinear_solver_id(init_image,hp_nn,data):
  loop_count = 7
  x = [0] *loop_count
  x[0] = init_image
  loss = lambda pp_image:screen_poisson_objective(pp_image,hp_nn,data)
  
  optim_cond = lambda w: jax.tree_util.tree_map(lambda x:(x**2).mean(),jax.grad(loss)(w))
  def loop_body(iter,args):
    x_list,count, gn_opt_err, gn_loss1,gn_loss2,gn_loss,linear_opt_err1,linear_opt_err2,data = args['x'],args['count'],args['gn_opt_err'],args['gn_loss1'],args['gn_loss2'],args['gn_loss'],args['linear_opt_err1'],args['linear_opt_err2'],args['data']
    iter_id = iter-1 if iter -1 >= 0 else 0
    x = x_list[iter_id]
    data['optimize_smooth'] = iter % 2 == 0
    gn_lr = data['gn_lr']
    data = compute_weights(x, data)
    d, linea_opt = linear_solver_id(None,x,hp_nn,data)
    # print('smooth ',d['smooth_image'].mean(),' scalemap ',d['scalemap_image'].mean())
    x = tree_multimap(lambda x,y:x+gn_lr *y,x, d)
    e = compute_terms(x,hp_nn,data)
    loss_i = [(i ** 2).mean() for i in e]
    linear_opt_err1[count] = linea_opt['scalemap_image']
    linear_opt_err2[count] = linea_opt['smooth_image']
    gn_opt_err[count] = optim_cond(x)
    gn_loss1[count] = loss_i[0]
    gn_loss2[count] = loss_i[1]
    gn_loss[count] = loss(x)
    x_list[iter] = x
    count += 1
    return {'x':x_list, 'count':count, 'gn_opt_err':gn_opt_err, 'gn_loss1':gn_loss1, 'gn_loss2':gn_loss2,'gn_loss':gn_loss,'linear_opt_err1':linear_opt_err1,'linear_opt_err2':linear_opt_err2,'data':data}

  
  # x,count, gn_opt_err, gn_loss, linear_opt_err,linea_opt= jax.lax.while_loop(lambda x:optim_cond(x[0]) >= 1e-10,loop_body,(x,0.0,-jnp.ones(200),-jnp.ones(200),-jnp.ones(200))) 
  aux = {'x':x,'count':0, 'gn_opt_err':[0] *loop_count, 'gn_loss1':[0] *loop_count, 'gn_loss2':[0] *loop_count, 'gn_loss':[0] *loop_count,'linear_opt_err1':[0] *loop_count,'linear_opt_err2':[0] *loop_count,'data':data}
  # aux = jax.lax.fori_loop(0,loop_count,lambda i,val:loop_body(val),aux)
  for i in tqdm.tqdm(range(loop_count)):
    aux = loop_body(i,aux)
  x = aux['x']
  # del aux['x']
  del aux['data']
  return x[-1], aux

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
  i_x, auxs = hierarchy(0,i_x, data,size_scale(data['flash_image'].shape,3),auxs)
  i_x, auxs = hierarchy(1,i_x, data,size_scale(data['flash_image'].shape,2),auxs)
  i_x, auxs = hierarchy(2,i_x, data,size_scale(data['flash_image'].shape,1),auxs)
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
    # f_v = ((x['smooth_image']-gt) ** 2).sum()
    return x['smooth_image'].sum(),(x,aux)


def testing():
  logger = cvgviz.logger('/home/mohammad/Projects/optimizer/logger/flash_no_flash','tb','autodiff','l1-M-normalize-flip-flop-6iter-1level-lambda1-009-lambda2-3-gnlr-04-lambda3-05-padding')
  # logger.addScalar(0,'hi')
  # logger.takeStep()
  # logger.addScalar(0,'hi')
  # logger.takeStep()
  # logger.addScalar(0,'hi')
  # logger.takeStep()
  # logger.addScalar(0,'hi')
  # logger.takeStep()
  # logger.writer.flush()

  rng = jax.random.PRNGKey(1)
  rng, init_rng = jax.random.split(rng)
  gt_im = cvgim.imread('/home/mohammad/Projects/optimizer/DifferentiableSolver/flash_no_flash/out.png')# ** 2.2
  flash = cvgim.imread('/home/mohammad/Projects/optimizer/DifferentiableSolver/flash_no_flash/cave_flash_1.png')# ** 2.2
  noflash = cvgim.imread('/home/mohammad/Projects/optimizer/DifferentiableSolver/flash_no_flash/cave_noflash_1.png')#** 2.2
  gt_im = gt_im[None,...]
  flash = flash[None,...]
  noflash = noflash[None,...]
  smooth_image_init = jax.device_put(noflash.copy())
  scalemap_init = jax.device_put(jnp.ones_like(noflash))
  
  inpt = {'smooth_image':smooth_image_init,'scalemap_image':scalemap_init}
  data = {'lambda1':0.009,'lambda2':3.0,'lambda3':0.5,'ambient_image':noflash, 'flash_image':flash,'gn_lr':1.}
  params = None
  
  valloss, (x,aux) = outer_objective_id(params,inpt,data)
  # scalemap = (x['scalemap_image'] - x['scalemap_image'].min()) / (x['scalemap_image'].max() - x['scalemap_image'].min())
  # scalemap = x['scalemap_image']
  # nom,denom = jnp.zeros_like(flash),jnp.zeros_like(flash)
  # nom = nom.at[:,:,1:,:].add(dx(x['smooth_image']))
  # nom = nom.at[:,1:,:,:].add(dy(x['smooth_image']))
  # denom = denom.at[:,:,1:,:].add(dx(flash))
  # denom = denom.at[:,1:,:,:].add(dy(flash))
  
  # sx = scalemap[:,:,1:,:] - safe_division(dx(x['smooth_image']),dx(flash),eps)
  # sy = scalemap[:,1:,:,:] - safe_division(dy(x['smooth_image']),dy(flash),0.004)

  # di_dg = safe_division(nom,denom,eps)
  # print('scalemap diff ',(sx**2).sum())
  for key in aux.keys():
    l = len(aux[key]['gn_opt_err'])
    for i in range(l):
      x = aux[key]['x'][-1]
      resized = resize_params({'gt_image':gt_im,'noflash_image':noflash,'flash_image':flash},x['smooth_image'].shape)
      imshow = jnp.concatenate((x['scalemap_image'],x['smooth_image'],resized['gt_image'],resized['noflash_image'],resized['flash_image']),axis=2)
      imshow = jnp.concatenate(imshow,axis=1)
      imshow = jnp.clip(imshow,0,1)# ** (1./2.2)
      logger.addImage(np.array(imshow).transpose(2,0,1),'Image')
      logger.addScalar(aux[key]['gn_opt_err'][i]['scalemap_image'],'/errors/gn_opt_err/scalemap')
      logger.addScalar(aux[key]['gn_opt_err'][i]['smooth_image'],'/errors/gn_opt_err/smooth')
      logger.addScalar(aux[key]['gn_loss'][i],'/errors/gn_loss')
      logger.addScalar(aux[key]['gn_loss1'][i],'/errors/gn_loss1')
      logger.addScalar(aux[key]['gn_loss2'][i],'/errors/gn_loss2')
      logger.addScalar(aux[key]['linear_opt_err1'][i],'/errors/linear_opt_err1')
      logger.addScalar(aux[key]['linear_opt_err2'][i],'/errors/linear_opt_err2')
      logger.takeStep()
  
  print(aux)
  print('hi')

################ outer model end #################################

# hyper_optimization()
testing()