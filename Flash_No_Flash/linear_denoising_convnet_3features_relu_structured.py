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
from cvgutils.nn import jaxutils
import tensorflow as tf
import deepfnf_utils.utils as ut
import deepfnf_utils.tf_utils as tfu
from deepfnf_utils.dataset import Dataset
import cvgutils.Linalg as linalg

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=1, type=int,help='Image count in a batch')
parser.add_argument('--image_size', default=448, type=int,help='Width and height of an image')
parser.add_argument('--displacement', default=2, type=float,help='Random shift in pixels')
parser.add_argument('--min_scale', default=0.98,type=float, help='Random shift in pixels')
parser.add_argument('--max_scale', default=1.02,type=float, help='Random shift in pixels')
parser.add_argument('--max_rotate', default=np.deg2rad(0.5),type=float, help='Maximum rotation')
parser.add_argument('--lr', default=1e-4, type=float,help='Maximum rotation')
parser.add_argument('--display_freq', default=1, type=int,help='Display frequency by iteration count')
parser.add_argument('--val_freq', default=101, type=int,help='Display frequency by iteration count')
parser.add_argument('--save_param_freq', default=5,type=int, help='Maximum rotation')
parser.add_argument('--in_features', default=3,type=int, help='Maximum rotation')
parser.add_argument('--out_features', default=6,type=int, help='Maximum rotation')
parser.add_argument('--nlin_iter', default=100,type=int, help='Maximum linear iterations')
parser.add_argument('--max_iter', default=1e6, type=float,help='Maximum rotation')
parser.add_argument('--TLIST', default='data/train.txt',type=str, help='Maximum rotation')
parser.add_argument('--VPATH', default='data/valset/', type=str,help='Maximum rotation')
parser.add_argument('--ngpus', type=int, default=1, help='use how many gpus')
parser.add_argument('--unet_depth', default=2, type=int,help='Display frequency by iteration count')
parser.add_argument('--model', type=str, default='overfit_unet',
choices=['overfit_straight','interpolate_straight','overfit_unet','interpolate_unet'],help='Which model to use')
parser.add_argument('--logdir', type=str, default='./logger/Flash_No_Flash',help='Direction to store log used as ')
parser.add_argument('--logger', type=str, default='tb',choices=['tb','filesystem'],help='Where to dump the logs')
parser.add_argument('--expname', type=str, default='implicit_diff',help='Name of the experiment used as logdir/exp_name')

opts = parser.parse_args()

################ inner loop model end ############################

BSZ = opts.batch_size
IMSZ = opts.image_size
displacement = opts.displacement
model = opts.model

if(opts.model == 'overfit_straight'):
    init_model = lambda rng, x: jaxutils.StraightCNN().init(rng, x)['params']
    # model = lambda params, batch: jaxutils.StraightCNN().apply({'params': params}, batch['net_input'])
    model = lambda params, batch: jaxutils.StraightCNN().apply({'params': params}, batch['net_input'])
elif(opts.model == 'overfit_unet'):
    init_model = lambda rng, x: jaxutils.UNet(opts.unet_depth,opts.in_features,opts.out_features).init(rng, x)['params']
    # model = lambda params, batch: jaxutils.UNet(opts.unet_depth,opts.in_features,opts.out_features).apply({'params': params}, batch['net_input'])
    model = lambda params, batch: jaxutils.UNet(opts.unet_depth,opts.in_features,opts.out_features).apply({'params': params}, batch['net_input'])
else:
    print('Model unrecognized')
    exit(0)

tf.config.set_visible_devices([], device_type='GPU')
dataset = Dataset(opts.TLIST, opts.VPATH, bsz=BSZ, psz=IMSZ,
                    ngpus=opts.ngpus, nthreads=4 * opts.ngpus,
                    jitter=displacement, min_scale=opts.min_scale, max_scale=opts.max_scale, theta=opts.max_rotate)

dataset.swap_train()

# @jax.jit
def dx(x,mode='NHWC'):
    if(mode == 'NHWC'):
        y = jnp.pad(x,((0,0),(0,0),(1,0),(0,0)))
        return y[:,:,1:,:] - y[:,:,:-1,:]
# @jax.jit
def dy(x,mode='NHWC'):
    if(mode == 'NHWC'):
        y = jnp.pad(x,((0,0),(1,0),(0,0),(0,0)))
        return y[:,1:,:,:] - y[:,:-1,:,:]
# @jax.jit
def camera_to_rgb(im,batch):
    return tfu.camera_to_rgb_jax(
      im, batch['color_matrix'], batch['adapt_matrix'])

def stencil_residual(pp_image, hp_nn, data):
  """Objective function."""
  avg_weight = (1. / 2.) ** 0.5 *  (1. / pp_image.reshape(-1).shape[0] ** 0.5)
  r1 =  pp_image - data['noisy']
  g = model(hp_nn,data)
  r2 = dx(pp_image) - g[...,:3]
  r3 = dy(pp_image) - g[...,3:]
  out = jnp.concatenate(( r1.reshape(-1), r2.reshape(-1),r3.reshape(-1)),axis=0)
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
  d = linear_solve.solve_cg(matvec=Ax,
                          b=-jtf(x),
                          init=d,
                          maxiter=opts.nlin_iter)
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
  return x,{'count':count,'gn_opt_err':gn_opt_err, 'gn_loss':gn_loss,'linear_opt_err':linear_opt_err}
################ linear and nonlinear solvers end #################


# Check for saved weights & optimizer states
def preprocess(example,keys):
    

    key1, key2, key3, key4, key5, key6, key7, key8, key9, key10= keys

    # # for i in range(opts.ngpus):
    #     # with tf.device('/gpu:%d' % i):
    alpha = example['alpha'][:, None, None, None]
    dimmed_ambient, _ = tfu.dim_image_jax(
        example['ambient'], key1,alpha=alpha)
    dimmed_warped_ambient, _ = tfu.dim_image_jax(
        example['warped_ambient'],key2, alpha=alpha)

    # Make the flash brighter by increasing the brightness of the
    # flash-only image.
    flash = example['flash_only'] * ut.FLASH_STRENGTH + dimmed_ambient
    warped_flash = example['warped_flash_only'] * \
        ut.FLASH_STRENGTH + dimmed_warped_ambient

    sig_read = example['sig_read'][:, None, None, None]
    sig_shot = example['sig_shot'][:, None, None, None]
    noisy_ambient, _, _ = tfu.add_read_shot_noise_jax(
        dimmed_ambient,key3,key4,key5,key6, sig_read=sig_read, sig_shot=sig_shot)
    noisy_flash, _, _ = tfu.add_read_shot_noise_jax(
        warped_flash,key7,key8,key9,key10, sig_read=sig_read, sig_shot=sig_shot)

    # noisy_ambient = jnp.zeros_like(example['ambient'])
    # noisy_flash = jnp.zeros_like(example['ambient'])
    # sig_shot = jnp.zeros((*example['ambient'].shape[:-1],6))
    # sig_read = jnp.zeros((*example['ambient'].shape[:-1],6))
    # sig_shot = jnp.zeros((*example['ambient'].shape[:-1],6))

    noisy = jnp.concatenate([noisy_ambient, noisy_flash], axis=-1)
    noise_std = jnp.zeros((*example['ambient'].shape[:-1],6)) #tfu.estimate_std_jax(noisy, sig_read, sig_shot)
    net_input = jnp.concatenate([noisy, noise_std], axis=-1)
    
    output = {
        'alpha':alpha,
        'ambient':example['ambient'],
        'flash':noisy_flash,
        'noisy':noisy_ambient,
        'net_input':net_input,
        'adapt_matrix':example['adapt_matrix'],
        'color_matrix':example['color_matrix']
    }

    return output


def loss(h_params,init_inner,batch_p):
    
    """Validation loss."""
    f = lambda h_params: nonlinear_solver_id(init_inner, h_params,batch_p)
    x, aux = f(h_params)
    x = camera_to_rgb(x/batch['alpha'],batch)
    f_v = ((x - batch_p['ambient']) ** 2).sum()
    psnr = linalg.get_psnr_jax(jax.lax.stop_gradient(x),batch_p['ambient'])
    aux.update({'x':x,'psnr':psnr})
    return f_v,aux


rng = jax.random.PRNGKey(1)
rng, init_rng = jax.random.split(rng)
batch = dataset.iterator.next()
batch = {k:jnp.array(v.numpy()) for k,v in batch.items()}
keys = [jax.random.PRNGKey(100),
        jax.random.PRNGKey(101),
        jax.random.PRNGKey(102),
        jax.random.PRNGKey(103),
        jax.random.PRNGKey(100),
        jax.random.PRNGKey(101),
        jax.random.PRNGKey(102),
        jax.random.PRNGKey(103),
        jax.random.PRNGKey(102),
        jax.random.PRNGKey(103)]
preprocessed = preprocess(batch,keys)
testim = preprocessed['net_input']
params = init_model(init_rng, jnp.array(testim))
flat_params = jax.tree_util.tree_flatten(params)
parameters_count = jnp.array([jnp.prod(jnp.array(flat_params[0][i].shape)) for i in range(len(flat_params[0]))]).sum()
tf.debugging.set_log_device_placement(True)

dataset.swap_train()
#########################################################################
# Main Training loop


info = opts.__dict__
info.update({'params_count':parameters_count})
lr = 1e-4
logger = cvgviz.logger(opts.logdir,opts.logger,'Flash_No_Flash',opts.expname,info)
solver = OptaxSolver(fun=loss, opt=optax.adam(lr),has_aux=True)
state = solver.init_state(params)

@jax.jit
def update(params_p,state_p,batch_p):    
    params_p, state_p = solver.update(params_p, state_p,batch_p=batch_p,init_inner=batch_p['noisy'])
    return params_p, state_p

def get_batch(val_iter,val_iterator,train_iterator):
    if(val_iter):
        try:
            batch = val_iterator.next()
        except StopIteration:
            val_iterator = iter(dataset.val.dataset)
            batch = val_iterator.next()
    else:
        try:
            batch = train_iterator.next()
        except StopIteration:
            train_iterator = iter(dataset.train.dataset)
            batch = train_iterator.next()
    return batch
    

@jax.jit
def visualize(x, params, batch):
    g = model(params,batch)
    ambient = camera_to_rgb(batch['ambient'],batch)
    flash = camera_to_rgb(batch['flash'],batch)

    x_x = jnp.abs(dx(x) * 10)
    x_y = jnp.abs(dy(x) * 10)
    g_x = jnp.abs(g[...,:3] * 10)
    g_y = jnp.abs(g[...,3:] * 10)


    imshow = jnp.concatenate((x,ambient,flash,x_x,g_x,x_y,g_y),axis=-2)
    imshow = jnp.clip(imshow,0,1)
    return imshow
################ outer model start ###############################
data = logger.load_params()
start_idx=0
val_iterator = iter(dataset.val.dataset)
train_iterator = iter(dataset.train.dataset)
val_iter = False#i % opts.val_freq == 0
mode = 'val' if val_iter else 'train'
if(data is not None):
    # state = data['state']
    batch = data['state']
    params = data['params']
    start_idx = data['idx']
with tqdm.trange(int(start_idx), int(opts.max_iter)) as t:
    for i in t:
        batch2 = get_batch(val_iter,val_iterator,train_iterator)
        batch2 = {k:jnp.array(v.numpy()) for k,v in batch2.items()}
        batch = preprocess(batch2,keys)
        if(val_iter):
            loss_state = loss(params,batch)
            count = state.aux['count']
            # loss_val,predicted,ambient,noisy,flash,psnr = loss_state[0], loss_state[1]['predicted'],loss_state[1]['ambient'],loss_state[1]['noisy'],loss_state[1]['flash'],loss_state[1]['psnr']
        else:
            params, state = update(params,state,batch)
            count = state.aux['count']
            imshow = visualize(state.aux['x'], params, batch)
            loss_val,predicted,ambient,noisy,flash,psnr = state.value, state.aux['x'],batch['ambient'],batch['noisy'],batch['flash'],state.aux['psnr']
        t.set_description('Error l2 '+str(np.array(loss_val))+' psnr '+str(psnr))
        print('count ',count)
        if(i % opts.display_freq == 0 or val_iter):
            # imshow = jnp.concatenate((predicted,ambient,noisy,flash),axis=2)
            # imshow = jnp.clip(imshow,0,1)
            logger.addImage(imshow[0],'imshow',mode=mode)
        if(i % opts.save_param_freq == 0):
            logger.save_params(params,batch,i)
        
        logger.addScalar(loss_val,'loss',mode=mode)
        logger.addScalar(psnr,'psnr',mode=mode)
        
        logger.takeStep()
################ outer model end #################################

