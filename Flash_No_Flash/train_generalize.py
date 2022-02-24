"""
In this file we are training a Unet to learn flash no flash denoising.
Input is concatenation of flash,noflash,shot_noise and read_noise and outputs
the denoised image
"""
#!/usr/bin/env python3
import optax
import tensorflow as tf
import deepfnf_utils.utils as ut
import deepfnf_utils.tf_utils as tfu
from deepfnf_utils.dataset import Dataset
import jax
from cvgutils.nn import jaxutils
from jaxopt import implicit_diff, OptaxSolver, linear_solve

import argparse
import jax.numpy as jnp
import tqdm
from jax.config import config
from jaxopt import OptaxSolver
import cvgutils.Viz as cvgviz
import cvgutils.Image as cvgim
import cvgutils.Utils as cvgutil
import numpy as np
import cvgutils.Linalg as linalg
config.update("jax_debug_nans", True)
# tf.compat.v1.disable_eager_execution()
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
# config = tf.config.experimental.set_memory_growth(physical_devices[1], True)

################ inner loop model end ############################

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=15, type=int,help='Image count in a batch')
parser.add_argument('--image_size', default=448, type=int,help='Width and height of an image')
parser.add_argument('--displacement', default=2, type=float,help='Random shift in pixels')
parser.add_argument('--min_scale', default=0.98,type=float, help='Random shift in pixels')
parser.add_argument('--max_scale', default=1.02,type=float, help='Random shift in pixels')
parser.add_argument('--max_rotate', default=np.deg2rad(0.5),type=float, help='Maximum rotation')
parser.add_argument('--lr', default=1e-4, type=float,help='Maximum rotation')
parser.add_argument('--display_freq', default=100, type=int,help='Display frequency by iteration count')
parser.add_argument('--val_freq', default=101, type=int,help='Display frequency by iteration count')
parser.add_argument('--save_param_freq', default=100,type=int, help='Maximum rotation')
parser.add_argument('--max_iter', default=1e6, type=float,help='Maximum rotation')
parser.add_argument('--unet_depth', default=4, type=int,help='Depth of neural net')
parser.add_argument('--TLIST', default='data/train.txt',type=str, help='Maximum rotation')
parser.add_argument('--VPATH', default='data/valset/', type=str,help='Maximum rotation')
parser.add_argument('--ngpus', type=int, default=1, help='use how many gpus')
parser.add_argument('--model', type=str, default='UNet_Hardcode',
choices=['overfit_straight','interpolate_straight','overfit_unet','interpolate_unet','UNet_Hardcode'],help='Which model to use')
parser.add_argument('--logdir', type=str, default='./logger/Flash_No_Flash',help='Direction to store log used as ')
parser.add_argument('--logger', type=str, default='tb',choices=['tb','filesystem'],help='Where to dump the logs')
parser.add_argument('--expname', type=str, default='unvet_overfit_hardcode_depth_1',help='Name of the experiment used as logdir/exp_name')
parser.add_argument('--min_alpha', default=0.02, type=float,help='Maximum rotation')
parser.add_argument('--max_alpha', default=0.2, type=float,help='Maximum rotation')
parser.add_argument('--min_read', default=-3., type=float,help='Maximum rotation')
parser.add_argument('--max_read', default=-2, type=float,help='Maximum rotation')
parser.add_argument('--min_shot', default=-2., type=float,help='Maximum rotation')
parser.add_argument('--max_shot', default=-1.3, type=float,help='Maximum rotation')

opts = parser.parse_args()
# cvgutil.savePickle('./params.pickle',opts)
# opts = cvgutil.loadPickle('./params.pickle')
BSZ = opts.batch_size
IMSZ = opts.image_size
displacement = opts.displacement
model = opts.model
LR = 1e-4
DROP = (1.1e6, 1.25e6) # Learning rate drop
MAXITER = 1.5e6

VALFREQ = 2e1
SAVEFREQ = 5e4

# sess = tf.compat.v1.Session(config=config)
# sess.run(tf.compat.v1.global_variables_initializer())
tf.config.set_visible_devices([], device_type='GPU')
dataset = Dataset(opts.TLIST, opts.VPATH, bsz=BSZ, psz=IMSZ,
                    ngpus=opts.ngpus, nthreads=4 * opts.ngpus,
                    jitter=displacement, min_scale=opts.min_scale, max_scale=opts.max_scale, theta=opts.max_rotate,
                    min_alpha=opts.min_alpha, max_alpha=opts.max_alpha,
                    min_read=opts.min_read, max_read=opts.max_read, min_shot=opts.min_shot, max_shot=opts.max_shot)

dataset.swap_train()

#########################################################################

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
    noise_std = tfu.estimate_std_jax(noisy, sig_read, sig_shot)
    net_input = jnp.concatenate([noisy_ambient], axis=-1)
    
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
if(opts.model == 'overfit_straight'):
    init_model = lambda rng, x: jaxutils.StraightCNN().init(rng, x)['params']
    model = lambda params, batch: jaxutils.StraightCNN().apply({'params': params}, batch['net_input'])
elif(opts.model == 'interpolate_straight'):
    init_model = lambda rng, x: jaxutils.StraightCNN().init(rng, x)['params']
    model = lambda params, batch: jaxutils.StraightCNN().apply({'params': params}, batch['net_input']) + batch['noisy']
elif(opts.model == 'overfit_unet'):
    init_model = lambda rng, x: jaxutils.UNet(opts.unet_depth,12,3).init(rng, x)['params']
    model = lambda params, batch: jaxutils.UNet(opts.unet_depth,12,3).apply({'params': params}, batch['net_input'])
elif(opts.model == 'interpolate_unet'):
    init_model = lambda rng, x: jaxutils.UNet(opts.unet_depth,12,3).init(rng, x)['params']
    model = lambda params, batch: jaxutils.UNet(opts.unet_depth,12,3).apply({'params': params}, batch['net_input']) + batch['noisy']
elif(opts.model == 'UNet_Hardcode'):
    init_model = lambda rng, x: jaxutils.UNet_Hardcode().init(rng, x)['params']
    model = lambda params, batch: jaxutils.UNet_Hardcode().apply({'params': params}, batch['net_input']) + batch['noisy']
else:
    print('Model unrecognized')
    exit(0)

params = init_model(init_rng, jnp.array(testim))
flat_params = jax.tree_util.tree_flatten(params)
parameters_count = jnp.array([jnp.prod(jnp.array(flat_params[0][i].shape)) for i in range(len(flat_params[0]))]).sum()
tf.debugging.set_log_device_placement(True)

dataset.swap_train()
#########################################################################
# Main Training loop


@jax.jit
def predict(im_p,params_p):
    return model(params_p, im_p)

@jax.jit
def loss(params_p,batch_p):
    predicted = model(params_p,batch_p)

    g = tfu.camera_to_rgb_jax(
      predicted/batch_p['alpha'], batch_p['color_matrix'], batch_p['adapt_matrix'])
    ambient = tfu.camera_to_rgb_jax(
        batch_p['ambient'],
        batch_p['color_matrix'], batch_p['adapt_matrix'])
    flash = tfu.camera_to_rgb_jax(
        batch_p['flash'],
        batch_p['color_matrix'], batch_p['adapt_matrix'])
    noisy = tfu.camera_to_rgb_jax(
        batch_p['noisy']/batch_p['alpha'],
        batch_p['color_matrix'], batch_p['adapt_matrix'])
    f = g
    diff = f - ambient
    loss = (diff ** 2).mean()
    psnr = linalg.get_psnr_jax(jax.lax.stop_gradient(f),ambient)
    return loss, {'psnr':psnr,'predicted':jax.lax.stop_gradient(f), 'ambient':ambient, 'flash':flash, 'noisy':noisy}


info = opts.__dict__
info.update({'params_count':parameters_count})
lr = 1e-4
logger = cvgviz.logger(opts.logdir,opts.logger,'Flash_No_Flash',opts.expname,info)
solver = OptaxSolver(fun=loss, opt=optax.adam(lr),has_aux=True)
state = solver.init_state(params)
# g = jax.grad(loss,has_aux=True)
@jax.jit
def update(params_p,state_p,batch_p):    
    params_p, state_p = solver.update(params_p, state_p,batch_p=batch_p)
    return params_p, state_p
data = logger.load_params()
start_idx=0
val_iterator = iter(dataset.val.dataset)
train_iterator = iter(dataset.train.dataset)
if(data is not None):
    # state = data['state']
    batch = data['state']
    params = data['params']
    start_idx = data['idx']

def get_batch(val_iter_p,val_iterator_p,train_iterator_p):
    if(val_iter_p):
        try:
            batch = val_iterator_p.next()
        except StopIteration:
            val_iterator_p = iter(dataset.val.dataset)
            batch = val_iterator_p.next()
    else:
        try:
            batch = train_iterator_p.next()
        except StopIteration:
            train_iterator_p = iter(dataset.train.dataset)
            batch = train_iterator_p.next()
    return batch
    


val_iter = False#i % opts.val_freq == 0
mode = 'val' if val_iter else 'train'

with tqdm.trange(int(start_idx), int(opts.max_iter)) as t:
    for i in t:
        batch2 = get_batch(val_iter,val_iterator,train_iterator)
        batch2 = {k:jnp.array(v.numpy()) for k,v in batch2.items()}
        batch = preprocess(batch2,keys)
        if(val_iter):
            loss_state = loss(params,batch)
            loss_val,predicted,ambient,noisy,flash,psnr = loss_state[0], loss_state[1]['predicted'],loss_state[1]['ambient'],loss_state[1]['noisy'],loss_state[1]['flash'],loss_state[1]['psnr']
        else:
            params, state = update(params,state,batch)
            loss_val,predicted,ambient,noisy,flash,psnr = state.value, state.aux['predicted'],state.aux['ambient'],state.aux['noisy'],state.aux['flash'],state.aux['psnr']
        t.set_description('Error l2 '+str(np.array(loss_val))+' psnr '+str(psnr))
        if(i % opts.display_freq == 0 or val_iter):
            imshow = jnp.concatenate((predicted,ambient),axis=2) #,batch['net_input'][...,:3],batch['net_input'][...,3:6],batch['net_input'][...,6:9]*100,batch['net_input'][...,9:12]*100),axis=2)
            imshow = jnp.clip(imshow,0,1)
            logger.addImage(imshow[0],'imshow',mode=mode)
        if(i % opts.save_param_freq == 0):
            logger.save_params(params,batch,i)
        
        logger.addScalar(loss_val,'loss',mode=mode)
        logger.addScalar(psnr,'psnr',mode=mode)
        
        logger.takeStep()
