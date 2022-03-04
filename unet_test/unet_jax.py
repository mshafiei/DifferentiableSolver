from cvgutils.nn.jaxUtils.unet_model import UNet
import jax.numpy as jnp
import jax
import optax
from jaxopt import OptaxSolver
import tensorflow as tf
import tqdm
import numpy as np
import cvgutils.Image as cvgim
from deepfnf_utils.dataset import Dataset
import cvgutils.Utils as cvgutil
import deepfnf_utils.utils as ut
import deepfnf_utils.tf_utils as tfu
import cvgutils.Viz as Viz
import cvgutils.Linalg as linalg
import argparse
from flax.core import freeze, unfreeze
import time


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

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=1, type=int,help='Image count in a batch')
parser.add_argument('--image_size', default=448, type=int,help='Width and height of an image')
parser.add_argument('--displacement', default=2, type=float,help='Random shift in pixels')
parser.add_argument('--min_scale', default=0.98,type=float, help='Random shift in pixels')
parser.add_argument('--max_scale', default=1.02,type=float, help='Random shift in pixels')
parser.add_argument('--max_rotate', default=np.deg2rad(0.5),type=float, help='Maximum rotation')
parser.add_argument('--lr', default=1e-4, type=float,help='Maximum rotation')
parser.add_argument('--display_freq', default=1000, type=int,help='Display frequency by iteration count')
parser.add_argument('--val_freq', default=101, type=int,help='Display frequency by iteration count')
parser.add_argument('--save_param_freq', default=100,type=int, help='Maximum rotation')
parser.add_argument('--max_iter', default=1000000, type=int,help='Maximum iteration count')
parser.add_argument('--unet_depth', default=4, type=int,help='Depth of neural net')
parser.add_argument('--TLIST', default='data/train.txt',type=str, help='Maximum rotation')
parser.add_argument('--VPATH', default='data/valset/', type=str,help='Maximum rotation')
parser.add_argument('--ngpus', type=int, default=1, help='use how many gpus')
parser.add_argument('--model', type=str, default='overfit_unet',
choices=['overfit_straight','interpolate_straight','overfit_unet','interpolate_unet','UNet_Hardcode'],help='Which model to use')
parser.add_argument('--logdir', type=str, default='./logger/Unet_test',help='Direction to store log used as ')
parser.add_argument('--logger', type=str, default='tb',choices=['tb','filesystem'],help='Where to dump the logs')
parser.add_argument('--expname', type=str, default='unvet_overfit_hardcode',help='Name of the experiment used as logdir/exp_name')
parser.add_argument('--min_alpha', default=0.02, type=float,help='Maximum rotation')
parser.add_argument('--max_alpha', default=0.2, type=float,help='Maximum rotation')
parser.add_argument('--min_read', default=-3., type=float,help='Maximum rotation')
parser.add_argument('--max_read', default=-2, type=float,help='Maximum rotation')
parser.add_argument('--min_shot', default=-2., type=float,help='Maximum rotation')
parser.add_argument('--max_shot', default=-1.3, type=float,help='Maximum rotation')

opts = parser.parse_args()

BSZ = opts.batch_size
IMSZ = opts.image_size
displacement = opts.displacement
model = opts.model


tf.config.set_visible_devices([], device_type='GPU')
dataset = Dataset(opts.TLIST, opts.VPATH, bsz=BSZ, psz=IMSZ,
                    ngpus=opts.ngpus, nthreads=4 * opts.ngpus,
                    jitter=displacement, min_scale=opts.min_scale, max_scale=opts.max_scale, theta=opts.max_rotate,
                    min_alpha=opts.min_alpha, max_alpha=opts.max_alpha,
                    min_read=opts.min_read, max_read=opts.max_read, min_shot=opts.min_shot, max_shot=opts.max_shot)

dataset.swap_train()
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
    batch = {k:jnp.array(v.numpy()) for k,v in batch.items()}
    keys = [jax.random.PRNGKey(time.time_ns()) + i for i in range(10)]
    batch = preprocess(batch,keys)
    return batch
    
dataset_train = iter(dataset.train.dataset)
dataset_val = iter(dataset.val.dataset)

# fn = '/home/mohammad/Projects/optimizer/baselines/dataset/flash_no_flash/merged/Objects_002_ambient.png'
# im = cvgim.imread(fn)[None,:448,:448,:]
logger = Viz.logger(opts.logdir,opts.logger,'Unet_test',opts.expname,opts.__dict__)

# cvgutil.savePickle('./params.pickle',opts)
# opts = cvgutil.loadPickle('./params.pickle')
# exit(0)



rng = jax.random.PRNGKey(1)
rng, init_rng = jax.random.split(rng)
def next_batch():
    batch = dataset.iterator.next()
    batch = {k:jnp.array(v.numpy()) for k,v in batch.items()}
    keys = [jax.random.PRNGKey(time.time_ns()) + i for i in range(10)]
    return preprocess(batch,keys)

im = next_batch()['net_input']
n_channels = 12
n_classes = 3
bilinear = True
def init_model(rng, x,test=False,group_norm=True):
    return UNet(n_channels,n_classes,bilinear,test,group_norm).init(rng, x)

def model_test(params, im,group_norm=True):
    return UNet(n_channels,n_classes,bilinear,True,group_norm).apply(params, im)
def model_train(params, im,group_norm=True):
    if(group_norm):
        return UNet(n_channels,n_classes,bilinear,False,group_norm).apply(params, im,mutable=['batch_stats'])
    else:
        return UNet(n_channels,n_classes,bilinear,False,group_norm).apply(params, im)
rng = jax.random.PRNGKey(1)
rng, init_rng = jax.random.split(rng)
params = init_model(init_rng, jnp.array(im))
lr = 1e-4

@jax.jit
def loss(params, batch,gt,test=False):
    pred, bn_state = model_train(params, batch)
    return ((gt - pred) ** 2).sum(), bn_state

@jax.jit
def update(params_p,state_p,batch_p,gt):    
    params_p, state_p = solver.update(params_p, state_p,batch=batch_p,gt=gt)
    return params_p, state_p

data = logger.load_params()
start_idx=0
if(data is not None):
    # state = data['state']
    batch = data['state']
    params = data['params']
    start_idx = data['idx']

solver = OptaxSolver(fun=loss, opt=optax.adam(lr),has_aux=True)
state = solver.init_state(params)

with tqdm.trange(start_idx,opts.max_iter) as t:
    for i in t:
        val_iter = i % opts.val_freq == 0
        mode = 'val' if val_iter else 'train'
        # batch = get_batch(val_iter,dataset_val,dataset_train)
        batch = next_batch()
        params, state = update(params,state,batch['net_input'],batch['ambient'])
        l,_ = loss(params,batch['net_input'],batch['ambient'])
        # params_p = params.pop('params')
        # batch_stats_p = state.aux.pop('batch_stats')
        # params = freeze({'params':params['params'],'batch_stats':state.aux['batch_stats']})

        # params = {'params':params['params'],'batch_stats':state.aux['batch_stats']}
        t.set_description('loss '+str(np.array(l)))

        if(i % opts.display_freq == 0 or val_iter):
            predicted = model_test(params, batch['net_input'])
            g = tfu.camera_to_rgb_jax(
            predicted, batch['color_matrix'], batch['adapt_matrix'])
            ambient = tfu.camera_to_rgb_jax(
                batch['ambient'],
                batch['color_matrix'], batch['adapt_matrix'])
            flash = tfu.camera_to_rgb_jax(
                batch['flash'],
                batch['color_matrix'], batch['adapt_matrix'])
            noisy = tfu.camera_to_rgb_jax(
                batch['noisy']/batch['alpha'],
                batch['color_matrix'], batch['adapt_matrix'])
            psnr = linalg.get_psnr_jax(jax.lax.stop_gradient(g),ambient)
            imshow = jnp.clip(jnp.concatenate((ambient,g,noisy,flash),axis=-2),0,1)
            logger.addImage(imshow[0],'image',mode=mode)
            logger.addScalar(psnr,'psnr',mode=mode)
        if(i % opts.save_param_freq == 0):
            logger.save_params(params,batch,i)

        logger.addScalar(l,'loss',mode=mode)
        logger.takeStep()
