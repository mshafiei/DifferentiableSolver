#!/usr/bin/env python3
import optax
import tensorflow as tf
import deepfnf_utils.utils as ut
import deepfnf_utils.tf_utils as tfu
from deepfnf_utils.dataset import Dataset
import jax
from cvgutils.nn import jaxutils

import argparse
import jax.numpy as jnp
import tqdm
from jax.config import config
from jaxopt import OptaxSolver
import cvgutils.Viz as cvgviz
import cvgutils.Image as cvgim
import numpy as np
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
parser.add_argument('--batch_size', default=1, help='Image count in a batch')
parser.add_argument('--image_size', default=448, help='Width and height of an image')
parser.add_argument('--displacement', default=0, help='Random shift in pixels')
parser.add_argument('--min_scale', default=1, help='Random shift in pixels')
parser.add_argument('--max_scale', default=1, help='Random shift in pixels')
parser.add_argument('--max_rotate', default=0, help='Maximum rotation')
parser.add_argument('--lr', default=1e-4, help='Maximum rotation')
parser.add_argument('--max_iter', default=1e6, help='Maximum rotation')
parser.add_argument('--viz_freq', default=20, help='Maximum rotation')
parser.add_argument('--save_param_freq', default=20, help='Maximum rotation')
parser.add_argument('--ngpus', type=int, default=1, help='use how many gpus')
parser.add_argument('--model', type=str, default='overfit_unet',
choices=['overfit_straight','interpolate_straight','overfit_unet','interpolate_unet'],help='Which model to use')
opts = parser.parse_args()

TLIST = 'data/train.txt'
VPATH = 'data/valset/'

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
dataset = Dataset(TLIST, VPATH, bsz=BSZ, psz=IMSZ,
                    ngpus=opts.ngpus, nthreads=4 * opts.ngpus,
                    jitter=displacement, min_scale=opts.min_scale, max_scale=opts.max_scale, theta=opts.max_rotate)

dataset.swap_train()

#########################################################################

# Check for saved weights & optimizer states
def preprocess(example):
    key1 = jax.random.PRNGKey(100)
    key2 = jax.random.PRNGKey(101)
    key3 = jax.random.PRNGKey(102)
    key4 = jax.random.PRNGKey(103)
    key5 = jax.random.PRNGKey(100)
    key6 = jax.random.PRNGKey(101)
    key7 = jax.random.PRNGKey(102)
    key8 = jax.random.PRNGKey(103)
    key9 = jax.random.PRNGKey(102)
    key10 = jax.random.PRNGKey(103)

    
    # for i in range(opts.ngpus):
        # with tf.device('/gpu:%d' % i):
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

    noisy = jnp.concatenate([noisy_ambient, noisy_flash], axis=-1)
    noise_std = tfu.estimate_std_jax(noisy, sig_read, sig_shot)
    net_input = jnp.concatenate([noisy, noise_std], axis=-1)
    
    output = {
        'alpha':alpha,
        'ambient':example['ambient'],
        'flash':noisy_flash,
        'noisy':noisy_ambient,
        'net_input':net_input,
        'adapt_matrix':jnp.array(example['adapt_matrix']),
        'color_matrix':jnp.array(example['color_matrix'])
    }

    return output



rng = jax.random.PRNGKey(1)
rng, init_rng = jax.random.split(rng)
batch = dataset.iterator.next()
batch = {k:jnp.array(v.numpy()) for k,v in batch.items()}
preprocessed = preprocess(batch)
testim = preprocessed['net_input']
if(opts.model == 'overfit_straight'):
    init_model = lambda rng, x: jaxutils.StraightCNN().init(rng, x)['params']
    model = lambda params, batch: jaxutils.StraightCNN().apply({'params': params}, batch['net_input'])
elif(opts.model == 'interpolate_straight'):
    init_model = lambda rng, x: jaxutils.StraightCNN().init(rng, x)['params']
    model = lambda params, batch: jaxutils.StraightCNN().apply({'params': params}, batch['net_input']) + batch['noisy']
elif(opts.model == 'overfit_unet'):
    init_model = lambda rng, x: jaxutils.UNet(4).init(rng, x)['params']
    model = lambda params, batch: jaxutils.UNet(4).apply({'params': params}, batch['net_input'])
elif(opts.model == 'interpolate_unet'):
    init_model = lambda rng, x: jaxutils.UNet(4).init(rng, x)['params']
    model = lambda params, batch: jaxutils.UNet(4).apply({'params': params}, batch['net_input']) + batch['noisy']
else:
    print('Model unrecognized')
    exit(0)

params = init_model(init_rng, jnp.array(testim))

tf.debugging.set_log_device_placement(True)

dataset.swap_train()
#########################################################################
# Main Training loop


@jax.jit
def predict(im,params):
    return model(params, im)

@jax.jit
def loss(params,batch):
    predicted = model(params,batch)
    g = tfu.camera_to_rgb_jax(
      predicted/batch['alpha'], batch['color_matrix'], batch['adapt_matrix'])
    ambient = tfu.camera_to_rgb_jax(
        batch['ambient'],
        batch['color_matrix'], batch['adapt_matrix'])
    flash = tfu.camera_to_rgb_jax(
        batch['flash']/batch['alpha'],
        batch['color_matrix'], batch['adapt_matrix'])
    noisy = tfu.camera_to_rgb_jax(
        batch['noisy']/batch['alpha'],
        batch['color_matrix'], batch['adapt_matrix'])
    f = g
    diff = f - ambient
    loss = (diff ** 2).mean()
    return loss, {'predicted':jax.lax.stop_gradient(f), 'ambient':ambient, 'flash':flash, 'noisy':noisy}



lr = 1e-4
logger = cvgviz.logger('./logger/Flash_No_Flash','filesystem','Flash_No_Flash','convnn_interpolation_256')
solver = OptaxSolver(fun=loss, opt=optax.adam(lr),has_aux=True)
state = solver.init_state(params)
# g = jax.grad(loss,has_aux=True)
@jax.jit
def update(params,state,batch):    
    params, state = solver.update(params, state,batch=batch)
    return params, state
batch = dataset.iterator.next()
batch = {k:jnp.array(v.numpy()) for k,v in batch.items()}
batch = preprocess(batch)
data = logger.load_params()
start_idx=0
if(data is not None):
    # state = data['state']
    batch = data['state']
    params = data['params']
    start_idx = data['idx']
with tqdm.trange(int(start_idx), int(opts.max_iter)) as t:
    for i in t:
        params, state = update(params,state,batch)
        t.set_description('Error '+str(np.array(state.value)))
        if(i % 1000 == 0):
            imshow = jnp.concatenate((state.aux['predicted'],state.aux['ambient'],state.aux['noisy'],state.aux['flash']),axis=2)
            imshow = jnp.clip(imshow,0,1)
            logger.addImage(imshow[0],'imshow')
            logger.save_params(params,batch,i)
        logger.addScalar(state.value,'loss')
        logger.takeStep()
