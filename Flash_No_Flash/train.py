#!/usr/bin/env python3
import optax
import tensorflow as tf
import deepfnf_utils.utils as ut
import deepfnf_utils.tf_utils as tfu
from deepfnf_utils.dataset import Dataset
import jax
from flax import linen as nn
import argparse
import jax.numpy as jnp
import tqdm
from jax.config import config
from jaxopt import OptaxSolver
import cvgutils.Viz as cvgviz
import cvgutils.Image as cvgim
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
class Conv3features(nn.Module):

  def setup(self):
    self.straight1       = nn.Conv(12,(3,3),strides=(1,1),use_bias=True)
    self.straight2       = nn.Conv(16,(3,3),strides=(1,1),use_bias=True)
    self.straight3       = nn.Conv(16,(3,3),strides=(1,1),use_bias=True)
    self.straight4       = nn.Conv(16,(3,3),strides=(1,1),use_bias=True)
    self.straight5       = nn.Conv(3,(3,3),strides=(1,1),use_bias=True)
    self.groupnorm1      = nn.GroupNorm(3)
    self.groupnorm2      = nn.GroupNorm(16)
    self.groupnorm3      = nn.GroupNorm(16)
    self.groupnorm4      = nn.GroupNorm(16)
    self.groupnorm5      = nn.GroupNorm(3)
  @nn.compact
  def __call__(self,x):
    l1 = self.groupnorm1(nn.softplus(self.straight1(x)))
    l2 = self.groupnorm2(nn.softplus(self.straight2(l1)))
    l3 = self.groupnorm3(nn.softplus(self.straight3(l2)))
    l4 = self.groupnorm4(nn.softplus(self.straight4(l3)))
    l5 = self.groupnorm5(nn.softplus(self.straight5(l4)))
    return nn.tanh(l5)

def outer_objective_id(inpt,outpt,params,data):
  """Validation loss."""

  out = Conv3features().apply({'params': params}, inpt)
  
  out = tfu.camera_to_rgb_jax(
      out/data['alpha'], data['color_matrix'], data['adapt_matrix'])
  gt = tfu.camera_to_rgb_jax(
      data['gt'],
      data['color_matrix'], data['adapt_matrix'])
  init = tfu.camera_to_rgb_jax(
      outpt/data['alpha'],
      data['color_matrix'], data['adapt_matrix'])
  l2 = ((out - gt) ** 2).sum()
  return l2, out

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
opts = parser.parse_args()

TLIST = 'data/train.txt'
VPATH = 'data/valset/'

BSZ = opts.batch_size
IMSZ = opts.image_size
displacement = opts.displacement
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
params = Conv3features().init(init_rng, jnp.array(testim))['params']

tf.debugging.set_log_device_placement(True)

dataset.swap_train()
#########################################################################
# Main Training loop


@jax.jit
def predict(im,params):
    return Conv3features().apply({'params': params}, im)

@jax.jit
def loss(params,batch):
    predicted = predict(batch['net_input'],params)
    # predicted = tfu.camera_to_rgb_jax(
    #   predicted/batch['alpha'], batch['color_matrix'], batch['adapt_matrix'])
    # ambient = tfu.camera_to_rgb_jax(
    #     batch['ambient'],
    #     batch['color_matrix'], batch['adapt_matrix'])
    # flash = tfu.camera_to_rgb_jax(
    #     batch['flash']/batch['alpha'],
    #     batch['color_matrix'], batch['adapt_matrix'])
    # noisy = tfu.camera_to_rgb_jax(
    #     batch['noisy']/batch['alpha'],
    #     batch['color_matrix'], batch['adapt_matrix'])
    ambient = batch['ambient']
    flash = batch['flash']
    noisy = batch['noisy']

    out = predicted - ambient
    loss = (out ** 2).mean()
    return loss, {'predicted':jax.lax.stop_gradient(predicted), 'ambient':ambient, 'flash':flash, 'noisy':noisy}



lr = 1e-4
logger = cvgviz.logger('./logger/Flash_No_Flash','filesystem','Flash_No_Flash','convnn_v1')
solver = OptaxSolver(fun=loss, opt=optax.sgd(lr),has_aux=True)
state = solver.init_state(params)
# g = jax.grad(loss,has_aux=True)
@jax.jit
def update(params,state,batch):    
    params, state = solver.update(params, state,batch=batch)
    return params, state
@jax.jit
def update2(params,batch):
    params = jax.tree_multimap(lambda x,y:x-lr*y, params, g(params,batch)[0])
    return params
print('before first iter')
batch = dataset.iterator.next()
print('before preprocess iter')
batch = {k:jnp.array(v.numpy()) for k,v in batch.items()}
batch = preprocess(batch)
print('loading params')
data = logger.load_params()
start_idx=0
if(data is not None):
    # state = data['state']
    batch = data['state']
    params = data['params']
    start_idx = data['idx']
# start_idx=0
# batch = {'alpha':jnp.ones([1,1,1,1]),
# 'ambient':jnp.ones([1,448,448,3]),
# 'noisy':jnp.ones([1,448,448,3]),
# 'flash':jnp.ones([1,448,448,3]),
# 'net_input':jnp.ones([1,448,448,12]),
# 'adapt_matrix':jnp.ones([1,3,3]),
# 'color_matrix':jnp.ones([1,3,3])}
for i in tqdm.trange(int(start_idx), int(opts.max_iter)):
    # Run training step and print losses
    # testim = sess.run(net_input)
    # params = update2(params,batch)
    print('before update')
    params, state = update(params,state,batch)
    print('after')
    print('after log')
    print(state.val)
    if(i % 100 == 0):
        imshow = jnp.concatenate((state.aux['predicted'],state.aux['ambient'],state.aux['noisy'],state.aux['flash']),axis=2)
        imshow = jnp.clip(imshow,0,1)
        logger.addImage(imshow[0],'imshow')
        logger.save_params(params,batch,i)
    logger.addScalar(state.val,'loss')
    logger.takeStep()
