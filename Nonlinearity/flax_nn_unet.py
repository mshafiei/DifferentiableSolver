import jax
import jax.numpy as jnp                # JAX NumPy

from flax import linen as nn           # The Linen API
from flax.training import train_state  # Useful dataclass to keep train state

import numpy as np                     # Ordinary NumPy
import optax                           # Optimizers
import tensorflow_datasets as tfds     # TFDS for MNIST
import cvgutils.Image as cvgim
import cvgutils.Viz as cvgviz

logger = cvgviz.logger('./logger','tb','autodiff','unet_overfit_test_rgb')

class UNet(nn.Module):

  def setup(self):
    self.layer1         = nn.Conv(3,(3,3),strides=1)
    self.group_l1         = nn.normalization.GroupNorm(3)
    self.down1          = nn.Conv(16,(3,3),strides=2)
    self.group1         = nn.normalization.GroupNorm(16)
    self.down2          = nn.Conv(32,(3,3),strides=2)
    self.group2         = nn.normalization.GroupNorm(32)
    self.down3          = nn.Conv(64,(3,3),strides=2)
    self.group3         = nn.normalization.GroupNorm(32)
    self.down4          = nn.Conv(128,(3,3),strides=2)
    self.group4         = nn.normalization.GroupNorm(32)
    self.latent         = nn.Conv(256,(1,1),strides=1)
    self.group_latent   = nn.normalization.GroupNorm(32)
    self.up4            = nn.ConvTranspose(256+128,(2,2),strides=(2,2))
    self.group_up4      = nn.normalization.GroupNorm(32)
    self.up3            = nn.ConvTranspose(128+64,(2,2),strides=(2,2))
    self.group_up3      = nn.normalization.GroupNorm(32)
    self.up2            = nn.ConvTranspose(64+32, (2,2),strides=(2,2))
    self.group_up2      = nn.normalization.GroupNorm(32)
    self.up1            = nn.ConvTranspose(32+16, (2,2),strides=(2,2))
    self.group_up1      = nn.normalization.GroupNorm(16)
    self.straight1       = nn.Conv(16+3,(3,3),strides=(1,1))
    self.group_straight1 = nn.normalization.GroupNorm(16+3)
    self.straight2       = nn.Conv(3,(3,3),strides=(1,1))
    self.group_straight2 = nn.normalization.GroupNorm(3)

  def __call__(self,x):
    out_l1 = nn.relu(self.group_l1(self.layer1(x)))
    out_1 = nn.relu(self.group1(self.down1(out_l1)))
    out_2 = nn.relu(self.group2(self.down2(out_1)))
    out_3 = nn.relu(self.group3(self.down3(out_2)))
    out_4 = nn.relu(self.group4(self.down4(out_3)))
    out_latent = nn.relu(self.group_latent(self.latent(out_4)))
    in_up4 = jnp.concatenate((out_4,out_latent),axis=-1)
    out_up4 = nn.relu(self.group_up4(self.up4(in_up4)))
    in_up3 = jnp.concatenate((out_3,out_up4),axis=-1)
    out_up3 = nn.relu(self.group_up3(self.up3(in_up3)))
    in_up2 = jnp.concatenate((out_2,out_up3),axis=-1)
    out_up2 = nn.relu(self.group_up2(self.up2(in_up2)))
    in_up1 = jnp.concatenate((out_1,out_up2),axis=-1)
    out_up1 = nn.relu(self.group_up1(self.up1(in_up1)))
    in_straight1 = jnp.concatenate((out_l1,out_up1),axis=-1)
    out_straight1 = nn.relu(self.group_straight1(self.straight1(in_straight1)))
    return nn.relu(self.group_straight2(self.straight2(out_straight1)))


# class CNN(nn.Module):
#   """A simple CNN model."""

#   @nn.compact
#   def __call__(self, x):
#     x = nn.Conv(features=32, kernel_size=(3, 3),strides=1)(x)
#     x = nn.relu(x)
#     return x


def get_datasets():
  """Load MNIST train and test datasets into memory."""
  ds_builder = tfds.builder('mnist')
  ds_builder.download_and_prepare()
  train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
  test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
  train_ds['image'] = jnp.float32(train_ds['image']) / 255.
  test_ds['image'] = jnp.float32(test_ds['image']) / 255.
  return train_ds, test_ds

def create_train_state(rng, learning_rate, momentum):
  """Creates initial `TrainState`."""
  cnn = UNet()
  params = cnn.init(rng, jnp.ones([1, 128, 128, 3]))['params']
  tx = optax.sgd(learning_rate, momentum)
  return train_state.TrainState.create(
      apply_fn=cnn.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch):
  """Train for a single step."""
  def loss_fn(params):
    unet_out = UNet().apply({'params': params}, batch['image'])
    # loss = cross_entropy_loss(logits=logits, labels=batch['label'])
    loss = ((unet_out - batch['image']) ** 2).mean()
    return loss, unet_out
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, unet_out), grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  return state, loss, unet_out

def train_epoch(state, train_ds, batch_size, epoch, rng):
  """Train for a single epoch."""
  train_ds_size = len(train_ds['image'])
  steps_per_epoch = train_ds_size // batch_size

  perms = jax.random.permutation(rng, train_ds_size)
  perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))
  batch_metrics = []
  for perm in perms:
    batch = {k: v[perm, ...] for k, v in train_ds.items()}
    state, loss, unet_out= train_step(state, batch)
  imshow = jnp.concatenate((unet_out,batch['image']),axis=2)[0]
  logger.addImage(jnp.clip(imshow,0,1).transpose(2,0,1),'output_input')
  logger.addScalar(loss,'training_loss')
  logger.takeStep()
  print('train epoch: %d, loss: %.4f' % (
      epoch, loss))

  return state

train_ds, test_ds = get_datasets()
gt_image = cvgim.imread('~/Projects/cvgutils/tests/testimages/wood_texture.jpg')
gt_image = cvgim.resize(gt_image,scale=0.10)[:128,:256,:] * 2
train_ds['image'] = jnp.array(gt_image)[None,...]
# x, y = jnp.meshgrid(jnp.linspace(0,1,128),jnp.linspace(0,1,128))
# xy = jnp.stack((x,y),axis=0)
# xy = jnp.stack((xy,xy,xy),axis=-1)
# a = jax.scipy.ndimage.map_coordinates(train_ds['image'],xy,order=1)

rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)

learning_rate = 0.01
momentum = 0.9
state = create_train_state(init_rng, learning_rate, momentum)
del init_rng  # Must not be used anymore.

num_epochs = 10000
batch_size = 1

for epoch in range(1, num_epochs + 1):
  # Use a separate PRNG key to permute image data during shuffling
  rng, input_rng = jax.random.split(rng)
  # Run an optimization step over a training batch
  state = train_epoch(state, train_ds, batch_size, epoch, input_rng)