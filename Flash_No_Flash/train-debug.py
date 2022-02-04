#!/usr/bin/env python3
import tensorflow as tf
import jax
from flax import linen as nn
import jax.numpy as jnp
import tqdm
from jax.config import config
config.update("jax_debug_nans", True)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"

################ inner loop model end ############################
class Conv3features(nn.Module):

  def setup(self):
    self.straight1       = nn.Conv(12,(3,3),strides=(1,1),use_bias=True)
    self.straight2       = nn.Conv(256,(3,3),strides=(1,1),use_bias=True)
    self.straight3       = nn.Conv(256,(3,3),strides=(1,1),use_bias=True)
    self.straight4       = nn.Conv(256,(3,3),strides=(1,1),use_bias=True)
    self.straight5       = nn.Conv(3,(3,3),strides=(1,1),use_bias=True)
    self.groupnorm1      = nn.GroupNorm(3)
    self.groupnorm2      = nn.GroupNorm(32)
    self.groupnorm3      = nn.GroupNorm(32)
    self.groupnorm4      = nn.GroupNorm(32)
    self.groupnorm5      = nn.GroupNorm(3)
  @nn.compact
  def __call__(self,x):
    l1 = self.groupnorm1(nn.softplus(self.straight1(x)))
    l2 = self.groupnorm2(nn.softplus(self.straight2(l1)))
    l3 = self.groupnorm3(nn.softplus(self.straight3(l2)))
    l4 = self.groupnorm4(nn.softplus(self.straight4(l3)))
    l5 = self.groupnorm5(nn.softplus(self.straight5(l4)))
    return nn.tanh(l5)


@jax.jit
def predict(im,params):
    return Conv3features().apply({'params': params}, im)

@jax.jit
def loss(params,batch):
    predicted = predict(batch['net_input'],params)
    ambient = batch['ambient']
    out = predicted - ambient
    loss = (out ** 2).mean()
    return loss

g = jax.grad(loss)
lr = 1e-4
@jax.jit
def update(params,batch):
    params = jax.tree_multimap(lambda x,y:x-lr*y, params, g(params,batch))
    return params
start_idx=0
max_iter=100000
batch = {'alpha':jnp.ones([1,1,1,1]),
'ambient':jnp.ones([1,448,448,3]),
'noisy':jnp.ones([1,448,448,3]),
'flash':jnp.ones([1,448,448,3]),
'net_input':jnp.ones([1,448,448,12]),
'adapt_matrix':jnp.ones([1,3,3]),
'color_matrix':jnp.ones([1,3,3])}

rng = jax.random.PRNGKey(1)
rng, init_rng = jax.random.split(rng)
params = Conv3features().init(init_rng, batch['net_input'])['params']
for i in tqdm.trange(0, max_iter):
    params = update(params,batch)