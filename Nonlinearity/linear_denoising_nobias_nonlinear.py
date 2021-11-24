from absl import app
import jax
from jax._src.numpy.lax_numpy import argsort, interp, zeros_like
import jax.numpy as jnp
from jaxopt import implicit_diff
from jaxopt import linear_solve
from jaxopt import OptaxSolver, GradientDescent
from matplotlib.pyplot import vlines
import optax
from sklearn import datasets
from sklearn import model_selection
from sklearn import preprocessing
import matplotlib.pylab as plt
import numpy as np
import jax.scipy as jsp
import tqdm
import cvgutils.Viz as cvgviz
import cvgutils.nn.jaxutils as jaxutils
from jax.experimental import stax
import cvgutils.Image as cvgim
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from flax import optim

from flax import linen as nn
import jax.numpy as jnp
import jax
from flax.training import train_state
import optax                          # The Optax gradient processing and optimization library
import numpy as np                    # Ordinary NumPy

class UNet(nn.Module):

  def setup(self):
    random_kernel = lambda rng, shape: jax.random.uniform(rng,shape)
    dx = lambda rng, shape: jnp.array([[0,0,0],[-1,1,0],[0,0,0]]).reshape(3,3,1,1).astype(jnp.float32)
    dy = lambda rng, shape: jnp.array([[0,-1,0],[0,1,0],[0,0,0]]).reshape(3,3,1,1).astype(jnp.float32)
    db = lambda rng, shape: jnp.array([0]).astype(jnp.float32)

    self.dx = nn.Conv(1,(3,3),strides=1,kernel_init=random_kernel,use_bias=False,padding='SAME')
    # self.dy = nn.Conv(1,(3,3),strides=1,kernel_init=random_kernel,bias_init=random_kernel,padding='SAME')
    

  def __call__(self,x):
    return self.dx(x)#, self.dy(x)

class UNet2(nn.Module):

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

@jax.jit
def train_step(state, batch):
  def loss_fn(params):
    out = UNet().apply({'params': params}, batch['image'])
    loss = ((out - batch['image']) ** 2).sum()
    return loss
  grad_fn = jax.value_and_grad(loss_fn)
  loss_val, grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  return state,loss_val

def train_epoch(state, train_ds, batch_size, epoch, rng):
  train_ds_size = len(train_ds['image'])
  steps_per_epoch = train_ds_size // batch_size

  perms = jax.random.permutation(rng, len(train_ds['image']))
  perms = perms[:steps_per_epoch * batch_size]  # Skip an incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))

  # batch_metrics = []

  for perm in perms:
    batch = {k: v[perm, ...] for k, v in train_ds.items()}
    state, loss_val = train_step(state, batch)
    # batch_metrics.append(metrics)

  # training_batch_metrics = jax.device_get(batch_metrics)
  # training_epoch_metrics = {
      # k: np.mean([metrics[k] for metrics in training_batch_metrics])
      # for k in training_batch_metrics[0]}

  print('Training - epoch: %d, loss: %.4f' % (epoch, loss_val))

  return state, loss_val

num_epochs = 10
batch_size = 1
h,w = 256, 256
cnn = UNet()
rng = jax.random.PRNGKey(1)
rng, init_rng = jax.random.split(rng)
params = cnn.init(init_rng, jnp.ones([1, h, w, 1]))['params']
nesterov_momentum = 0.9
learning_rate = 0.01
tx = optax.sgd(learning_rate=learning_rate, nesterov=nesterov_momentum)
state = train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=tx)
unet = UNet()
train_ds = {'image':jnp.ones((1,256,256,1))}
params = unet.init(jax.random.PRNGKey(0),train_ds['image'])

for epoch in range(1, num_epochs + 1):
  # Use a separate PRNG key to permute image data during shuffling
  rng, input_rng = jax.random.split(rng)
  # Run an optimization step over a training batch
  state, train_metrics = train_epoch(state, train_ds, batch_size, epoch, input_rng)
  # Evaluate on the test set after each training epoch
  # test_loss, test_accuracy = eval_model(state.params, test_ds)
  # print('Testing - epoch: %d, loss: %.2f, accuracy: %.2f' % (epoch, test_loss, test_accuracy * 100))
  
print('hi')

class deriv(nn.Module):

  def setup(self):
    random_kernel = lambda rng, shape: jax.random.uniform(rng,shape)
    dx = lambda rng, shape: jnp.array([[0,0,0],[-1,1,0],[0,0,0]]).reshape(3,3,1,1).astype(jnp.float32)
    dy = lambda rng, shape: jnp.array([[0,-1,0],[0,1,0],[0,0,0]]).reshape(3,3,1,1).astype(jnp.float32)
    db = lambda rng, shape: jnp.array([0]).astype(jnp.float32)

    self.dx = nn.Conv(1,(3,3),strides=1,kernel_init=random_kernel,use_bias=True,padding='SAME')
    # self.dy = nn.Conv(1,(3,3),strides=1,kernel_init=random_kernel,bias_init=random_kernel,padding='SAME')
    

  def __call__(self,x):
    return self.dx(x)#, self.dy(x)

@jax.jit
def stencil_residual(pp_image, hp_nn, data):
  dw, h, w, inpt = data
  """Objective function."""
  avg_weight = (1. / 2.) ** 0.5 *  (1. / pp_image.reshape(-1).shape[0] ** 0.5)
  r1 =  pp_image - inpt
  flag = False
  if(flag):
    dy = pp_image[1:,:,:] - pp_image[:-1,:,:]
    dx = pp_image[:,1:,:] - pp_image[:,:-1,:]
  else:
    dx1 = deriv().apply({'params': hp_nn}, pp_image[None,:,:,0:1])
    dx2 = deriv().apply({'params': hp_nn}, pp_image[None,:,:,1:2])
    dx3 = deriv().apply({'params': hp_nn}, pp_image[None,:,:,2:])
    dx = jnp.concatenate((dx1,dx2,dx3),axis=-1)
    # dy = jnp.concatenate((dy1,dy2,dy3),axis=-1)

  out = jnp.concatenate(( r1.reshape(-1), dx.reshape(-1)),axis=0)
  return avg_weight * out


@jax.jit
def screen_poisson_objective(pp_image, hp_nn, data):
  """Objective function."""
  return (stencil_residual(pp_image, hp_nn, data) ** 2).sum()


@implicit_diff.custom_root(jax.grad(screen_poisson_objective))
def screen_poisson_solver(init_image,hp_nn, data):
    f = lambda pp_image:stencil_residual(pp_image,hp_nn,data)
    loss = lambda pp_image:screen_poisson_objective(pp_image,hp_nn,data)
    def matvec(pp_image):
        jtd = jax.jvp(f,(init_image,),(pp_image,))[1]
        return jax.vjp(f,init_image)[1](jtd)[0]
    def jtf(x):
      return jax.vjp(f,x)[1](f(x))[0]

    gn_iters = 3
    x = init_image
    for i in range(gn_iters):
        x += linear_solve.solve_cg(matvec=matvec,
                                b=-jtf(x),
                                init=x,
                                maxiter=100)

    return x

@jax.jit
def outer_objective(hp_nn, init_inner, data):
    """Validation loss."""
    _,_,_,_, gt = data
    f = lambda hp_nn: screen_poisson_solver(init_inner, hp_nn, data[:-1])
    f_v = ((f(hp_nn) - gt) ** 2).mean()
    return f_v

def fd(hyper_params, init_inner, data,delta):
  from jax.tree_util import tree_flatten, tree_unflatten
  grad_flat, grad_tree = tree_flatten(hyper_params)
  for i in tqdm.trange(len(grad_flat)):
    value_flat, value_tree = tree_flatten(hyper_params)
    shape = value_flat[i].shape
    for j in tqdm.trange(value_flat[i].reshape(-1).shape[0]):
      vff = value_flat.copy()
      vfb = value_flat.copy()
      vff[i] = vff[i].reshape(-1).at[j].set(vff[i].reshape(-1)[j] + delta/2)
      vfb[i] = vfb[i].reshape(-1).at[j].set(vfb[i].reshape(-1)[j] - delta/2)
      vff[i] = vff[i].reshape(*shape)
      vfb[i] = vfb[i].reshape(*shape)
      vff_tree = tree_unflatten(value_tree, vff)
      vfb_tree = tree_unflatten(value_tree, vfb)
      ff = outer_objective(vff_tree, init_inner, data)
      fb = outer_objective(vfb_tree, init_inner, data)
      grad_flat[i] = grad_flat[i].reshape(-1).at[j].set((ff - fb) / delta)
    grad_flat[i] = grad_flat[i].reshape(*shape)
  grad_tree = tree_unflatten(grad_tree, grad_flat)
  return grad_tree

def hyper_optimization():
    dw = 3
    key4 = jax.random.PRNGKey(45)
    gt_image = cvgim.imread('~/Projects/cvgutils/tests/testimages/wood_texture.jpg')
    gt_image = cvgim.resize(gt_image,scale=0.10) * 2
    noise = jax.random.normal(key4,gt_image.shape) * 0.3
    noisy_image = jnp.clip(gt_image + noise,0,1)
    
    # noisy_image = jnp.zeros_like(gt_image)
    # noisy_image = noisy_image.at[100,100,:].set(1)
    init_inpt = jnp.zeros_like(gt_image)
    # init_inpt = init_inpt.at[100,100,:].set(1)
    im_gt = jnp.array(gt_image)
    h,w = gt_image.shape[0],gt_image.shape[1]

    cnn = deriv()
    rng = jax.random.PRNGKey(1)
    testim = jax.random.uniform(rng,[1, h, w, 1])
    rng, init_rng = jax.random.split(rng)
    params = cnn.init(init_rng, testim)['params']

    rng = jax.random.PRNGKey(0)
    logger = cvgviz.logger('./logger','tb','autodiff','autodiff_withbias')
    data = [dw,h,w,noisy_image, im_gt]


    f = lambda hp_nn:outer_objective(hp_nn, init_inpt, data)
    lr = 0.03
    delta = 0.0001
    solver = OptaxSolver(fun=f, opt=optax.adam(lr), implicit_diff=True)
    state = solver.init_state(params)
    # result, _ = solver.run(init_params = state)
    f_t = jax.jit(screen_poisson_solver)

    for i in tqdm.trange(10000):
    #   # a = fd(params, init_inpt, data,0.01)
      
      # loss,grad = jax.value_and_grad(f)(params)
      params, state = solver.update(params=params, state=state)
      # params = optax.apply_updates(params, updates)
      loss = f(params)
      print('loss ',loss)
      logger.addScalar(loss,'loss_GD')
      if(i%10 == 0):
        output = f_t(init_inpt, params, data[:-1])
        imshow = jnp.concatenate((output,noisy_image,im_gt),axis=1)
        imshow = jnp.clip(imshow,0,1)
        logger.addImage(np.array(imshow).transpose(2,0,1),'Image')
      logger.takeStep()

      
    #   print('loss ', loss)
    #   params = jax.tree_multimap(lambda x,dfx:  x - lr * dfx, params, grad)


hyper_optimization()
