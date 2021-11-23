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
class deriv(nn.Module):

  def setup(self):
    random_kernel = lambda rng, shape: jax.random.uniform(rng,shape)
    dx = lambda rng, shape: jnp.array([[0,0,0],[-1,1,0],[0,0,0]]).reshape(3,3,1,1).astype(jnp.float32)
    dy = lambda rng, shape: jnp.array([[0,-1,0],[0,1,0],[0,0,0]]).reshape(3,3,1,1).astype(jnp.float32)
    db = lambda rng, shape: jnp.array([0]).astype(jnp.float32)

    self.dx = nn.Conv(1,(3,3),strides=1,kernel_init=random_kernel,use_bias=False,padding='SAME')
    # self.dy = nn.Conv(1,(3,3),strides=1,kernel_init=random_kernel,bias_init=random_kernel,padding='SAME')
    

  def __call__(self,x):
    return nn.relu(self.dx(x))#, self.dy(x)

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
    logger = cvgviz.logger('./logger','tb','autodiff','autodiff_nobias_nonlinear')
    data = [dw,h,w,noisy_image, im_gt]


    f = lambda hp_nn:outer_objective(hp_nn, init_inpt, data)
    lr = 0.01
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
