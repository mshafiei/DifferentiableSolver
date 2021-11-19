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

class deriv(nn.Module):

  def setup(self):
    dh = lambda rng, shape: 10*jnp.array([[0,-1,0],[-1,2,0],[0,0,0]]).reshape(3,3,1,1)
    db = lambda rng, shape: jnp.array([0])

    self.layer1 = nn.Conv(1,(3,3),strides=1,kernel_init=dh,bias_init=db,padding='VALID')
    

  def __call__(self,x):
    return self.layer1(x)

# h, w = 4, 4
# cnn = deriv()
# rng = jax.random.PRNGKey(1)
# testim = jax.random.uniform(rng,[1, h, w, 1])
# rng, init_rng = jax.random.split(rng)
# params = cnn.init(init_rng, testim)['params']
# out = deriv().apply({'params': params}, testim)
# dx = testim[0,:,1:,:] - testim[0,:,:-1,:]
# dx[1:-1,:-1,0]
# @jax.jit
def stencil_residual(pp_image, hp_nn, data):
  dw, h, w, inpt = data
  """Objective function."""
  avg_weight =  1.#1. / 425040.# ** 0.5
  r1 =  pp_image - inpt
  # regularizer = jsp.signal.convolve(pp_image.reshape(int(h),int(w),3), hp_nn.reshape(int(dw),int(dw),3) , mode='same')
  dh = jnp.repeat(jnp.array([[0,0,0],[-1,1,0],[0,0,0]])[:,:,None],3,-1)
  dv = jnp.repeat(jnp.array([[0,-1,0],[0,1,0],[0,0,0]])[:,:,None],3,-1)
  r2 = pp_image[1:,:,:] - pp_image[:-1,:,:]
  r3 = pp_image[:,1:,:] - pp_image[:,:-1,:]
  
  out = jnp.concatenate(( r1.reshape(-1), r2.reshape(-1),r3.reshape(-1)),axis=0)
  # c2 = jsp.signal.convolve(pp_image.reshape(int(h),int(w),3), dh.reshape(int(dw),int(dw),3) , mode='same')
  # c3 = jsp.signal.convolve(pp_image.reshape(int(h),int(w),3), dv.reshape(int(dw),int(dw),3) , mode='same')
  # out = jnp.concatenate((r1.reshape(-1), r2.reshape(-1),r3.reshape(-1)),axis=0)
  # d1 = deriv().apply({'params': hp_nn}, pp_image[None,:,:,0:1])
  # d2 = deriv().apply({'params': hp_nn}, pp_image[None,:,:,1:2])
  # d3 = deriv().apply({'params': hp_nn}, pp_image[None,:,:,2:])
  # d = jnp.concatenate((d1,d2,d3),axis=-1)

  # out = jnp.concatenate((r1.reshape(-1), d.reshape(-1)),axis=0)
  return avg_weight * out


# @jax.jit
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
    # step = logger.step
    # logger.addScalar(np.float32(loss(x).sum()),'loss_GN_%04i'%(step))
    for i in range(gn_iters):
        x += linear_solve.solve_cg(matvec=matvec,
                                b=-jtf(x),
                                init=x,
                                maxiter=100)
        # logger.addScalar(np.float32(loss(x).sum()),'loss_GN_%04i'%(step))
        # logger.takeStep()
    # print(x[:3,:3])
    return x

# @jax.jit
def outer_objective(hp_nn, init_inner, data):
    """Validation loss."""
    _,_,_,_, gt = data
    f = lambda hp_nn: screen_poisson_solver(init_inner, hp_nn, data[:-1])
    # a = f(hp_nn)
    f_v = (f(hp_nn) - gt) ** 2
    return f_v.sum()

def fd(hyper_params, init_inner, data,delta):
  dw = data[0]
  grad = jnp.zeros_like(hyper_params.reshape(-1))
  for i in tqdm.trange(hyper_params.reshape(-1).shape[0]):
    winf = hyper_params.reshape(-1)
    winb = hyper_params.reshape(-1)
    winf = winf.at[i].add(delta/2)
    winb = winb.at[i].add(-delta/2)
    f = outer_objective(winf.reshape(dw,dw,3), init_inner, data)
    b = outer_objective(winb.reshape(dw,dw,3), init_inner, data)
    grad = grad.at[i].set((f - b) / delta)
  return grad.reshape(dw,dw,3)

def hyper_optimization():
    dw = 3
    key1 = jax.random.PRNGKey(42)
    key2 = jax.random.PRNGKey(43)
    key3 = jax.random.PRNGKey(44)
    key4 = jax.random.PRNGKey(45)
    gt_image = cvgim.imread('~/Projects/cvgutils/tests/testimages/wood_texture.jpg')
    gt_image = cvgim.resize(gt_image,scale=0.10)
    noise = jax.random.uniform(key4,gt_image.shape)
    noisy_image = jnp.clip(gt_image + noise,0,1)
    
    noisy_image = jnp.zeros_like(gt_image)
    noisy_image = noisy_image.at[100,100,:].set(1)
    # init_inpt = jax.random.uniform(key3,noisy_image.shape)
    init_inpt = jnp.zeros_like(gt_image)
    init_inpt = init_inpt.at[100,100,:].set(1)
    im_gt = jnp.array(gt_image)
    h,w = gt_image.shape[0],gt_image.shape[1]

    cnn = deriv()
    rng = jax.random.PRNGKey(1)
    testim = jax.random.uniform(rng,[1, h, w, 1])
    rng, init_rng = jax.random.split(rng)
    params = cnn.init(init_rng, testim)['params']

    rng = jax.random.PRNGKey(0)
    in_shape = (-1, 1, 1)

    x = jnp.linspace(-1, 1, dw)
    window_gt = jsp.stats.norm.pdf(x) * jsp.stats.norm.pdf(x[:, None])
    window_gt /= jnp.linalg.norm(window_gt)
    window_gt = jnp.repeat(window_gt[:,:,None],3,axis=-1)
    # window = jnp.zeros_like(window_gt)
    window = jnp.array([[0,-1,0],[-1,2,0],[0,0,0]])
    window = jnp.repeat(window[:,:,None],3,axis=-1)
    logger = cvgviz.logger('./logger','filesystem','autodiff','autodiff')

    data = [dw,h,w,noisy_image, im_gt]
    
    lr = 0.002
    f = lambda hp_nn:outer_objective(hp_nn, init_inpt, data)
    imshow = screen_poisson_solver(init_inpt, params, data[:-1])
    imshow2 = jnp.concatenate((imshow,noisy_image))
    logger.addImage(np.array(imshow2*20).transpose(2,0,1),'Image')
    for i in tqdm.trange(2000):
      a = fd(window, init_inpt, data,0.0001)
      loss,grad = jax.value_and_grad(f)(params)
      logger.addScalar(loss,'loss_GD')
      if(i%100 == 0):
        output = screen_poisson_solver(init_inpt, window, data[:-1])
        imshow = jnp.concatenate((np.clip(output,0,1),noisy_image,im_gt),axis=1)
        logger.addImage(np.array(imshow).transpose(2,0,1),'Image')
      logger.takeStep()
      # plt.figure(figsize=(24,7))
      # plt.subplot(1,4,1)
      # plt.plot(lmbdas,valid_loss,'r')
      # plt.scatter(lmbda_init,valid_loss[0],color='r')
      # plt.scatter(lmbda_gt,valid_loss[-1],color='g')
      # plt.arrow(window,loss,1,g[0],color='b')
      # plt.subplot(1,4,2)
      # plt.imshow(im_gt)
      # plt.subplot(1,4,3)
      # plt.imshow(inpt[0])
      # plt.subplot(1,4,4)
      # plt.imshow(inpt[1])
      # plt.savefig('out/plt_%04d.png'%i)
      
      print('loss ', loss)
      window += lr * grad


hyper_optimization()
