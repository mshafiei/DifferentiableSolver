import jax
import jax.numpy as jnp
from jaxopt import implicit_diff
from jaxopt import linear_solve
import numpy as np
import tqdm
import cvgutils.Viz as cvgviz
from jax.experimental import stax
from jax.config import config
import cvgutils.Image as cvgim
from jax_resnet import pretrained_resnest
config.update("jax_debug_nans", True)

lmbda_init = 0.1
lmbda_gt = 0.9
h,w = 100,100
dw = 3
key1 = jax.random.PRNGKey(42)
key2 = jax.random.PRNGKey(43)
key3 = jax.random.PRNGKey(44)
key4 = jax.random.PRNGKey(45)
rng = jax.random.PRNGKey(0)


net_init, net_apply = stax.serial(stax.Conv(32, (3, 3),strides=2, padding='SAME'),stax.Relu,
                                  stax.Conv(3, (3, 3), padding='SAME'),stax.Relu)
out_shape, net_params = net_init(rng, (-1, h, w, 3))

logger = cvgviz.logger('./logger','filesystem','autodiff','autodiff')

@jax.jit
def stencil_residual(pp_image, hp_nn, y):
  """Objective function."""
  avg_weight = 0.5 ** 0.5 * 1 / len(y.reshape(-1)) ** 0.5
  r1 =  pp_image[None,:,:,:] - y[None,:,:,:]
  r2 =  net_apply(hp_nn,pp_image[None,:,:,:])
  out = jnp.concatenate((r1.reshape(-1),r2.reshape(-1)),axis=0)
  return avg_weight * out


@jax.jit
def screen_poisson_objective(pp_image, hp_nn, y):
  """Objective function."""
  return (stencil_residual(pp_image, hp_nn, y) ** 2).sum()


@implicit_diff.custom_root(jax.grad(screen_poisson_objective))
def screen_poisson_solver(init_image,hp_nn, y):
    f = lambda pp_image:stencil_residual(pp_image,hp_nn,y)
    loss = lambda pp_image:screen_poisson_objective(pp_image,hp_nn,y)

    gn_iters = 3
    x = init_image
    step = logger.step
    print('Start GN')
    for k in range(gn_iters):
      def jtj(d):
          jtd = jax.jvp(f,(x,),(d,))[1]
          ret = jax.vjp(f,x)[1](jtd)[0]
          return ret
      def jtf(x):
        ret = jax.vjp(f,x)[1](f(x))[0]
        return ret
      x += linear_solve.solve_cg(matvec=jtj,
                              b=-jtf(x),
                              init=jnp.zeros_like(x),
                              maxiter=200)
    return x

@jax.jit
def outer_objective(hp_nn, init_inner, x,y):
    """Validation loss."""
    f = lambda hp_nn: screen_poisson_solver(init_inner, hp_nn, y)
    f_v = (f(hp_nn) - x) ** 2
    return f_v.mean()


def hyper_optimization():
    
    gt_image = cvgim.imread('~/Projects/cvgutils/tests/testimages/wood_texture.jpg')
    gt_image = cvgim.resize(gt_image,scale=0.10)
    noise = jax.random.uniform(key4,gt_image.shape)
    noisy_image = jnp.clip(gt_image + noise,0,1)
    init_inpt = jax.random.uniform(key3,noisy_image.shape)

    
    lr = 0.2
    f = lambda hp_nn:outer_objective(hp_nn, init_inpt, gt_image,noisy_image)
    
    for i in tqdm.trange(2000):
      loss,grad = jax.value_and_grad(f)(net_params)
      logger.addScalar(loss,'loss_GD')
      if(i%100 == 0):
        output = screen_poisson_solver(init_inpt, net_params, noisy_image)
        imshow = jnp.concatenate((gt_image,np.clip(output,0,1),noisy_image),axis=1)
        logger.addImage(np.array(imshow.transpose(2,0,1)),'Image')
      logger.takeStep()
      net_params[0] = (net_params[0][0] + lr * grad[0][0],net_params[0][1] + lr * grad[0][1])
      print('loss ', loss)
      # print('params ', net_params)


hyper_optimization()
