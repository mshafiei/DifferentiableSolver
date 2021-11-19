import jax
import jax.numpy as jnp
from jaxopt import implicit_diff
from jaxopt import linear_solve
import numpy as np
import tqdm
import cvgutils.Viz as cvgviz
from jax.experimental import stax
from jax.config import config
import matplotlib.pyplot as plt
config.update("jax_debug_nans", True)

lmbda_init = 0.1
lmbda_gt = 0.9
h,w = 100,100
dw = 3
key1 = jax.random.PRNGKey(42)
key2 = jax.random.PRNGKey(43)
rng = jax.random.PRNGKey(0)


net_init, net_apply = stax.serial(stax.Conv(1, (3, 3), padding='SAME'))

in_shape = (-1, 1, 1)
out_shape, net_params = net_init(rng, (-1, h, w, 1))

inpt = jax.random.uniform(key1,(h,w))
init_inpt = jax.random.uniform(key2,(h,w))
logger = cvgviz.logger('./logger','filesystem','autodiff','autodiff')

# @jax.jit
def stencil_residual(pp_image, hp_nn, data):
  """Objective function."""
  avg_weight = 1 / len(data.reshape(-1)) ** 0.5
  r1 =  pp_image - data
  return avg_weight * r1


# @jax.jit
def screen_poisson_objective(pp_image, hp_nn, data):
  """Objective function."""
  return (stencil_residual(pp_image, hp_nn, data) ** 2).sum()


@implicit_diff.custom_root(jax.grad(screen_poisson_objective))
def screen_poisson_solver(init_image,hp_nn, data):
    f = lambda pp_image:stencil_residual(pp_image,hp_nn,data)
    loss = lambda pp_image:screen_poisson_objective(pp_image,hp_nn,data)

    gn_iters = 10
    x = init_image
    print('loss ', loss(x))
    step = logger.step
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
                              maxiter=100)
      print('loss ', loss(x))
    return x

key1 = jax.random.PRNGKey(42)
key2 = jax.random.PRNGKey(43)
init = jax.random.uniform(key1,(h,w))
data = jax.random.uniform(key2,(h,w))
screen_poisson_solver(init,0.1,data)