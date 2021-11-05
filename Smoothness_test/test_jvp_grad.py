from absl import app
import jax
from jax._src.numpy.lax_numpy import argsort
import jax.numpy as jnp
from jaxopt import implicit_diff
from jaxopt import linear_solve
from jaxopt import OptaxSolver
from matplotlib.pyplot import vlines
import optax
from sklearn import datasets
from sklearn import model_selection
from sklearn import preprocessing

lmbda_init = 0.1
lmbda_gt = 2.

key = jax.random.PRNGKey(10)
data = jax.random.uniform(key,(4,))
init_params = jnp.zeros_like(data)
lmbda = 1.
@jax.jit
def screen_poisson_objective(params, lmbda, data):
  """Objective function."""
  return 0.5 * ((params - data) ** 2).sum() + 0.5 * lmbda * ((params[1:] - params[:-1]) ** 2).sum()

def screen_poisson_residual(params, lmbda, data):
  """Objective function."""
  return jnp.concatenate((params - data, lmbda ** 0.5 * (params[1:] - params[:-1])),axis=0)


def matvec(u):
    a = u
    a = a.at[1:].set(a[1:] + lmbda * (u[1:] - u[:-1]))
    a = a.at[:-1].set(a[:-1] + lmbda * (u[:-1] - u[1:]))
    return a

def matvec2(u):
    f = lambda u:screen_poisson_residual(u,lmbda,data)

    jtd = jax.jvp(f,(init_params,),(u,))[1]
    return jax.vjp(f,init_params)[1](jtd)[0]

matvec(jnp.ones_like(data))
matvec2(jnp.ones_like(data))
print('hi')