import jax.scipy as jsp
import jax.numpy as jnp
import jax
import imageio
from jaxopt import linear_solve

# h,w = 100,100
# key = jax.random.PRNGKey(42)
# inpt = jax.random.uniform(key,(h,w))
# x = jnp.linspace(-3, 3, 7)
# window = jsp.stats.norm.pdf(x) * jsp.stats.norm.pdf(x[:, None])
# smooth_image = jsp.signal.convolve(inpt, window, mode='same')
# outim = jnp.concatenate((inpt,smooth_image))
# imageio.imwrite('./out/smooth.png',outim)

def A(u):
    return (u[0] ** 2).sum() + (u[1] ** 2).sum()

def matvec(u):
    return u[0],u[1]

a = jnp.ones((10))
b = jnp.ones((5))
linear_solve.solve_cg(matvec=matvec,
                   b=(a,b),
                   init=(a,b),
                   maxiter=100)