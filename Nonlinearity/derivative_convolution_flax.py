#this sample code reproduces a backward derivative by convolution
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
    dh = lambda rng, shape: jnp.array([[0,-1,0],[-1,2,0],[0,0,0]]).reshape(3,3,1,1)
    db = lambda rng, shape: jnp.array([0])

    self.layer1 = nn.Conv(1,(3,3),strides=1,kernel_init=dh,bias_init=db,padding='VALID')
    

  def __call__(self,x):
    return self.layer1(x)

h, w = 100, 100
cnn = deriv()
rng = jax.random.PRNGKey(1)
testim = jax.random.uniform(rng,[1, h, w, 3])
rng, init_rng = jax.random.split(rng)
params = cnn.init(init_rng, testim[:,:,:,0:1])['params']
out1 = deriv().apply({'params': params}, testim[:,:,:,0:1])
out2 = deriv().apply({'params': params}, testim[:,:,:,1:2])
out3 = deriv().apply({'params': params}, testim[:,:,:,2:])
out = jnp.concatenate((out1,out2,out3),axis=-1)
dx = testim[0,:,1:,:] - testim[0,:,:-1,:]
dy = testim[0,1:,:,:] - testim[0,:-1,:,:]
d = dy[:-1,1:-1,:]  + dx[1:-1,:-1,:] 
print(((d - out[0,:,:,:])**2).sum())