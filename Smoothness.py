
import os
import jax
import tqdm
import jax.numpy as np
from jax import random
import cvgutils.Image as cvgim

outdir = 'out'
key = random.PRNGKey(42)
#initialize source, target, delta image respectively I, I0 and d
I0 = np.ones((100,100))
I0 = I0.at[:50,:50].set(0)
I0 = I0.at[50:,50:].set(0)
h,w = I0.shape
I = random.uniform(key,(I0.shape[0],I0.shape[1]))
d = random.uniform(key,(I0.shape[0],I0.shape[1]))
I = I.reshape(-1)
I0 = I0.reshape(-1)
d = d.reshape(-1)

# Initialize the fidelity, smoothness, linear and non linear iteration counts
# respectively alpha, lmbda, lin_iters and nonlin_iters
alpha = 1
lmbda = 10
lin_iters = 50
nonlin_iters = 20

# Define energy terms
def dataTerm(I): return alpha * (I-I0)
def smoothnessTerm(I): return lmbda ** 0.5 * ((I.reshape(h,w)[1:,:-1] - I.reshape(h,w)[:-1,:-1]) ** 2 + (I.reshape(h,w)[:-1,1:] - I.reshape(h,w)[:-1,:-1]) ** 2).reshape(-1)

#define JTF and JTJ functions
def jtf_eval(v):
  jtfd = -jax.vjp(dataTerm,v)[1](dataTerm(v))[0]
  jtfs = -jax.vjp(smoothnessTerm,v)[1](smoothnessTerm(v))[0]
  return jtfd + jtfs

def jtjv_eval(v,d):
    jtjdd = jax.vjp(dataTerm,v)[1](jax.jvp(dataTerm, (v,), (d,))[1])[0]
    jtjds = jax.vjp(smoothnessTerm,v)[1](jax.jvp(smoothnessTerm,(v,),(d,))[1])[0]
    return jtjdd + jtjds

#Gauss Neuton iterations
for li in tqdm.tqdm(range(lin_iters)):
  a = 1
  #pcg iterations
  r = jtf_eval(I) - jtjv_eval(I,d)
  z = r
  p = z
  for ni in tqdm.tqdm(range(nonlin_iters)):
    jtjp = jtjv_eval(I,p)
    a = np.matmul(r,z) / np.matmul(p,jtjp)
    d = d + a * p
    last_rz = np.matmul(r,z)
    if(last_rz < 1e-6):
      break
    r = r - a * jtjp
    z = r
    beta = np.matmul(r,z)/last_rz
    p = z + beta * p

  I = I + d
  
  #store the image
  outim = np.concatenate((I.reshape(h,w),I0.reshape(h,w)),axis=1)
  if(not os.path.exists(outdir)):
    os.makedirs(outdir)
  cvgim.imwrite(os.path.join(outdir,'im_%04i.png'%li), outim[...,None][...,[0,0,0]])