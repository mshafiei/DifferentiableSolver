import jax
from jax import tree_util
import jax.numpy as jnp
from jaxopt import implicit_diff, OptaxSolver, linear_solve
import optax
import numpy as np
import tqdm
import cvgutils.Viz as cvgviz
import cvgutils.Image as cvgim
from flax import linen as nn
import tensorflow as tf
import timeit
from jax.tree_util import tree_map, tree_reduce, tree_multimap
import jax.scipy as jsp
import json
from scipy.io import loadmat
from scipy import sparse
eps=0.004
eps_phi = 1e-3
# eta = 0.01
eta2 = 0.001#eta**2
alpha = 0.9

def gauss(im):
  return im

# def dx2(x):
#   x = jnp.pad(x,((0,0),(0,0),(1,1),(0,0)),mode='reflect')
#   return 2*gauss(x)[...,:,1:-1,:] - gauss(x)[...,:,:-2,:] - gauss(x)[...,:,2:,:]

# def dy2(x):
#   x = jnp.pad(x,((0,0),(1,1),(0,0),(0,0)),mode='reflect')
#   return 2*gauss(x)[...,1:-1,:,:] - gauss(x)[...,:-2,:,:] - gauss(x)[...,2:,:,:]

# def dytdx(x):
#   x = jnp.pad(x,((0,0),(0,1),(1,0),(0,0)),mode='reflect')
#   dx = gauss(x)[...,:,1:,:] - gauss(x)[...,:,:-1,:]
#   dx1 = gauss(x)[...,1:,1:,:] - gauss(x)[...,1:,:-1,:]
#   return dx+dx1

def dx(x):
  y = jnp.pad(x,((0,0),(0,0),(1,0),(0,0)),mode='symmetric')
  return gauss(y)[...,:,1:,:] - gauss(y)[...,:,:-1,:]

def dy(x):
  y = jnp.pad(x,((0,0),(1,0),(0,0),(0,0)),mode='symmetric')
  return gauss(y)[...,1:,:,:] - gauss(y)[...,:-1,:,:]

def dxt(x):
  x = jnp.pad(x,((0,0),(0,0),(0,1),(0,0)),mode='symmetric')
  return gauss(x)[...,:,:-1,:] - gauss(x)[...,:,1:,:]

def dyt(x):
  x = jnp.pad(x,((0,0),(0,1),(0,0),(0,0)),mode='symmetric')
  return gauss(x)[...,:-1,:,:] - gauss(x)[...,1:,:,:]

def sumx_bck(x):
  x = jnp.pad(x,((0,0),(0,0),(1,0),(0,0)))
  return gauss(x)[...,:,1:,:] + gauss(x)[...,:,:-1,:]

def sumy_bck(x):
  x = jnp.pad(x,((0,0),(1,0),(0,0),(0,0)))
  return gauss(x)[...,1:,:,:] + gauss(x)[...,:-1,:,:]

def sumx_fwd(x):
  x = jnp.pad(x,((0,0),(0,0),(0,1),(0,0)))
  return gauss(x)[...,:,:-1,:] + gauss(x)[...,:,1:,:]

def sumy_fwd(x):
  x = jnp.pad(x,((0,0),(0,1),(0,0),(0,0)))
  return gauss(x)[...,:-1,:,:] + gauss(x)[...,1:,:,:]

def sign(x):
  return jnp.sign(x) + jnp.where(x==0,1.,0.)

def safe_division(nom,denom,eps):
  return nom / (sign(denom) * jnp.maximum(jnp.abs(denom), eps))

def phi(nom,denom,eps):
  return nom / (jnp.abs(denom) ** (2 - alpha) + eps)

logger = cvgviz.logger('/home/mohammad/Projects/optimizer/logger/flash_no_flash','filesystem','autodiff','cross-field-direct')
gt_im = cvgim.imread('/home/mohammad/Projects/optimizer/DifferentiableSolver/flash_no_flash/out.png').astype(np.float64)# ** 2.2
G = cvgim.imread('/home/mohammad/Projects/optimizer/DifferentiableSolver/flash_no_flash/cave_flash_1.png').astype(np.float64)# ** 2.2
I0 = cvgim.imread('/home/mohammad/Projects/optimizer/DifferentiableSolver/flash_no_flash/cave_noflash_1.png').astype(np.float64)#** 2.2

gt_im = gt_im[None,...]
G = G[None,...]
I0 = I0[None,...]
size =  I0.shape[0],I0.shape[1]//2**6,I0.shape[2]//2**6,I0.shape[3]
# I0 = jax.image.resize(I0,size,'trilinear')
# G = jax.image.resize(G,size,'trilinear')

I = jax.device_put(I0.copy()).astype(jnp.float64)
s = jax.device_put(jnp.ones_like(I0))
Gy, Gx, Iy, Ix = dy(G), dx(G), lambda i:dy(i), lambda i:dx(i)
Gx2y2 = Gx**2 + Gy**2
Gx1y1 = Gx2y2 ** 0.5
Px,Py = safe_division(1.,Gx,eps), safe_division(1.,Gy,eps)
sig1, sig2, Vx,Vy = eta2 / (Gx2y2 + 2*eta2), (Gx2y2 + eta2) / (Gx2y2 + 2*eta2), safe_division(Gx,Gx1y1,eps), safe_division(Gy,Gx1y1,eps)
Ax, Ay, B = lambda s,x: phi(1.,s-Px*Ix(x),eps_phi), lambda s,x: phi(1.,s-Py*Iy(x),eps_phi), lambda x: phi(1.,x-I0,eps_phi)
# Ax, Ay, B = lambda s,x: 1., lambda s,x: 1., lambda x: 1.
matvars = loadmat('/home/mohammad/Projects/optimizer/baselines/cross-field-source/iter1.mat')
gmask = jnp.where(Gx1y1 < 0.005)
nhierarchy = 4
max_iter = 20
lmbda = 3.
beta = 0.5
lmbda1 = 1.


# @jax.jit
def Mxs(st,it,s):
  denom = lmbda1 * Ax(st,it)+ lmbda1 * Ay(st,it)
  # idx =jnp.where(denom <= 0.004)
  ax = safe_division(s,denom,eps)

  # ax = ax.at[idx].set(1)
  return ax
# @jax.jit
def Axs(st,it,s):
  ax = lmbda1 * Ax(st,it) * s
  ax += lmbda1 * Ay(st,it) * s
  
  ax += beta * dxt((sig1 * Vx ** 2 + sig2 * Vy ** 2) * dx(s))
  ax += beta * dyt((sig2 * Vx ** 2 + sig1 * Vy ** 2) * dy(s))
  ax += beta * 2*dyt(((sig1 - sig2) * Vx * Vy) * dx(s))
  return ax



# @jax.jit
def bs(s,i):
    return lmbda1 * Ax(s,i) * Px * dx(i) + lmbda1 * Ay(s,i) * Py * dy(i)
    
# @jax.jit
def Mxi(st,it,x):
    ax = lmbda * B(it)
    ax += lmbda1 * sumy_fwd(Py ** 2 * Ay(st,it))
    ax += lmbda1 * sumx_fwd(Px ** 2 * Ax(st,it))
    # idx = jnp.where(ax <= eps)
    ax = safe_division(x,ax,eps)
    # ax = ax.at[idx].set(1)
    return ax

# @jax.jit
def Axi(st,it,x):
    ax = lmbda * B(it) * x
    ax += lmbda1 * dxt(Px ** 2 * Ax(st,it) * dx(x))
    ax += lmbda1 * dyt(Py ** 2 * Ay(st,it) * dy(x))
    return ax

# @jax.jit
def bi(s,i):
    return lmbda1 * dxt(Px * Ax(s,i) * s) + lmbda1 * dyt(Py * Ay(s,i) * s) + lmbda *B(i)*I0

# # @jax.jit
def gn_iter(s,I):
    st,It = s,I
    # st1 = st
    # ds = 0
    # lhs_s = Axs(st,It,s)
    # rhs_s = bs(st,It)
    st1 = linear_solve.solve_cg(matvec=lambda s:Axs(st,It,s), b=bs(st,It), init=st, maxiter=100)#,M=lambda s: Mxs(st,It,s))
    # st1 = jax.scipy.sparse.linalg.cg(A=lambda s:Axs(st,It,s), b=bs(st,It), x0=st, maxiter=10000)
    ds = ((Axs(st,It,st1) - bs(st,It)) ** 2).mean()
    logger.addScalar(ds,'s')
    st1 = st1.at[gmask].set(0)

    # lhs_i = Axi(st1,It,I)
    # rhs_i = bi(st1,It)
    It1 = linear_solve.solve_cg(matvec=lambda i:Axi(st1,It,i), b=bi(st1,It), init=It, maxiter=100)#,M=lambda i:Mxi(st1,It,i))
    # It1 = jax.scipy.sparse.linalg.cg(A=lambda i:Axi(st1,It,i), b=bi(st1,It), x0=It, maxiter=10000)
    di = ((Axi(st1,It,It1) - bi(st1,It)) ** 2).mean()
    It1 = jnp.clip(It1,0,1)
    logger.addScalar(di,'I')
    return st1,It1, ds, di


for i in tqdm.tqdm(range(max_iter)):
    s,I,ds,di = gn_iter(s,I)
    print('di ', di, ' ds ',ds)

    imshow = jnp.concatenate((s,I,I0,G),axis=2)
    imshow = jnp.concatenate(imshow,axis=1)
    imshow = jnp.clip(imshow,0,1)
    logger.addImage(np.array(imshow).astype(np.float32).transpose(2,0,1),'Image')
    logger.takeStep()
    psnr = 20 * jnp.log10(1) -10 * jnp.log10(((I - gt_im) ** 2).mean())


    


#gn loop
#write f and J
#ax = JTJd, b = JTf, solve