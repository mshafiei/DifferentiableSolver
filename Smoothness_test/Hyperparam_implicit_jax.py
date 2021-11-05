import os
import jax
import tqdm
import jax.numpy as np
from jax import random
import cvgutils.Image as cvgim
import cvgutils.Viz as cvgviz
from jax import jit 
from jax.experimental import optimizers
from jaxopt.implicit_diff import custom_fixed_point

alpha = 1.0
w,h = 2,2
gt_lmbda = 2.0
lin_iters = 1
h_iters = 50
nonlin_iters = 10
# Define energy terms
def dataTerm(I,lmbdaa): return alpha ** 0.5 * (I-P)
def smoothnessTermDx(I,lmbdaa): return lmbdaa * (I.reshape(h,w,-1)[:-1,1:,:] - I.reshape(h,w,-1)[:-1,:-1,:]).reshape(-1)
def smoothnessTermDy(I,lmbdaa): return lmbdaa * (I.reshape(h,w,-1)[1:,:-1,:] - I.reshape(h,w,-1)[:-1,:-1,:]).reshape(-1)

def loss(v,alpha):
  return (dataTerm(v,alpha) ** 2).sum() + (smoothnessTermDx(v,alpha) ** 2).sum() + (smoothnessTermDy(v,alpha) ** 2).sum()

#define JTF and JTJ functions
def jtf_eval(v,v_lmbda):
  f, jt = jax.vjp(dataTerm,v,v_lmbda)
  jtfd = jt(f)[0]
  f, jt = jax.vjp(smoothnessTermDx,v,v_lmbda)
  jtfdx = jt(f)[0]
  f, jt = jax.vjp(smoothnessTermDy,v,v_lmbda)
  jtfdy = jt(f)[0]
  return -(jtfd + jtfdx + jtfdy)

def jtjv_eval(v,v_lmbda,d):
  f, jt = jax.vjp(dataTerm,v,v_lmbda)
  _,jd = jax.jvp(dataTerm, (v,v_lmbda), (d,0.0))
  jtjdd = jt(jd)[0]
  f, jt = jax.vjp(smoothnessTermDx,v,v_lmbda)
  _,jd = jax.jvp(smoothnessTermDx, (v,v_lmbda), (d,0.0))
  jtjdx = jt(jd)[0]
  f, jt = jax.vjp(smoothnessTermDy,v,v_lmbda)
  _,jd = jax.jvp(smoothnessTermDy, (v,v_lmbda), (d,0.0))
  jtjdy = jt(jd)[0]
  return jtjdd + jtjdx + jtjdy


F = jax.grad(loss)

@custom_fixed_point(F)
def gn_solver(v, lmbda):
  v_lmbda = lmbda
  #Gauss Newton iterations
  for li in tqdm.tqdm(range(lin_iters)):
    a = 1
    d = np.zeros_like(P)
    #pcg iterations
    r = jtf_eval(v,v_lmbda) - jtjv_eval(v,v_lmbda,d)
    z = r
    p = z
    for ni in tqdm.tqdm(range(nonlin_iters)):
      jtjp = jtjv_eval(v,v_lmbda,p)
      a = (np.matmul(r,z)) / (np.matmul(p,jtjp) + 1e-8)
      d = d + a * p
      last_rz = np.matmul(r,z)
      r = r - a * jtjp

      if(last_rz < 1e-8):
        print('Reached minima')
        break

      z = r
      beta = np.matmul(r,z) / (last_rz + 1e-8)
      p = z + beta * p
      
    v = v + d
  return v

def v_loss(lmbda):
  v = np.zeros_like(P)
  x_star = gn_solver(v,lmbda)
  return ((x_star - Q) **2).sum()

def fd(lmbda,delta):
  l1 = v_loss(lmbda-delta/2)
  l2 = v_loss(lmbda+delta/2)
  return (l2 - l1) / delta



outdir = 'out_alpha'
key = random.PRNGKey(42)
batch_size = 100

P = random.uniform(key,(w,h,batch_size))
P = P.reshape(-1)
Q = gn_solver(P.reshape(-1),gt_lmbda)
Q = Q.reshape(-1)



# Initialize the fidelity, smoothness, linear and non linear iteration counts
# respectively alpha, lmbda, lin_iters and nonlin_iters
opt_init, opt_update, get_params = optimizers.adam(step_size=1e-2)
v_lmbda = 3.0
opt_state = opt_init(v_lmbda)
# Define a compiled update step

@jit
def step(i, opt_state):
    p = get_params(opt_state)
    d = jax.grad(v_loss)(p)
    return opt_update(i, d, opt_state)

for i in range(100):
    opt_state = step(i, opt_state)
net_params = get_params(opt_state)
