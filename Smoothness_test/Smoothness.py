
import os
import jax
import tqdm
import jax.numpy as np
from jax import random
import cvgutils.Image as cvgim
import cvgutils.Viz as cvgviz

logger = cvgviz.logger('logger','tb','default','cg')
outdir = 'out'
key = random.PRNGKey(42)
batch_size = 1
w,h = 2,2
#initialize source, target, delta image respectively I, I0 and d
I0 = random.uniform(key,(w,h,batch_size))
h,w,_ = I0.shape
I0 = I0.reshape(-1,batch_size)
I = I0.copy()



# Initialize the fidelity, smoothness, linear and non linear iteration counts
# respectively alpha, lmbda, lin_iters and nonlin_iters
alpha = 1
lmbda = 100
lin_iters = 50
nonlin_iters = 1
# Define energy terms
def dataTerm(I): return alpha ** 0.5 * (I-I0)
def dataPre(I): return (I-I0)
def smoothnessTermDx(I): return lmbda * (I.reshape(h,w,-1)[:-1,1:,:] - I.reshape(h,w,-1)[:-1,:-1,:]).reshape(-1)
def smoothnessTermDy(I): return lmbda * (I.reshape(h,w,-1)[1:,:-1,:] - I.reshape(h,w,-1)[:-1,:-1,:]).reshape(-1)
def smoothnessPreDx(I): return (I.reshape(h,w,-1)[:-1,1:,:] - I.reshape(h,w,-1)[:-1,:-1,:]).reshape(-1)
def smoothnessPreDy(I): return (I.reshape(h,w,-1)[1:,:-1,:] - I.reshape(h,w,-1)[:-1,:-1,:]).reshape(-1)

def loss(v):
  return (dataTerm(v) ** 2).sum() + (smoothnessTermDx(v) ** 2).sum() + (smoothnessTermDy(v) ** 2).sum()

#define JTF and JTJ functions
def jtf_eval(v):
  jtfd = -jax.vjp(dataTerm,v)[1](dataTerm(v))[0]
  jtfsdx = -jax.vjp(smoothnessTermDx,v)[1](smoothnessTermDx(v))[0]
  jtfsdy = -jax.vjp(smoothnessTermDy,v)[1](smoothnessTermDy(v))[0]
  return jtfd + jtfsdx + jtfsdy

def jtjv_eval(v,d):
    jtjdd = jax.vjp(dataTerm,v)[1](jax.jvp(dataTerm, (v,), (d,))[1])[0]
    jtjdsdx = jax.vjp(smoothnessTermDx,v)[1](jax.jvp(smoothnessTermDx,(v,),(d,))[1])[0]
    jtjdsdy = jax.vjp(smoothnessTermDy,v)[1](jax.jvp(smoothnessTermDy,(v,),(d,))[1])[0]
    return jtjdd + jtjdsdx + jtjdsdy

def pre(v):
  # jtfd = -jax.vjp(dataTerm,v)[1](dataTerm(v))[0]
  # jtfsdx = -jax.vjp(smoothnessTermDx,v)[1](smoothnessTermDx(v))[0]
  # jtfsdy = -jax.vjp(smoothnessTermDy,v)[1](smoothnessTermDy(v))[0]
  # prejtfd = -jax.vjp(dataPre,v)[1](dataPre(v))[0]
  # prejtfdx = -jax.vjp(smoothnessPreDx,v)[1](smoothnessPreDx(v))[0]
  # prejtfdy = -jax.vjp(smoothnessPreDy,v)[1](smoothnessPreDy(v))[0]
  # pred = jtfd / (prejtfd + 1e-10)
  # presdx = jtfsdx / (prejtfdx + 1e-10)
  # presdy = jtfsdy / (prejtfdy + 1e-10)
  # return 1 / (pred + presdx + presdy + 1e-10)
  return 1


#Gauss Newton iterations
for li in tqdm.tqdm(range(lin_iters)):
  a = 1
  d = np.zeros_like(I0)
  #pcg iterations
  r = jtf_eval(I) - jtjv_eval(I,d)
  z = pre(I) * r
  p = z
  for ni in tqdm.tqdm(range(nonlin_iters)):
    jtjp = jtjv_eval(I,p)
    a = (r*z).sum(axis=0) / (p*jtjp).sum(axis=0)
    d = d + a * p
    last_rz = (r*z).sum(axis=0)
    r = r - a * jtjp
    # logger.addScalar(last_rz,'rz_%04i' % li)
    logger.takeStep()
    if(last_rz.sum() < 1e-6):
      print('Reached minima')
      break

    z = pre(I) * r
    beta = (r*z).sum(axis=0) / last_rz
    p = z + beta * p

  I = I + d
  print('loss ',loss(I))
  for b in range(batch_size):

  # logger.addScalar(loss(I),'linear_loss')
  
  # #store the image
    outim = np.concatenate((I.reshape(h,w,batch_size),I0.reshape(h,w,batch_size)),axis=1)
    if(not os.path.exists(outdir)):
      os.makedirs(outdir)
    cvgim.imwrite(os.path.join(outdir,'im_%04i_%04i.png'%(b,li)), outim[...,None][:,:,b,[0,0,0]])