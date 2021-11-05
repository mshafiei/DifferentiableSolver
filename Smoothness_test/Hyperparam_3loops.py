
import os
import jax
import tqdm
import jax.numpy as np
from jax import random
import cvgutils.Image as cvgim
import cvgutils.Viz as cvgviz

def load_images(dr,n):
  im = []
  for i in range(n):
    im.append(cvgim.imread(os.path.join(dr,'im_%04i_%04i.png'%(i,1))))
  return np.concatenate(im,axis=-1)

logger = cvgviz.logger('logger','tb','default','cg')
outdir = 'out'
key = random.PRNGKey(42)
batch_size = 500
#initialize source, target, delta image respectively I, I0 and d
I0 = load_images('out',batch_size)
h,w,_ = I0.shape
I0 = I0.reshape(-1)



# Initialize the fidelity, smoothness, linear and non linear iteration counts
# respectively alpha, lmbda, lin_iters and nonlin_iters
alpha = 1
lin_iters = 50
nonlin_iters = 50
h_iters = 50
# Define energy terms
def dataTerm(lmbda,I): return alpha ** 0.5 * (I-I0)
def dataPre(I): return (I-I0)
def smoothnessTermDx(lmbda,I): return lmbda * (I.reshape(h,w,-1)[:-1,1:,:] - I.reshape(h,w,-1)[:-1,:-1,:]).reshape(-1)
def smoothnessTermDy(lmbda,I): return lmbda * (I.reshape(h,w,-1)[1:,:-1,:] - I.reshape(h,w,-1)[:-1,:-1,:]).reshape(-1)
def smoothnessPreDx(I): return (I.reshape(h,w,-1)[:-1,1:,:] - I.reshape(h,w,-1)[:-1,:-1,:]).reshape(-1)
def smoothnessPreDy(I): return (I.reshape(h,w,-1)[1:,:-1,:] - I.reshape(h,w,-1)[:-1,:-1,:]).reshape(-1)

def loss(lmbda,v):
  return (dataTerm(lmbda,v) ** 2).sum() + (smoothnessTermDx(lmbda,v) ** 2).sum() + (smoothnessTermDy(lmbda,v) ** 2).sum()

#define JTF and JTJ functions
def jtf_eval(v):
  f, jt = jax.vjp(dataTerm,*v.toTuple())
  jtfd = Parameters(*jt(f))
  f, jt = jax.vjp(smoothnessTermDx,*v.toTuple())
  jtfdx = Parameters(*jt(f))
  f, jt = jax.vjp(smoothnessTermDy,*v.toTuple())
  jtfdy = Parameters(*jt(f))
  return -(jtfd + jtfsdx + jtfsdy)

def jtjv_eval(v,d):
  f, jt = jax.vjp(dataTerm,*v.toTuple())
  _,jd = jax.jvp(dataTerm, v.toTuple(), d.toTuple())
  jtjdd = Parameters(*jt(jd))
  f, jt = jax.vjp(smoothnessTermDx,*v.toTuple())
  _,jd = jax.jvp(smoothnessTermDx, v.toTuple(), d.toTuple())
  jtjdx = Parameters(*jt(jd))
  f, jt = jax.vjp(smoothnessTermDy,*v.toTuple())
  _,jd = jax.jvp(smoothnessTermDy, v.toTuple(), d.toTuple())
  jtjdy = Parameters(*jt(jd))

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

v_I = I0.copy()
v_lambda = 0
d_lambda = 0

#hyper parameter gradient decent
for hi in tqdm.tqdm(range(h_iters)):
  
  d_I = np.zeros_like(I0)
  #Gauss Newton iterations
  for li in tqdm.tqdm(range(lin_iters)):
    a = 1
    #pcg iterations
    r = jtf_eval((v_lambda,v_I)) - jtjv_eval((v_lambda,v_I),(d_lambda,d_I))
    z = pre((v_lambda,v_I)) * r
    p = z
    for ni in tqdm.tqdm(range(nonlin_iters)):
      jtjp = jtjv_eval((v_lambda,v_I),p)
      a = np.matmul(r,z) / np.matmul(p,jtjp)
      d = d + a * p
      last_rz = np.matmul(r,z)
      r = r - a * jtjp
      logger.addScalar(last_rz,'rz_%04i' % li)
      logger.takeStep()
      if(last_rz < 1e-6):
        print('Reached minima')
        continue

      z = pre((v_lambda,v_I)) * r
      beta = np.matmul(r,z) / last_rz
      p = z + beta * p

    v_I = v_I + d_I
    v_lambda = v_lambda + d_lambda
    
    # logger.addScalar(loss(I),'linear_loss')
    
    #store the image
    # outim = np.concatenate((v_I.reshape(h,w),I0.reshape(h,w)),axis=1)
    # if(not os.path.exists(outdir)):
    #   os.makedirs(outdir)
    # cvgim.imwrite(os.path.join(outdir,'im_%04i.png'%li), outim[...,None][...,[0,0,0]])