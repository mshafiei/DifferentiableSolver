import os
import jax
import tqdm
import jax.numpy as np
from jax import random
import cvgutils.Image as cvgim
import cvgutils.Viz as cvgviz

class Parameters:
    def __init__(self,h_alpha,p_I) -> None:
        self.h_alpha = h_alpha
        self.p_I = p_I

    def __add__(self,other):
        # h_alpha = self.h_alpha + other.h_alpha
        p_I = self.p_I + other.p_I
        return Parameters(self.h_alpha,p_I)

    def __sub__(self,other):
        # h_alpha = self.h_alpha - other.h_alpha
        p_I = self.p_I - other.p_I
        return Parameters(self.h_alpha,p_I)
    def __mul__(self,other):
        if(isinstance(other,int) or isinstance(other,float)):
            # h_alpha = self.h_alpha * other.h_alpha
            p_I = self.p_I * other.p_I
        else:
            # h_alpha = self.h_alpha * other.h_alpha
            p_I = self.p_I * other.p_I
        return Parameters(self.h_alpha,p_I)

    def __neg__(self):
        return Parameters(self.h_alpha,-self.p_I)

    def toTuple(self):
        return (self.h_alpha,self.p_I)

def load_images(dr,n):
  im = []
  for i in range(n):
    im.append(cvgim.imread(os.path.join(dr,'im_%04i_%04i.png'%(i,0))))
  return np.concatenate(im,axis=-1)

logger = cvgviz.logger('logger','tb','default','cg')
outdir = 'out_alpha'
key = random.PRNGKey(42)
batch_size = 500
#initialize source, target, delta image respectively I, I0 and d
P = load_images('out/',batch_size)
# I0 = I0.at[:50,:50].set(0)
# I0 = I0.at[50:,50:].set(0)
# I0 = cvgim.imread('/home/mohammad/Projects/optimizer/out_5/im_0004.png')[:,:100,0]
Q,P = P[:,:100,:], P[:,100:,:]
h,w,_ = P.shape
P = P.reshape(-1)
Q = Q.reshape(-1)



# Initialize the fidelity, smoothness, linear and non linear iteration counts
# respectively alpha, lmbda, lin_iters and nonlin_iters
alpha = 1
lmbda = 1000
lin_iters = 2
h_iters = 50
nonlin_iters = 50
# Define energy terms
def dataTerm(lmbdaa,I): return alpha ** 0.5 * (I-P)
def dataPre(I): return (I-P)
def smoothnessTermDx(lmbdaa,I): 
    # print('lmbdaa, ',lmbdaa)
    return lmbdaa ** 2 * (I.reshape(h,w,-1)[:-1,1:,:] - I.reshape(h,w,-1)[:-1,:-1,:]).reshape(-1)
def smoothnessTermDy(lmbdaa,I): return lmbdaa ** 2 * (I.reshape(h,w,-1)[1:,:-1,:] - I.reshape(h,w,-1)[:-1,:-1,:]).reshape(-1)
def smoothnessPreDx(I): return (I.reshape(h,w,-1)[:-1,1:,:] - I.reshape(h,w,-1)[:-1,:-1,:]).reshape(-1)
def smoothnessPreDy(I): return (I.reshape(h,w,-1)[1:,:-1,:] - I.reshape(h,w,-1)[:-1,:-1,:]).reshape(-1)

def loss(alpha,v):
  return (dataTerm(alpha,v) ** 2).sum() + (smoothnessTermDx(alpha,v) ** 2).sum() + (smoothnessTermDy(alpha,v) ** 2).sum()

#define JTF and JTJ functions
def jtf_eval(v):
#   f, jt = jax.vjp(f_Dx,d_alpha,d_I)
#   jtf_alpha, jtf_I = jt(f)
#   _,jd = jax.jvp(f_Dx, (jtf_alpha,jtf_I), (d_alpha,d_I))
#   jtjd_alpha, jtjd_I = jt(jd)
  f, jt = jax.vjp(dataTerm,*v.toTuple())
  jtfd = Parameters(*jt(f))
  f, jt = jax.vjp(smoothnessTermDx,*v.toTuple())
  jtfdx = Parameters(*jt(f))
  f, jt = jax.vjp(smoothnessTermDy,*v.toTuple())
  jtfdy = Parameters(*jt(f))
#   jtfd = -jax.vjp(dataTerm,v)[1](dataTerm(v))[0]
#   jtfdx = -jax.vjp(smoothnessTermDx,v)[1](smoothnessTermDx(v))[0]
#   jtfdy = -jax.vjp(smoothnessTermDy,v)[1](smoothnessTermDy(v))[0]
  return -(jtfd + jtfdx + jtfdy)

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
    # jtjdd = jax.vjp(dataTerm,v)[1](jax.jvp(dataTerm, (v,), (d,))[1])[0]
    # jtjdsdx = jax.vjp(smoothnessTermDx,v)[1](jax.jvp(smoothnessTermDx,(v,),(d,))[1])[0]
    # jtjdsdy = jax.vjp(smoothnessTermDy,v)[1](jax.jvp(smoothnessTermDy,(v,),(d,))[1])[0]
  return jtjdd + jtjdx + jtjdy

def pre(v):
#   jtfd = -jax.vjp(dataTerm,v)[1](dataTerm(v))[0]
#   jtfsdx = -jax.vjp(smoothnessTermDx,v)[1](smoothnessTermDx(v))[0]
#   jtfsdy = -jax.vjp(smoothnessTermDy,v)[1](smoothnessTermDy(v))[0]
#   prejtfd = -jax.vjp(dataPre,v)[1](dataPre(v))[0]
#   prejtfdx = -jax.vjp(smoothnessPreDx,v)[1](smoothnessPreDx(v))[0]
#   prejtfdy = -jax.vjp(smoothnessPreDy,v)[1](smoothnessPreDy(v))[0]
#   pred = jtfd / (prejtfd + 1e-10)
#   presdx = jtfsdx / (prejtfdx + 1e-10)
#   presdy = jtfsdy / (prejtfdy + 1e-10)
  # return 1 / (pred + presdx + presdy + 1e-10)
  return 1

v_lmbda=0.0
for hi in tqdm.tqdm(range(h_iters)):

  v = Parameters(v_lmbda, np.zeros_like(P))
  #Gauss Newton iterations
  for li in tqdm.tqdm(range(lin_iters)):
    a = 1
    d = Parameters(0.0, np.zeros_like(P))
    #pcg iterations
    r = jtf_eval(v) - jtjv_eval(v,d)
  #   z = pre(v) * r
    z = r
    p = z
    for ni in tqdm.tqdm(range(nonlin_iters)):
      jtjp = jtjv_eval(v,p)
      a = (np.matmul(r.p_I,z.p_I)) / (np.matmul(p.p_I,jtjp.p_I) + 1e-8)
      d = d + Parameters(v_lmbda,a * p.p_I)
      last_rz = np.matmul(r.p_I,z.p_I)
      r = r - Parameters(v_lmbda,a * jtjp.p_I)
      logger.addScalar(last_rz,'rz_%04i' % li)
      logger.takeStep()
      # if(last_rz.alpha + last_rz.I < 1e-8):
      #   print('Reached minima')
      #   continue

      # z = pre(I) * r
      z = r
      beta = np.matmul(r.p_I,z.p_I) / (last_rz + 1e-8)
      p = z + Parameters(v_lmbda,beta * p.p_I)
      
    v.p_I = v.p_I + d.p_I
    
    logger.addScalar(loss(v.h_alpha,v.p_I),'linear_loss')
    
    #store the image
    for b in range(0,10):
      outim = np.concatenate((v.p_I.reshape(h,w,-1)[...,b],Q.reshape(h,w,-1)[...,b],P.reshape(h,w,-1)[...,b],),axis=1)
      if(not os.path.exists(outdir)):
        os.makedirs(outdir)
      cvgim.imwrite(os.path.join(outdir,'im_%04i_%04i.png'%(b,li)), outim[...,None][...,[0,0,0]])

  
  
  def alphaTerm(v):
    return ((v - Q) **2).mean()
  f, jt = jax.vjp(alphaTerm, v.p_I)
  dalpha = jt(f)[0].sum()
  lr = 0.1
  v_lmbda = v_lmbda - lr * dalpha
  print('Loss %04d, lambda %02d dalpha %02d' % (loss(v_lmbda,v.p_I),v_lmbda,dalpha))