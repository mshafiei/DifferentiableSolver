import os
import jax
import tqdm
import jax.numpy as np
from jax import random
import cvgutils.Image as cvgim
import cvgutils.Viz as cvgviz
import argparse


class Solver:
  def __init__(self,logger,opt) -> None:

    self.gn_iters = opt.gn_iters
    self.cg_iters = opt.cg_iters
    self.logger = logger
    self.outdir = 'out'
    key = random.PRNGKey(42)
    #initialize source, target, delta image respectively I, I0 and d
    self.I0 = np.ones((100,100))
    self.I0 = self.I0.at[:50,:50].set(0)
    self.I0 = self.I0.at[50:,50:].set(0)
    self.h,self.w = self.I0.shape
    self.I0 = self.I0.reshape(-1)

  @staticmethod
  def parse_arguments(parser):
    parser.add_argument('--gn_iters',type=int, default=50, help='Gauss Newton iteration count')
    parser.add_argument('--cg_iters',type=int, default=50, help='Conjugate Gradient iteration count')
    return parser


  def loss(self,v):
    l = 0
    for f in dir(self):
      if(f.startswith('f_')):
        f_func = getattr(self,f)
        l += (f_func(v) ** 2).sum()

    return l    

  #define JTF and JTJ functions
  def jtf_eval(self,v):
    jtfd = 0
    for f in dir(self):
      if(f.startswith('f_')):
        f_func = getattr(self,f)
        jtfd += -jax.vjp(f_func,v)[1](f_func(v))[0]

    return jtfd

  def jtjv_eval(self,v,d):
    jtjd = 0
    for f in dir(self):
      if(f.startswith('f_')):
        f_func = getattr(self,f)
        jtjd += jax.vjp(f_func,v)[1](jax.jvp(f_func, (v,), (d,))[1])[0]

    return jtjd

  def pre(self,v):
    return 1

  def solve(self):
    I = self.I0.copy()
    d = np.zeros_like(self.I0)
    I = I.reshape(-1)
    d = d.reshape(-1)

    # Initialize the fidelity, smoothness, linear and non linear iteration counts
    # respectively alpha, lmbda, lin_iters and nonlin_iters

    #Gauss Newton iterations
    for li in tqdm.tqdm(range(self.gn_iters)):
      a = 1
      #pcg iterations
      r = self.jtf_eval(I) - self.jtjv_eval(I,d)
      z = self.pre(I) * r
      p = z
      for ni in tqdm.tqdm(range(self.cg_iters)):
        jtjp = self.jtjv_eval(I,p)
        a = np.matmul(r,z) / np.matmul(p,jtjp)
        d = d + a * p
        last_rz = np.matmul(r,z)
        r = r - a * jtjp
        self.logger.addScalar(last_rz,'rz_%04i' % li)
        self.logger.takeStep()
        if(last_rz < 1e-6):
          continue

        z = self.pre(I) * r
        beta = np.matmul(r,z) / last_rz
        p = z + beta * p

      I = I + d
      
      self.logger.addScalar(self.loss(I),'linear_loss')
      
      #store the image
      outim = np.concatenate((I.reshape(self.h,self.w),self.I0.reshape(self.h,self.w)),axis=1)
      if(not os.path.exists(self.outdir)):
        os.makedirs(self.outdir)
      cvgim.imwrite(os.path.join(self.outdir,'im_%04i.png'%li), outim[...,None][...,[0,0,0]])

class ImageSmoothing(Solver):
  def __init__(self,logger,opt) -> None:
      super().__init__(logger,opt)
      self.alpha = opt.alpha
      self.lmbda = opt.lmbda
  # Define energy terms
  def f_data(self,I): return self.alpha ** 0.5 * (I-self.I0)
  def f_Dx(self,I): return self.lmbda ** 0.5 * (I.reshape(self.h,self.w)[:-1,1:] - I.reshape(self.h,self.w)[:-1,:-1]).reshape(-1)
  def f_Dy(self,I): return self.lmbda ** 0.5 * (I.reshape(self.h,self.w)[1:,:-1] - I.reshape(self.h,self.w)[:-1,:-1]).reshape(-1)
  
  @staticmethod
  def parse_arguments(parser):
    parser = Solver.parse_arguments(parser)
    parser.add_argument('--alpha',type=float, default=1, help='Weight for the data term')
    parser.add_argument('--lmbda',type=float, default=1000, help='Weight for the smoothness term')
    return parser

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='')
  parser = ImageSmoothing.parse_arguments(parser)
  args = parser.parse_args()
  logger = cvgviz.logger('logger','tb','default','cg')

  Solver = ImageSmoothing(logger,args)
  Solver.solve()