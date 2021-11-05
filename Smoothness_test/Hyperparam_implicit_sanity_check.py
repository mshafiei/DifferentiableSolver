import os
import jax
import tqdm
import jax.numpy as np
from jax import random
import cvgutils.Image as cvgim
import cvgutils.Viz as cvgviz
from jaxopt.implicit_diff import custom_fixed_point
import argparse
import matplotlib.pyplot as plt

def main(args):
  alpha = args.alpha
  w =args.w
  gt_lmbda = args.gt_lmbda
  lin_iters = args.lin_iters
  h_iters = args.h_iters
  nonlin_iters = args.nonlin_iters
  # Define energy terms
  def dataTerm(I,lmbdaa): return alpha ** 0.5 * (I-P)
  def smoothnessTermDx(I,lmbdaa): return lmbdaa * (I.reshape(w,-1)[1:,:] - I.reshape(w,-1)[:-1,:]).reshape(-1)


  def loss(v,alpha):
    return (dataTerm(v,alpha) ** 2).sum() + (smoothnessTermDx(v,alpha) ** 2).sum()

  #define JTF and JTJ functions
  def jtf_eval(v,v_lmbda):
    f, jt = jax.vjp(dataTerm,v,v_lmbda)
    jtfd = jt(f)[0]
    f, jt = jax.vjp(smoothnessTermDx,v,v_lmbda)
    jtfdx = jt(f)[0]
    return -(jtfd + jtfdx)

  def jtjv_eval(v,v_lmbda,d):
    f, jt = jax.vjp(dataTerm,v,v_lmbda)
    _,jd = jax.jvp(dataTerm, (v,v_lmbda), (d,0.0))
    jtjdd = jt(jd)[0]
    f, jt = jax.vjp(smoothnessTermDx,v,v_lmbda)
    _,jd = jax.jvp(smoothnessTermDx, (v,v_lmbda), (d,0.0))
    jtjdx = jt(jd)[0]
    return jtjdd + jtjdx


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
    return ((x_star - Q) **2).mean()


  def fd(lmbda,delta):
    l1 = v_loss(lmbda-delta/2)
    l2 = v_loss(lmbda+delta/2)
    if(delta = 0):
      d = 0
    else:
      d = (l2 - l1) / delta
    return d 

  def derivatives(est_lambda):
    v = np.zeros_like(P)
    x_star = gn_solver(v,est_lambda)
    

    dftdx = np.eye(len(x_star)) * (1+2 * est_lambda)
    dftdx = dftdx.at[0,0].set(1+est_lambda)
    dftdx = dftdx.at[-1,-1].set(1+est_lambda)
    # dftdx = dftdx.at[1:,:-1].set(-np.eye(len(x_star)-1))
    diag_idx = np.diag_indices(len(x_star)-1)
    dftdx = dftdx.at[1:,:-1].set(dftdx[1:,:-1] + np.eye(len(x_star)-1) * -est_lambda)
    dftdx = dftdx.at[:-1,1:].set(dftdx[:-1,1:] + np.eye(len(x_star)-1) * -est_lambda)

    dftdl = np.zeros(len(x_star))
    dftdl = dftdl.at[1:].set(x_star[1:] - x_star[:-1])
    dftdl = dftdl.at[:-1].set(x_star[:-1] - x_star[1:])
    
    dxdl = -np.matmul(np.linalg.inv(dftdx),dftdl)
    dfv = x_star - Q

    return np.dot(dfv,dxdl)

  #compute df_t/dx ^-1 . df_t/dlambda
  #compute <df_v/dx, df_t/dx ^-1 . df_t/dlambda>

  outdir = 'out_alpha'
  key = random.PRNGKey(42)
  batch_size = args.batch_size

  P = random.uniform(key,(w,batch_size))
  P = P.reshape(-1)
  Q = gn_solver(P.reshape(-1),gt_lmbda)
  Q = Q.reshape(-1)

  grid = np.linspace(0,2,20)
  analytic_val = np.stack([derivatives(i) for i in grid],axis=-1)
  loss_val = np.stack([v_loss(i) for i in grid])
  grad_val = np.stack([jax.grad(v_loss)(i) for i in grid],axis=-1)
  fd_val = np.stack([fd(i,0.01) for i in grid],axis=-1)
  dxl = np.stack((grad_val,-np.sign(grad_val) * np.ones_like(grad_val)),axis=-1)
  dxl_fd = np.stack((fd_val,-np.sign(fd_val) * np.ones_like(fd_val)),axis=-1)
  dxl_analytic = np.stack((analytic_val,-np.sign(analytic_val) * np.ones_like(analytic_val)),axis=-1)
  dxl = dxl / np.linalg.norm(dxl,axis=-1)[:,None] * np.linalg.norm(grad_val[:,None],axis=1,keepdims=True)
  dxl_fd = dxl_fd / np.linalg.norm(dxl_fd,axis=-1)[:,None] * np.linalg.norm(fd_val[:,None],axis=1,keepdims=True)
  dxl_analytic = dxl_analytic / np.linalg.norm(dxl_analytic,axis=-1)[:,None] * np.linalg.norm(analytic_val[:,None],axis=1,keepdims=True)

  plt.plot(grid,loss_val,'r')
  for i in range(grid.shape[0]):
    plt.arrow(grid[i],loss_val[i],dxl[i,1], dxl[i,0],color='purple')
    plt.arrow(grid[i],loss_val[i],dxl_fd[i,1],dxl_fd[i,0],color='lightgreen')
    plt.arrow(grid[i],loss_val[i],dxl_analytic[i,1],dxl_analytic[i,0],color='blue')
    
  plt.legend(['Validation loss','autodiff','Finite difference','Analytic gradient'])
  # plt.legend(['Validation loss','Finite difference','Analytic gradient'])
  # plt.legend(['Validation loss','Analytic gradient'])
  plt.savefig('out/plot.pdf')
  plt.close()
  # Initialize the fidelity, smoothness, linear and non linear iteration counts
  # respectively alpha, lmbda, lin_iters and nonlin_iters
  lr = args.lr
  v_lmbda = args.v_lmbda
  for hi in tqdm.tqdm(range(h_iters)):
    
    d = jax.grad(v_loss)(v_lmbda)
    v_lmbda = v_lmbda - lr * d
    lss = v_loss(v_lmbda)
    print('loss, ',lss, ' lambda pred ',v_lmbda,' gt lambda ',gt_lmbda,' d ',d)
      
  x_star = gn_solver(np.zeros_like(P), v_lmbda)

  #store the image
  for b in range(0,10): 
    outim = np.concatenate((x_star.reshape(w,-1)[...,b],Q.reshape(w,-1)[...,b],P.reshape(w,-1)[...,b]),axis=-1)
    if(not os.path.exists(outdir)):
      os.makedirs(outdir)
    cvgim.imwrite(os.path.join(outdir,'im_%04i.png'%(b)), outim[...,None][...,[0,0,0]])


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('--alpha',type=float, default=1.0, help='Weight for the data term')
  parser.add_argument('--v_lmbda',type=float, default=3.0, help='Weight for the data term')
  parser.add_argument('--gt_lmbda',type=float, default=1.0, help='Weight for the data term')
  parser.add_argument('--w',type=float, default=24, help='Weight for the data term')
  parser.add_argument('--h',type=float, default=24, help='Weight for the data term')
  parser.add_argument('--lin_iters',type=float, default=1, help='Weight for the data term')
  parser.add_argument('--h_iters',type=float, default=50, help='Weight for the data term')
  parser.add_argument('--nonlin_iters',type=float, default=10, help='Weight for the data term')
  parser.add_argument('--lr',type=float, default=10.0, help='Weight for the data term')
  parser.add_argument('--batch_size',type=int, default=1, help='Weight for the data term')
  args = parser.parse_args()
  main(args)