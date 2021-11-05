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

# define interpolation function
# f_gt = interpolate(alpha_gt)
# x_0 = 0
# f_0 = interpolate(x_0)
# define fd 
# evaluate finite_grad(x_0)
# define implicit grad
# evaluate implicit_analytic_grad(x_0)
# plot

i1 = 0
i2 = 1
alpha_gt = 0.8

def f_t(x,alpha):
    return (x-i1) ** 2 + alpha * (x-i2) ** 2

def interpolate(alpha):
    return (1-alpha) * i1 + alpha * i2

x_gt = interpolate(alpha_gt)

def f_v(alpha):
    x = interpolate(alpha)
    return (x - x_gt) ** 2

def finite_grad(alpha,delta):
    d_r = f_v(alpha+delta/2)
    d_l = f_v(alpha-delta/2)
    return (d_r - d_l) / delta

def implicit_analytic_grad(alpha):
    return 2*(interpolate(alpha) - x_gt) * (-i1 + i2)

grid = np.linspace(0.1,2,20)
fd_val = np.stack([finite_grad(i,0.01) for i in grid],axis=-1)
analytic_val = np.stack([implicit_analytic_grad(i) for i in grid],axis=-1)

dxl_fd = np.stack((fd_val,np.ones_like(fd_val)),axis=-1)
dxl_analytic = np.stack((analytic_val,np.ones_like(analytic_val)),axis=-1)
dxl_fd = dxl_fd / np.linalg.norm(dxl_fd,axis=-1)[:,None] * np.linalg.norm(fd_val[:,None],axis=1,keepdims=True)
dxl_analytic = dxl_analytic / np.linalg.norm(dxl_analytic,axis=-1)[:,None] * np.linalg.norm(analytic_val[:,None],axis=1,keepdims=True)
loss_val = np.stack([f_v(i) for i in grid])

plt.plot(grid,loss_val,'r')
for i in range(grid.shape[0]):
  plt.arrow(grid[i],loss_val[i],dxl_fd[i,1],dxl_fd[i,0],color='lightgreen')
  plt.arrow(grid[i],loss_val[i],dxl_analytic[i,1],dxl_analytic[i,0],color='blue')
  
plt.legend(['Validation loss','Finite difference','Analytic gradient'])
plt.savefig('out/plot.pdf')
plt.close()


# solve interpolation by grad decent solver
# define implicit_jax_grad
# evaluate implicit_jax_grad(x_0)
# plot
# write the same functions for 3pix
# write the same functions for 3pix with smoothness term
# write the same functions for 2d
# create stencil

