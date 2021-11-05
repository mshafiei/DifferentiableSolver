import os
import jax
from jaxopt._src import gradient_descent
import tqdm
import jax.numpy as np
from jax import random
import cvgutils.Image as cvgim
import cvgutils.Viz as cvgviz
from jaxopt.implicit_diff import custom_fixed_point
import argparse
import matplotlib.pyplot as plt
from jax.experimental import optimizers
from jax import grad, jit, vmap, value_and_grad

# solve interpolation by grad decent solver
# define implicit_jax_grad
# evaluate implicit_jax_grad(x_0)
# plot
# write the same functions for 3pix
# write the same functions for 3pix with smoothness term
# write the same functions for 2d
# create stencil

i1 = 0
i2 = 1
alpha_gt = 0.8
lr = 0.6
maxiter=20
x_0=0.
def f_t(x,alpha):
    return (1-alpha)*(x-i1) ** 2 + alpha * (x-i2) ** 2

F = jax.grad(f_t,argnums=0)

@custom_fixed_point(F)
def x(alpha):
    g_f_t = jax.grad(f_t,argnums=0)
    x = x_0
    for _ in range(maxiter):
        x = x - lr * g_f_t(x,alpha)
    return x

def f_v(alpha):
    return (x(alpha) - x(alpha_gt))**2

def finite_grad(alpha,delta):
    d_r = f_v(alpha+delta/2)
    d_l = f_v(alpha-delta/2)
    return (d_r - d_l) / delta

def implicit_analytic_grad(alpha):
    return 2*(x(alpha) - x(alpha_gt)) * (-i1 + i2)

step_size = 0.5
opt_init, opt_update, get_params = optimizers.adam(step_size)
opt_state = opt_init(params)

implicit_autodiff_grad = jax.grad(f_v)
grid = np.linspace(0.1,2,20)
fd_val = np.stack([finite_grad(i,0.01) for i in tqdm.tqdm(grid)],axis=-1)
analytic_val = np.stack([implicit_analytic_grad(i) for i in tqdm.tqdm(grid)],axis=-1)
autodiff_val = np.stack([implicit_autodiff_grad(i) for i in tqdm.tqdm(grid)],axis=-1)

dxl_fd = np.stack((fd_val,np.ones_like(fd_val)),axis=-1)
dxl_analytic = np.stack((analytic_val,np.ones_like(analytic_val)),axis=-1)
dxl_autodiff = np.stack((autodiff_val,np.ones_like(autodiff_val)),axis=-1)
dxl_fd = dxl_fd / np.linalg.norm(dxl_fd,axis=-1)[:,None] * np.linalg.norm(fd_val[:,None],axis=1,keepdims=True)
dxl_analytic = dxl_analytic / np.linalg.norm(dxl_analytic,axis=-1)[:,None] * np.linalg.norm(analytic_val[:,None],axis=1,keepdims=True)
dxl_autodiff = dxl_autodiff / np.linalg.norm(dxl_autodiff,axis=-1)[:,None] * np.linalg.norm(analytic_val[:,None],axis=1,keepdims=True)
loss_val = np.stack([f_v(i) for i in tqdm.tqdm(grid)])

plt.plot(grid,loss_val,'r')
for i in range(grid.shape[0]):
  plt.arrow(grid[i],loss_val[i],dxl_fd[i,1],dxl_fd[i,0],color='lightgreen')
  plt.arrow(grid[i],loss_val[i],dxl_analytic[i,1],dxl_analytic[i,0],color='blue')
  plt.arrow(grid[i],loss_val[i],dxl_autodiff[i,1],dxl_autodiff[i,0],color='blue')
  
  
plt.legend(['Validation loss','Finite difference','Analytic gradient','Autodiff gradient'])
# plt.legend(['Validation loss','Finite difference'])
plt.savefig('out/plot.pdf')
plt.close()