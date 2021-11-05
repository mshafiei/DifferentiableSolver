import os
import jax
from jaxopt._src import gradient_descent
import tqdm
import jax.numpy as np
from jax import random
import cvgutils.Image as cvgim
import cvgutils.Viz as cvgviz
from jaxopt.implicit_diff import custom_fixed_point, custom_root
import argparse
import matplotlib.pyplot as plt
from jax.experimental import optimizers
from jax import grad, jit, vmap, value_and_grad
import optax
import jaxopt

# solve interpolation by grad decent solver
# define implicit_jax_grad
# evaluate implicit_jax_grad(x_0)
# plot
# write the same functions for 3pix
# write the same functions for 3pix with smoothness term
# write the same functions for 2d
# create stencil
w=3
i1 = np.array([0,1,0])
alpha_gt = 0.8
alpha_0 = 0.1
maxiter=10
lr = 0.6
x_0=np.zeros(w)

# def f_data(x):
#   return ((x - i1) ** 2).mean()

# def f_dx(x,alpha):
#   return alpha * ((x[1:] - x[:-1]) ** 2).mean()

def f_t(x,alpha):
    return 0.5 * ((x - i1) ** 2).mean() + 0.5 * alpha * ((x[1:] - x[:-1]) ** 2).mean()

@custom_root(jax.grad(f_t,argnums=0))
def x(x,alpha):
    gd = jaxopt.GradientDescent(f_t,stepsize=lr,maxiter=maxiter)
    return gd.run(x,alpha).params

def f_v(alpha):
    return ((x(x_0,alpha) - x(x_0,alpha_gt)) ** 2).mean()

def f_v_fd(alpha,delta):
    f1 = f_v(alpha - delta/2)
    f2 = f_v(alpha + delta/2)
    return (f2 - f1) / delta

f_v_autodiff = jax.grad(f_v)
# f_v_autodiff(0.1)
# f_v_fd(0.1, 0.01)
# gd_v = jaxopt.GradientDescent(f_v,stepsize=lr,maxiter=maxiter)
# gd_v.optimality_fun = jax.grad(f_t,argnums=0)
# gd_v.run(alpha_0)
grid = np.linspace(0.01,2,20)

fd_val = np.stack([f_v_fd(i, 0.001) for i in tqdm.tqdm(grid)],axis=-1)
autodiff_val = np.stack([f_v_autodiff(i) for i in tqdm.tqdm(grid)],axis=-1)
dxl_fd = np.stack((fd_val,np.ones_like(fd_val)),axis=-1)
dxl_autodiff_val = np.stack((autodiff_val,np.ones_like(autodiff_val)),axis=-1)
# dxl_autodiff_val = dxl_autodiff_val / np.linalg.norm(dxl_autodiff_val,axis=-1)[:,None] * np.linalg.norm(dxl_autodiff_val[:,None],axis=1,keepdims=True)

loss_val = np.stack([f_v(i) for i in tqdm.tqdm(grid)])
plt.figure(figsize=(24,7))
plt.subplot(1,3,1)
plt.plot(grid,loss_val,'r')
for i in range(grid.shape[0]):
  plt.arrow(grid[i],loss_val[i],dxl_fd[i,1],dxl_fd[i,0],color='lightgreen')
plt.xlabel('Smoothness weight')
plt.ylabel('Validation loss')
plt.ylim((-10,70))


plt.subplot(1,3,2)
plt.plot(grid,loss_val,'r')
for i in range(grid.shape[0]):
  plt.arrow(grid[i],loss_val[i],dxl_autodiff_val[i,1],dxl_autodiff_val[i,0],color='blue')
plt.xlabel('Smoothness weight')
plt.ylim((-10,70))

plt.subplot(1,3,3)
plt.plot(grid,loss_val,'r')
for i in range(grid.shape[0]):
  plt.arrow(grid[i],loss_val[i],dxl_fd[i,1],dxl_fd[i,0],color='lightgreen')
  plt.arrow(grid[i],loss_val[i],dxl_autodiff_val[i,1],dxl_autodiff_val[i,0],color='blue')

plt.xlabel('Smoothness weight')
plt.ylim((-10,70))

plt.legend(['Validation loss','Finite difference','Autodiff gradient'])
plt.savefig('out/plot.pdf')
plt.close()