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
import optax

# solve interpolation by grad decent solver
# define implicit_jax_grad
# evaluate implicit_jax_grad(x_0)
# plot
# write the same functions for 3pix
# write the same functions for 3pix with smoothness term
# write the same functions for 2d
# create stencil

i1 = 0.
i2 = 1.
alpha_gt = 0.8
lr = 0.6
x_0=0.
def f_t(x,i1,i2,alpha):
    return (1-alpha)*(x-i1) ** 2 + alpha * (x-i2) ** 2


def x_val(x,i1,i2,alpha):
    @jit
    def update(params, i1,i2,alpha, opt_state):
        """ Compute the gradient for a batch and update the parameters """
        value, grads = value_and_grad(f_t,argnums=0)(params,i1,i2,alpha)
        opt_state = opt_update(0, grads, opt_state)
        return opt_state, opt_state, value
    step_size = 0.010
    maxiters = 200
    params = x
    opt_init, opt_update, get_params = optimizers.adam(step_size)
    opt_state = opt_init(params)

    for i in tqdm.trange(maxiters):
        params, opt_state, loss = update(params, i1,i2,alpha, opt_state)
        params = get_params(opt_state)
        print('x ',params,' alpha ',alpha, ' loss ', loss)
    return params

x_gt = x_val(x_0,i1,i2,alpha_gt)
def f_v(x,i1,i2,alpha):
    return (x_val(x,i1,i2,alpha) - x_gt) ** 2

f_v(x_0,i1,i2,0.1)