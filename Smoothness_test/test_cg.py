from absl import app
import jax
from jax._src.numpy.lax_numpy import argsort
import jax.numpy as jnp
from jaxopt import implicit_diff
from jaxopt import linear_solve
from jaxopt import OptaxSolver
from matplotlib.pyplot import vlines
import optax
from sklearn import datasets
from sklearn import model_selection
from sklearn import preprocessing

lmbda_init = 0.1
lmbda_gt = 2.
inpt = jnp.array([0.,10.,0.])
init_inpt = jnp.zeros_like(inpt)
@jax.jit
def screen_poisson_objective(params, lmbda, data):
  """Objective function."""
  return 0.5 * ((params - data) ** 2).sum() + 0.5 * lmbda * ((params[1:] - params[:-1]) ** 2).sum()

def screen_poisson_residual(params, lmbda, data):
  """Objective function."""
  return jnp.concatenate((params - data, lmbda ** 0.5 * (params[1:] - params[:-1])),axis=0)


@implicit_diff.custom_root(jax.grad(screen_poisson_objective))
def screen_poisson_solver(init_params,lmbda, data):
    def matvec(u):
        a = u
        a = a.at[1:].set(a[1:] + lmbda * (u[1:] - u[:-1]))
        a = a.at[:-1].set(a[:-1] + lmbda * (u[:-1] - u[1:]))
        return a

    return linear_solve.solve_cg(matvec=matvec,
                                b=data,
                                init=init_params,
                                maxiter=100)

@implicit_diff.custom_root(jax.grad(screen_poisson_objective))
def screen_poisson_solver2(init_params,lmbda, data):
    # grad_f = jax.grad(screen_poisson_objective,argnums=0)
    f = lambda u:screen_poisson_residual(u,lmbda,data)
    def matvec2(u):
        jtd = jax.jvp(f,(init_params,),(u,))[1]
        return jax.vjp(f,init_params)[1](jtd)[0]
    
    jtf = jax.vjp(f,init_params)[1](f(init_params))[0]

    return linear_solve.solve_cg(matvec=matvec2,
                                b=-jtf,
                                init=init_params,
                                maxiter=100)
                                

def outer_objective(lmbda, init_inner, data):
    """Validation loss."""
    inpt, gt = data
    # We use the bijective mapping l2reg = jnp.exp(theta)
    # both to optimize in log-space and to ensure positivity.
    param = screen_poisson_solver2(init_inner, lmbda, inpt)
    f_v = (param - gt) ** 2

    return f_v.sum()

def plot_tangent():
    
    im_gt = screen_poisson_solver2(init_inpt,lmbda_gt,inpt)
    data = [inpt, im_gt]
    
    count = 20
    lmbdas = jnp.linspace(0.5,4,count)
    delta = 0.0001
    valid_loss = jnp.array([outer_objective(lmbdas[i], init_inpt, data) for i in range(count)])
    grad_lmbdas = jnp.array([jax.grad(outer_objective,argnums=0)(lmbdas[i], init_inpt, data) for i in range(count)])
    fd_lmbdas = jnp.array([(outer_objective(lmbdas[i] + delta/2, init_inpt, data) - outer_objective(lmbdas[i] - delta/2, init_inpt, data)) / delta for i in range(count)])
    import matplotlib.pylab as plt
    import tqdm
    plt.figure(figsize=(24,7))
    plt.subplot(1,3,1)
    plt.plot(lmbdas,valid_loss,'r')
    for i in tqdm.trange(valid_loss.shape[0]):
        plt.arrow(lmbdas[i],valid_loss[i],1,grad_lmbdas[i],color='g')
    plt.ylabel('Validation loss')
    plt.xlabel('Smoothness weight')
    plt.subplot(1,3,2)
    plt.plot(lmbdas,valid_loss,'r')
    for i in tqdm.trange(valid_loss.shape[0]):
        plt.arrow(lmbdas[i],valid_loss[i],1,fd_lmbdas[i],color='b')
    plt.xlabel('Smoothness weight')
    plt.subplot(1,3,3)
    plt.plot(lmbdas,valid_loss,'r')
    for i in tqdm.trange(valid_loss.shape[0]):
        plt.arrow(lmbdas[i],valid_loss[i],1,grad_lmbdas[i],color='g')
        plt.arrow(lmbdas[i],valid_loss[i],1,fd_lmbdas[i],color='b')
    plt.legend(['Finite difference','Autodiff'])
    plt.xlabel('Smoothness weight')
    plt.savefig('out/plot.pdf')
    plt.close()
    # jax.check_grads(outer_objective,(lmbdas[0],init_inpt, data),order=1)

plot_tangent()