from absl import app
import jax
from jax._src.numpy.lax_numpy import argsort
import jax.numpy as jnp
from jaxopt import implicit_diff
from jaxopt import linear_solve
from jaxopt import OptaxSolver, GradientDescent
from matplotlib.pyplot import vlines
import optax
from sklearn import datasets
from sklearn import model_selection
from sklearn import preprocessing
import matplotlib.pylab as plt
import numpy as np
import jax.scipy as jsp
import tqdm

lmbda_init = 0.1
lmbda_gt = 20.
h,w = 100,100
dw = 1
key = jax.random.PRNGKey(42)

# x = jnp.linspace(-3, 3, dw)
inpt = [jax.random.uniform(key,(h,w,)),jax.random.uniform(key,(h,w,))]
x = jnp.array([1])
window_gt = jsp.stats.norm.pdf(x) * jsp.stats.norm.pdf(x[:, None])
image_gt = jsp.signal.convolve(inpt, window_gt, mode='same').reshape(-1)
init_inpt = jnp.zeros_like(inpt)
init_window = jnp.zeros_like(window_gt)

# @jax.jit
def stencil_residual(image, window, data):
  """Objective function."""
  r1 = 1 / len(data[0].reshape(-1)) ** 0.5 * (image.reshape(h,w) - data[0].reshape(h,w))
#   r2 = 1 / len(data.reshape(-1)) ** 0.5 * lmbda ** 0.5 * (params[0].reshape(h,w)[1:,:] - params[0].reshape(h,w)[:-1,:])
#   r3 = 1 / len(data.reshape(-1)) ** 0.5 * lmbda ** 0.5 * (params[0].reshape(h,w)[:,1:] - params[0].reshape(h,w)[:,:-1])
  
#   smooth_image = 1 / len(data) ** 0.5 * image * window
  smooth_image = 1 / len(data[0]) ** 0.5 * jsp.signal.convolve((image - data[1]).reshape(h,w), window.reshape(dw,dw), mode='same')
  out = jnp.concatenate((0.5 ** 0.5 * r1.reshape(-1),0.5 ** 0.5 * smooth_image.reshape(-1)),axis=0)
  return out


# @jax.jit
def screen_poisson_objective(image, window, data):
  """Objective function."""
  return (stencil_residual(image, window, data) ** 2).sum()


@implicit_diff.custom_root(jax.grad(screen_poisson_objective))
def screen_poisson_solver(init_image,window, data):
    f = lambda u:stencil_residual(u,window,data)
    def matvec(u):
        jtd = jax.jvp(f,(init_image,),(u,))[1]
        return jax.vjp(f,init_image)[1](jtd)[0]
    
    jtf = jax.vjp(f,init_image)[1](f(init_image))[0]

    #gauss newton loop
    gn_iters = 3
    x = init_image
    for i in range(gn_iters):
        x += linear_solve.solve_cg(matvec=matvec,
                                b=-jtf,
                                init=x,
                                maxiter=100)
    return x
                                
# @jax.jit
def outer_objective(window, init_inner, data):
    """Validation loss."""
    inpt, gt = data
    # We use the bijective mapping l2reg = jnp.exp(theta)
    # both to optimize in log-space and to ensure positivity.
    f = lambda u: screen_poisson_solver(init_inner, u, inpt)
    f_v = (f(window) - gt) ** 2
    print('outer_obj ',f(window))
    return f_v.mean()

def hyper_optimization():
    im_gt = jsp.signal.convolve(inpt, window_gt, mode='same')
    window = init_window
    data = [inpt, im_gt]
    lr = .9
    import cv2
    for i in tqdm.trange(2000):
        # a = outer_objective(window, init_inpt, data)
        g = jax.grad(outer_objective,argnums=0)(window, init_inpt, data)
        window -= lr * g
        if(i%100 == 0):
            im_gt = jsp.signal.convolve(init_inpt, window_gt, mode='same')
            im = screen_poisson_solver(init_inpt, window, inpt)
            imshow = jnp.concatenate((im.reshape(h,w),im_gt.reshape(h,w)),axis=1)
            cv2.imwrite('./out/%05i.png' % i,np.array(imshow * 255).astype(np.uint8))
        print('g ',g,' window ', window, ' loss ', outer_objective(window, init_inpt, data))


# plot_tangent()
hyper_optimization()
# hyper_optimization_jaxopt()
# hyper_optimization_jaxopt_adam()

# def plot_tangent():

#     im_gt = screen_poisson_solver(init_inpt,window_gt,inpt)
#     data = [inpt, im_gt]
    
#     count = 20
#     lmbdas = jnp.linspace(0.5,4,count)
#     delta = 0.0001
#     valid_loss = jnp.array([outer_objective(lmbdas[i], init_inpt, data) for i in range(count)])
#     grad_lmbdas = jnp.array([jax.grad(outer_objective,argnums=0)(lmbdas[i], init_inpt, data) for i in range(count)])
#     fd_lmbdas = jnp.array([(outer_objective(lmbdas[i] + delta/2, init_inpt, data) - outer_objective(lmbdas[i] - delta/2, init_inpt, data)) / delta for i in range(count)])
    
#     plt.figure(figsize=(24,7))
#     plt.subplot(1,3,1)
#     plt.plot(lmbdas,valid_loss,'r')
#     for i in tqdm.trange(valid_loss.shape[0]):
#         plt.arrow(lmbdas[i],valid_loss[i],1,grad_lmbdas[i],color='g')
#     plt.ylabel('Validation loss')
#     plt.xlabel('Smoothness weight')
#     plt.subplot(1,3,2)
#     plt.plot(lmbdas,valid_loss,'r')
#     for i in tqdm.trange(valid_loss.shape[0]):
#         plt.arrow(lmbdas[i],valid_loss[i],1,fd_lmbdas[i],color='b')
#     plt.xlabel('Smoothness weight')
#     plt.subplot(1,3,3)
#     plt.plot(lmbdas,valid_loss,'r')
#     for i in tqdm.trange(valid_loss.shape[0]):
#         plt.arrow(lmbdas[i],valid_loss[i],1,grad_lmbdas[i],color='g')
#         plt.arrow(lmbdas[i],valid_loss[i],1,fd_lmbdas[i],color='b')
#     plt.legend(['Finite difference','Autodiff'])
#     plt.xlabel('Smoothness weight')
#     plt.savefig('out/plot.pdf')
#     plt.close()
#     # jax.check_grads(outer_objective,(lmbdas[0],init_inpt, data),order=1)

# def hyper_optimization_jaxopt():
#     lr = 0.05
#     im_gt = screen_poisson_solver(init_inpt,lmbda_gt,inpt)
#     data = [inpt, im_gt]
#     lmbda = lmbda_init
#     f_v = lambda u: outer_objective(u,init_inpt,data)
#     gd = GradientDescent(f_v,stepsize=lr,maxiter=50)
#     print('solution ',gd.run(lmbda).params)
#     return gd.run(0.1).params

# def hyper_optimization_jaxopt_adam():
#     lr = 0.05
#     im_gt = screen_poisson_solver(init_inpt,lmbda_gt,inpt)
#     data = [inpt, im_gt]
#     lmbda = lmbda_init
#     f_v = lambda u: outer_objective(u,init_inpt,data)
#     gd = OptaxSolver(opt=optax.adam(0.1),fun=f_v)
#     print('solution ',gd.run(lmbda).params)
#     return gd.run(0.1).params

