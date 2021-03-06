from absl import app
import jax
from jax._src.numpy.lax_numpy import argsort, interp, zeros_like
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
import cvgutils.Viz as cvgviz
import cvgutils.nn.jaxutils as jaxutils
from jax.experimental import stax

lmbda_init = 0.1
lmbda_gt = 0.9
h,w = 100,100
dw = 3
key1 = jax.random.PRNGKey(42)
key2 = jax.random.PRNGKey(43)

inpt = jax.random.uniform(key1,(h,w))
hp_w_gt = jnp.array([[0,-1,0],[-1,2,0],[0,0,0]])
hp_b_gt = jnp.array([[0,-1,0],[-1,2,0],[0,0,0]])
# window_gt = jnp.ones((3,3))
# x = jnp.linspace(-1, 1, dw)
# window_gt = jsp.stats.norm.pdf(x) * jsp.stats.norm.pdf(x[:, None])
# window_gt /= jnp.linalg.norm(window_gt)
# image_gt = jsp.signal.convolve(inpt, window_gt, mode='same')
init_inpt = jnp.zeros_like(inpt)
hp_w_init,hp_b_init = hp_w_gt +jax.random.uniform(key1,(dw,dw))*0.1, jax.random.uniform(key1,inpt.shape)
logger = cvgviz.logger('./logger','filesystem','autodiff','autodiff')

@jax.jit
def stencil_residual(pp_image, hp_w,hp_b, data):
  """Objective function."""
  avg_weight = 0.5 ** 0.5 * 1 / len(data.reshape(-1)) ** 0.5
  r1 =  pp_image - data
  relu_image = jsp.signal.convolve(pp_image.reshape(h,w), hp_w.reshape(dw,dw) , mode='same')
  # relu_image = jaxutils.relu(conv_image)
  out = jnp.concatenate(( r1.reshape(-1), relu_image.reshape(-1)),axis=0)
  return avg_weight * out


@jax.jit
def screen_poisson_objective(pp_image, hp_w,hp_b, data):
  """Objective function."""
  return (stencil_residual(pp_image, hp_w,hp_b, data) ** 2).sum()


@implicit_diff.custom_root(jax.grad(screen_poisson_objective))
def screen_poisson_solver(init_image,hp_w,hp_b, data):
    f = lambda pp_image:stencil_residual(pp_image,hp_w,hp_b,data)
    loss = lambda pp_image:screen_poisson_objective(pp_image,hp_w,hp_b,data)
    def matvec(pp_image):
        jtd = jax.jvp(f,(init_image,),(pp_image,))[1]
        return jax.vjp(f,init_image)[1](jtd)[0]
    def jtf(x):
      return jax.vjp(f,x)[1](f(x))[0]

    #gauss newton loop
    gn_iters = 3
    x = init_image
    step = logger.step
    # logger.addScalar(np.float32(loss(x).sum()),'loss_GN_%04i'%(step))
    for i in range(gn_iters):
        x += linear_solve.solve_cg(matvec=matvec,
                                b=-jtf(x),
                                init=x,
                                maxiter=100)
        # logger.addScalar(np.float32(loss(x).sum()),'loss_GN_%04i'%(step))
        # logger.takeStep()
    return x

@jax.jit
def outer_objective(hp_w,hp_b, init_inner, data):
    """Validation loss."""
    inpt, gt = data
    f = lambda hp_w,hp_b: screen_poisson_solver(init_inner, hp_w,hp_b, inpt)
    f_v = (f(hp_w,hp_b) - gt) ** 2
    return f_v.mean()

def fd(hyper_params, init_inner, data,delta):
  grad = jnp.zeros_like(hyper_params.reshape(-1))
  for i in range(hyper_params.reshape(-1).shape[0]):
    winf = hyper_params.reshape(-1)
    winb = hyper_params.reshape(-1)
    winf = winf.at[i].add(delta/2)
    winb = winb.at[i].add(-delta/2)
    f = outer_objective(winf.reshape(dw,dw), init_inner, data)
    b = outer_objective(winb.reshape(dw,dw), init_inner, data)
    grad = grad.at[i].set((f - b) / delta)
  return grad.reshape(dw,dw)

def hyper_optimization():
    
    # im_gt = jsp.signal.convolve(inpt, window_gt, mode='same')
    im_gt = screen_poisson_solver(init_inpt, hp_w_gt,hp_b_gt, inpt)
    data = [inpt, im_gt]
    count = 20
    # lmbdas = jnp.linspace(lmbda_init,lmbda_gt,count)
    # valid_loss = jnp.array([outer_objective(lmbdas[i], init_inpt, data) for i in range(count)])
    hp_w,hp_b = hp_w_init, hp_b_init
    
    lr = 0.002
    #compare to fd
    # fd_grad = fd(window, init_inpt, data,0.00001)
    # g_grad = jax.grad(outer_objective,argnums=0)(window, init_inpt, data)
    # print('tst')
    f = lambda hp_w,hp_b:outer_objective(hp_w,hp_b, init_inpt, data)
    
    # solver = OptaxSolver(fun=f, opt=optax.adam(3e-4), implicit_diff=True)
    # initial, _ =solver.init(init_params=init_window)
    # result, _ = solver.run(init_params = initial)
    # print('result ',result)

    for i in tqdm.trange(2000):
      loss,grad = jax.value_and_grad(f,argnums=(0,1))(hp_w,hp_b)
      logger.addScalar(loss,'loss_GD')
      if(i%100 == 0):
        output = screen_poisson_solver(init_inpt, hp_w,hp_b, inpt)
        imshow = jnp.concatenate((np.clip(output,0,1),inpt,im_gt),axis=1)
        logger.addImage(np.array(imshow[None,...]),'Image')
      logger.takeStep()
      # plt.figure(figsize=(24,7))
      # plt.subplot(1,4,1)
      # plt.plot(lmbdas,valid_loss,'r')
      # plt.scatter(lmbda_init,valid_loss[0],color='r')
      # plt.scatter(lmbda_gt,valid_loss[-1],color='g')
      # plt.arrow(window,loss,1,g[0],color='b')
      # plt.subplot(1,4,2)
      # plt.imshow(im_gt)
      # plt.subplot(1,4,3)
      # plt.imshow(inpt[0])
      # plt.subplot(1,4,4)
      # plt.imshow(inpt[1])
      # plt.savefig('out/plt_%04d.png'%i)
      
      # print('g \n',g)
      print('loss ', loss)
      print('window pred \n', hp_w)
      # print('window gt \n', window_gt)
      hp_w -= lr * grad[0]
      hp_b -= lr * grad[1]


hyper_optimization()
