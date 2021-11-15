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
import tqdm
import cvgutils.Image as cvgim

lmbda_init = 0.1
lmbda_gt = 4.
h,w = 100,100
key = jax.random.PRNGKey(42)

# inpt = jnp.array([[0.,10.,0.],[10.,0.,10.],[0.,10.,0.]])
inpt = jax.random.uniform(key,(h*w,))
init_inpt = jnp.zeros_like(inpt)

@jax.jit
def screen_poisson_residual(params, lmbda, data):
  """Objective function."""
  r1 = 1 / len(data.reshape(-1)) ** 0.5 * (params.reshape(h,w) - data.reshape(h,w))
  r2 = 1 / len(data.reshape(-1)) ** 0.5 * lmbda ** 0.5 * (params.reshape(h,w)[1:,:] - params.reshape(h,w)[:-1,:])
  r3 = 1 / len(data.reshape(-1)) ** 0.5 * lmbda ** 0.5 * (params.reshape(h,w)[:,1:] - params.reshape(h,w)[:,:-1])
  return jnp.concatenate((0.5 ** 0.5 * r1.reshape(-1),0.5 ** 0.5 * r2.reshape(-1),0.5 ** 0.5 * r3.reshape(-1)),axis=0)

@jax.jit
def screen_poisson_objective(params, lmbda, data):
  """Objective function."""
  return (screen_poisson_residual(params, lmbda, data) ** 2).sum()



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

    #gauss newton loop
    gn_iters = 3
    x = init_params
    for i in range(gn_iters):
        x += linear_solve.solve_cg(matvec=matvec2,
                                b=-jtf,
                                init=x,
                                maxiter=100)
    return x
                                
@jax.jit
def outer_objective(lmbda, init_inner, data):
    """Validation loss."""
    inpt, gt = data
    # We use the bijective mapping l2reg = jnp.exp(theta)
    # both to optimize in log-space and to ensure positivity.
    f = lambda u: screen_poisson_solver2(init_inner, u, inpt)
    f_v = (f(lmbda) - gt) ** 2

    return f_v.mean()

def plot_tangent():

    im_gt = screen_poisson_solver2(init_inpt,lmbda_gt,inpt)
    data = [inpt, im_gt]
    
    count = 20
    delta = 0.0001
    lmbdas = jnp.linspace(0.5,4,count)
    valid_loss = jnp.array([outer_objective(lmbdas[i], init_inpt, data) for i in range(count)])
    grad_lmbdas = jnp.array([jax.grad(outer_objective,argnums=0)(lmbdas[i], init_inpt, data) for i in range(count)])
    fd_lmbdas = jnp.array([(outer_objective(lmbdas[i] + delta/2, init_inpt, data) - outer_objective(lmbdas[i] - delta/2, init_inpt, data)) / delta for i in range(count)])
    
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
def hyper_optimization_jaxopt():
    lr = 0.05
    im_gt = screen_poisson_solver2(init_inpt,lmbda_gt,inpt)
    data = [inpt, im_gt]
    lmbda = lmbda_init
    f_v = lambda u: outer_objective(u,init_inpt,data)
    gd = GradientDescent(f_v,stepsize=lr,maxiter=50)
    print('solution ',gd.run(lmbda).params)
    return gd.run(0.1).params

def hyper_optimization_jaxopt_adam():
    lr = 0.05
    im_gt = screen_poisson_solver2(init_inpt,lmbda_gt,inpt)
    data = [inpt, im_gt]
    lmbda = lmbda_init
    f_v = lambda u: outer_objective(u,init_inpt,data)
    gd = OptaxSolver(opt=optax.adam(0.1),fun=f_v)
    print('solution ',gd.run(lmbda).params)
    return gd.run(0.1).params

def hyper_optimization():
    im_gt = screen_poisson_solver2(init_inpt,lmbda_gt,inpt)
    count = 20
    lmbda = lmbda_init
    data = [inpt, im_gt]
    lmbdas = jnp.linspace(lmbda_init,lmbda_gt,count)
    valid_loss = jnp.array([outer_objective(lmbdas[i], init_inpt, data) for i in range(count)])
    

    lr = .9
    import cv2
    for i in tqdm.trange(2000):
        g = jax.grad(outer_objective,argnums=0)(lmbda, init_inpt, data)
        
        if(i%100 == 0):
            plt.figure(figsize=(12*1.3,3.5*1.3))
            im = screen_poisson_solver2(init_inpt, lmbda, inpt)
            imshow = jnp.concatenate((im.reshape(h,w),im_gt.reshape(h,w)),axis=1)
            imshow = np.array(imshow * 255).astype(np.uint8)
            # cv2.imwrite('./out/%05i.png' % i,)
            plt.subplot(1,3,1)
            plt.imshow((im.reshape(h,w) * 255))
            plt.title('Prediction')
            plt.xlabel('Iteration %04i, lambda %05d'%(i,lmbda))
            plt.subplot(1,3,2)
            plt.imshow((im_gt.reshape(h,w) * 255))
            plt.title('Ground truth')
            plt.xlabel('lambda = ' + '%05d'%lmbda_gt)
            plt.subplot(1,3,3)
            plt.plot(lmbdas,valid_loss,'r')
            loss = outer_objective(lmbda, init_inpt, data)
            loss_gt = outer_objective(lmbda_gt, init_inpt, data)
            nrm = jnp.linalg.norm(jnp.array([1,g]))
            plt.arrow(lmbda,loss,1/nrm,g/nrm,color='b')
            plt.scatter(lmbda,loss,color='r',marker='o')
            plt.scatter(lmbda_gt,loss_gt,color='g',marker='o')
            plt.legend(['Validation loss','Tangent vector','Prediction loss','Ground truth loss'])
            plt.ylabel('Validation loss')
            plt.xlabel('Smoothness weight')
            plt.savefig('./out/plt_%04i.png'%i)
            plt.close()
        
        lmbda -= lr * g
        print('g ',g,' lmbda ', lmbda, ' loss ', outer_objective(lmbda, init_inpt, data))
    import glob
    imfns = sorted(glob.glob('./out/plt_*.png'))
    fn = './out/plt.avi'
    im = cv2.imread(imfns[0])
    out = cv2.VideoWriter(fn,cv2.VideoWriter_fourcc(*'DIVX'), 5, (im.shape[1],im.shape[0]))
    for imfn in imfns:
        im = cv2.imread(imfn)
        out.write(im)
    out.release()
    # cvgim.imageseq2avi(fn,cvgim.loadImageSeq(sorted(imfns)))
    

# plot_tangent()
hyper_optimization()
# hyper_optimization_jaxopt()
# hyper_optimization_jaxopt_adam()
import matplotlib.pyplot as plt
plt.figure((24,10))
plt.subplot()