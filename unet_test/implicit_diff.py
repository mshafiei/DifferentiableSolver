from tkinter import Variable
import jax
import jax.numpy as jnp
from flax import linen as nn
from jaxopt import implicit_diff, linear_solve
from cvgutils.nn.jaxUtils import utils, unet_model
from cvgutils.nn.jaxUtils.unet_parts import Sequential

class diff_solver:
    """Differentiable solver class
    Input: Quadratic objective and initialization
    Output: Minimizer of the objective found by gauss newton method
    """
    opts = None
    obj = None
    init_point = None
    @staticmethod
    def init(opts,obj,init_point):
        diff_solver.opts = opts
        diff_solver.obj = obj
        diff_solver.init_point = init_point
        
    
    @staticmethod
    def parse_arguments(parser):
        parser.add_argument('--nlin_iter', type=int, default=1, help='Number of linear (Conjugate Gradient) iterations')
        parser.add_argument('--nnonlin_iter', type=int, default=1, help='Number of non linear (Gauss Newton) iterations')
        return parser
    
    @staticmethod
    def stencil_residual(pp_image,hp_nn, data):
        return diff_solver.obj(hp_nn,pp_image,data)
        ################ inner loop model end ############################

    ################ linear and nonlinear solvers begin ##############
    @staticmethod
    def screen_poisson_objective(pp_image,hp_nn, data):
        """Objective function."""
        return (diff_solver.stencil_residual(pp_image,hp_nn, data) ** 2).sum()

    # @implicit_diff.custom_root(jax.grad(screen_poisson_objective),has_aux=True)
    @staticmethod
    def nonlinear_solver_id(hp_nn,inpt):
        """Gauss newton solver

        Args:
            hp_nn (_type_): hyper parameters
            inpt (_type_): input

        Returns:
            _type_: minimizer
        """
        x = diff_solver.init_point(inpt)
        loss = lambda pp_image:diff_solver.screen_poisson_objective(pp_image,hp_nn,inpt)
        optim_cond = lambda x: (jax.grad(loss)(x) ** 2).sum()

            
        def cg_optimality(d,x,hp_nn,inpt):
            f = lambda pp_image:diff_solver.stencil_residual(pp_image,hp_nn,inpt)
            def Ax(pp_image):
                jtd = jax.jvp(f,(x,),(pp_image,))[1]
                return jax.vjp(f,x)[1](jtd)[0]
            def jtf(x):
                return jax.vjp(f,x)[1](f(x))[0]
            cg = Ax(d) + jtf(x)
            return cg

        # @implicit_diff.custom_root(cg_optimality,has_aux=True)
        def linear_solver_id(d,x,hp_nn,inpt):
            f = lambda pp_image:diff_solver.stencil_residual(pp_image,hp_nn,inpt)
            def Ax(pp_image):
                jtd = jax.jvp(f,(x,),(pp_image,))[1]
                return jax.vjp(f,x)[1](jtd)[0]
            def jtf(x):
                return jax.vjp(f,x)[1](f(x))[0]
            d = linear_solve.solve_cg(matvec=Ax,
                                    b=-jtf(x),
                                    init=d,
                                    maxiter=diff_solver.opts.nlin_iter)
            aux = ((Ax(d) +jtf(x)) ** 2).sum()
            return d, aux

        def loop_body(args):
            x,count, gn_opt_err, gn_loss,linear_opt_err = args
            d, linea_opt = linear_solver_id(None,x,hp_nn,inpt)
            x += 1.0 * d

            linear_opt_err = linear_opt_err.at[count.astype(int)].set(linea_opt)
            gn_opt_err = gn_opt_err.at[count.astype(int)].set(optim_cond(x))
            gn_loss = gn_loss.at[count.astype(int)].set(diff_solver.screen_poisson_objective(x,hp_nn,inpt))
            count += 1
            return (x,count, gn_opt_err, gn_loss,linear_opt_err)

        loop_count = diff_solver.opts.nnonlin_iter
        val = (x,jnp.array([0.0]),-jnp.ones(loop_count),-jnp.ones(loop_count),-jnp.ones(loop_count))
        for i in range(loop_count):
            val = loop_body(val)
        x,count, gn_opt_err, gn_loss,linear_opt_err = val
        # x,count, gn_opt_err, gn_loss,linear_opt_err = jax.lax.fori_loop(0,loop_count,lambda i,val:loop_body(val),(x,0.0,-jnp.ones(loop_count),-jnp.ones(loop_count),-jnp.ones(loop_count))) 
        return x,{'count':count,'gn_opt_err':gn_opt_err, 'gn_loss':gn_loss,'linear_opt_err':linear_opt_err}
        ###############`# linear and nonlinear solvers end #################

#Models
class fnf_regularizer(nn.Module):
    # weight_init: float
    unet: unet_model.UNet
    def setup(self):
        self.alpha = Sequential([nn.Dense(3,use_bias=False),nn.softplus])
        # self.weight = 
    @staticmethod
    def init_point(batch):
        return batch['noisy']
    
    def __call__(self,primal_param,inpt):
        avg_weight = (1. / 2.) ** 0.5 *  (1. / primal_param.reshape(-1).shape[0] ** 0.5)
        r1 =  primal_param - inpt['noisy']
        g = self.unet(inpt['net_input'])
        r2 = utils.dx(primal_param) - g[...,:3]
        r3 = utils.dy(primal_param) - g[...,3:]  
        alpha = self.alpha(inpt['net_input']).reshape(-1)      
        out = jnp.concatenate(( r1.reshape(-1), alpha * r2.reshape(-1), alpha * r3.reshape(-1)),axis=0)
        return out * avg_weight

class screen_poisson(nn.Module):
    # weight_init: float
    def setup(self):
        pass

    @staticmethod
    def init_primal(batch):
        return batch['noisy']

    @staticmethod
    def init_hyper(key,batch,dtype):
        return jnp.array([1.0])

    @nn.compact
    def __call__(self,primal_param,inpt):
        alpha = self.param('alpha',
                    screen_poisson.init_hyper,
                    (1),
                    jnp.array)

        avg_weight = (1. / 2.) ** 0.5 *  (1. / primal_param.reshape(-1).shape[0] ** 0.5)
        r1 =  primal_param - inpt['noisy']
        r2 = alpha * utils.dx(primal_param)
        r3 = alpha * utils.dy(primal_param)
        out = jnp.concatenate(( r1.reshape(-1), r2.reshape(-1), r3.reshape(-1)),axis=0)
        return out * avg_weight
