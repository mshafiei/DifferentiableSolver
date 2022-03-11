from tkinter import Variable
import jax
import jax.numpy as jnp
from flax import linen as nn
from jaxopt import implicit_diff, linear_solve
from cvgutils.nn.jaxUtils import utils, unet_model
from cvgutils.nn.jaxUtils.unet_parts import Sequential
from typing import Any
from jax import random


#diff solve(module)
#setup()
#  self.quadratic_model with primal parameters
#__call__
#  solve quadratic model by gauss newton

#Quadratic model
#Note: only hyper is a flax parameter. primal is not
#def init_primal()
#def init_hyper
#def __call__
#  quadratic model. combination of hyper and primal


class Quad_model(nn.Module):
    
    @staticmethod
    def init_primal(batch):
        pass

    @staticmethod
    def init_hyper(key,batch,dtype):
        pass

    @nn.compact
    def __call__(self,primal_param,inpt):
        pass
    
    def visualize(self,primal_param,inpt):
        pass

class diff_solver(nn.Module):
    """Differentiable solver class
    Input: Quadratic objective and initialization
    Output: Minimizer of the objective found by gauss newton method
    """
    opts: Any
    quad_model : Quad_model

    
    
    @staticmethod
    def parse_arguments(parser):
        parser.add_argument('--nlin_iter', type=int, default=1, help='Number of linear (Conjugate Gradient) iterations')
        parser.add_argument('--nnonlin_iter', type=int, default=1, help='Number of non linear (Gauss Newton) iterations')
        return parser
    
    def visualize(self,inpt):
        predict = self(inpt)
        return self.quad_model.visualize(predict[0],inpt)

    # @implicit_diff.custom_root(jax.grad(screen_poisson_objective),has_aux=True)
    
    def __call__(self,inpt):
        """Gauss newton solver

        Args:
            inpt (_type_): input

        Returns:
            _type_: minimizer
        """

        #Resudual
        r = lambda pp_image: self.quad_model(pp_image,inpt)
        #Quadratic objective
        r2sum = lambda pp_image: (r(pp_image) ** 2).sum()
        x = self.quad_model.init_primal(inpt)
        optim_cond = lambda x: (jax.grad(r2sum)(x) ** 2).sum()

            
        def Ax(pp_image):
            jtd = jax.jvp(r,(x,),(pp_image,))[1]
            return jax.vjp(r,x)[1](jtd)[0]
        def jtf(x):
            return jax.vjp(r,x)[1](r(x))[0]
        def Axb(x,d):
            return Ax(d) + jtf(x)

        # @implicit_diff.custom_root(Axb,has_aux=True)
        def linear_solver_id(d,x):
            d = linear_solve.solve_cg(matvec=Ax,
                                    b=-jtf(x),
                                    init=d,
                                    maxiter=self.opts.nlin_iter)
            aux = (Axb(x,d) ** 2).sum()
            return d, aux

        def loop_body(args):
            x,count, gn_opt_err, gn_loss,linear_opt_err = args
            d, linea_opt = linear_solver_id(None,x)
            x += 1.0 * d

            linear_opt_err = linear_opt_err.at[count.astype(int)].set(linea_opt)
            gn_opt_err = gn_opt_err.at[count.astype(int)].set(optim_cond(x))
            gn_loss = gn_loss.at[count.astype(int)].set(r2sum(x))
            count += 1
            return (x,count, gn_opt_err, gn_loss,linear_opt_err)

        loop_count = self.opts.nnonlin_iter
        val = (x,jnp.array([0.0]),-jnp.ones(loop_count),-jnp.ones(loop_count),-jnp.ones(loop_count))
        for i in range(loop_count):
            val = loop_body(val)
        x,count, gn_opt_err, gn_loss,linear_opt_err = val
        # x,count, gn_opt_err, gn_loss,linear_opt_err = jax.lax.fori_loop(0,loop_count,lambda i,val:loop_body(val),(x,0.0,-jnp.ones(loop_count),-jnp.ones(loop_count),-jnp.ones(loop_count))) 
        return x,{'count':count,'gn_opt_err':gn_opt_err, 'gn_loss':gn_loss,'linear_opt_err':linear_opt_err}
        ###############`# linear and nonlinear solvers end #################

#Models
class fnf_regularizer(Quad_model):
    # weight_init: float
    unet: unet_model.UNet
    def setup(self):
        self.alpha = Sequential([nn.Dense(3,use_bias=False),nn.softplus])
        # self.weight = 
    @staticmethod
    def init_primal(batch):
        return batch['noisy']

    @staticmethod
    def init_hyper(key,val,dtype):
        rand = random.uniform(key)
        return jnp.array(rand)

    def __call__(self,primal_param,inpt):
        avg_weight = (1. / 2.) ** 0.5 *  (1. / primal_param.reshape(-1).shape[0] ** 0.5)
        r1 =  primal_param - inpt['noisy']
        g = self.unet(inpt['net_input'])
        r2 = utils.dx(primal_param) - g[...,:3]
        r3 = utils.dy(primal_param) - g[...,3:]  
        alpha = self.alpha(inpt['net_input']).reshape(-1)      
        out = jnp.concatenate(( r1.reshape(-1), alpha * r2.reshape(-1), alpha * r3.reshape(-1)),axis=0)
        return out * avg_weight

class implicit_sanity_model(Quad_model):
    unet: unet_model.UNet
    def setup(self):
        self.alpha = self.param('alpha',
            implicit_sanity_model.init_hyper,
            None,
            jnp.array)

    @staticmethod
    def init_primal(batch):
        return batch['noisy']

    @staticmethod
    def init_hyper(key,val,dtype):
        rand = random.uniform(key)
        return jnp.array(rand)

    @nn.compact
    def __call__(self,primal_param,inpt):
        avg_weight = (1. / 2.) ** 0.5 *  (1. / primal_param.reshape(-1).shape[0] ** 0.5)
        r1 =  primal_param - inpt['noisy']
        g = self.unet(inpt['net_input'])
        r2 = self.alpha * (primal_param - g)
        out = jnp.concatenate(( r1.reshape(-1), r2.reshape(-1)),axis=0)
        return out * avg_weight
    
    def visualize(self,primal_param,inpt):
        r1 =  primal_param - inpt['noisy']
        g = self.unet(inpt['net_input'])
        r2 = self.alpha * (primal_param - g)
        return r1, g, r2

class implicit_poisson_model(Quad_model):
    unet: unet_model.UNet
    def setup(self):
        self.alpha = self.param('alpha',
            implicit_poisson_model.init_hyper,
            None,
            jnp.array)

    @staticmethod
    def init_primal(batch):
        return batch['noisy']

    @staticmethod
    def init_hyper(key,val,dtype):
        rand = random.uniform(key)
        return jnp.array(rand)

    @nn.compact
    def __call__(self,primal_param,inpt):
        avg_weight = (1. / 2.) ** 0.5 *  (1. / primal_param.reshape(-1).shape[0] ** 0.5)
        r1 =  primal_param - inpt['noisy']
        g = self.unet(inpt['net_input'])
        r2 = self.alpha * (utils.dx(primal_param) - g[...,:3])
        r3 = self.alpha * (utils.dy(primal_param) - g[...,3:])
        out = jnp.concatenate(( r1.reshape(-1), r2.reshape(-1), r3.reshape(-1)),axis=0)
        return out * avg_weight
    
    def visualize(self,primal_param,inpt):
        r1 =  primal_param - inpt['noisy']
        g = self.unet(inpt['net_input'])
        r2 = self.alpha * (utils.dx(primal_param) - g[...,:3])
        r3 = self.alpha * (utils.dy(primal_param) - g[...,3:])
        return r1, g[...,:3], g[...,3:], r2, r3
        

class screen_poisson(Quad_model):
    @staticmethod
    def init_primal(batch):
        return batch['noisy']

    @staticmethod
    def init_hyper(key,val,dtype):
        rand = random.uniform(key)
        return jnp.array(rand)

    @nn.compact
    def __call__(self,primal_param,inpt):
        alpha = self.param('alpha',
                    screen_poisson.init_hyper,
                    None,
                    jnp.array)

        avg_weight = (1. / 2.) ** 0.5 *  (1. / primal_param.reshape(-1).shape[0] ** 0.5)
        r1 =  primal_param - inpt['noisy']
        r2 = nn.softplus(alpha) * utils.dx(primal_param)
        r3 = nn.softplus(alpha) * utils.dy(primal_param)
        out = jnp.concatenate(( r1.reshape(-1), r2.reshape(-1), r3.reshape(-1)),axis=0)
        return out * avg_weight
