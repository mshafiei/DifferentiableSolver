from tkinter import Variable
import jax
import jax.numpy as jnp
from flax import linen as nn
from jaxopt import implicit_diff, linear_solve
from cvgutils.nn.jaxUtils import utils, unet_model
from cvgutils.nn.jaxUtils.unet_parts import Sequential
from typing import Any
import deepfnf_utils.tf_utils as tfu
from jax import random
import cvgutils.Linalg as linalg
import cvgutils.Image as cvgim
from functools import partial
import cvgutils.nn.jaxUtils.utils as jaxutils
from collections import OrderedDict
import numpy as np
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

class direct_model(nn.Module):
    """Differentiable solver class
    Input: Quadratic objective and initialization
    Output: Minimizer of the objective found by gauss newton method
    """
    opts: Any
    quad_model : nn.Module

    @staticmethod
    def parse_arguments(parser):
        return parser
    
    def visualize(self,inpt):
        # _,aux = self(inpt)
        return {}

    def __call__(self,inpt):
        pred = self.quad_model(inpt['net_input'])
        aux = {}
        aux['pred'] = pred
        solution = jnp.clip(tfu.camera_to_rgb_batch(pred/inpt['alpha'],inpt),0,1)
        ambient = jnp.clip(tfu.camera_to_rgb_batch(inpt['ambient'],inpt),0,1)
        aux['sse_loss'] = ((ambient - solution) ** 2).sum()
        loss_val = self.opts.sse_weight * aux['sse_loss']

        return loss_val, aux

    def labels(self):
        return {}

class fft_solver(nn.Module):
    """Solves a screen poisson equation by an fft solver
    Input: gradient for screen poisson equation
    """
    opts: Any
    quad_model : Quad_model
    alpha_type: str
    alpha_map: Quad_model
    fft_model: str
    delta_phi_init: float
    delta_psi_init: float
    fixed_delta: bool
    @staticmethod
    def parse_arguments(parser):
        parser.add_argument('--delta_phi_init', type=float, default=0.00001, help='Initial value for delta phi')
        parser.add_argument('--delta_psi_init', type=float, default=0.00001, help='Initial value for delta psi')
        parser.add_argument('--fixed_delta', action='store_true',help='Do not change the delta value')
        return parser

    def setup(self):
        if(self.alpha_type == 'map_2d'):
            self.alpha = lambda x: nn.softplus(self.alpha_map(x))
        elif(self.alpha_type == 'scalar'):
            self.alpha = self.param('alpha',
                implicit_sanity_model.init_hyper,
                None,
                jnp.array)
        if(self.fft_model == 'fft_helmholz' and (not self.fixed_delta)):

            self.delta_phi = self.param('delta_phi',
                implicit_sanity_model.init_hyper,
                jnp.log(jnp.exp(jnp.double(self.delta_phi_init)) - 1),
                jnp.array,0,0)
            self.delta_psi = self.param('delta_psi',
                implicit_sanity_model.init_hyper,
                jnp.log(jnp.exp(jnp.double(self.delta_psi_init)) - 1),
                jnp.array,0,0)
    def visualize(self,inpt):
        predict,g,aux = self.fft(inpt)
        div_func = lambda x,y:utils.dx(x) + utils.dy(y)
        curl_func = lambda x,y:utils.dx(y) - utils.dy(x)
        wb_noscale = lambda x : tfu.camera_to_rgb_batch(x,inpt)
        wb = lambda x : tfu.camera_to_rgb_batch(x/inpt['alpha'],inpt)
        # g = self.quad_model(inpt['net_input'])
        if(self.fft_model == 'fft_filters'):
            gx = predict * 0
            gy = predict * 0
        elif(self.fft_model == 'fft_highdim'):
            gx = aux['gx'][...,:5:2]
            gy = aux['gy'][...,1:6:2]
            phix = aux['phix'][...,0:5:2]
            phiy = aux['phiy'][...,1:6:2]
            ax = aux['psix'][...,0:5:2]
            ay = aux['psiy'][...,1:6:2]
            phi_heatmap = aux['phi'][...,0:5:2]
            a_heatmap = aux['psi'][...,1:6:2]
            curlrota = curl_func(-aux['psiy'],aux['psix'])
            divgradphi = div_func(aux['phix'],aux['phiy'])
            divcurl = jnp.concatenate((curlrota,divgradphi),axis=-1)
            divcurlmin, divcurlmax = float(np.array(divcurl.min())),float(np.array(divcurl.max()))
        elif(self.fft_model == 'fft_highdim_nohelmholz'):
            gx = aux['g'][...,0:5:2]
            gy = aux['g'][...,1:6:2]
        elif(self.fft_model == 'fft_helmholz'):
            phi = g[...,:3]
            a = g[...,3:]
            phi_heatmap = phi
            a_heatmap = a
            phix = utils.dx(phi)
            phiy = utils.dy(phi)
            ax = utils.dx(a)
            ay = utils.dy(a)
            curlrota = curl_func(-ay,ax)
            divgradphi = div_func(phix,phiy)
            divcurl = jnp.concatenate((curlrota,divgradphi),axis=-1)
            divcurlmin, divcurlmax = float(np.array(divcurl.min())),float(np.array(divcurl.max()))
            if(self.fixed_delta):
                gx = self.delta_psi_init * phix - self.delta_phi_init * ay
                gy = self.delta_psi_init * phiy + self.delta_phi_init * ax
            else:
                gx = nn.softplus(self.delta_psi_init) * phix - nn.softplus(self.delta_phi_init) * ay
                gy = nn.softplus(self.delta_psi_init) * phiy + nn.softplus(self.delta_phi_init) * ax
            
        elif(self.fft_model == 'fft_image_grad'):
            gx = utils.dx(g)
            gy = utils.dy(g)
        elif(self.fft_model == 'fft'):
            gx = g[...,:3]
            gy = g[...,3:]
        else:
            print('Error: no such fft model ', self.fft_model)
            exit(0)
        dx = utils.dx(predict)
        dy = utils.dy(predict)

        dxx = utils.dx(dx)
        dyy = utils.dy(dy)
        gxx = utils.dx(gx)
        gyy = utils.dy(gy)
        out = OrderedDict()
        out['pred2'] = wb(predict)
        out['gxx'] = wb(gxx) * 0.5 + 0.5
        out['dxx'] = wb(dxx) * 0.5 + 0.5
        out['gyy'] = wb(gyy) * 0.5 + 0.5
        out['dyy'] = wb(dyy) * 0.5 + 0.5
        out['gx'] = wb(gx) * 0.5 + 0.5
        out['dx'] = wb(dx) * 0.5 + 0.5
        out['gy'] = wb(gy) * 0.5 + 0.5
        out['dy'] = wb(dy) * 0.5 + 0.5
        out['noisyx'] = wb(utils.dx(inpt['noisy'])) * 0.5 + 0.5
        out['noisyy'] = wb(utils.dy(inpt['noisy'])) * 0.5 + 0.5
        out['ambientx'] = wb_noscale(utils.dx(inpt['ambient'])) * 0.5 + 0.5
        out['ambienty'] = wb_noscale(utils.dy(inpt['ambient'])) * 0.5 + 0.5
        out['flashx'] = wb_noscale(utils.dx(inpt['flash'])) * 0.5 + 0.5
        out['flashy'] = wb_noscale(utils.dy(inpt['flash'])) * 0.5 + 0.5
        div = utils.dx(gx) + utils.dy(gy)
        curl = utils.dy(gx) - utils.dx(gy)

        divcurl = jnp.concatenate((div[0].mean(-1),curl[0].mean(-1)),axis=-1)
        
        if(self.fft_model == 'fft_helmholz' or self.fft_model == 'fft_highdim' or self.fft_model == 'fft_highdim_nohelmholz'):
            out['phix'] = wb(phix) * 0.5 + 0.5
            out['phiy'] = wb(phiy) * 0.5 + 0.5
            out['ax'] = wb(ax) * 0.5 + 0.5
            out['ay'] = wb(ay) * 0.5 + 0.5
            divcurl = jnp.concatenate((divcurl,phi_heatmap[0].mean(-1),a_heatmap[0].mean(-1)),axis=-1)
            min_val,max_val = float(np.array(divcurl.min())),float(np.array(divcurl.max()))
            out['phi_heatmap'] = cvgim.heatmap(phi_heatmap[0].mean(-1),zmin=min_val,zmax=max_val)[None,...]
            out['psi_heatmap'] = cvgim.heatmap(a_heatmap[0].mean(-1),zmin=min_val,zmax=max_val)[None,...]
            out['rotax'] = wb(-ay) * 0.5 + 0.5
            out['rotay'] = wb(ax) * 0.5 + 0.5
            out['divgradphi'] = cvgim.heatmap(divgradphi[0].mean(-1),zmin=divcurlmin, zmax=divcurlmax)[None,...]
            out['curlrota'] = cvgim.heatmap(curlrota[0].mean(-1),zmin=divcurlmin, zmax=divcurlmax)[None,...]
        
        if(self.fft_model == 'fft' or self.fft_model == 'fft_image_grad' or self.fft_model == 'fft_helmholz'  or self.fft_model == 'fft_highdim' or self.fft_model == 'fft_highdim_nohelmholz'):
            min_val,max_val = float(np.array(divcurl.min())),float(np.array(divcurl.max()))
            out['div'] = cvgim.heatmap(div[0].mean(-1),zmin=min_val,zmax=max_val)[None,...]
            out['curl'] = cvgim.heatmap(curl[0].mean(-1),zmin=min_val,zmax=max_val)[None,...]

        elif(self.fft_model == 'fft_image_grad'):
            out.update({'g':wb(g) * 0.5 + 0.5})
        if(self.alpha_type == 'map_2d'):
            alpha = self.alpha(inpt['net_input'])
            out.update({'alpha_map':jnp.concatenate([alpha,alpha,alpha],axis=-1)})
        return out
        # predict,(gt,grad_x,dx,grad_y,dy) = self(inpt)
        # return [predict,gt,grad_x[None,...],dx,grad_y[None,...],dy]

    def labels(self):
        out = OrderedDict()
        out['pred2'] = r'$I$'
        out['gxx'] = r'$Unet~output (g^x_x)$'
        out['dxx'] = r'$I_{xx}$'
        out['gyy'] = r'$Unet~output (g^y_y)$'
        out['dyy'] = r'$I_{yy}$'
        out['gx'] = r'$Unet~output (g^x)$'
        out['dx'] = r'$I_{x}$'
        out['gy'] = r'$Unet~output (g^y)$'
        out['dy'] = r'$I_{y}$'
        out['noisyx'] = r'$\nabla_x I_{noisy}$'
        out['noisyy'] = r'$\nabla_y I_{noisy}$'
        out['ambientx'] = r'$\nabla_x I_{ambient}$'
        out['ambienty'] = r'$\nabla_y I_{ambient}$'
        out['flashx'] = r'$\nabla_x I_{flash}$'
        out['flashy'] = r'$\nabla_y I_{flash}$'

        if(self.fft_model == 'fft_helmholz'  or self.fft_model == 'fft_highdim' or self.fft_model == 'fft_highdim_nohelmholz'):
            out['phix'] = r'$\nabla_x \phi$'
            out['phiy'] = r'$\nabla_y \phi$'
            out['ax'] = r'$\nabla_x \psi$'
            out['ay'] = r'$\nabla_y \psi$'
            out['phi_heatmap'] = r'$\phi$'
            out['psi_heatmap'] = r'$\psi$'
            out['rotax'] = r'$rot(\nabla \psi)^x$'
            out['rotay'] = r'$rot(\nabla \psi)^y$'
            out['divgradphi'] = r'$div(\phi))$'
            out['curlrota'] = r'$curl(rot(\nabla \psi))$'
        
        if(self.fft_model == 'fft' or self.fft_model == 'fft_image_grad' or self.fft_model == 'fft_helmholz'  or self.fft_model == 'fft_highdim' or self.fft_model == 'fft_highdim_nohelmholz'):
            out['div'] = r'$\nabla \cdot g$'
            out['curl'] = r'$\nabla \times g$'

        elif(self.fft_model == 'fft_image_grad'):
            out['g'] = '$Unet output$'
        if(self.alpha_type == 'map_2d'):
            out['alpha_map'] = '$\lambda$'
        return out
        # return [r'$I$',r'$I_{ambient}$',r'$g^x$',r'$\nabla_x I$',r'$g^y$',r'$\nabla_y I$']
        
        
        
    def __call__(self,inpt):
        pred,g,aux_fft= self.fft(inpt)
        aux = {}
        fft_solution = jnp.clip(tfu.camera_to_rgb_batch(pred/inpt['alpha'],inpt),0,1)
        ambient = jnp.clip(tfu.camera_to_rgb_batch(inpt['ambient'],inpt),0,1)
        if(self.fft_model == 'fft_filters'):
            pass
        elif(self.fft_model == 'fft_highdim'):
            gx = aux_fft['gx'][...,:5:2]
            gy = aux_fft['gy'][...,1:6:2]
            phix = aux_fft['phix'][...,0:5:2]
            phiy = aux_fft['phiy'][...,1:6:2]
            ax = aux_fft['psix'][...,0:5:2]
            ay = aux_fft['psiy'][...,1:6:2]
        elif(self.fft_model == 'fft_highdim_nohelmholz'):
            gx = aux_fft['g'][...,0:5:2]
            gy = aux_fft['g'][...,1:6:2]
        elif(self.fft_model == 'fft_helmholz'):
            phi = g[...,:3]
            a = g[...,3:]
            phix = utils.dx(phi)
            phiy = utils.dy(phi)
            ax = utils.dx(a)
            ay = utils.dy(a)
            if(self.fixed_delta):
                gx = self.delta_psi_init * phix - self.delta_phi_init * ay
                gy = self.delta_psi_init * phiy + self.delta_phi_init * ax
            else:
                gx = nn.softplus(self.delta_psi_init) * phix - nn.softplus(self.delta_phi_init) * ay
                gy = nn.softplus(self.delta_psi_init) * phiy + nn.softplus(self.delta_phi_init) * ax
        elif(self.fft_model == 'fft_image_grad'):
            gx = utils.dx(g)
            gy = utils.dy(g)
        elif(self.fft_model == 'fft'):
            gx = g[...,:3]
            gy = g[...,3:]
        loss_val = 0.
        aux['pred'] = pred
        if(self.fft_model == 'fft' or self.fft_model == 'fft_image_grad' or self.fft_model == 'fft_helmholz' or self.fft_model == 'fft_highdim' or self.fft_model == 'fft_highdim_nohelmholz'):
            aux['div'] = utils.dx(gx) + utils.dy(gy)
            aux['curl'] = utils.dy(gx) - utils.dx(gy)
        sg = lambda x: jax.lax.stop_gradient(x)
        if(self.opts.sse_weight > 0):
            aux['sse_loss'] = ((ambient - fft_solution) ** 2).sum()
            loss_val += self.opts.sse_weight * aux['sse_loss']

        if(self.opts.curl_1 > 0):
            aux['curl_1'] = ((jaxutils.dx(gy) + jaxutils.dy(gx)) ** 2).sum()
            loss_val += self.opts.curl_1 * aux['curl_1']
        # else:
        #     aux['curl_1'] = ((jaxutils.dx(sg(gy)) + jaxutils.dy(sg(gx))) ** 2).sum()


        if(self.opts.div_1 > 0):
            aux['div_1'] = ((jaxutils.dx(gx) + jaxutils.dy(gy)) ** 2).sum()
            loss_val += self.opts.div_1 * aux['div_1']
        # else:
        #     aux['div_1'] = ((jaxutils.dx(sg(gx)) + jaxutils.dy(sg(gy))) ** 2).sum()

        if(self.opts.div_2 > 0):
            aux['div_2'] = ((jaxutils.dxx(gx) + jaxutils.dyx(gy)) ** 2 + (jaxutils.dxy(gx) + jaxutils.dyy(gy)) ** 2).sum()
            loss_val += self.opts.div_2 * aux['div_2']
            
        if(self.opts.curl_2 > 0):
            aux['curl_2'] = ((jaxutils.dyx(gx) + jaxutils.dxx(gy)) ** 2 + (jaxutils.dyy(gx) + jaxutils.dxy(gy)) ** 2).sum()
            loss_val += self.opts.curl_2 * aux['curl_2']

        return loss_val, aux


    def fft(self,inpt,inim=None):
        b,h,w,c = inpt['noisy'].shape
        aux = {}
        if(self.alpha_type == 'map_2d'):
            alpha = self.alpha(inpt['net_input']).transpose(0,3,1,2).reshape(-1,h,w)
        elif(self.alpha_type == 'scalar'):
            alpha = self.alpha
        psp = partial(linalg.screen_poisson,alpha)
        if(self.fft_model == 'fft_highdim'):
            g = self.quad_model(inpt['net_input'])#b,h,w,3 <- b,h,w,12
            lowdim_inpt = jnp.concatenate((inpt['noisy'],inpt['net_input'][...,:6]),axis=-1)
            noisy_high = self.quad_model.low2highdim(lowdim_inpt).transpose(0,3,1,2).reshape(-1,h,w)#b,h,w,64 <- b,h,w,9
            phi = g[...,::2]
            a = g[...,1::2]

            phix = utils.dx(phi)
            phiy = utils.dy(phi)
            ax = utils.dx(a)
            ay = utils.dy(a)
            if(self.fixed_delta):
                dx = self.delta_psi_init * phix - self.delta_phi_init * ay
                dy = self.delta_psi_init * phiy + self.delta_phi_init * ax
            else:
                dx = nn.softplus(self.delta_psi_init) * phix - nn.softplus(self.delta_phi_init) * ay
                dy = nn.softplus(self.delta_psi_init) * phiy + nn.softplus(self.delta_phi_init) * ax
            dx = dx.transpose(0,3,1,2).reshape(-1,h,w)
            dy = dy.transpose(0,3,1,2).reshape(-1,h,w)

            func = map(psp,noisy_high,dx,dy)#b,h,w,64 <- b,h,w,64, b,h,w,64, b,h,w,64
            i_denoise_highdim = jnp.stack(list(func)).reshape(b,64,h,w).transpose(0,2,3,1)
            i_denoise = self.quad_model.high2lowdim(i_denoise_highdim)
            aux['gx'] = dx.reshape(b,phix.shape[-1],h,w).transpose(0,2,3,1)
            aux['gy'] = dy.reshape(b,phix.shape[-1],h,w).transpose(0,2,3,1)
            aux['phi'] = phi
            aux['psi'] = a
            aux['phix'] = phix
            aux['phiy'] = phiy
            aux['psix'] = ax
            aux['psiy'] = ay
        elif(self.fft_model == 'fft_highdim_nohelmholz'):
            g = self.quad_model(inpt['net_input'])#b,h,w,3 <- b,h,w,12
            lowdim_inpt = jnp.concatenate((inpt['noisy'],inpt['net_input'][...,:6]),axis=-1)
            noisy_high = self.quad_model.low2highdim(lowdim_inpt).transpose(0,3,1,2).reshape(-1,h,w)#b,h,w,64 <- b,h,w,9
            dx = g[...,::2]
            dy = g[...,1::2]

            dx = dx.transpose(0,3,1,2).reshape(-1,h,w)
            dy = dy.transpose(0,3,1,2).reshape(-1,h,w)

            func = map(psp,noisy_high,dx,dy)#b,h,w,64 <- b,h,w,64, b,h,w,64, b,h,w,64
            i_denoise_highdim = jnp.stack(list(func)).reshape(b,64,h,w).transpose(0,2,3,1)
            i_denoise = self.quad_model.high2lowdim(i_denoise_highdim)
            aux['gx'] = g[...,::2]
            aux['gy'] = g[...,1::2]
        elif(self.fft_model == 'fft_filters'):
            assert self.opts.out_features == self.opts.kernel_count * self.opts.kernel_channels
            g, kernel = self.quad_model(inpt['net_input'])
        else:
            if(inim is None):
                g = self.quad_model(inpt['net_input'])
            else:
                g = inim
        # lambda_d = 0.00000001
        img = inpt['noisy'].transpose(0,3,1,2).reshape(-1,h,w)
        if(self.fft_model == 'fft_filters'):
            i_denoise = linalg.screened_poisson_multi_kernel(self.alpha,inpt['noisy'],g,kernel)
        elif(self.fft_model == 'fft_helmholz'):
            phi = g[...,:3]
            a = g[...,3:]
            phix = utils.dx(phi)
            phiy = utils.dy(phi)
            ax = utils.dx(a)
            ay = utils.dy(a)
            if(self.fixed_delta):
                dx = self.delta_psi_init * phix - self.delta_phi_init * ay
                dy = self.delta_psi_init * phiy + self.delta_phi_init * ax
            else:
                dx = nn.softplus(self.delta_psi_init) * phix - nn.softplus(self.delta_phi_init) * ay
                dy = nn.softplus(self.delta_psi_init) * phiy + nn.softplus(self.delta_phi_init) * ax
            dx = dx.transpose(0,3,1,2).reshape(-1,h,w)
            dy = dy.transpose(0,3,1,2).reshape(-1,h,w)

        elif(self.fft_model == 'fft_image_grad'):
            dx = utils.dx(g).transpose(0,3,1,2).reshape(-1,h,w)
            dy = utils.dy(g).transpose(0,3,1,2).reshape(-1,h,w)
        elif(self.fft_model == 'fft' or self.fft_model == 'fft_alphamap'):
            dx = g[...,:3].transpose(0,3,1,2).reshape(-1,h,w)
            dy = g[...,3:].transpose(0,3,1,2).reshape(-1,h,w)
        if(self.fft_model != 'fft_filters'):
            func = map(psp,img,dx,dy)
            i_denoise = jnp.stack(list(func)).reshape(b,c,h,w).transpose(0,2,3,1)
        # dx = g[...,:3].transpose(0,3,1,2).reshape(-1,h,w)
        # dy = g[...,3:].transpose(0,3,1,2).reshape(-1,h,w)
        return i_denoise, g, aux
        # noisy = cvgim.imread('/home/mohammad/Projects/optimizer/DifferentiableSolver/logger/fft_solver/noisy.png')
        # gt = cvgim.imread('/home/mohammad/Projects/optimizer/DifferentiableSolver/logger/fft_solver/gt.png')

        # grad_x = g[0,...,:3]
        # grad_y = g[0,...,3:]

        # sp = map(linalg.screen_poisson,[lambda_d,lambda_d,lambda_d], noisy.transpose(2,0,1),grad_x.transpose(2,0,1),grad_y.transpose(2,0,1))
        # recon = jnp.stack(list(sp),axis=-1)[None,...]
        # dx = jnp.roll(recon, 1, axis=[1]) - recon
        # dy = jnp.roll(recon, 1, axis=[0]) - recon
        # return recon , [gt[None,...], grad_x,dx, grad_y,dy]
        

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
        out = OrderedDict()
        out['pred'] = self.quad_model.visualize(predict[0],inpt)
        return out

    def labels(self):
        return self.quad_model.labels()

    def termLabels(self):
        return self.quad_model.termLabels()
        
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
        r_terms = lambda pp_image: self.quad_model.terms(pp_image,inpt)
        #Quadratic objective
        r2sum = lambda pp_image: (r(pp_image) ** 2).sum()
        r2sum_terms = lambda pp_image: [(i ** 2).sum() for i in r_terms(pp_image)]
        
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
        def linear_solver_id(x):
            d = linear_solve.solve_cg(matvec=Ax,
                                    b=-jtf(x),
                                    maxiter=self.opts.nlin_iter)#,tol=1e-25)
            aux = (Axb(x,d) ** 2).sum()
            return d, aux

        def loop_body(args):
            x,xs,count, gn_opt_err, gn_loss,gn_loss_terms,linear_opt_err = args
            d, linea_opt = linear_solver_id(x)
            # assert linea_opt < 1e-8, 'linear system is not converging, optimality error: ' + str(linea_opt)
            x += 1.0 * d
            xs = xs.at[count[0].astype(int)+1,...].set(jax.lax.stop_gradient(x))
            linear_opt_err = linear_opt_err.at[count.astype(int)].set(linea_opt)
            gn_opt_err = gn_opt_err.at[count.astype(int)].set(optim_cond(x))
            gn_loss = gn_loss.at[count.astype(int)].set(r2sum(x))
            for i,ri in enumerate(r2sum_terms(x)):
                gn_loss_terms = gn_loss_terms.at[count.astype(int),i].set(ri)
            count += 1
            return (x,xs,count, gn_opt_err, gn_loss,gn_loss_terms,linear_opt_err)
        loop_count = self.opts.nnonlin_iter
        xs = jnp.ones((loop_count+1,*x.shape))
        xs = xs.at[0,...].set(jax.lax.stop_gradient(x))
        val = (x,xs,jnp.array([0.0]),-jnp.ones(loop_count),-jnp.ones(loop_count),-jnp.ones((loop_count,self.quad_model.nterms())),-jnp.ones(loop_count))
        for i in range(loop_count):
            val = loop_body(val)
        x,xs,count, gn_opt_err, gn_loss,gn_loss_terms,linear_opt_err = val
        # x,count, gn_opt_err, gn_loss,linear_opt_err = jax.lax.fori_loop(0,loop_count,lambda i,val:loop_body(val),(x,0.0,-jnp.ones(loop_count),-jnp.ones(loop_count),-jnp.ones(loop_count))) 
        return x,{'xs':xs,'count':count,'gn_opt_err':gn_opt_err, 'gn_loss':gn_loss,'gn_loss_terms':gn_loss_terms,'linear_opt_err':linear_opt_err}
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

    def nterms(self):
        return 3
    def termLabels(self):
        return 'dataTerm', 'termX', 'termY'

    def __call__(self,primal_param,inpt):
        avg_weight = (1. / 2.) ** 0.5 *  (1. / primal_param.reshape(-1).shape[0] ** 0.5)
        tfm = lambda x : tfu.camera_to_rgb_jax(x/inpt['alpha'],inpt)

        r1 =  tfm(primal_param) - tfm(inpt['noisy'])
        g = self.unet(inpt['net_input'])
        r2 = tfm(utils.dx(primal_param)) - tfm(g[...,:3])
        r3 = tfm(utils.dy(primal_param)) - tfm(g[...,3:])
        alpha = self.alpha(inpt['net_input']).reshape(-1)      
        out = jnp.concatenate(( r1.reshape(-1), alpha * r2.reshape(-1), alpha * r3.reshape(-1)),axis=0)
        return out * avg_weight,{}

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
    def init_hyper(key,val,dtype,min=0,max=1):
        if(val is None):
            rand = random.uniform(key,minval=min,maxval=max)
        else:
            rand = val
        return jnp.array(rand)

    def terms(self,primal_param,inpt):
        avg_weight = (1. / 2.) ** 0.5 *  (1. / primal_param.reshape(-1).shape[0] ** 0.5)
        tfm = lambda x : tfu.camera_to_rgb_batch(x/inpt['alpha'],inpt)
        r1 =  tfm(primal_param) - tfm(inpt['noisy'])
        g = self.unet(inpt['net_input'])
        r2 = self.alpha * (tfm(primal_param) - tfm(g))
        return r1.reshape(-1) * avg_weight, r2.reshape(-1) * avg_weight
    def termLabels(self):
        return 'dataTerm', 'smoothnessTerm'
        
    def nterms(self):
        return 2
        
    @nn.compact
    def __call__(self,primal_param,inpt):
        r1, r2 = self.terms(primal_param,inpt)
        out = jnp.concatenate((r1, r2),axis=0)
        return out,{}
    
    def visualize(self,primal_param,inpt):
        tfm = lambda x : tfu.camera_to_rgb_batch(x/inpt['alpha'],inpt)
        r1 =  tfm(primal_param) - tfm(inpt['noisy'])
        g = self.unet(inpt['net_input'])
        r2 = self.alpha * (tfm(primal_param) - tfm(g))
        imgs = [r1, g, r2]
        return imgs
    def labels(self):
        return [r'$Fidelity~res~(I-I_{noisy})$', r'$Unet~output$', r'$Regularizer~res~(\lambda(\partial_x I - g))$']

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

    def nterms(self):
        return 3
    def terms(self,primal_param,inpt):
        avg_weight = (1. / 2.) ** 0.5 *  (1. / primal_param.reshape(-1).shape[0] ** 0.5)
        tfm = lambda x : tfu.camera_to_rgb_batch(x/inpt['alpha'],inpt)
        r1 =  tfm(primal_param) - tfm(inpt['noisy'])
        g = self.unet(inpt['net_input'])
        r2 = self.alpha * (tfm(utils.dx(primal_param)) - tfm(g[...,:3]))
        r3 = self.alpha * (tfm(utils.dy(primal_param)) - tfm(g[...,3:]))
        return r1.reshape(-1)*avg_weight, r2.reshape(-1)*avg_weight, r3.reshape(-1)*avg_weight
    def termLabels(self):
        return 'dataTerm', 'smoothnessTermX', 'smoothnessTermY'
    @nn.compact
    def __call__(self,primal_param,inpt):
        r1, r2, r3 = self.terms(primal_param,inpt)
        out = jnp.concatenate((r1, r2, r3),axis=0)
        return out,{}
    
    def visualize(self,primal_param,inpt):
        tfm = lambda x : tfu.camera_to_rgb_batch(x/inpt['alpha'],inpt)
        r1 =  primal_param - inpt['noisy']
        g = self.unet(inpt['net_input'])
        dx = utils.dx(primal_param)
        dy = utils.dy(primal_param)
        gx = g[...,:3]
        gy = g[...,3:]
        r2 = self.alpha * (dx - gx)
        r3 = self.alpha * (dy - gy)
        imgs = [tfm(r1), tfm(gx), tfm(dx), tfm(gy), tfm(dy), tfm(r2), tfm(r3)]
        return imgs
    
    def labels(self):
        return [r'$Fidelity~res~(I-I_{noisy})$', r'$Unet~output~(g[1:3])$', r'$\nabla_x I$', r'$Unet~output~(g[3:6])$', r'$\nabla_y I$', r'$x~regularizer~res~(\lambda(\partial_x I - g[1:3]))$', r'$y~regularizer~res~(\lambda(\partial_y I - g[3:6]))$']


class screen_poisson(Quad_model):
    @staticmethod
    def init_primal(batch):
        return batch['noisy']

    @staticmethod
    def init_hyper(key,val,dtype):
        rand = random.uniform(key)
        return jnp.array(rand)

    def terms(self,primal_param,inpt):
        alpha = self.param('alpha',
                    screen_poisson.init_hyper,
                    None,
                    jnp.array)

        avg_weight = (1. / 2.) ** 0.5 *  (1. / primal_param.reshape(-1).shape[0] ** 0.5)
        tfm = lambda x : tfu.camera_to_rgb_batch(x/inpt['alpha'],inpt)
        r1 =  (tfm(primal_param) - tfm(inpt['noisy']))
        r2 = nn.softplus(alpha) * tfm(utils.dx(primal_param))
        r3 = nn.softplus(alpha) * tfm(utils.dy(primal_param))
        return r1.reshape(-1) * avg_weight, r2.reshape(-1) * avg_weight, r3.reshape(-1) * avg_weight
    def nterms(self):
        return 3

    def termLabels(self):
        return 'dataTerm', 'termX', 'termY'

    @nn.compact
    def __call__(self,primal_param,inpt):
        r1, r2, r3 = self.terms(primal_param,inpt)
        out = jnp.concatenate((r1, r2, r3),axis=0)
        return out,{}