from cvgutils.nn.jaxUtils.unet_model import UNet, DummyConv
import jax.numpy as jnp
import jax
import optax
from jaxopt import OptaxSolver
import tensorflow as tf
import tqdm
import numpy as np
from deepfnf_utils.dataset import Dataset
import cvgutils.Utils as cvgutil
import cvgutils.Image as cvgim
import deepfnf_utils.tf_utils as tfu
import cvgutils.Viz as Viz
import cvgutils.Linalg as linalg
import argparse
from jaxopt import implicit_diff, linear_solve
from implicit_diff_module import diff_solver, fnf_regularizer, implicit_sanity_model, implicit_poisson_model, direct_model, fft_solver
from flax import linen as nn
import deepfnf_utils.utils as ut
import time
import os
import cvgutils.nn.jaxUtils.utils as jaxutils
import functools
import deepfnf_utils.np_utils as np_utils
from collections import OrderedDict
from clu import parameter_overview
def parse_arguments(parser):
    parser.add_argument('--model', type=str, default='implicit_sanity_model',
    choices=['implicit_sanity_model','fft_highdim_nohelmholz','implicit_poisson_model','unet','fft','fft_alphamap','fft_image_grad','fft_helmholz','fft_filters','fft_highdim'],help='Which model to use')
    parser.add_argument('--nn_model', type=str, default='unet', choices=['linear','unet'],help='Which model to use')
    parser.add_argument('--lr', default=1e-4, type=float,help='Maximum rotation')
    parser.add_argument('--display_freq', default=50000, type=int,help='Display frequency by iteration count')
    parser.add_argument('--display_freq_test', default=10, type=int,help='Display frequency by iteration count')
    parser.add_argument('--val_freq', default=10001, type=int,help='Display frequency by iteration count')
    parser.add_argument('--save_param_freq', default=50000,type=int, help='Maximum rotation')
    parser.add_argument('--max_iter', default=1500000, type=int,help='Maximum iteration count')
    parser.add_argument('--unet_depth', default=4, type=int,help='Depth of neural net')
    parser.add_argument('--mode', default='train', type=str,choices=['train','test'],help='Should we train or test the model?')
    parser.add_argument('--debug', default='none', type=str,help='What should we debug')
    parser.add_argument('--alpha_thickness', default=4, type=int,help='Thickness of layers in alpha map')
    parser.add_argument('--sse_weight', default=1., type=float,help='Weight of the Sum Squared Error loss')
    parser.add_argument('--grad_weight', default=0., type=float,help='Weight of the Sum Squared Error loss')
    parser.add_argument('--curl_1', default=0., type=float,help='First order curl regularizer')
    parser.add_argument('--curl_2', default=0., type=float,help='Second order curl regularizer')
    parser.add_argument('--div_1', default=0., type=float,help='First order Div. regularizer')
    parser.add_argument('--div_2', default=0., type=float,help='Second order Div. regularizer')
    parser.add_argument('--activation', default='relu', type=str,help='Activation function of the neural network')
    
    return parser


parser = argparse.ArgumentParser()
parser = parse_arguments(parser)
parser = Viz.logger.parse_arguments(parser)
parser = Dataset.parse_arguments(parser)
parser = diff_solver.parse_arguments(parser)
parser = fft_solver.parse_arguments(parser)
parser = UNet.parse_arguments(parser)
opts = parser.parse_args()


logger = Viz.logger(opts,opts.__dict__)
opts = logger.opts
tf.config.set_visible_devices([], device_type='GPU')
dataset = Dataset(opts)

batch,_ = dataset.next_batch(False,0)
# dr = '/home/mohammad/Projects/optimizer/DifferentiableSolver/logger/fft_solver'
# noisy = jnp.clip(tfu.camera_to_rgb_batch(batch['noisy']/batch['alpha'],batch),0,1)
# ambient = jnp.clip(tfu.camera_to_rgb_batch(batch['ambient'],batch),0,1)
# cvgim.imwrite(os.path.join(dr,'noisy.png'),noisy[0])
# cvgim.imwrite(os.path.join(dr,'gt.png'),ambient[0])


im = batch['net_input']
if(opts.nn_model == 'unet'):
    nn_model = UNet(opts.in_features,opts.out_features,opts.bilinear,
    opts.mode == 'test',opts.group_norm,opts.num_groups,opts.thickness,
    opts.activation,opts.model,opts.kernel_channels,opts.kernel_count,opts.kernel_size,opts.unet_factor,opts.high_dim,opts.outc_kernel_size)
elif(opts.nn_model == 'linear'):
    nn_model = DummyConv(opts.in_features,opts.out_features)
else:
    print('Unknowns neural network')

if(opts.model == 'implicit_sanity_model'):
    diffable_solver = diff_solver(opts=opts, quad_model=implicit_sanity_model(nn_model))
elif(opts.model == 'implicit_poisson_model'):
    diffable_solver = diff_solver(opts=opts, quad_model=implicit_poisson_model(nn_model))
elif(opts.model == 'unet'):
    diffable_solver = direct_model(opts=opts, quad_model=nn_model)
elif(opts.model == 'fft' or opts.model == 'fft_image_grad' or opts.model == 'fft_helmholz' or opts.model == 'fft_filters' or opts.model == 'fft_highdim' or opts.model == 'fft_highdim_nohelmholz'):
    diffable_solver = fft_solver(opts=opts, quad_model=nn_model,alpha_type='scalar',alpha_map=None,fft_model=opts.model,delta_phi_init=opts.delta_phi_init,delta_psi_init=opts.delta_psi_init,fixed_delta=opts.fixed_delta)
elif(opts.model == 'fft_alphamap'):
    alpha_model = UNet(opts.in_features,1,opts.bilinear,
    opts.mode == 'test',opts.group_norm,2,opts.alpha_thickness,
    'softplus',opts.model,opts.kernel_channels,opts.kernel_count,opts.kernel_size,opts.unet_factor)

    diffable_solver = fft_solver(opts=opts, quad_model=nn_model,alpha_type='map_2d',alpha_map=alpha_model,fft_model=opts.model,delta_phi_init=opts.delta_phi_init,delta_psi_init=opts.delta_psi_init,fixed_delta=opts.fixed_delta)
elif(opts.model == 'dummy'):
    diffable_solver = diff_solver(opts=opts, quad_model=nn.Module())
else:
    print('Cannot recognize model')
    exit(0)

rng = jax.random.PRNGKey(2)
rng, init_rng = jax.random.split(rng)


visualize_model = lambda params,batch :diffable_solver.apply(params, batch, method=diffable_solver.visualize)
apply = jax.jit(lambda params,batch :diffable_solver.apply(params,batch))
# visualize_model = (lambda params,batch :diffable_solver.apply(params, batch, method=diffable_solver.visualize))
# apply = (lambda params,batch :diffable_solver.apply(params,batch))
ssim_fn = jax.jit(jaxutils.ssim)
# ssim_fn = jax.jit(functools.partial(jaxutils.compute_ssim, max_val=1.))

params = diffable_solver.init(rng,batch)
params_overview = parameter_overview.get_parameter_overview(params)
print(params_overview)
flat_tree, _ = jax.tree_flatten(params)
nparams = np.sum([jnp.prod(np.array(i.shape)) for i in flat_tree])
logger.addDict({'nparams':nparams,'architecture':params_overview},'model_details')
pred, aux = apply(params,batch)
# @jax.jit
def metrics(preds,gts,ignorelist=''):
    mtrcs = {}
    for pred,gt in zip(preds,gts):
        pred = jnp.clip(pred,0,1)
        gt = jnp.clip(gt,0,1)
        mtrcs.update({'mse':np_utils.get_mse_jax(pred,gt)})
        mtrcs.update({'psnr':np_utils.get_psnr_jax(pred,gt)})
        if(not('ssim' in ignorelist)):
            mtrcs.update({'ssim':np_utils.get_ssim(np.array(pred),np.array(gt))})
    return mtrcs


@jax.jit
def update(params_p,state_p,batch_p):
    params_p, state_p = solver.update(params_p, state_p,batch=batch_p)
    return params_p, state_p

# DROP = (1.1e6, 1.25e6) # Learning rate drop
def get_lr(niter):
    # cond = lambda x: jax.lax.cond(x >= 1.1e6 and x < 1.25e6,lambda y:lr,lambda y:lr / 10.,x)
    # return cond(niter)
    # return jax.lax.cond(niter < 1.1e6,lambda x:lr,lambda x: #jax.lax.cond(x >= 1.1e6 and x < 1.25e6,lambda x:lr / jnp.sqrt(10.),lr/10.,x),niter)
    return jnp.where(niter < 1.1e6,opts.lr,jnp.where(niter > 1.25e6,opts.lr / 10.,opts.lr / jnp.sqrt(10.)))
    # if niter < 1.1e6:
    #     return lr
    # elif niter >= 1.1e6 and niter < 1.25e6:
    #     return lr / jnp.sqrt(10.)
    # else:
    #     return lr / 10.

data = logger.load_params()
solver = OptaxSolver(fun=apply, opt=optax.adam(get_lr),has_aux=True)
state = solver.init_state(params)
start_idx=0
if(data is not None):
    # state = data['state']
    if(type(state) == type(data['state'])):
        state = data['state']
    params = data['params']
    start_idx = data['idx']
# start_idx = 1000
# print('Catching up with the current batch idx')
# for i in tqdm.trange(start_idx):
#     batch,_ = dataset.next_batch(False,i)

#     print('Parameters loaded successfully')


def eval_visualize(params,batch,logger,mode,display,save_params,state,erreval=None,add_scalars=True,ignorelist='',t=None,method_name=''):
    _,aux = apply(params,batch)

    pred = tfu.camera_to_rgb_batch(aux['pred']/batch['alpha'],batch)
    noisy_dim = tfu.camera_to_rgb_batch(batch['noisy'],batch)
    noisy = tfu.camera_to_rgb_batch(batch['noisy']/batch['alpha'],batch)
    ambient = tfu.camera_to_rgb_batch(batch['ambient'],batch)
    mtrcs = metrics(pred,ambient,ignorelist)
    mtrcs_noisy = metrics(noisy,ambient,ignorelist)
    if(erreval != None):
        piq_metrics_pred = erreval.eval(batch['ambient'],pred,dtype='jax')
        piq_metrics_noisy = erreval.eval(batch['ambient'],noisy,dtype='jax')
        mtrcs.update({'msssim':piq_metrics_pred['msssim'],'lpipsVGG':piq_metrics_pred['lpipsVGG'],'lpipsAlex':piq_metrics_pred['lpipsAlex']})
        mtrcs_noisy.update({'msssim':piq_metrics_noisy['msssim'],'lpipsVGG':piq_metrics_noisy['lpipsVGG'],'lpipsAlex':piq_metrics_noisy['lpipsAlex']})

    mtrcs_str = ''.join([' %s:%.5f, lr:%.7f' % (k,v,get_lr(state[0])) for k,v in mtrcs.items()])
    if(t is not None):
        t.set_description(mtrcs_str)
    if(display):
        viz_imgs = visualize_model(params,batch)
        labels_ret = diffable_solver.labels()
        flash = tfu.camera_to_rgb_batch(batch['flash'],batch)
        imgs,labels = OrderedDict(),OrderedDict()
        imgs['pred'], imgs['ambient'], imgs['noisy'], imgs['noisy_dim'], imgs['flash'] = pred, ambient, noisy, noisy_dim, flash
        if('div' in viz_imgs.keys()):
            div_sum = (aux['div'] ** 2).sum()
            curl_sum = (aux['curl'] ** 2).sum()
            more_info = r' (\frac{1}{n}||\nabla \cdot g||^2 : %.02f, \frac{1}{n}||\nabla \times g||^2 : %.02f' % (div_sum, curl_sum)
        else:
            more_info = ''
        imgs.update(viz_imgs)
        annotation = {}
        if(erreval):
            annotation = {'pred':'%s<br>PSNR:%.3f<br>SSIM:%.3f'%(method_name,mtrcs['psnr'],mtrcs['ssim']),
                        'noisy':'Noisy input<br>PSNR:%.3f<br>SSIM:%.3f'%(mtrcs_noisy['psnr'],mtrcs_noisy['ssim']),
                        'ambient':'Ground truth',
                        'flash':'Flash input'}
        labels['pred'] = r'$Prediction~(I),~PSNR:~%.3f,~MSE:~%.5f$'%(mtrcs['psnr'],mtrcs['mse'])
        labels['ambient'] = r'$Ground~Truth~(I_{ambient})$'
        labels['noisy'] = r'$Noisy~input~(I_{noisy}),~PSNR: %.3f,~MSE:~%.5f,\alpha:~%.2f$'%(mtrcs_noisy['psnr'], mtrcs_noisy['mse'],1./batch['alpha'])
        labels['noisy_dim'] = r'$Noisy~input~dim$'
        labels['flash'] = r'$Flash~input~(I_{flash})$'
        if('dx' in labels_ret.keys()):
            labels_ret['dx'] = labels_ret['dx'][:-1] + more_info + labels_ret['dx'][-1]
                    
        labels.update(labels_ret)
        if('alpha' in params['params'].keys()):
            strlambda = params['params']['alpha']
        else:
            strlambda = 'N/A'
        if('delta' in params['params'].keys()):
            strdelta = nn.softplus(params['params']['delta'])
        else:
            strdelta = 'N/A'
        
        if(mode == 'test'):
            imgs_sl,labels_sl = OrderedDict(), OrderedDict()
            imgs_sl['pred'],labels_sl['pred'] = imgs['pred'],labels['pred']
            imgs_sl['ambient'],labels_sl['ambient'] = imgs['ambient'],labels['ambient']
            imgs_sl['noisy'],labels_sl['noisy'] = imgs['noisy'],labels['noisy']
            imgs_sl['flash'],labels_sl['flash'] = imgs['flash'],labels['flash']
            logger.addIndividualImages(imgs,labels,'image',dim_type='BHWC',mode=mode,text=r'$\lambda=%s, \delta$=%s'%(strlambda,strdelta),annotation=annotation,ltype='filesystem')
            # logger.addImage(imgs,labels,'image',dim_type='BHWC',mode=mode,text=r'$\lambda=%s, \delta$=%s'%(strlambda,strdelta),annotation=annotation)
            # logger.addImage(imgs_sl,labels_sl,'image',dim_type='BHWC',mode=mode,text=r'$\lambda=%s, \delta$=%s'%(strlambda,strdelta),annotation=annotation)
            # logger.addImage(imgs,labels,'image_inset',dim_type='BHWC',mode=mode,text=r'$\lambda=%s, \delta$=%s'%(strlambda,strdelta),addinset=True)
        else:
            logger.addImage(imgs,labels,'image',dim_type='BHWC',mode=mode,text=r'$\lambda=%s, \delta$=%s'%(strlambda,strdelta),ltype='html')
        if(opts.model != 'unet'):
            logger.createTeaser(imgs,labels,'Teaser',dim_type='BHWC',mode=mode)

    if(save_params):
        logger.save_params(params,state,i)
    mtrcs = {k:v for k,v in mtrcs.items()}
    if('div' in aux.keys()):
        div_sum = (aux['div'] ** 2).sum()
        curl_sum = (aux['curl'] ** 2).sum()
        mtrcs.update({'div':div_sum,'curl':curl_sum})
    
    if('div_1'in aux.keys()):
        mtrcs.update({'div_1':aux['div_1']})
    if('curl_1'in aux.keys()):
        mtrcs.update({'curl_1':aux['curl_1']})
    if('alpha' in params['params'].keys()):
        mtrcs.update({'lambda':params['params']['alpha']})
    if('delta' in params['params'].keys()):
        mtrcs.update({'delta':nn.softplus(params['params']['delta'])})
    if(add_scalars):
        logger.addMetrics(mtrcs,mode=mode)
    # termNames = diffable_solver.termLabels()
    # for step in range(opts.nnonlin_iter):
    #     logger.addScalar(aux['gn_loss'][step],'gn_loss_overall/step%i'%step,mode=mode)
    #     for term in range(len(termNames)):
    #         logger.addScalar(aux['gn_loss_terms'][step,term],'gn_loss_%s/step%i'%(termNames[term],step),mode=mode,display_name='gn_loss_%s'%termNames[term])
    return mtrcs
    

start_time = time.time()
_,aux = apply(params,batch)
metrics(np.array(aux['pred'][0]/batch['alpha'][0]),np.array(batch['ambient'][0]))
visualize_model(params,batch)
eval_visualize(params,batch,logger,'val',True,False,state,ignorelist='ssim')
if(opts.mode == 'train'):
    #compile
    update(params,state,batch)
    end_time = time.time()
    print('compile time ',end_time - start_time)
    with tqdm.trange(start_idx, opts.max_iter) as t:
        for i in t:
            #train_display and validation are mutually exclusive
            val_iter = i % opts.val_freq == 0
            train_display = i % opts.display_freq == 0
            save_params = i % opts.save_param_freq == 0
            if(val_iter):
                mtrcs = []
                for c in tqdm.trange(128):
                    batch,_ = dataset.next_batch(True,c)
                    _,aux = apply(params,batch)
                    pred = tfu.camera_to_rgb_batch(aux['pred']/batch['alpha'],batch)
                    noisy = tfu.camera_to_rgb_batch(batch['noisy']/batch['alpha'],batch)
                    ambient = tfu.camera_to_rgb_batch(batch['ambient'],batch)
                    mtrcs.append(metrics(pred,ambient))
                psnr = np.mean([i['psnr'] for i in mtrcs])
                mse = np.mean([i['mse'] for i in mtrcs])
                ssim = np.mean([i['ssim'] for i in mtrcs])
                logger.addMetrics({'psnr':psnr,'mse':mse,'ssim':ssim},mode='val')
                eval_visualize(params,batch,logger,'val',True,False,state,add_scalars=False)
                

            batch,_ = dataset.next_batch(False,i)
            params, state = update(params,state,batch)
            eval_visualize(params,batch,logger,'train',train_display,save_params,state=state,ignorelist='ssim',t=t)
            logger.takeStep()
            
elif(opts.mode == 'test'):
    end_time = time.time()
    print('compile time ',end_time - start_time)
    errors = {}
    errors_dict = {}
    erreval = linalg.ErrEval('cuda:0')
    for k in range(4):
        mtrcs = {}
        for c in tqdm.trange(128):
            data = np.load('%s/%d/%d.npz' % (opts.TESTPATH, k, c))
            keys = [jax.random.PRNGKey(c*10 + i) for i in range(10)]
            if(len(data['alpha'].shape) == 1):
                alpha = data['alpha'][None, None, None, None]
            else:
                alpha = data['alpha']
            ambient = data['ambient']
            dimmed_ambient, _ = tfu.dim_image_jax(data['ambient'],keys[0], alpha=alpha)
            dimmed_warped_ambient, _ = tfu.dim_image_jax(
                data['warped_ambient'], keys[1], alpha=alpha)

            # Make the flash brighter by increasing the brightness of the
            # flash-only image.
            flash = data['flash_only'] * ut.FLASH_STRENGTH + dimmed_ambient
            warped_flash = data['warped_flash_only'] * \
                ut.FLASH_STRENGTH + dimmed_warped_ambient

            noisy_ambient = data['noisy_ambient']
            noisy_flash = data['noisy_warped_flash']

            noisy = jnp.concatenate([noisy_ambient, noisy_flash], axis=-1)
            noise_std = tfu.estimate_std_jax(
                noisy, data['sig_read'], data['sig_shot'])
            net_input = jnp.concatenate([noisy, noise_std], axis=-1)

            batch = {'net_input':net_input,'noisy':noisy_ambient,'ambient':data['ambient'],'flash':noisy_flash,'alpha':data['alpha'],'noise_std':noise_std,'color_matrix':data['color_matrix'],'adapt_matrix':data['adapt_matrix']}
            method_name = ''
            if('fft' in opts.model):
                method_name = 'Ours'
            else:
                method_name = 'U-Net'

            mt = eval_visualize(params,batch,logger,'test', c % opts.display_freq_test == 0 ,False,state,erreval=erreval,method_name=method_name)
            logger.takeStep()
            for key,v in mt.items():
                if(not(key in mtrcs.keys())):
                    mtrcs[key] = []

            [mtrcs[key].append(v) for key,v in mt.items()]

        mean_mtrcs = {key:'%.4f'%np.mean(np.array(v)) for key,v in mtrcs.items()}
        errstr = ['%s: %s' %(key,v) for key,v in mean_mtrcs.items()]
        errors_dict['Level %d' % (4 - k)] = mean_mtrcs
        errors['Level %d' % (4 - k)] = ', '.join(errstr)
        print(errors['Level %d' % (4 - k)])
    logger.dumpDictJson(errors_dict,'test_errors',opts.mode)
    
else:
    print('Unknown mode ',opts.mode)
    exit(0)