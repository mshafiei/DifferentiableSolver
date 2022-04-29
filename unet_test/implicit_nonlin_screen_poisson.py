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
def parse_arguments(parser):
    parser.add_argument('--model', type=str, default='implicit_sanity_model',
    choices=['implicit_sanity_model','implicit_poisson_model','unet','fft','fft_alphamap'],help='Which model to use')
    parser.add_argument('--nn_model', type=str, default='unet', choices=['linear','unet'],help='Which model to use')
    parser.add_argument('--lr', default=1e-4, type=float,help='Maximum rotation')
    parser.add_argument('--display_freq', default=1000, type=int,help='Display frequency by iteration count')
    parser.add_argument('--val_freq', default=101, type=int,help='Display frequency by iteration count')
    parser.add_argument('--save_param_freq', default=100,type=int, help='Maximum rotation')
    parser.add_argument('--max_iter', default=100000000, type=int,help='Maximum iteration count')
    parser.add_argument('--unet_depth', default=4, type=int,help='Depth of neural net')
    parser.add_argument('--mode', default='train', type=str,choices=['train','test'],help='Should we train or test the model?')
    parser.add_argument('--debug', default='none', type=str,help='What should we debug')
    parser.add_argument('--alpha_thickness', default=4, type=int,help='Thickness of layers in alpha map')
    parser.add_argument('--sse_weight', default=1., type=float,help='Weight of the Sum Squared Error loss')
    parser.add_argument('--curl_1', default=0., type=float,help='First order curl regularizer')
    parser.add_argument('--curl_2', default=0., type=float,help='Second order curl regularizer')
    parser.add_argument('--div_1', default=0., type=float,help='First order Div. regularizer')
    parser.add_argument('--div_2', default=0., type=float,help='Second order Div. regularizer')
    
    return parser


parser = argparse.ArgumentParser()
parser = parse_arguments(parser)
parser = Viz.logger.parse_arguments(parser)
parser = Dataset.parse_arguments(parser)
parser = diff_solver.parse_arguments(parser)
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
    nn_model = UNet(opts.in_features,opts.out_features,opts.bilinear,opts.mode == 'test',opts.group_norm,opts.num_groups,opts.thickness,'softplus')
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
elif(opts.model == 'fft'):
    diffable_solver = fft_solver(opts=opts, quad_model=nn_model,alpha_type='scalar',alpha_map=None)
elif(opts.model == 'fft_alphamap'):
    alpha_model = UNet(opts.in_features,1,opts.bilinear,opts.mode == 'test',opts.group_norm,2,opts.alpha_thickness,'softplus')
    diffable_solver = fft_solver(opts=opts, quad_model=nn_model,alpha_type='map_2d',alpha_map=alpha_model)
elif(opts.model == 'dummy'):
    diffable_solver = diff_solver(opts=opts, quad_model=nn.Module())
else:
    print('Cannot recognize model')
    exit(0)

rng = jax.random.PRNGKey(2)
rng, init_rng = jax.random.split(rng)


visualize_model = jax.jit(lambda params,batch :diffable_solver.apply(params, batch, method=diffable_solver.visualize))
apply = jax.jit(lambda params,batch :diffable_solver.apply(params,batch))
# visualize_model = (lambda params,batch :diffable_solver.apply(params, batch, method=diffable_solver.visualize))
# apply = (lambda params,batch :diffable_solver.apply(params,batch))
ssim_fn = jax.jit(jaxutils.ssim)
# ssim_fn = jax.jit(functools.partial(jaxutils.compute_ssim, max_val=1.))

params = diffable_solver.init(rng,batch)
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

data = logger.load_params()
solver = OptaxSolver(fun=apply, opt=optax.adam(opts.lr),has_aux=True)
state = solver.init_state(params)
start_idx=0
if(data is not None):
    # state = data['state']
    batch = data['state']
    params = data['params']
    start_idx = data['idx']
    print('Parameters loaded successfully')


def eval_visualize(params,batch,logger,mode,display,save_params,ignorelist='',t=None):
    _,aux = apply(params,batch)

    pred = tfu.camera_to_rgb_batch(aux['pred']/batch['alpha'],batch)
    noisy = tfu.camera_to_rgb_batch(batch['noisy']/batch['alpha'],batch)
    ambient = tfu.camera_to_rgb_batch(batch['ambient'],batch)
    mtrcs = metrics(pred,ambient,ignorelist)
    mtrcs_noisy = metrics(noisy,ambient,ignorelist)
    mtrcs_str = ''.join([' %s:%.5f' % (k,v) for k,v in mtrcs.items()])
    if(t is not None):
        t.set_description(mtrcs_str)
    if(display):
        viz_imgs = visualize_model(params,batch)
        labels = diffable_solver.labels()
        flash = tfu.camera_to_rgb_batch(batch['flash'],batch)
        imgs = [pred,ambient,noisy,flash,*viz_imgs]
        labels = [r'$Prediction~(I),~PSNR:~%.3f,~MSE:~%.5f$'%(mtrcs['psnr'],
                    mtrcs['mse']),
                    r'$Ground~Truth~(I_{ambient})$',
                    r'$Noisy~input~(I_{noisy}),~PSNR: %.3f,~MSE:~%.5f$'%(mtrcs_noisy['psnr'],
                    mtrcs_noisy['mse']),
                    r'$Flash~input~(I_{flash})$',
                    *labels]
        logger.addImage(imgs,labels,'image',dim_type='BHWC',mode=mode)

    if(save_params):
        logger.save_params(params,batch,i)
    mtrcs = {k:v for k,v in mtrcs.items()}
    if('div_1'in aux.keys()):
        mtrcs.update({'div_1':aux['div_1']})
    if('curl_1'in aux.keys()):
        mtrcs.update({'curl_1':aux['curl_1']})

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
eval_visualize(params,batch,logger,'val',True,False,ignorelist='ssim')
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
                batch,_ = dataset.next_batch(True,i)
                _,aux = apply(params,batch)
                eval_visualize(params,batch,logger,'val',True,False)

            batch,_ = dataset.next_batch(False,i)
            params, state = update(params,state,batch)
            eval_visualize(params,batch,logger,'train',train_display,save_params,ignorelist='ssim',t=t)
            logger.takeStep()
elif(opts.mode == 'test'):
    end_time = time.time()
    print('compile time ',end_time - start_time)
    errors = {}
    for k in range(4):
        mtrcs = []
        for c in tqdm.trange(128):
            try:
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
                mt = eval_visualize(params,batch,logger,'test', c % 10 == 0 ,False)
                logger.takeStep()
                mtrcs.append(mt)
            except:
                pass
        
        psnr = np.mean([i['psnr'] for i in mtrcs])
        mse = np.mean([i['mse'] for i in mtrcs])
        ssim = np.mean([i['ssim'] for i in mtrcs])

        errors['Level %d' % (4 - k)] = 'PSNR: %.3f, MSE: %.4f,SSIM: %.4f' % (psnr,mse,ssim)
        print(errors['Level %d' % (4 - k)])
    logger.addDict(errors,'test_errors',opts.mode)
else:
    print('Unknown mode ',opts.mode)
    exit(0)