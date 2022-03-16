from cvgutils.nn.jaxUtils.unet_model import UNet
import jax.numpy as jnp
import jax
import optax
from jaxopt import OptaxSolver
import tensorflow as tf
import tqdm
import numpy as np
from deepfnf_utils.dataset import Dataset
import cvgutils.Utils as cvgutil
import deepfnf_utils.tf_utils as tfu
import cvgutils.Viz as Viz
import cvgutils.Linalg as linalg
import argparse
from jaxopt import implicit_diff, linear_solve
from implicit_diff_module import diff_solver, fnf_regularizer, implicit_sanity_model, implicit_poisson_model
from flax import linen as nn
import deepfnf_utils.utils as ut
import time

def parse_arguments(parser):
    parser.add_argument('--model', type=str, default='implicit_sanity_model',
    choices=['implicit_sanity_model','implicit_poisson_model'],help='Which model to use')
    parser.add_argument('--lr', default=1e-4, type=float,help='Maximum rotation')
    parser.add_argument('--display_freq', default=1000, type=int,help='Display frequency by iteration count')
    parser.add_argument('--val_freq', default=101, type=int,help='Display frequency by iteration count')
    parser.add_argument('--save_param_freq', default=100,type=int, help='Maximum rotation')
    parser.add_argument('--max_iter', default=100000000, type=int,help='Maximum iteration count')
    parser.add_argument('--unet_depth', default=4, type=int,help='Depth of neural net')
    parser.add_argument('--mode', default='train', type=str,choices=['train','test'],help='Should we train or test the model?')
    
    return parser


parser = argparse.ArgumentParser()
parser = parse_arguments(parser)
parser = Viz.logger.parse_arguments(parser)
parser = Dataset.parse_arguments(parser)
parser = diff_solver.parse_arguments(parser)
parser = UNet.parse_arguments(parser)
opts = parser.parse_args()


# opts = cvgutil.loadPickle('./params.pickle')
# cvgutil.savePickle('./params.pickle',opts)
# exit(0)
tf.config.set_visible_devices([], device_type='GPU')
dataset = Dataset(opts)
logger = Viz.logger(opts,opts.__dict__)

batch = dataset.next_batch(False,0)
im = batch['net_input']
if(opts.model == 'implicit_sanity_model'):
    diffable_solver = diff_solver(opts=opts, quad_model=implicit_sanity_model(UNet(opts.in_features,opts.out_features,opts.bilinear,opts.test,opts.group_norm,'softplus')))
elif(opts.model == 'implicit_poisson_model'):
    diffable_solver = diff_solver(opts=opts, quad_model=implicit_poisson_model(UNet(opts.in_features,opts.out_features,opts.bilinear,opts.test,opts.group_norm,'softplus')))
else:
    print('Cannot recognize model')
    exit(0)

rng = jax.random.PRNGKey(2)
rng, init_rng = jax.random.split(rng)
params = diffable_solver.init(rng,batch)
pred, aux = diffable_solver.apply(params,batch)


visualize_model = jax.jit(lambda params,batch :diffable_solver.apply(params, batch, method=diffable_solver.visualize))
apply = jax.jit(lambda params,batch :diffable_solver.apply(params,batch))

@jax.jit
def loss(params,batch):
    pred, aux = apply(params,batch)
    pred = jnp.clip(tfu.camera_to_rgb_batch(pred/batch['alpha'],batch),0,1)
    ambient = jnp.clip(tfu.camera_to_rgb_batch(batch['ambient'],batch),0,1)

    return ((ambient - pred) ** 2).sum(), aux

@jax.jit
def metrics(pred,gt):
    pred = jnp.clip(pred,0,1)
    gt = jnp.clip(gt,0,1)
    mse = ((gt - pred) ** 2).mean([1,2,3])
    psnr = -10. * jnp.log10(mse) / jnp.log10(10.)
    return {'mse':mse,'psnr':psnr}


@jax.jit
def update(params_p,state_p,batch_p):
    params_p, state_p = solver.update(params_p, state_p,batch=batch_p)
    return params_p, state_p

data = logger.load_params()
solver = OptaxSolver(fun=loss, opt=optax.adam(opts.lr),has_aux=True)
state = solver.init_state(params)
start_idx=0
if(data is not None):
    # state = data['state']
    batch = data['state']
    params = data['params']
    start_idx = data['idx']


def eval_visualize(params,batch,logger,mode,display,save_params,t=None):
    pred,_ = apply(params,batch)
    pred = tfu.camera_to_rgb_batch(pred/batch['alpha'],batch)
    noisy = tfu.camera_to_rgb_batch(batch['noisy']/batch['alpha'],batch)
    ambient = tfu.camera_to_rgb_batch(batch['ambient'],batch)
    mtrcs = metrics(pred,ambient)
    mtrcs_noisy = metrics(noisy,ambient)
    mtrcs_str = ''.join([' %s:%.5f' % (k,v[0]) for k,v in mtrcs.items()])
    if(t is not None):
        t.set_description(mtrcs_str)
    if(display):
        imgs = visualize_model(params,batch)
        labels = diffable_solver.quad_model.labels()
        imgs = [tfu.camera_to_rgb_batch(i/batch['alpha'],batch) for i in imgs]
        imgs = [pred,ambient,noisy,tfu.camera_to_rgb_batch(batch['flash'],batch),*imgs]
        labels = [r'$Prediction~(I),~PSNR:~%.3f,~MSE:~%.5f$'%(mtrcs['psnr'][0],mtrcs['mse'][0]),r'$Ground~Truth~(I_{ambient})$',r'$Noisy~input~(I_{noisy}),~PSNR: %.3f,~MSE:~%.5f$'%(mtrcs_noisy['psnr'][0],mtrcs_noisy['mse'][0]),r'$Flash~input~(I_{flash})$',*labels]
        logger.addImage(imgs,labels,'image',dim_type='BHWC',mode=mode)

    if(save_params):
        logger.save_params(params,batch,i)

    logger.addMetrics(mtrcs,mode=mode)
    

start_time = time.time()
pred,_ = apply(params,batch)
metrics(pred/batch['alpha'],batch['ambient'])
visualize_model(params,batch)
eval_visualize(params,batch,logger,'val',True,False)
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
                batch = dataset.next_batch(True,i)
                eval_visualize(params,batch,logger,'val',True,False)

            batch = dataset.next_batch(False,i)
            params, state = update(params,state,batch)
            eval_visualize(params,batch,logger,'train',train_display,save_params,t)
            logger.takeStep()
elif(opts.mode == 'test'):
    end_time = time.time()
    print('compile time ',end_time - start_time)
    for k in range(4):
        mtrcs = []
        for c in range(128):
            data = np.load('%s/%d/%d.npz' % (opts.TESTPATH, k, c))

            alpha = data['alpha'][None, None, None, None]
            ambient = data['ambient']
            dimmed_ambient, _ = tfu.dim_image_jax(data['ambient'], alpha=alpha)
            dimmed_warped_ambient, _ = tfu.dim_image_jax(
                data['warped_ambient'], alpha=alpha)

            # Make the flash brighter by increasing the brightness of the
            # flash-only image.
            flash = data['flash_only'] * ut.FLASH_STRENGTH + dimmed_ambient
            warped_flash = data['warped_flash_only'] * \
                ut.FLASH_STRENGTH + dimmed_warped_ambient

            noisy_ambient = data['noisy_ambient']
            noisy_flash = data['noisy_warped_flash']

            noisy = tf.concat([noisy_ambient, noisy_flash], axis=-1)
            noise_std = tfu.estimate_std(
                noisy, data['sig_read'], data['sig_shot'])
            net_input = tf.concat([noisy, noise_std], axis=-1)

            batch = {'net_input':net_input,'noisy':noisy,'noise_std':noise_std,'flash':flash}
            denoise = apply(params,batch)
            eval_visualize(params,batch,logger,'test',True,False)
            logger.takeStep()

            mtrcs.append(metrics(params,batch))
        
        psnr = np.mean([i['psnr'] for i in mtrcs])
        mse = np.mean([i['mse'] for i in mtrcs])

        print('\nLevel %d' % (4 - k) +
            ': PSNR: %.3f, SSIM: %.4f' % (psnr,mse))
else:
    print('Unknown mode ',opts.mode)
    exit(0)