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

def parse_arguments(parser):
    parser.add_argument('--model', type=str, default='overfit_unet',
    choices=['overfit_straight','interpolate_straight','overfit_unet','interpolate_unet','UNet_Hardcode'],help='Which model to use')
    parser.add_argument('--lr', default=1e-4, type=float,help='Maximum rotation')
    parser.add_argument('--display_freq', default=1000, type=int,help='Display frequency by iteration count')
    parser.add_argument('--val_freq', default=101, type=int,help='Display frequency by iteration count')
    parser.add_argument('--save_param_freq', default=100,type=int, help='Maximum rotation')
    parser.add_argument('--max_iter', default=1000000000, type=int,help='Maximum iteration count')
    parser.add_argument('--unet_depth', default=4, type=int,help='Depth of neural net')
    return parser


parser = argparse.ArgumentParser()
parser = parse_arguments(parser)
parser = Viz.logger.parse_arguments(parser)
parser = Dataset.parse_arguments(parser)
opts = parser.parse_args()

tf.config.set_visible_devices([], device_type='GPU')
dataset = Dataset(opts)
logger = Viz.logger(opts,opts.__dict__)

# cvgutil.savePickle('./params.pickle',opts)
# opts = cvgutil.loadPickle('./params.pickle')
# exit(0)



im = dataset.next_batch(False)['net_input']
n_channels = 12
n_classes = 3
bilinear = True

def init_model(rng, x,test=False,group_norm=True):
    return UNet(n_channels,n_classes,bilinear,test,group_norm).init(rng, x)
def model_test(params, im,group_norm=True):
    return UNet(n_channels,n_classes,bilinear,True,group_norm).apply(params, im)
def model_train(params, im,group_norm=True):
    if(group_norm):
        return UNet(n_channels,n_classes,bilinear,False,group_norm).apply(params, im,mutable=['batch_stats'])
    else:
        return UNet(n_channels,n_classes,bilinear,False,group_norm).apply(params, im)


rng = jax.random.PRNGKey(1)
rng, init_rng = jax.random.split(rng)
params = init_model(init_rng, jnp.array(im))

@jax.jit
def loss(params, batch,gt,test=False):
    pred, bn_state = model_train(params, batch)
    return ((gt - pred) ** 2).sum(), bn_state

@jax.jit
def update(params_p,state_p,batch_p,gt):    
    params_p, state_p = solver.update(params_p, state_p,batch=batch_p,gt=gt)
    return params_p, state_p
def camera_to_rgb(im,batch):
    return tfu.camera_to_rgb_jax(
      im, batch['color_matrix'], batch['adapt_matrix'])

data = logger.load_params()
start_idx=0
if(data is not None):
    # state = data['state']
    batch = data['state']
    params = data['params']
    start_idx = data['idx']

solver = OptaxSolver(fun=loss, opt=optax.adam(opts.lr),has_aux=True)
state = solver.init_state(params)

with tqdm.trange(start_idx,opts.max_iter) as t:
    for i in t:
        val_iter = i % opts.val_freq == 0
        mode = 'val' if val_iter else 'train'
        
        batch = dataset.next_batch(val_iter)

        params, state = update(params,state,batch['net_input'],batch['ambient'])
        l,_ = loss(params,batch['net_input'],batch['ambient'])
        t.set_description('loss '+str(np.array(l)))

        if(i % opts.display_freq == 0 or val_iter):
            predicted = model_test(params, batch['net_input'])
            g = camera_to_rgb(predicted, batch)
            ambient = camera_to_rgb(batch['ambient'], batch)
            flash = camera_to_rgb(batch['flash'], batch)
            noisy = camera_to_rgb(batch['noisy']/batch['alpha'], batch)
            psnr = linalg.get_psnr_jax(jax.lax.stop_gradient(g),ambient)
            imshow = jnp.clip(jnp.concatenate((ambient,g,noisy,flash),axis=-2),0,1)
            logger.addImage(imshow[0],'image',mode=mode)
            logger.addScalar(psnr,'psnr',mode=mode)
        if(i % opts.save_param_freq == 0):
            logger.save_params(params,batch,i)

        logger.addScalar(l,'loss',mode=mode)
        logger.takeStep()
