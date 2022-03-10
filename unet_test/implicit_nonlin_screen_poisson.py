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
from implicit_diff_module import diff_solver, fnf_regularizer, implicit_sanity_model
from flax import linen as nn

def parse_arguments(parser):
    parser.add_argument('--model', type=str, default='overfit_unet',
    choices=['overfit_straight','interpolate_straight','overfit_unet','interpolate_unet','UNet_Hardcode'],help='Which model to use')
    parser.add_argument('--lr', default=1e-4, type=float,help='Maximum rotation')
    parser.add_argument('--display_freq', default=1000, type=int,help='Display frequency by iteration count')
    parser.add_argument('--val_freq', default=101, type=int,help='Display frequency by iteration count')
    parser.add_argument('--save_param_freq', default=100,type=int, help='Maximum rotation')
    parser.add_argument('--max_iter', default=1000000, type=int,help='Maximum iteration count')
    parser.add_argument('--unet_depth', default=4, type=int,help='Depth of neural net')
    
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

batch = dataset.next_batch(False)
im = batch['net_input']

diffable_solver = diff_solver(opts=opts, quad_model=implicit_sanity_model(UNet(opts.in_features,opts.out_features,opts.bilinear,opts.test,opts.group_norm,'softplus')))
rng = jax.random.PRNGKey(2)
rng, init_rng = jax.random.split(rng)
params = diffable_solver.init(rng,batch)
pred, aux = diffable_solver.apply(params,batch)


visualize_model = jax.jit(lambda params,batch :diffable_solver.apply(params, batch, method=diffable_solver.visualize))
apply = jax.jit(lambda params,batch :diffable_solver.apply(params,batch))

@jax.jit
def loss(params,batch):
    pred, aux = apply(params,batch)
    return ((batch['ambient'] - pred/batch['alpha']) ** 2).sum(), aux

@jax.jit
def update(params_p,state_p,batch_p):
    params_p, state_p = solver.update(params_p, state_p,batch=batch_p)
    return params_p, state_p

data = logger.load_params()
solver = OptaxSolver(fun=loss, opt=optax.adam(opts.lr),has_aux=True)
state = solver.init_state(params)

with tqdm.trange(opts.max_iter) as t:
    for i in t:
        val_iter = i+1 % opts.val_freq == 0
        mode = 'val' if val_iter else 'train'
        batch = dataset.next_batch(val_iter)

        params, state = update(params,state,batch)
        l,_ = loss(params,batch)
        t.set_description('loss '+str(np.array(l)))
        if(i % opts.display_freq == 0 or val_iter):
            predicted = apply(params,batch)
            l1,l2,l3 = visualize_model(params,batch)
            l1 = tfu.camera_to_rgb_batch(l1/batch['alpha'], batch)
            l2 = tfu.camera_to_rgb_batch(l2/batch['alpha'], batch)
            l3 = tfu.camera_to_rgb_batch(l3/batch['alpha'], batch)
            noisy = tfu.camera_to_rgb_batch(batch['noisy']/batch['alpha'], batch)
            flash = tfu.camera_to_rgb_batch(batch['flash'], batch)
            g = tfu.camera_to_rgb_batch(predicted[0]/batch['alpha'], batch)
            ambient = tfu.camera_to_rgb_batch(batch['ambient'], batch)
            psnr = linalg.get_psnr_jax(jax.lax.stop_gradient(g),ambient)
            imshow = jnp.clip(jnp.concatenate((g,ambient,noisy,flash,l1,l2,l3),axis=-2),0,1)
            logger.addImage(imshow[0],'image',mode=mode)
            logger.addScalar(psnr,'psnr',mode=mode)
        if(i % opts.save_param_freq == 0):
            logger.save_params(params,batch,i)

        logger.addScalar(l,'loss',mode=mode)
        logger.takeStep()
