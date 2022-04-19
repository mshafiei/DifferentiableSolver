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
from implicit_diff_module import diff_solver, fnf_regularizer, implicit_sanity_model, implicit_poisson_model, direct_model
from flax import linen as nn
import deepfnf_utils.utils as ut
import time

def parse_arguments(parser):
    parser.add_argument('--model', type=str, default='implicit_sanity_model',
    choices=['implicit_sanity_model','implicit_poisson_model','unet'],help='Which model to use')
    parser.add_argument('--lr', default=1e-4, type=float,help='Maximum rotation')
    parser.add_argument('--display_freq', default=1000, type=int,help='Display frequency by iteration count')
    parser.add_argument('--val_freq', default=101, type=int,help='Display frequency by iteration count')
    parser.add_argument('--save_param_freq', default=100,type=int, help='Maximum rotation')
    parser.add_argument('--max_iter', default=100000000, type=int,help='Maximum iteration count')
    parser.add_argument('--unet_depth', default=4, type=int,help='Depth of neural net')
    parser.add_argument('--debug_keywords', default='None', type=str,help='Debug keywords')
    parser.add_argument('--mode', default='train', type=str,choices=['train','test'],help='Should we train or test the model?')
    
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

batch = dataset.next_batch(False,0)

data = logger.load_params()

def eval_visualize(batch,logger):
    inpt1 = tfu.camera_to_rgb_batch(batch['net_input'][...,:3]/batch['alpha'],batch)
    inpt2 = tfu.camera_to_rgb_batch(batch['net_input'][...,3:6],batch)
    imgs = [inpt1,inpt2]
    labels = ['inpt1','inpt2']
    logger.addImage(imgs,labels,'image',dim_type='BHWC',mode='train')

for i in tqdm.trange(100):
    batch = dataset.next_batch(False,i)
    eval_visualize(batch,logger)
    logger.takeStep()
