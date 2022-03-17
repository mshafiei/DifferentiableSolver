import glob
import functools

import numpy as np
import tensorflow as tf
from six.moves import cPickle as pkl
from deepfnf_utils.tf_spatial_transformer import transformer
import deepfnf_utils.utils as ut
import deepfnf_utils.tf_utils as tfu
import time
import jax
import jax.numpy as jnp
with open('./data/exifs.pkl', 'rb') as f:
    COLOR_MAP_DATA = pkl.load(f)

DATA_NAMES = [
    'ambient', 'warped_ambient', 'flash_only', 'warped_flash_only',
    'color_matrix', 'adapt_matrix', 'alpha', 'sig_read', 'sig_shot',
]


def load_image(filename, color_matrix, adapt_matrix):
    '''Load image and its camera matrices'''
    example = {}
    ambient = tf.io.read_file(filename + '_ambient.png')
    ambient = tf.image.decode_png(ambient, channels=3, dtype=tf.uint16)
    example['ambient'] = tf.cast(ambient, tf.float32) / 65535.

    flash_only = tf.io.read_file(filename + '_flash.png')
    flash_only = tf.image.decode_png(flash_only, channels=3, dtype=tf.uint16)
    example['flash_only'] = tf.cast(flash_only, tf.float32) / 65535.
    example['color_matrix'] = color_matrix
    example['adapt_matrix'] = adapt_matrix
    return example


def gen_homography(
        example, jitter, min_scale, max_scale, theta, psz, is_val=False):
    '''Randomly warp the image'''
    ambient = tf.clip_by_value(example['ambient'], 0., 1.)
    flash_only = tf.clip_by_value(example['flash_only'], 0., 1.)
    height, width = tf.shape(ambient)[0], tf.shape(ambient)[1]

    valid = int(jitter / min_scale)
    v_error = tf.maximum((psz + 2 * valid - height + 1) // 2, 0)
    h_error = tf.maximum((psz + 2 * valid - width + 1) // 2, 0)
    ambient = tf.pad(ambient, [[v_error, v_error], [h_error, h_error], [0, 0]])
    flash_only = tf.pad(
        flash_only, [[v_error, v_error], [h_error, h_error], [0, 0]])
    height = height + 2 * v_error
    width = width + 2 * h_error

    if not is_val:
        y = tf.random.uniform([], valid, tf.shape(
            ambient)[0] - valid - psz + 1, tf.int32)
        x = tf.random.uniform([], valid, tf.shape(
            ambient)[1] - valid - psz + 1, tf.int32)
    else:
        y = valid
        x = valid

    fov = np.deg2rad(90)
    f = psz / 2 / np.tan(fov / 2.)
    intrinsic = tf.convert_to_tensor([
        [f, 0, tf.cast(x, tf.float32) + psz / 2.],
        [0, f, tf.cast(y, tf.float32) + psz / 2.],
        [0, 0, 1]])
    intrinsic_inv = tf.linalg.inv(intrinsic)

    curr = tf.eye(3)
    scale = tf.random.uniform([], min_scale, max_scale)
    theta_x = tf.random.uniform([], -theta, theta)
    theta_y = tf.random.uniform([], -theta, theta)
    theta_z = tf.random.uniform([], -theta, theta)
    shift_x = tf.random.uniform([], -jitter, jitter)
    shift_y = tf.random.uniform([], -jitter, jitter)

    rotate_x = tf.convert_to_tensor([
        [1, 0, 0],
        [0, tf.cos(theta_x), -tf.sin(theta_x)],
        [0, tf.sin(theta_x), tf.cos(theta_x)]])
    rotate_y = tf.convert_to_tensor([
        [tf.cos(theta_y), 0, -tf.sin(theta_y)],
        [0, 1, 0],
        [tf.sin(theta_y), 0, tf.cos(theta_y)]])
    rotate_z = tf.convert_to_tensor([
        [tf.cos(theta_z), -tf.sin(theta_z), 0],
        [tf.sin(theta_z), tf.cos(theta_z), 0],
        [0, 0, 1]])
    rotate = tf.matmul(tf.matmul(rotate_x, rotate_y), rotate_z)
    rotate_homo = tf.matmul(tf.matmul(intrinsic, rotate), intrinsic_inv)

    scale_shift = tf.convert_to_tensor(
        [[scale, 0, -shift_x], [0, scale, -shift_y], [0, 0, 1]])

    H = tf.matmul(rotate_homo, scale_shift)
    H = tf.matmul(H, curr)
    H = tf.reshape(H, [1, 9])

    warped_flash_only, _ = transformer(flash_only[None], H, [height, width])
    warped_flash_only = tf.squeeze(warped_flash_only, axis=0)
    warped_flash_only = warped_flash_only[y:y + psz, x:x + psz, :]
    # due to numerical issue, might be values that are slightly larger than 1.0
    example['warped_flash_only'] = tf.clip_by_value(warped_flash_only, 0., 1.)

    warped_ambient, _ = transformer(ambient[None], H, [height, width])
    warped_ambient = tf.squeeze(warped_ambient, axis=0)
    warped_ambient = warped_ambient[y:y + psz, x:x + psz, :]
    example['warped_ambient'] = tf.clip_by_value(warped_ambient, 0., 1.)

    example['ambient'] = ambient[y:y + psz, x:x + psz, :]
    example['flash_only'] = flash_only[y:y + psz, x:x + psz, :]

    return example


def gen_random_params(
        example, min_alpha, max_alpha,
        min_read, max_read, min_shot, max_shot):
    '''Random noise parameters'''
    example['alpha'] = tf.pow(
        10., tf.random.uniform([], np.log10(min_alpha), np.log10(max_alpha)))
    example['sig_read'] = tf.pow(
        10., tf.random.uniform([], min_read, max_read))
    example['sig_shot'] = tf.pow(
        10., tf.random.uniform([], min_shot, max_shot))
    return example


def valset_generator(data_path):
    l = len(glob.glob(data_path+'/*.npz'))
    for i in range(l):
        data = np.load('%s/%d.npz' % (data_path, i))
        example = {}
        for name in DATA_NAMES:
            example[name] = np.squeeze(data[name])

        yield example



class Dataset:
    def __init__(self, opts, onfly_val=False):
        self.opts = opts
        train_list = opts.TLIST
        val_path = opts.VPATH
        bsz=opts.batch_size
        psz=opts.image_size
        ngpus=opts.ngpus
        nthreads=4 * opts.ngpus
        jitter=opts.displacement
        min_scale=opts.min_scale
        max_scale=opts.max_scale
        theta=opts.max_rotate
        min_alpha=opts.min_alpha
        max_alpha=opts.max_alpha
        min_read=opts.min_read
        max_read=opts.max_read
        min_shot=opts.min_shot
        max_shot=opts.max_shot

        self.train = TrainSet(
            train_list, bsz, psz, jitter,
            min_scale, max_scale, theta, ngpus, nthreads,min_alpha, max_alpha,
        min_read, max_read, min_shot, max_shot)
        if onfly_val:
            self.val = _OnFlyValSet(
                val_path, bsz, psz, jitter, min_scale, 
                max_scale, theta, ngpus, nthreads,min_alpha, max_alpha,
        min_read, max_read, min_shot, max_shot)
        else:
            self.val = ValSet(val_path, bsz, ngpus)

        self.train_iter = iter(self.train.dataset)
        self.val_iter = iter(self.val.dataset)
    
    # Check for saved weights & optimizer states
    def preprocess(self,example,keys):
        

        key1, key2, key3, key4, key5, key6, key7, key8, key9, key10= keys

        # # for i in range(opts.ngpus):
        #     # with tf.device('/gpu:%d' % i):
        alpha = example['alpha'][:, None, None, None]
        dimmed_ambient, _ = tfu.dim_image_jax(
            example['ambient'], key1,alpha=alpha)
        dimmed_warped_ambient, _ = tfu.dim_image_jax(
            example['warped_ambient'],key2, alpha=alpha)

        # Make the flash brighter by increasing the brightness of the
        # flash-only image.
        flash = example['flash_only'] * ut.FLASH_STRENGTH + dimmed_ambient
        warped_flash = example['warped_flash_only'] * \
            ut.FLASH_STRENGTH + dimmed_warped_ambient

        sig_read = example['sig_read'][:, None, None, None]
        sig_shot = example['sig_shot'][:, None, None, None]
        
        noisy_ambient, _, _ = tfu.add_read_shot_noise_jax(
            dimmed_ambient,key3,key4,key5,key6, min_read=self.opts.min_read, max_read=self.opts.max_read, min_shot=self.opts.min_shot, max_shot=self.opts.max_shot,sig_read=sig_read, sig_shot=sig_shot)
        noisy_flash, _, _ = tfu.add_read_shot_noise_jax(
            warped_flash,key7,key8,key9,key10, min_read=self.opts.min_read, max_read=self.opts.max_read, min_shot=self.opts.min_shot, max_shot=self.opts.max_shot,sig_read=sig_read, sig_shot=sig_shot)

        # noisy_ambient = jnp.zeros_like(example['ambient'])
        # noisy_flash = jnp.zeros_like(example['ambient'])
        # sig_shot = jnp.zeros((*example['ambient'].shape[:-1],6))
        # sig_read = jnp.zeros((*example['ambient'].shape[:-1],6))
        # sig_shot = jnp.zeros((*example['ambient'].shape[:-1],6))

        noisy = jnp.concatenate([noisy_ambient, noisy_flash], axis=-1)
        noise_std = tfu.estimate_std_jax(noisy, sig_read, sig_shot)
        net_input = jnp.concatenate([noisy,noise_std], axis=-1)
        
        output = {
            'alpha':alpha,
            'ambient':example['ambient'],
            'flash':noisy_flash,
            'noisy':noisy_ambient,
            'net_input':net_input,
            'adapt_matrix':example['adapt_matrix'],
            'color_matrix':example['color_matrix']
        }
        
        return output
    def next_batch(self,val_iter_p,iter_c):
        if(val_iter_p):
            try:
                batch = self.val_iter.next()
            except StopIteration:
                self.val_iter = iter(self.val.dataset)
                batch = self.val_iter.next()
        else:
            try:
                batch = self.train_iter.next()
            except StopIteration:
                self.train_iter = iter(self.train.dataset)
                batch = self.train_iter.next()
        batch = {k:jnp.array(v.numpy()) for k,v in batch.items()}
        keys = [jax.random.PRNGKey(iter_c*10 + i) for i in range(10)]
        
        
        return self.preprocess(batch,keys)

    @staticmethod
    def parse_arguments(parser):
        parser.add_argument('--ngpus', type=int, default=1, help='use how many gpus')
        parser.add_argument('--displacement', default=2, type=float,help='Random shift in pixels')
        parser.add_argument('--TLIST', default='data/train.txt',type=str, help='Maximum rotation')
        parser.add_argument('--VPATH', default='data/valset/', type=str,help='Maximum rotation')
        parser.add_argument('--TESTPATH', default='data/testset/', type=str,help='Maximum rotation')
        parser.add_argument('--batch_size', default=1, type=int,help='Image count in a batch')
        parser.add_argument('--min_scale', default=0.98,type=float, help='Random shift in pixels')
        parser.add_argument('--max_scale', default=1.02,type=float, help='Random shift in pixels')
        parser.add_argument('--image_size', default=448,type=int, help='Image size')
        parser.add_argument('--max_rotate', default=np.deg2rad(0.5),type=float, help='Maximum rotation')
        parser.add_argument('--min_alpha', default=0.02, type=float,help='Maximum rotation')
        parser.add_argument('--max_alpha', default=0.2, type=float,help='Maximum rotation')
        parser.add_argument('--min_read', default=-3., type=float,help='Maximum rotation')
        parser.add_argument('--max_read', default=-2, type=float,help='Maximum rotation')
        parser.add_argument('--min_shot', default=-2., type=float,help='Maximum rotation')
        parser.add_argument('--max_shot', default=-1.3, type=float,help='Maximum rotation')
        return parser


class TrainSet:
    def __init__(
            self, file_list, bsz, psz, jitter,
            min_scale, max_scale, theta, ngpus, nthreads,
            min_alpha, max_alpha,
            min_read, max_read, min_shot, max_shot):
        files = [l.strip() for l in open(file_list)]

        gen_homography_fn = functools.partial(
            gen_homography, jitter=jitter, min_scale=min_scale,
            max_scale=max_scale, theta=theta, psz=psz, is_val=True)
        gen_random_params_fn = functools.partial(
            gen_random_params, min_alpha=min_alpha, max_alpha=max_alpha,
        min_read=min_read, max_read=max_read, min_shot=min_shot, max_shot=max_shot)

        color_matrices = np.stack(
            [COLOR_MAP_DATA[nm][0] for nm in files],
            axis=0).astype(np.float32)
        adapt_matrices = np.stack(
            [COLOR_MAP_DATA[nm][1] for nm in files],
            axis=0).astype(np.float32)

        dataset = tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(files),
            tf.data.Dataset.from_tensor_slices(color_matrices),
            tf.data.Dataset.from_tensor_slices(adapt_matrices)
        ))
        self.dataset = (dataset
                        .repeat()
                        .shuffle(buffer_size=len(files))
                        .map(load_image, num_parallel_calls=nthreads)
                        .map(gen_homography_fn, num_parallel_calls=nthreads)
                        .map(gen_random_params_fn, num_parallel_calls=nthreads)
                        .batch(bsz)
                        .prefetch(ngpus)
                        )
        self.iterator = iter(self.dataset)
        
    def initialize(self):
        self.iterator = iter(self.dataset)

class ValSet:
    def __init__(self, val_path, bsz, ngpus):
        generator = functools.partial(valset_generator, data_path=val_path)
        dataset = tf.data.Dataset.from_generator(
            generator,
            {name: tf.float32 for name in DATA_NAMES})

        self.dataset = (dataset
                        .batch(bsz, drop_remainder=True)
                        .prefetch(ngpus)
                        )
        self.iterator = iter(self.dataset)

    def initialize(self):
        self.iterator = iter(self.dataset)
        

class _OnFlyValSet:
    def __init__(
            self, file_list, bsz, psz, jitter,
            min_scale, max_scale, theta, ngpus, nthreads,
            min_alpha, max_alpha,
            min_read, max_read, min_shot, max_shot):
        files = [l.strip() for l in open(file_list)]

        gen_homography_fn = functools.partial(
            gen_homography, jitter=jitter, min_scale=min_scale,
            max_scale=max_scale, theta=theta, psz=psz, is_val=True)
        gen_random_params_fn = functools.partial(
            gen_random_params, min_alpha=min_alpha, max_alpha=max_alpha,
        min_read=min_read, max_read=max_read, min_shot=min_shot, max_shot=max_shot)

        color_matrices = np.stack(
            [COLOR_MAP_DATA[nm][0] for nm in files],
            axis=0).astype(np.float32)
        adapt_matrices = np.stack(
            [COLOR_MAP_DATA[nm][1] for nm in files],
            axis=0).astype(np.float32)

        dataset = tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(files),
            tf.data.Dataset.from_tensor_slices(color_matrices),
            tf.data.Dataset.from_tensor_slices(adapt_matrices)
        ))
        self.dataset = (dataset
                        .map(load_image, num_parallel_calls=nthreads)
                        .map(gen_homography_fn, num_parallel_calls=nthreads)
                        .map(gen_random_params_fn, num_parallel_calls=nthreads)
                        .batch(bsz, drop_remainder=True)
                        .prefetch(ngpus)
                        )
        self.iterator = iter(self.dataset)

    def initialize(self, sess):
        self.iterator = iter(self.dataset)
