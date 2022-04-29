from skimage.metrics import structural_similarity
import numpy as np
import imageio
import jax.numpy as jnp

def imsave(nm, img):
    if len(img.shape) == 4:
        img = np.squeeze(img, 0)
    img = np.uint8(np.clip(img,0,1) * 255.)
    imageio.imsave(nm, img)


def get_mse(pred, gt):
    return np.mean(np.square(pred-gt))
def get_mse_jax(pred, gt):
    return jnp.mean(jnp.square(pred-gt))


def get_psnr(pred, gt):
    pred = pred.clip(0., 1.)
    gt = gt.clip(0., 1.)
    mse = np.mean((pred-gt)**2.0)
    psnr = -10. * np.log10(mse)
    return psnr
def get_psnr_jax(pred, gt):
    pred = pred.clip(0., 1.)
    gt = gt.clip(0., 1.)
    mse = jnp.mean((pred-gt)**2.0)
    psnr = -10. * jnp.log10(mse)
    return psnr


def get_ssim(pred, gt):
    ssim = structural_similarity(
        pred,
        gt,
        win_size=11,
        gaussian_weights=True,
        multichannel=True,
        K1=0.01,
        K2=0.03,
        sigma=1.5)
    return ssim
