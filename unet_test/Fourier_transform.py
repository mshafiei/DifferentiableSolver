import jax.numpy as jnp
import jax
from jax.scipy.fft import dctn as jax_dctn
from jax.numpy.fft import ifftn as jax_ifftn
from scipy.fftpack import dctn as scp_dctn, idctn as scp_idctn, ifftn as scp_ifftn
import cvgutils.Image as cvgim

fn = '/home/mohammad/Projects/optimizer/DifferentiableSolver/testImages1/Flash.jpg'
im = cvgim.imread(fn)
scp_dct_im = scp_dctn(im)
jax_dct_im = jax_dctn(im)

scp_ifft_im = scp_ifftn(scp_dct_im)
scp_idct_im = scp_idctn(scp_dct_im)
jax_ifft_im = jax_dctn(jax_dct_im,2)

print('hi')