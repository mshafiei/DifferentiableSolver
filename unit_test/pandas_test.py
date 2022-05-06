from tabulate import tabulate
import pandas as pd
import numpy as np
from cvgutils.Utils import createTexTable, loadJson

metrics = ['PSNR','SSIM']
level_name = ['Level 4','Level 3','Level 2','Level 1']
noise_levels = ['100x','50x','25x','12x']
# row_title = ['U-Net',r'Ours w/o decomposition',' Ours with decomposition','deepFnF']
#128
row_title = ['U-Net',r'Ours $\phi=1,\psi=1$ fixed',r'Ours $\phi,\psi$ learned',r'Ours $\phi=1,\psi=0$ fixed',r'Ours unconstrained $\bm{g}$']
fns = []
fns.append('logger/fft_solver_largeds-relu-128/unet/test/test_errors.txt')
fns.append('logger/fft_solver_largeds_relu-128/fft_helmholze-phi1-psi1-fixed-128/test/test_errors.txt')
fns.append('logger/fft_solver_largeds_relu-128/fft_helmholze-phi1-psi1-128/test/test_errors.txt')
fns.append('logger/fft_solver_largeds_relu-128/fft_l0/test/test_errors.txt')
fns.append('logger/fft_solver_largeds_relu-128/fft/test/test_errors.txt')

#64
fns.append('logger/fft_solver_largeds-relu/unet/test/test_errors.txt')
fns.append('logger/fft_solver_largeds_relu/fft_helmholze-phi1-psi1-fixed-128/test/test_errors.txt')
fns.append('logger/fft_solver_largeds_relu/fft_helmholze-phi1-psi1-128/test/test_errors.txt')
fns.append('logger/fft_solver_largeds_relu/fft_l0/test/test_errors.txt')
fns.append('logger/fft_solver_largeds_relu/fft/test/test_errors.txt')


ress = [loadJson(i) for i in fns]
data = np.zeros((len(row_title),len(noise_levels)*len(metrics)))
for i, ress in enumerate(ress):
    c = 0
    for _, metric in enumerate(metrics):
        for _, (name, level) in enumerate(zip(level_name,noise_levels)):
            errval = ress[name][metric]
            data[i,c] = float(errval)
            c += 1
# data_psnr = np.array([[19.441,30.448,30.242,30.818],
# [16.062,29.290,28.990,29.642],
# [13.068,28.958,28.484,29.404],
# [11.497, 27.113,26.362,28.018]])

# data_ssim = np.array([[0.7359,0.8555,0.8516,0.8544],
# [0.6266,0.8315,0.8251,0.8313],
# [0.5493,0.8276,0.8196, 0.8303],
# [0.3788,0.7882,0.7798, 0.8073]])
# data = np.concatenate((data_psnr.transpose(1,0),data_ssim.transpose(1,0)),axis=1)


tex = createTexTable(row_title,metrics,noise_levels,data)
print(tex)
