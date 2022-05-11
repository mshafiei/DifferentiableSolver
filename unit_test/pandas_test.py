import pandas as pd
import numpy as np
from cvgutils.Utils import createTexTable, loadJson

metrics = ['PSNR','SSIM']
level_name = ['Level 4','Level 3','Level 2','Level 1']
noise_levels = ['100x','50x','25x','12x']
# row_title = ['U-Net',r'Ours w/o decomposition',' Ours with decomposition','deepFnF']
#128
# row_title = ['U-Net',
# r'Ours $\lambda_{\phi}=1,\lambda_{\psi}=1$ fixed',
# r'Ours $\lambda_{\phi}=1,\lambda_{\psi}=0$ fixed',
# r'Ours $\lambda_{\phi}=1,\lambda_{\psi}=1$ learned',
# r'Ours $\lambda_{\phi}=1,\lambda_{\psi}=0.01$ learned',
# r'Ours $\lambda_{\phi}=0.01,\lambda_{\psi}=1$ learned',
# r'Ours unconstrained $\bm{g}$']
# fns = []
# #128
# fns.append('logger/fft_solver_largeds-relu-128/unet/test/test_errors.txt')
# fns.append('logger/fft_solver_largeds_relu-128/fft_helmholze-phi1-psi1-fixed-128/test/test_errors.txt')
# fns.append('logger/fft_solver_largeds-relu-128/fft_l0/test/test_errors.txt')
# fns.append('logger/fft_solver_largeds_relu-128/fft_helmholze-phi1-psi1-128/test/test_errors.txt')
# fns.append('logger/fft_solver_largeds_relu-128/fft_helmholze-phi1-psi1e2-128/test/test_errors.txt')
# fns.append('logger/fft_solver_largeds_relu-128/fft_helmholze-phi1e2-psi1-128/test/test_errors.txt')
# fns.append('logger/fft_solver_largeds-relu-128/fft/test/test_errors.txt')

#64
#fns.append('logger/fft_solver_largeds-relu/unet/test/test_errors.txt')
#fns.append('logger/fft_solver_largeds_relu/fft_helmholze-phi1-psi1-fixed/test/test_errors.txt')
#fns.append('logger/fft_solver_largeds-relu/fft_l0/test/test_errors.txt')
#fns.append('logger/fft_solver_largeds_relu/fft_helmholze-phi1-psi1/test/test_errors.txt')
#fns.append('logger/fft_solver_largeds_relu/fft_helmholze-phi1-psi1e2/test/test_errors.txt')
#fns.append('logger/fft_solver_largeds_relu/fft_helmholze-phi1e2-psi1/test/test_errors.txt')
#fns.append('logger/fft_solver_largeds-relu/fft/test/test_errors.txt')

# ress = [loadJson(i) for i in fns]
# data = np.zeros((len(row_title),len(noise_levels)*len(metrics)))
# print(len(ress))
# print(len(data))
# for i, ress in enumerate(ress):
#     c = 0
#     for _, metric in enumerate(metrics):
#         for _, (name, level) in enumerate(zip(level_name,noise_levels)):
#             errval = ress[name][metric]
#             data[i,c] = float(errval)
#             c += 1

row_title = ['U-Net',
r'Ours $\lambda_{\phi}=1,\lambda_{\psi}=1$ fixed',
r'Ours $\lambda_{\phi}=1,\lambda_{\psi}=0.01$ learned',
r'deepfnf']

data = np.zeros((len(row_title),len(noise_levels)*len(metrics)))


        
data[0,:] = np.array([24.393, 26.134, 26.749, 27.233, 0.770, 0.811, 0.817, 0.842])
data[1,:] = np.array([26.462, 28.650, 29.214, 30.514, 0.782, 0.823, 0.828, 0.854])
data[2,:] = np.array([26.085, 28.579,29.252, 30.601, 0.773, 0.820,0.828,0.856])
data[3,:] = np.array([28.018,29.404,29.642,30.818,0.807,0.830,0.831,0.854])
tex = createTexTable(row_title,metrics,noise_levels,data)
print(tex)