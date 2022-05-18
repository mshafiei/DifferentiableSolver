import pandas as pd
import numpy as np
from cvgutils.Utils import createTexTable, loadJson


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
def createDataFile(method_fns,method_name,noise_levels,metrics,metrics_id,noise_levels_id):
    ress = [loadJson(i) for i in method_fns]
    data = np.zeros((len(method_name),len(noise_levels)*len(metrics)))
    print(len(ress))
    print(len(data))
    for i, ress in enumerate(ress):
        c = 0
        for _, metric in enumerate(metrics_id):
            for _, (name, level) in enumerate(zip(noise_levels_id,noise_levels)):
                errval = ress[name][metric]
                data[i,c] = float(errval)
                c += 1
    return data


noise_levels = ['100x','50x','25x','12x']
noise_levels_id = ['Level 4','Level 3','Level 2','Level 1']

# method_name = [
# r'Ours $\lambda_{\phi}=1,\lambda_{\psi}=1$',
# r'Ours $\lambda^0_{\phi}=0.01,\lambda^0_{\psi}=1$',
# r'Ours $\lambda^0_{\phi}=1,\lambda^0_{\psi}=1$',
# r'Ours $\lambda^0_{\phi}=1,\lambda^0_{\psi}=0.01$']

# method_fns = [
# 'logger/fft_solver_largeds_relu-128/fft_helmholze-phi1-psi1-fixed-128/test/test_errors.txt',
# 'logger/fft_solver_largeds_relu-128/fft_helmholze-phi1e2-psi1-128/test/test_errors.txt',
# 'logger/fft_solver_largeds_relu-128/fft_helmholze-phi1-psi1-128/test/test_errors.txt',
# 'logger/fft_solver_largeds_relu-128/fft_helmholze-phi1-psi1e2-128/test/test_errors.txt']

# method_name += [
# r'Ours HD $\lambda^0_{\phi}=0.01,\lambda^0_{\psi}=0.01$',
# r'Ours HD $\lambda^0_{\phi}=0.01,\lambda^0_{\psi}=1$',
# r'Ours HD $\lambda^0_{\phi}=1,\lambda^0_{\psi}=0.01$',
# r'Ours HD $\lambda_{\phi}=1,\lambda_{\psi}=1$',]

# method_fns += [
# 'logger/fft-highdim-phi01-psi01-64/fft-highdim-phi01-psi01-64/test/test_errors.txt',
# 'logger/fft-highdim-phi01-psi1-64/fft-highdim-phi01-psi1-64/test/test_errors.txt',
# 'logger/fft-highdim-phi1-psi01-64/fft-highdim-phi1-psi01-64/test/test_errors.txt',
# 'logger/fft-highdim-phi1-psi1-fixed-64/fft-highdim-phi1-psi1-fixed-64/test/test_errors.txt',
# '/mshvol2/users/mohammad/optimization/DifferentiableSolver/logger/fft-highdim-nohelmholz-64/fft-highdim-nohelmholz-64/test/test_errors.txt']


method_name = [
'Ours (learned Fixed HP)',
'Ours (learned HP)',
'Ours (unconstrained)']

method_fns = [
'logger/fft-highdim-phi1-psi1-fixed-64/fft-highdim-phi1-psi1-fixed-64/test/test_errors.txt',
'logger/fft-highdim-phi01-psi01-64/fft-highdim-phi01-psi01-64/test/test_errors.txt',
'logger/fft_solver_largeds-relu/fft/test/test_errors.txt']
# '/mshvol2/users/mohammad/optimization/DifferentiableSolver/logger/fft-highdim-nohelmholz-64/fft-highdim-nohelmholz-64/test/test_errors.txt']

method_name += ['U-Net','DeepFnF']
method_fns += ['logger/fft_solver_largeds-relu-128/unet/test/test_errors.txt',
'/mshvol2/users/mohammad/optimization/deepfnf_fork/logs/deepfnf/test/test_errors.txt']

# metrics = ['LPIPS-VGG','LPIPS-Alex']
# metrics_id = ['lpipsVGG','lpipsAlex']
# data = createDataFile(method_fns,method_name,noise_levels,metrics,metrics_id,noise_levels_id)
# tex = createTexTable(method_name,metrics,noise_levels,data)
# print('******* LPIPS-VGG, LPIPS-Alex *************')
# print(tex)

# metrics = ['PSNR','MSE']
# metrics_id = ['psnr','mse']
# data = createDataFile(method_fns,method_name,noise_levels,metrics,metrics_id,noise_levels_id)
# tex = createTexTable(method_name,metrics,noise_levels,data)
# print('******* PSNR, MSE *************')
# print(tex)

# metrics = ['MS-SSIM','SSIM']
# metrics_id = ['msssim','ssim']
# data = createDataFile(method_fns,method_name,noise_levels,metrics,metrics_id,noise_levels_id)
# tex = createTexTable(method_name,metrics,noise_levels,data)
# print('******* MS-SSIM, SSIM *************')
# print(tex)

metrics = ['PSNR','SSIM']
metrics_id = ['psnr','ssim']
data = createDataFile(method_fns,method_name,noise_levels,metrics,metrics_id,noise_levels_id)
tex = createTexTable(method_name,metrics,noise_levels,data)
print('******* PSNR, SSIM *************')
print(tex)

metrics = ['PSNR','SSIM']
metrics_id = ['psnr','ssim']
data = createDataFile(method_fns[:-1],method_name[:-1],noise_levels,metrics,metrics_id,noise_levels_id)
tex = createTexTable(method_name[:-1],metrics,noise_levels,data)
print('******* PSNR, SSIM *************')
print(tex)

