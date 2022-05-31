#!/bin/bash
source ./experiments/homography_params_static.sh
source ./experiments/logger_params_train_tb.sh
source ./experiments/noise_params_deepfnf.sh
source ./experiments/solver_params.sh
mode=test
fcount=64
fcount_suffix=

expname=fft-highdim-phi01-psi01-$fcount
helmholz="--delta_phi_init .01 --delta_psi_init .01 --model fft_highdim"

# expname=fft-highdim-phi01-psi1-$fcount
# helmholz="--delta_phi_init .01 --delta_psi_init 1. --model fft_highdim"

# expname=fft-highdim-phi1-psi01-$fcount
# helmholz="--delta_phi_init 1. --delta_psi_init .01 --model fft_highdim"

# expname=fft-highdim-phi1-psi1-fixed-$fcount
# helmholz="--delta_phi_init 1. --delta_psi_init 1. --fixed_delta --model fft_highdim"

# expname=fft-highdim-nohelmholz-$fcount
# helmholz="--model fft_highdim_nohelmholz"

name=msh-$mode-$expname

exp_params="\
--mode $mode \
--TLIST data/train.txt \
--TESTPATH data/testset_nojitter \
--logdir logger/highdim-dynamiclr \
--expname $expname \
--batch_size 1 \
--thickness $fcount \
--in_features 12 \
--out_features 128 \
--unet_factor 1 \
--activation relu \
--kernel_channels 3 \
--kernel_count 90 \
--kernel_size 15 \
--high_dim \
--max_iter 2500000"

priority='normal'

# name=msh-fft-solver-train-helmholze-3
scriptFn="unet_test/implicit_nonlin_screen_poisson.py $exp_params $helmholz $homography_params $logger_params $noise_params $solver_params"

# ./experiments/run_local.sh "$scriptFn"
./experiments/run_server.sh "$scriptFn" "$name" "$priority"

#unconstrained 
# mse: 0.0021, psnr: 27.6517, ssim: 0.8031, msssim: 0.7323, lpipsVGG: 0.4461, lpipsAlex: 0.3189, div: 0.0304, curl: 0.0070, lambda: 0.0346
# 100% 128/128 [07:01<00:00,  3.29s/it]
# mse: 0.0014, psnr: 29.3920, ssim: 0.8375, msssim: 0.7479, lpipsVGG: 0.3944, lpipsAlex: 0.2910, div: 0.1236, curl: 0.0134, lambda: 0.0346
# 100% 128/128 [07:08<00:00,  3.35s/it]
# mse: 0.0013, psnr: 29.7017, ssim: 0.8400, msssim: 0.7399, lpipsVGG: 0.3612, lpipsAlex: 0.2813, div: 0.5176, curl: 0.0464, lambda: 0.0346
# 100% 128/128 [07:14<00:00,  3.39s/it]
# mse: 0.0010, psnr: 30.8202, ssim: 0.8615, msssim: 0.7664, lpipsVGG: 0.3465, lpipsAlex: 0.2641, div: 2.3994, curl: 0.1598, lambda: 0.0346

#0.01, 0.01

# mse: 0.0021, psnr: 27.6808, ssim: 0.8043, msssim: 0.7422, lpipsVGG: 0.4237, lpipsAlex: 0.3030, div: 0.0602, curl: 0.2194, lambda: 0.0051
# 100% 128/128 [10:02<00:00,  4.71s/it]
# mse: 0.0014, psnr: 29.4968, ssim: 0.8391, msssim: 0.7513, lpipsVGG: 0.3805, lpipsAlex: 0.2850, div: 0.1212, curl: 0.3092, lambda: 0.0051
# 100% 128/128 [09:51<00:00,  4.62s/it]
# mse: 0.0013, psnr: 29.7891, ssim: 0.8407, msssim: 0.7419, lpipsVGG: 0.3501, lpipsAlex: 0.2766, div: 0.3723, curl: 0.7227, lambda: 0.0051
# 100% 128/128 [10:02<00:00,  4.71s/it]
# mse: 0.0010, psnr: 30.8726, ssim: 0.8627, msssim: 0.7677, lpipsVGG: 0.3358, lpipsAlex: 0.2617, div: 1.4652, curl: 2.2484, lambda: 0.0051