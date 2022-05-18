#!/bin/bash
source ./experiments/homography_params_static.sh
source ./experiments/logger_params_train_tb.sh
source ./experiments/noise_params_deepfnf.sh
source ./experiments/solver_params.sh
mode=test
fcount=64
fcount_suffix=

# expname=fft-highdim-phi01-psi01-$fcount
# helmholz="--delta_phi_init .01 --delta_psi_init .01 --model fft_highdim"

# expname=fft-highdim-phi01-psi1-$fcount
# helmholz="--delta_phi_init .01 --delta_psi_init 1. --model fft_highdim"

# expname=fft-highdim-phi1-psi01-$fcount
# helmholz="--delta_phi_init 1. --delta_psi_init .01 --model fft_highdim"

# expname=fft-highdim-phi1-psi1-fixed-$fcount
# helmholz="--delta_phi_init 1. --delta_psi_init 1. --fixed_delta --model fft_highdim"

expname=fft-highdim-nohelmholz-$fcount
helmholz="--model fft_highdim_nohelmholz"

name=msh-$mode-$expname

exp_params="\
--mode $mode \
--TLIST data/train.txt \
--TESTPATH data/testset_nojitter \
--logdir logger/$expname \
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
--high_dim"

priority='normal'

# name=msh-fft-solver-train-helmholze-3
scriptFn="unet_test/implicit_nonlin_screen_poisson.py $exp_params $helmholz $homography_params $logger_params $noise_params $solver_params"

# ./experiments/run_local.sh "$scriptFn"
./experiments/run_server.sh "$scriptFn" "$name" "$priority"