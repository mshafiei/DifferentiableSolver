#!/bin/bash
source ./experiments/homography_params_static.sh
source ./experiments/logger_params_train_tb.sh
source ./experiments/noise_params_deepfnf.sh
source ./experiments/solver_params.sh
mode=train
fcount=64
fcount_suffix=
factor=2
# out_features=330
# kernel_count=110
# outc_kernel_size=3

out_features=240
kernel_count=80
outc_kernel_size=3

expname=fft-solver-filters-$kernel_count-factor$factor
name=msh-$mode-$expname

exp_params="\
--mode $mode \
--model fft_filters \
--TLIST data/train.txt \
--TESTPATH data/testset_nojitter \
--logdir logger/$expname \
--expname $expname \
--batch_size 1 \
--thickness $fcount \
--in_features 12 \
--out_features $out_features \
--unet_factor $factor \
--activation relu \
--kernel_channels 3 \
--kernel_count $kernel_count \
--outc_kernel_size $outc_kernel_size \
--kernel_size 15 \
--max_iter 2500000"

priority='normal'

# name=msh-fft-solver-train-helmholze-3
scriptFn="unet_test/implicit_nonlin_screen_poisson.py $exp_params $homography_params $logger_params $noise_params $solver_params"

# ./experiments/run_local.sh "$scriptFn"
./experiments/run_server.sh "$scriptFn" "$name" "$priority"
