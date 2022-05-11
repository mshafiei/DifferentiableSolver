#!/bin/bash
source ./experiments/homography_params_static.sh
source ./experiments/logger_params_train_tb.sh
source ./experiments/noise_params_deepfnf.sh
source ./experiments/solver_params.sh
mode=train
fcount=64
fcount_suffix=

expname=fft-solver-filters
name=msh-$expname

exp_params="\
--mode $mode \
--model fft_filters \
--TLIST data/train.txt \
--TESTPATH data/testset_nojitter \
--logdir logger/$expname \
--expname $expname
--batch_size 1 \
--thickness $fcount \
--in_features 12 \
--activation relu \
--kernel_channels 3 \
--kernel_count 50 \
--kernel_size 15"

priority='nice'

# name=msh-fft-solver-train-helmholze-3
scriptFn="unet_test/implicit_nonlin_screen_poisson.py $exp_params $homography_params $logger_params $noise_params $solver_params"

# ./experiments/run_local.sh "$scriptFn"
./experiments/run_server.sh "$scriptFn" "$name" "$priority"