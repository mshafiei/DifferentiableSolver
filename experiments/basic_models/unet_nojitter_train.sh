#!/bin/bash
source ./experiments/homography_params_static.sh
source ./experiments/logger_params_train_tb.sh
source ./experiments/noise_params_deepfnf.sh
source ./experiments/solver_params.sh
mode=test
fcount=64
fcount_suffix=

exp_params="\
--mode $mode \
--model unet \
--TLIST data/train.txt \
--TESTPATH data/testset_nojitter \
--logdir logger/fft_solver_largeds-relu$fcount_suffix \
--expname unet \
--batch_size 1 \
--out_features 3 \
--in_features 12 \
--thickness $fcount \
--activation relu --display_freq_test 1000"

priority='nice'


name=msh-$mode-unet-relu-$fcount
scriptFn="unet_test/implicit_nonlin_screen_poisson.py $exp_params $homography_params $logger_params $noise_params $solver_params"

# ./experiments/run_local.sh "$scriptFn" "$name"
./experiments/run_server.sh "$scriptFn" "$name" "$priority"