#!/bin/bash
source ./experiments/homography_params_static.sh
source ./experiments/logger_params_train_tb.sh
source ./experiments/noise_params_deepfnf.sh
source ./experiments/solver_params.sh
exp_params="\
--mode test \
--model fft \
--TLIST data/train_1600.txt \
--logdir logger/fft_solver \
--expname fft \
--batch_size 1 \
--out_features 6 \
--in_features 12"



name=msh-fft-solver-test
scriptFn="unet_test/implicit_nonlin_screen_poisson.py $exp_params $homography_params $logger_params $noise_params $solver_params"

# ./experiments/run_local.sh "$scriptFn"
./experiments/run_server.sh "$scriptFn" "$name"