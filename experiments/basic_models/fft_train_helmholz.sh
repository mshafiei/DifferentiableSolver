#!/bin/bash
source ./experiments/homography_params_static.sh
source ./experiments/logger_params_train_tb.sh
source ./experiments/noise_params_deepfnf.sh
source ./experiments/solver_params.sh
exp_params="\
--mode train \
--model fft_helmholz \
--TLIST data/train_1600.txt \
--TESTPATH data/testset_nojitter \
--logdir logger/fft_solver_gradfixed \
--expname fft_helmholz \
--batch_size 1 \
--out_features 6 \
--in_features 12 \
--max_delta 0.00001"


name=msh-fft-solver-train-helmholz
scriptFn="unet_test/implicit_nonlin_screen_poisson.py $exp_params $homography_params $logger_params $noise_params $solver_params"

./experiments/run_local.sh "$scriptFn"
# ./experiments/run_server.sh "$scriptFn" "$name"
