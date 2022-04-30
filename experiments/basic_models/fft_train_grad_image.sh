#!/bin/bash
source ./experiments/homography_params_static.sh
source ./experiments/logger_params_train_tb.sh
source ./experiments/noise_params_deepfnf.sh
source ./experiments/solver_params.sh
exp_params="\
--mode train \
--model fft_image_grad \
--TLIST data/train_1600.txt \
--TESTPATH data/testset_nojitter \
--logdir logger/fft_solver \
--expname fft_grad_image \
--batch_size 1 \
--out_features 3 \
--in_features 12 --store_params"



name=msh-fft-solver-train-grad-image1
scriptFn="unet_test/implicit_nonlin_screen_poisson.py $exp_params $homography_params $logger_params $noise_params $solver_params"

./experiments/run_local.sh "$scriptFn"
# ./experiments/run_server.sh "$scriptFn" "$name"
