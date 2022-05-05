#!/bin/bash
source ./experiments/homography_params_static.sh
source ./experiments/logger_params_train_tb.sh
source ./experiments/noise_params_deepfnf.sh
source ./experiments/solver_params.sh
exp_params="\
--mode train \
--model fft_image_grad \
--TLIST data/train.txt \
--TESTPATH data/testset_nojitter \
--logdir logger/fft_solver_largeds-relu-128 \
--expname fft_l0 \
--batch_size 1 \
--out_features 3 \
--in_features 12 \
--thickness 128 \
--activation relu"

priority='nice'



name=msh-fft-train-l0-gfixed-large0-relu-128
scriptFn="unet_test/implicit_nonlin_screen_poisson.py $exp_params $homography_params $logger_params $noise_params $solver_params"

# ./experiments/run_local.sh "$scriptFn"
./experiments/run_server.sh "$scriptFn" "$name" "$priority"
