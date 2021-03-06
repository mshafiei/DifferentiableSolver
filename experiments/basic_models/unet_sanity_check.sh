#!/bin/bash
source ./experiments/homography_params_static.sh
source ./experiments/logger_params_train_tb.sh
source ./experiments/noise_params_noisefree.sh
source ./experiments/solver_params.sh
exp_params="\
--mode train \
--model unet \
--TLIST data/train_1.txt \
--logdir logger/Unet_test \
--expname unet-sanity-softplus-64 \
--batch_size 1 \
--out_features 3 \
--thickness 64 \
--in_features 12 \
--activation softplus"



name=msh-unet-sanity-check
scriptFn="unet_test/implicit_nonlin_screen_poisson.py $exp_params $homography_params $logger_params $noise_params $solver_params"

./experiments/run_local.sh "$scriptFn" "$name"
# ./experiments/run_server.sh "$scriptFn" "$name"