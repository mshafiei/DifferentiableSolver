#!/bin/bash
source ./experiments/homography_params_static.sh
source ./experiments/logger_params_train_tb.sh
source ./experiments/noise_params_deepfnf.sh

solver_params="\
--nlin_iter 100 \
--nnonlin_iter 3"

exp_params="\
--mode train \
--nn_model linear \
--model implicit_poisson_model \
--TLIST data/train_1600.txt \
--logdir logger/nonlin-poisson-nojitter-fixed2 \
--expname unet-poisson-deriv \
--batch_size 1 \
--out_features 6 \
--in_features 12 --store_params"



name=msh-deep-nonlin-screen-poisson-generalize-nojitter-fixed2-train
scriptFn="unet_test/implicit_nonlin_screen_poisson.py $exp_params $homography_params $logger_params $noise_params $solver_params"

./experiments/run_local.sh "$scriptFn"
# ./experiments/run_server.sh "$scriptFn" "$name"