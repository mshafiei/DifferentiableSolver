#!/bin/bash
source ./experiments/homography_params_static.sh
source ./experiments/logger_params_train_tb.sh
source ./experiments/noise_params_deepfnf.sh
source ./experiments/solver_params.sh
exp_params="\
--mode train \
--model unet \
--TLIST data/train_1600.txt \
--logdir logger/nonlin-poisson-nojitter-fixed2 \
--expname unet \
--batch_size 1 \
--out_features 3 \
--in_features 12"



name=msh-deep-nonlin-unet-generalize-nojitter-fixed2-train
scriptFn="unet_test/implicit_nonlin_screen_poisson.py $exp_params $homography_params $logger_params $noise_params $solver_params"

# ./experiments/run_local.sh "$scriptFn" "$name"
./experiments/run_server.sh "$scriptFn" "$name"