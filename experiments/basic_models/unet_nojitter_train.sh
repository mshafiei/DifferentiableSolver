#!/bin/bash
source ./experiments/homography_params_static.sh
source ./experiments/logger_params_train_tb.sh
source ./experiments/noise_params_deepfnf.sh
source ./experiments/solver_params.sh
exp_params="\
--mode test \
--model unet \
--TLIST data/train_1600.txt \
--logdir logger/Unet_test \
--expname unet-generalize-nojitter-1bsz-tmp \
--batch_size 1 \
--out_features 3 \
--in_features 12 --store_params"



name=msh-deep-nonlin-unet-generalize-nojitter-fixed2-test2
scriptFn="unet_test/implicit_nonlin_screen_poisson.py $exp_params $homography_params $logger_params $noise_params $solver_params"

./experiments/run_local.sh "$scriptFn" "$name"
# ./experiments/run_server.sh "$scriptFn" "$name"