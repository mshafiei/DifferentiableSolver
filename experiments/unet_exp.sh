#!/bin/bash
source experiments/homography_params_statis.sh
source experiments/logger_params_train_tb.sh
source experiments/noise_params_deepfnf.sh
exp_params="\
--mode train \
--model unet \
--TLIST data/train_1600.txt \
--logdir logger/nonlin-poisson \
--expname jitter-test \
--batch_size 1 \
--out_features 3 \
--in_features 12"

params="$homography_params $logger_params $noise_params $exp_params"
echo $params