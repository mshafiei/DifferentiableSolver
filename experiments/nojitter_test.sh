#!/bin/bash
source ./experiments/homography_params_static.sh
source ./experiments/logger_params_train_filesystem.sh
source ./experiments/noise_params_deepfnf.sh
exp_params="\
--mode train \
--model unet \
--TLIST data/train_1600.txt \
--logdir logger/unit-tests \
--expname jitter-test \
--batch_size 1 \
--store_params"



scriptFn="unit_test/jitter_test.py $exp_params $homography_params $logger_params $noise_params"

./experiments/run_local.sh "$scriptFn $name"
# ./run_server.sh $scriptFn $name