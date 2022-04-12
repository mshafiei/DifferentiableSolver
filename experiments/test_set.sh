#!/bin/bash
source ./experiments/homography_params_static.sh
source ./experiments/logger_params_train_tb.sh
source ./experiments/noise_params_deepfnf.sh
exp_params="\
--logdir logger/dataset \
--expname nojitter-testset \
--batch_size 1"


scriptFn="./gen_valset_trainset_organized.py $exp_params $homography_params $logger_params $noise_params"

./experiments/run_local.sh $scriptFn $name
# ./run_server.sh $scriptFn $name