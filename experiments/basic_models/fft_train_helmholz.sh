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
--expname fft_helmholze-3 \
--batch_size 1 \
--out_features 6 \
--in_features 12 \
--min_delta -1.0502254"
# --min_delta -0.19587028"
# --min_delta -2.252168"
# --min_delta -9.210175" 
# --min_delta -15.942385"
# jnp.log(jnp.exp(jnp.double(3e-1)) - 1)
# DeviceArray(-1.0502254, dtype=float32)
# jnp.log(jnp.exp(jnp.double(6e-1)) - 1)
# DeviceArray(-0.19587028, dtype=float32)
# jnp.log(jnp.exp(jnp.double(1e-1)) - 1)
# DeviceArray(-2.252168, dtype=float32)
# jnp.log(jnp.exp(jnp.double(1e-4)) - 1)
# DeviceArray(-9.210175, dtype=float32)
# jnp.log(jnp.exp(jnp.double(1e-7)) - 1)
# DeviceArray(-15.942385, dtype=float32)
name=msh-fft-solver-train-helmholze-3
scriptFn="unet_test/implicit_nonlin_screen_poisson.py $exp_params $homography_params $logger_params $noise_params $solver_params"

# ./experiments/run_local.sh "$scriptFn"
./experiments/run_server.sh "$scriptFn" "$name"
