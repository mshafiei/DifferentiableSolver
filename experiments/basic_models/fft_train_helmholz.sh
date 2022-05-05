#!/bin/bash
source ./experiments/homography_params_static.sh
source ./experiments/logger_params_train_tb.sh
source ./experiments/noise_params_deepfnf.sh
source ./experiments/solver_params.sh
# name=msh-fft-solver-train-helmholze-phi5e1-psi5e1
# helmholz='--delta_phi_init -1.0502254 --delta_psi_init -1.0502254 --expname fft_helmholze-phi5e1-psi5e1'
# name=msh-fft-solver-train-helmholze-phi5e1-psi1e1
# helmholz='--delta_phi_init -1.0502254 --delta_psi_init -2.252168 --expname fft_helmholze-phi5e1-psi1e1'
# name=msh-fft-solver-train-helmholze-phi5e1-psi1e7
# helmholz='--delta_phi_init -1.0502254 --delta_psi_init -15.942385 --expname fft_helmholze-phi5e1-psi1e7'
# name=msh-fft-solver-train-helmholze-phi1e1-psi5e1
# helmholz='--delta_phi_init -2.252168 --delta_psi_init -1.0502254 --expname fft_helmholze-phi1e1-psi5e1'
# name=msh-fft-solver-train-helmholze-phi1e1-psi1e1
# helmholz='--delta_phi_init -2.252168 --delta_psi_init -2.252168 --expname fft_helmholze-phi1e1-psi1e1'
# name=msh-fft-solver-train-helmholze-phi1e1-psi1e7
# helmholz='--delta_phi_init 0.1 --delta_psi_init 0.0000001 --expname fft_helmholze-phi1e1-psi1e7'
# name=msh-fft-solver-train-helmholze-phi1e7-psi5e1
# helmholz='--delta_phi_init 0.0000001 --delta_psi_init 0.5 --expname fft_helmholze-phi1e7-psi5e1'
# name=msh-fft-solver-train-helmholze-phi1e7-psi1e1
# helmholz='--delta_phi_init -15.942385 --delta_psi_init -2.252168 --expname fft_helmholze-phi1e7-psi1e1'
# name=msh-fft-solver-train-helmholze-phi1e7-psi1e7
# helmholz='--delta_phi_init -15.942385 --delta_psi_init -15.942385 --expname fft_helmholze-phi1e7-psi1e7'

name=msh-fft-solver-train-helmholze-phi1-psi1e2-relu-128
helmholz='--delta_phi_init 1. --delta_psi_init 0.01 --expname fft_helmholze-phi1-psi1e2-128'
# name=msh-fft-solver-train-helmholze-phi1e2-psi1-relu-128
# helmholz='--delta_phi_init 0.01 --delta_psi_init 1. --expname fft_helmholze-phi1e2-psi1-128'
# name=msh-fft-solver-train-helmholze-phi1-psi1-relu-128
# helmholz='--delta_phi_init 1. --delta_psi_init 1. --expname fft_helmholze-phi1-psi1-128'
# name=msh-fft-solver-train-helmholze-phi1-psi1-fixed-relu-128
# helmholz='--delta_phi_init 1. --delta_psi_init 1. --fixed_delta --expname fft_helmholze-phi1-psi1-fixed-128'

exp_params="\
--mode train \
--model fft_helmholz \
--TLIST data/train.txt \
--TESTPATH data/testset_nojitter \
--logdir logger/fft_solver_largeds_relu-128 \
--batch_size 1 \
--out_features 6 \
--thickness 128 \
--in_features 12 \
--activation relu"

priority='nice'



# name=msh-fft-solver-train-helmholze-3
scriptFn="unet_test/implicit_nonlin_screen_poisson.py $exp_params $homography_params $logger_params $noise_params $solver_params $helmholz"

# ./experiments/run_local.sh "$scriptFn"
./experiments/run_server.sh "$scriptFn" "$name" "$priority"
