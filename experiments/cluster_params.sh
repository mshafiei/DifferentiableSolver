#!/bin/bash
ngpus=1
ncpus=1
meml=22G
memr=20G

local=1
if [ $local == 1 ]
then
server_path=./
else
server_path=/mshvol2/users/mohammad/optimization/DifferentiableSolver/
fi

# name=msh-deep-nonlin-screen-poisson-generalize-nojitter-fixed-train
# scriptFn="unet_test/implicit_nonlin_screen_poisson.py --mode train --val_freq 1001 --model implicit_sanity_model --TLIST data/train_1600.txt --display_freq 1000 --logdir logger/nonlin-poisson-nojitter-fixed --expname unet-generalize --save_param_freq 10000 --batch_size 1 --logger tb --displacement 0 --min_scale 1. --max_scale 1. --max_rotate 0 --min_alpha 0.02 --max_alpha 0.2 --min_read -3 --max_read -2 --min_shot -2 --max_shot -1.3 --nlin_iter 100 --nnonlin_iter 3 --out_features 3 --in_features 12"

if [ $local == 1 ]
then
python3 $scriptFn
else
cd /home/mohammad/Projects/cvgutils/cluster_control/deployments/
python deploy --name $name \
--ngpus $ngpus --cpu $ncpus --meml $meml --memr $memr --type job_deeplearning \
--cmd "$server_path/run.sh $scriptFn"
fi