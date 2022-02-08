#!/bin/bash
name=msh-autodiff-overfitunet
ngpus=1
ncpus=2
meml=15G
memr=12G
scriptFn="/mshvol2/users/mohammad/optimization/DifferentiableSolver/run.sh --model overfit_unet --expname "

cd /home/mohammad/Projects/cvgutils/cluster_control/deployments/
python deploy --name $name \
--ngpus $ngpus --cpu $ncpus --meml $meml --memr $memr --type job_deeplearning \
--cmd "$scriptFn"