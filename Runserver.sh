#!/bin/bash
name=msh-autodiff-interpolateunet
ngpus=1
ncpus=2
meml=15G
memr=12G
scriptFn="/mshvol2/users/mohammad/optimization/DifferentiableSolver/run.sh --model interpolate_unet --expname interpolate_unet"

cd /home/mohammad/Projects/cvgutils/cluster_control/deployments/
python deploy --name $name \
--ngpus $ngpus --cpu $ncpus --meml $meml --memr $memr --type job_deeplearning \
--cmd "$scriptFn"