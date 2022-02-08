#!/bin/bash
name=msh-autodiff-interpolation
ngpus=1
ncpus=2
meml=15G
memr=12G
scriptFn=/mshvol2/users/mohammad/optimization/DifferentiableSolver/run.sh

cd /home/mohammad/Projects/cvgutils/cluster_control/deployments/
python deploy --name $name \
--ngpus $ngpus --cpu $ncpus --meml $meml --memr $memr --type job_deeplearning \
--cmd "$scriptFn"