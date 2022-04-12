#!/bin/bash
ngpus=1
ncpus=1
meml=22G
memr=20G
server_path=/mshvol2/users/mohammad/optimization/DifferentiableSolver/

cd /home/mohammad/Projects/cvgutils/cluster_control/deployments/
python deploy --name $name \
--ngpus $ngpus --cpu $ncpus --meml $meml --memr $memr --type job_deeplearning \
--cmd "$server_path/run.sh $scriptFn"
