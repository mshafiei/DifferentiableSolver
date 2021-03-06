#!/bin/bash
ngpus=1
ncpus=1
meml=22G
memr=20G

server_path=/mshvol2/users/mohammad/optimization/DifferentiableSolver
cmd1="python deploy --image docker.io/mohammadsh/deeplearning:latest --priority $3 --key kub --name $2 --ngpus $ngpus --cpu $ncpus --meml $meml --memr $memr --type job_deeplearning --cmd"
cmd2="$server_path/run.sh $1"
echo $cmd1 "$cmd2"

cd /home/mohammad/Projects/cvgutils/cluster_control/deployments/
$cmd1 "$cmd2"