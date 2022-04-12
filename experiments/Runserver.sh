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

# scriptFn="$server_path/run.sh Flash_No_Flash/train.py --model overfit_unet --TLIST data/train.txt --expname unet_overfit_1 --save_param_freq 10000 --batch_size 1 --logger tb --displacement 0 --min_scale 0.99999 --max_scale 1.000001 --max_rotate 0"
# name=msh-autodiff-overfit-1

# scriptFn="$server_path/run.sh Flash_No_Flash/train.py --model overfit_unet --TLIST data/train_1600.txt --expname unet_generalize --save_param_freq 1000 --batch_size 25 --logger tb"
# name=msh-autodiff-overfit-1600

# name=msh-autodiff-overfit-interpolate-1600
# scriptFn="$server_path/run.sh Flash_No_Flash/train.py --model interpolate_unet --TLIST data/train_1600.txt --expname unet_generalize_interpolate --save_param_freq 1000 --batch_size 25 --logger tb"

# name=msh-deep-regularizer-overfit-2
# scriptFn="$server_path/run.sh Flash_No_Flash/linear_denoising_convnet_3features_relu_structured.py --model UNet_Hardcode --TLIST data/train.txt --expname UNet_Hardcode_overfit --save_param_freq 10000 --batch_size 1 --logger tb --displacement 0 --min_scale 0.99999 --max_scale 1.000001 --max_rotate 0"

# name=msh-deep-regularizer-hardcode-2
# scriptFn="$server_path/run.sh Flash_No_Flash/train_generalize.py --model UNet_Hardcode --TLIST data/train.txt --display_freq 1000 --expname unet_overfit_hardcode --save_param_freq 10000 --batch_size 1 --logger tb --displacement 0 --min_scale 0.99999 --max_scale 1.000001 --max_rotate 0 --min_alpha 0.2 --max_alpha 0.2 --min_read -2 --max_read -2 --min_shot -2 --max_shot -2"

# name=msh-deep-jax-image
# scriptFn="$server_path/run.sh unet_test/unet_jax.py  --val_freq 101 --model overfit_unet --TLIST data/train_1600.txt --display_freq 100 --expname unvet_overfit_images_rand --save_param_freq 10000 --batch_size 4 --logger tb --displacement 0 --min_scale 0.99999 --max_scale 1.000001 --max_rotate 0 --min_alpha 0.2 --max_alpha 0.2 --min_read -2 --max_read -2 --min_shot -2 --max_shot -2"

# name=msh-deep-jax-alpha-images-rand
# scriptFn="$server_path/run.sh unet_test/unet_jax.py --val_freq 101 --model overfit_unet --TLIST data/train_1600.txt --display_freq 100 --expname unvet_overfit_alpha_images_rand --save_param_freq 10000 --batch_size 4 --logger tb --displacement 0 --min_scale 0.99999 --max_scale 1.000001 --max_rotate 0 --min_alpha 0.02 --max_alpha 0.2 --min_read -2 --max_read -2 --min_shot -2 --max_shot -2 --out_features 3 --in_features 3"

# name=msh-deep-nonlin-screen-poisson-overfit
# scriptFn="$server_path/run.sh unet_test/implicit_nonlin_screen_poisson.py --val_freq 10001 --model overfit_unet --TLIST data/train.txt --display_freq 10000 --logdir logger/nonlin-poisson --expname unet-overfit --save_param_freq 10000 --batch_size 1 --logger tb --displacement 0 --min_scale 0.99999 --max_scale 1.000001 --max_rotate 0 --min_alpha 0.02 --max_alpha 0.2 --min_read -2 --max_read -2 --min_shot -2 --max_shot -2 --nlin_iter 100 --nnonlin_iter 3 --out_features 3 --in_features 12"

# name=msh-deep-deriv-poisson-generalize
# scriptFn="unet_test/implicit_nonlin_screen_poisson.py --val_freq 1001 --model implicit_poisson_model --TLIST data/train_1600.txt --display_freq 1000 --logdir logger/nonlin-poisson --expname unet-poisson-deriv --save_param_freq 10000 --batch_size 1 --logger tb --displacement 0 --min_scale 1. --max_scale 1. --max_rotate 0 --min_alpha 0.02 --max_alpha 0.2 --min_read -3 --max_read -2 --min_shot -2 --max_shot -1.3 --nlin_iter 100 --nnonlin_iter 3 --out_features 6 --in_features 12"

# name=msh-deep-nonlin-screen-poisson-generalize-nojitter-fixed-train
# scriptFn="unet_test/implicit_nonlin_screen_poisson.py --mode train --val_freq 1001 --model implicit_sanity_model --TLIST data/train_1600.txt --display_freq 1000 --logdir logger/nonlin-poisson-nojitter-fixed --expname unet-generalize --save_param_freq 10000 --batch_size 1 --logger tb --displacement 0 --min_scale 1. --max_scale 1. --max_rotate 0 --min_alpha 0.02 --max_alpha 0.2 --min_read -3 --max_read -2 --min_shot -2 --max_shot -1.3 --nlin_iter 100 --nnonlin_iter 3 --out_features 3 --in_features 12"

# name=msh-deep-deriv-poisson-generalize-nojitter-fixed-train
# scriptFn="unet_test/implicit_nonlin_screen_poisson.py --mode train --val_freq 1001 --model implicit_poisson_model --TLIST data/train_1600.txt --display_freq 1000 --logdir logger/nonlin-poisson-nojitter-fixed --expname unet-poisson-deriv --save_param_freq 10000 --batch_size 1 --logger tb --displacement 0 --min_scale 1. --max_scale 1. --max_rotate 0 --min_alpha 0.02 --max_alpha 0.2 --min_read -3 --max_read -2 --min_shot -2 --max_shot -1.3 --nlin_iter 100 --nnonlin_iter 3 --out_features 6 --in_features 12"

# name=msh-deep-deriv-poisson-generalize-nojitter-wb-train
# scriptFn="unet_test/implicit_nonlin_screen_poisson.py --mode train --val_freq 1001 --model implicit_poisson_model --TLIST data/train_1600.txt --display_freq 1000 --logdir logger/nonlin-poisson-nojitter-fixed --expname unet-poisson-deriv-wb --save_param_freq 10000 --batch_size 1 --logger tb --displacement 0 --min_scale 1. --max_scale 1. --max_rotate 0 --min_alpha 0.02 --max_alpha 0.2 --min_read -3 --max_read -2 --min_shot -2 --max_shot -1.3 --nlin_iter 100 --nnonlin_iter 3 --out_features 6 --in_features 12"


# name=msh-deep-unet-generalize-nojitter-1bsz
# scriptFn="unet_test/implicit_nonlin_screen_poisson.py --val_freq 1001 --model unet --TLIST data/train_1600.txt --display_freq 1000 --logdir logger/Unet_test --expname unet-generalize-nojitter-1bsz --save_param_freq 10000 --batch_size 1 --logger tb --displacement 0 --min_scale 1. --max_scale 1. --max_rotate 0 --min_alpha 0.02 --max_alpha 0.2 --min_read -3 --max_read -2 --min_shot -2 --max_shot -1.3 --nlin_iter 100 --nnonlin_iter 3 --out_features 3 --in_features 12"

# name=msh-deep-unet-generalize-nojitter
# scriptFn="unet_test/implicit_nonlin_screen_poisson.py --val_freq 1001 --model unet --TLIST data/train_1600.txt --display_freq 1000 --logdir logger/Unet_test --expname unet-generalize-nojitter --save_param_freq 10000 --batch_size 4 --logger tb --displacement 0 --min_scale 1. --max_scale 1. --max_rotate 0 --min_alpha 0.02 --max_alpha 0.2 --min_read -3 --max_read -2 --min_shot -2 --max_shot -1.3 --nlin_iter 100 --nnonlin_iter 3 --out_features 3 --in_features 12"

# scriptFn="unet_test/implicit_nonlin_screen_poisson.py --mode train --val_freq 1001 --model implicit_sanity_model --TLIST data/train_1600.txt --display_freq 2 --logdir logger/nonlin-poisson --expname jitter-test --save_param_freq 10000 --batch_size 1 --logger tb --displacement 0 --min_scale 1. --max_scale 1. --max_rotate 0 --min_alpha 1 --max_alpha 1 --min_read -3 --max_read -2 --min_shot -2 --max_shot -1.3 --nlin_iter 100 --nnonlin_iter 3 --out_features 3 --in_features 12"
# scriptFn="unet_test/implicit_nonlin_screen_poisson.py --mode train --val_freq 1001 --model unet --TLIST data/train_1600.txt --display_freq 1000 --logdir logger/nonlin-poisson --expname jitter-test --save_param_freq 10000 --batch_size 1 --logger tb --displacement 0 --min_scale 1. --max_scale 1. --max_rotate 0 --min_alpha 1 --max_alpha 1 --min_read -3 --max_read -2 --min_shot -2 --max_shot -1.3 --nlin_iter 100 --nnonlin_iter 3 --out_features 3 --in_features 12"

scriptFn="unet_test/implicit_nonlin_screen_poisson.py --store_params --mode test --val_freq 1001 --model implicit_sanity_model --TLIST data/train_1600.txt --display_freq 1000 --logdir logger/nonlin-poisson-nojitter-fixed --expname unet-generalize --save_param_freq 10000 --batch_size 1 --logger tb --displacement 0 --min_scale 1. --max_scale 1. --max_rotate 0 --min_alpha 0.02 --max_alpha 0.2 --min_read -3 --max_read -2 --min_shot -2 --max_shot -1.3 --nlin_iter 100 --nnonlin_iter 3 --out_features 3 --in_features 12"



if [ $local == 1 ]
then
python3 $scriptFn
else
cd /home/mohammad/Projects/cvgutils/cluster_control/deployments/
python deploy --name $name \
--ngpus $ngpus --cpu $ncpus --meml $meml --memr $memr --type job_deeplearning \
--cmd "$server_path/run.sh $scriptFn"
fi