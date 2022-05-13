#!/bin/bash
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/root/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/root/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/root/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/root/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
export XLA_PYTHON_CLIENT_PREALLOCATE=False
# <<< conda initialize <<<
conda env create -f req.yml
conda activate deepfnf
cp /root/ssh_mount/id_rsa* /root/.ssh/
chmod 400 ~/.ssh/id_rsa
apt-get update
pip3 install imageio
sudo apt-get -y install exiftool
pip3 install PyExifTool piq lpips plotly==5.6.0 pandas kaleido jinja2
pip3 install --upgrade pip
pip3 install setuptools
pip3 install imageio tensorflow-gpu==1.13.1 scikit-image==0.16.2 tqdm PyExifTool piq lpips plotly==5.6.0 pandas kaleido
python3 -c """import imageio
imageio.plugins.freeimage.download()
"""
cd /mshvol2/users/mohammad/optimization/DifferentiableSolver
export PYTHONPATH=`pwd`:/mshvol2/users/mohammad/cvgutils/
echo command:
echo $@
python3 $@
