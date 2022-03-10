#!/bin/bash
cp /root/ssh_mount/id_rsa* /root/.ssh/
chmod 400 ~/.ssh/id_rsa
pip3 install imageio
sudo apt-get -y install exiftool
pip3 install PyExifTool piq
python3 -c """import imageio
imageio.plugins.freeimage.download()
"""
cd /mshvol2/users/mohammad/optimization/DifferentiableSolver
export PYTHONPATH=`pwd`:/mshvol2/users/mohammad/cvgutils/
echo command:
echo $@
python3 $@
