#!/bin/bash
pip3 install imageio
python3 -c """import imageio
imageio.plugins.freeimage.download()
"""
cd /mshvol2/users/mohammad/optimization/DifferentiableSolver
export PYTHONPATH=`pwd`:/mshvol/users/mohammad/cvgutils/
python3 Flash_No_Flash/train.py
