#!/bin/bash
set -exu

sudo apt-get -y install python3
sudo apt-get -y install python3-pip
sudo apt -y install libjpeg-dev zlib1g-dev
pip3 install matplotlib
pip3 install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install tensorboard==1.15.0
