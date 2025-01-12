#! /bin/bash

sudo apt-get install libglfw3-dev libboost-all-dev libeigen3-dev libpcl-dev libopencv-dev
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh

conda env create -f environment.yaml
cd ./easy_pbr
cmake --version || sudo snap install cmake --classic || sudo apt-get install cmake
make