#/bin/bash

sudo apt-get install cmake

git submodule init
git submodule update

mkdir ~/local/
cd ./repo/torch
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=~/local/
make install

