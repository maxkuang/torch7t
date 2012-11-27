#/bin/bash
cd ~
mkdir ~/local/
cd src/torch7
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=~/local/
