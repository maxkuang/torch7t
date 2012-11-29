#/bin/bash

# 3rd
sudo apt-get install cmake
sudo apt-get install libzmq-dev
# src
git submodule init
git submodule update

# installer 
GO() {
        mkdir -p $DEST
        cd $SRC
        mkdir -p build
        cd build
        cmake .. -DCMAKE_INSTALL_PREFIX=$DEST
        make install
}

ROOT=`pwd`
DEST=~/local
BIN=$DEST/bin
if [ -d "$BIN" ] && [[ ":$PATH:" != *":$BIN:"* ]]; then
        echo "export PATH=\$PATH:$BIN" >> ~/.bashrc
fi

# torch
SRC=$ROOT/repo/torch
GO

# sys
SRC=$ROOT/repo/sys
GO

# nnx
SRC=$ROOT/repo/nnx
GO

# parallel
SRC=$ROOT/repo/parallel
GO

# optim
SRC=$ROOT/repo/optim
GO

# DONE
cat <<DONE

Okay, Congratulations ! 
PLEASE put "export PATH=\$PATH:$BIN" in ~/.bashrc, and run "source ~/.bashrc" under current shell to activate the new PATH
DONE
