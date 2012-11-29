#/bin/bash

# 3rd
sudo yum install cmake
#http://download.opensuse.org/repositories/home:/fengshuo:/zeromq/CentOS_CentOS-6/home:fengshuo:zeromq.repo
sudo yum install zeromq-devel

# src
git submodule init
git submodule update

# installer 
GO() {
        mkdir -p $DEST
        cd $SRC
        mkdir -p build
        cd build
	rm -rf *
        cmake .. -DCMAKE_INSTALL_PREFIX=$DEST
        make install
}

ROOT=`pwd`
DEST=~/local
BIN=$DEST/bin
LIB=$DEST/lib
if [ -d "$BIN" ] && [[ ":$PATH:" != *":$BIN:"* ]]; then
        echo "export PATH=\$PATH:$BIN" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$LIB" >>~/.bashrc
	export PATH=$PATH:$BIN
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIB
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

Good, Congratulations ! 
PLEASE run "source ~/.bashrc" under current shell to activate the new PATH
DONE
