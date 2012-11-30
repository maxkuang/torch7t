#/bin/bash

ROOT=`pwd`

# 3rd
sudo yum install cmake
# YUM REPO
# http://download.opensuse.org/repositories/home:/fengshuo:/zeromq/CentOS_CentOS-6/home:fengshuo:zeromq.repo
sudo yum install zeromq-devel

# src
git submodule init
git submodule update

REPO=$ROOT/repo
PKG=$ROOT/src
rm -rf $PKG
mkdir -p $PKG
# torch7
cp -r $REPO/torch $PKG/torch7
# sys
cp -r $REPO/sys $PKG/sys
# nnx
cp -r $REPO/nnx $PKG/nnx
# parallel
cp -r $REPO/parallel $PKG/parallel
# optim
cp -r $REPO/optim $PKG/optim
cp $REPO/optim2/vsgd.lua $PKG/optim/vsgd.lua
echo "-- supp" >> $PKG/optim/init.lua
echo "torch.include('optim', 'vsgd.lua')" >> $PKG/optim/init.lua
cp $REPO/optim2/test/noisyl2.lua $PKG/optim/test/noisyl2.lua
cp $REPO/optim2/test/test_vsgd.lua $PKG/optim/test/test_vsgd.lua

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
SRC=$PKG/torch7
#GO

# sys
SRC=$PKG/sys
#GO

# nnx
SRC=$PKG/nnx
#GO

# parallel
SRC=$PKG/parallel
#GO

# optim
SRC=$PKG/optim
GO

# DONE
cat <<DONE

Good, Congratulations ! 
PLEASE run "source ~/.bashrc" under current shell to activate the new PATH
DONE
