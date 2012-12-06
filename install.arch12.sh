#/bin/bash

ROOT=`pwd`

# 3rd
#sudo pacman -Syy
#sudo pacman -S cmake
# https://aur.archlinux.org/packages/zeromq-dev/
#mkdir -p 3rd
#cd 3rd
#wget https://aur.archlinux.org/packages/ze/zeromq-dev/zeromq-dev.tar.gz
#tar xf zeromq-dev.tar.gz
#cd zeromq-dev
#makepkg
#sudo pacman -U zeromq-dev-3.2.0-1-i686.pkg.tar.xz 

# src
cd $ROOT
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
# xlua
cp -r $REPO/xlua $PKG/xlua
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
	echo "append PATH and LD_LIBRARY_PATH .."
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

# xlua
SRC=$PKG/xlua
GO

# nnx
#SRC=$PKG/nnx
#GO

# parallel
SRC=$PKG/parallel
#GO

# optim
SRC=$PKG/optim
#GO

# DONE
cat <<DONE

Good, Congratulations ! 
PLEASE run "source ~/.bashrc" under current shell to activate the new PATH
DONE
