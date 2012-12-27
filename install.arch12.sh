#/bin/bash

INSTALL=(torch7 sys xlua nnx parallel optim mattorch)

# DEPENDENCY
#sudo pacman -Syy
#sudo pacman -S cmake
####https://aur.archlinux.org/packages/zeromq-dev/
#mkdir -p 3rd
#cd 3rd
#wget https://aur.archlinux.org/packages/ze/zeromq-dev/zeromq-dev.tar.gz
#tar xf zeromq-dev.tar.gz
#cd zeromq-dev
#makepkg
#sudo pacman -U zeromq-dev-3.2.0-1-i686.pkg.tar.xz 

# SETUP
ROOT=`pwd`
ROOT_REPO=$ROOT/repo
ROOT_SRC=$ROOT/src
DEST=~/local/torch7t

# REPO
git submodule init
git submodule update

# INSTALL
source $ROOT/install.repo.sh
for item in ${INSTALL[*]}
do
    SRC=$ROOT_SRC/$item
    if [ ! -d $SRC ]; then
        printf "INSTALLING %s from %s ...\n" $item $SRC
        eval $(printf "install_%s" $item)
    else
        printf "ALREADY INSTALLED %s from %s ...\n" $item $SRC
#       printf "RE-INSTALL %s from %s ...\n" $item $SRC
#       eval $(printf "install_%s" $item)
    fi
done

# ENV
ENV_BIN=$DEST/bin
ENV_LIB=$DEST/lib
if [ -d "$ENV_BIN" ] && [[ ":$PATH:" != *":$ENV_BIN:"* ]]; then
    echo "export PATH=\$PATH:$ENV_BIN" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$ENV_LIB" >>~/.bashrc
fi

# DONE
cat <<DONE

Good, Congratulations !
PLEASE run "source ~/.bashrc" under current shell to activate the new PATH
DONE

