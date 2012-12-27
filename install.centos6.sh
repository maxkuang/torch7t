#/bin/bash

INSTALL=(torch7 sys xlua nnx parallel optim mattorch image)

# DEPENDENCY
# sudo yum install cmake
# sudo yum install zeromq-devel
# (http://download.opensuse.org/repositories/home:/fengshuo:/zeromq/CentOS_CentOS-6/home:fengshuo:zeromq.repo)

# SETUP
ROOT=`pwd`
ROOT_REPO=$ROOT/repo
ROOT_SRC=$ROOT/src
DEST=~/local

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
#	printf "RE-INSTALL %s from %s ...\n" $item $SRC
#	eval $(printf "install_%s" $item)
    fi
done

# ENV 
ENV_BIN=$DEST/bin
ENV_LIB=$DEST/lib
if [ -d "$ENV_BIN" ] && [[ ":$PATH:" != *":$ENV_BIN:"* ]]; then
#    if [ -f "~/.bashrc" ]; then
#	pattern="export PATH=\$PATH:$BIN"
#	result=`grep -i "$pattern" ~/.bashrc` # TODO check duplication
#	echo $result;
#    else
    echo "export PATH=\$PATH:$ENV_BIN" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$ENV_LIB" >>~/.bashrc	
#    fi
fi

# DONE
cat <<DONE

Good, Congratulations ! 
PLEASE run "source ~/.bashrc" under current shell to activate the new PATH
DONE
