#/bin/bash

GO() {
    mkdir -p $DEST
    cd $SRC
    mkdir -p build
    cd build
    rm -r *
    cmake .. -DCMAKE_INSTALL_PREFIX=$DEST
    make install
}

install_torch7() {
    # repo -> src
    rm -r $SRC
    mkdir -p $SRC
    cp -r $ROOT_REPO/torch/* $SRC
    # install
    GO
}

install_sys() {
    # repo -> src
    rm -r $SRC
    mkdir -p $SRC
    cp -r $ROOT_REPO/sys/* $SRC
    # install
    GO
}

install_xlua() {
    # repo -> src
    rm -r $SRC
    mkdir -p $SRC
    cp -r $ROOT_REPO/xlua/* $SRC
    # install
    GO
}

install_nnx() {
    # repo -> src
    rm -r $SRC
    mkdir -p $SRC
    cp -r $ROOT_REPO/nnx/* $SRC
    # install
    GO
}

install_parallel() {
    # repo -> src
    rm -r $SRC
    mkdir -p $SRC
    cp -r $ROOT_REPO/parallel/* $SRC
    # install
    GO
}

install_optim() {
    # repo -> src
    rm -r $SRC
    mkdir -p $SRC
    cp -r $ROOT_REPO/optim/* $SRC
    cp $ROOT_REPO/optim2/vsgd.lua $SRC/vsgd.lua   
    cp $ROOT_REPO/optim2/test/noisyl2.lua $SRC/test/noisyl2.lua
    cp $ROOT_REPO/optim2/test/test_vsgd.lua $SRC/test/test_vsgd.lua
    echo "-- register" >> $SRC/init.lua
    echo "torch.include('optim', 'vsgd.lua')" >> $SRC/init.lua
    GO
}
