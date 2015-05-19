#!/bin/bash -e 

cd ~/proj

if [ -d torch ]
then
    ( cd torch && git pull )
else
    git clone https://github.com/torch/distro.git ~/torch --recursive
fi

cd torch
./install.sh

cd ~/
ln -sf ~/proj/torch
