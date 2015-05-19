#!/bin/bash -e

cd ~/proj

if [ -d nn ]
then
    ( cd nn && git pull )
else
    git clone https://github.com/torch/nn 
fi
    
cd nn && git checkout getParamsByDevice && luarocks make rocks/nn-scm-1.rockspec

if [ -d fbtorch ]
then
    ( cd fbtorch && git pull )
else
    git clone https://github.com/facebook/fbtorch.git
fi
cd fbtorch && luarocks make rocks/fbtorch-scm-1.rockspec

if [ -d fbnn ]
then
    ( cd fbnn && git pull )
else
    git clone https://github.com/facebook/fbnn.git
fi
cd fbnn && luarocks make rocks/fbnn-scm-1.rockspec

if [ -d fbcunn ]
then
    ( cd fbcunn && git pull )
else
    git clone https://github.com/facebook/fbcunn.git
fi
cd fbcunn && luarocks make rocks/fbcunn-scm-1.rockspec
