#!/bin/bash -e

dir=$(mktemp --tmpdir -d fbthrift-build.XXXXXX)
cd $dir
git clone https://github.com/facebook/fbthrift

cd fbthrift/thrift
autoreconf -ivf
./configure
make -j8
sudo make install
cd /tmp
sudo rm -fr $dir
