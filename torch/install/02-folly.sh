#!/bin/bash -e

dir=$(mktemp --tmpdir -d folly-build.XXXXXX)
cd $dir
git clone https://github.com/facebook/folly
cd folly/folly
autoreconf -ivf
./configure
make
sudo make install
sudo ldconfig
sudo ldconfig -v | grep folly
cd /tmp
sudo rm -fr $dir
