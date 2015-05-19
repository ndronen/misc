#!/bin/bash -e

dir=$(mktemp --tmpdir -d thpp-build.XXXXXX)
cd $dir
git clone https://github.com/facebook/thpp
cd thpp/thpp
./build.sh
cd /tmp
sudo rm -fr $dir
