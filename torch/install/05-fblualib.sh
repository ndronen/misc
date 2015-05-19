#!/bin/bash -e

dir=$(mktemp --tmpdir -d fblualib-build.XXXXXX)
cd $dir
git clone https://github.com/soumith/fblualib
cd fblualib/fblualib
./build.sh
cd /tmp
sudo rm -fr $dir
