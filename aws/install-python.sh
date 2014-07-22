#!/bin/bash

if [ $# -ne 1 ]
then
    echo "usage "$(basename $0)": PYTHON_VERSION" >&2
    exit 1
fi

version=$1

tmp=$(mktemp -d)
cd $tmp
wget http://www.python.org/ftp/python/$version/Python-$version.tgz
tar -zxvf Python-$version.tgz
cd Python-$version
./configure --prefix=/home/ubuntu/proj/Python-$version
make -j 8
make install
