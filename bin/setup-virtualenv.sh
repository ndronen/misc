#!/bin/bash -e

function install_package {
    pkg=$1
    dpkg -l $pkg | grep -q ^ii 
    if [ $? == 0 ]
    then
        echo $pkg is installed
    else
        echo sudo apt-get install $pkg
    fi
}

source bin/activate

echo "using pip "$(which pip)

if [ "$(which pip)" == "/usr/bin/pip" ]
then
    echo "system pip $(which pip) is inappropriate for virtual envs" >&2
    exit 1
fi

install_package liblapack-dev
install_package gfortran

export LAPACK=/usr/lib/liblapack.so
export ATLAS=/usr/lib/libatlas.so
export BLAS=/usr/lib/libblas.so

pip install numpy
pip install scipy
pip install joblib
pip install pandas
pip install ipython

pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.7.1-cp34-none-linux_x86_64.whl
