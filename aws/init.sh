#!/bin/bash -x

metadata_url=http://169.254.169.254/latest/meta-data/

function num_cores {
    grep -c ^processor /proc/cpuinfo 
}

function init_ami {
    (
        cd 
        cp -r proj/misc/dotfiles/.[a-z][a-z]* .
    )
    git config --global user.name "Nicholas Dronen"
    git config --global user.email ndronen@gmail.com

    git clone git://github.com/kennethreitz/autoenv.git ~/.autoenv
    if [[ $(curl $metadata_url/instance-type) == g* ]]
    then
        echo "This is an Ubuntu instance; GPU support unlikely" >2
    fi
}

function init_ubuntu_ami {
    sudo apt-get install -y gcc g++ libgfortran3 liblapack3 parallel 
    sudo apt-get install -y graphicsmagick python-pgmagick python-dev
    sudo apt-get install -y git
    init_ami
}

function init_amazon_ami {
    sudo vi /etc/yum.repos.d/epel.repo
    sudo yum install -y gcc gcc-c++ libgfortran lapack lapack-devel 
    ( 
        cd /tmp
        wget http://git.savannah.gnu.org/cgit/parallel.git/plain/src/parallel
        chmod 755 parallel
        sudo mv parallel /usr/bin
    )
    sudo yum install -y GraphicsMagick GraphicsMagick-devel
    sudo yum install -y GraphicsMagick-c++ GraphicsMagick-c++-devel
    sudo yum install -y python-devel boost-python boost-devel
    sudo yum install -y git numpy scipy python-sphinx python-pygments

    sudo easy_install pip
    sudo pip install pgmagick
    sudo pip install scikit-learn
    
    sudo groupmod --gid 1000 ec2-user
    sudo usermod --uid 1000 ec2-user

    init_ami
}

if [[ $(curl $metadata_url/ami-id) == ami-9c13ecf4 ]]
then
    init_amazon_ami
else
    init_ubuntu_ami
fi

# These are commands to run to set up GPU drivers in Ubuntu.  They might
# be outdated.  Even if they're not, it appears that saving the boot
# volume and turning it into an AMI isn't guaranteed to work.  I did so
# and the first few instances I booted using my AMI booted without error.
# Subsequent instances wouldn't boot far enough for anything to be
# registered in the kernel error log.  So caveat emptor.
# wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1204/x86_64/cuda-repo-ubuntu1204_5.5-0_amd64.deb
# sudo dpkg -i cuda-repo-ubuntu1204_5.5-0_amd64.deb
# sudo apt-get update
# sudo apt-get install cuda
