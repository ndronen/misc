#!/bin/sh

sudo apt-get install git apt-file
sudo apt-file update
sudo apt-get install gcc g++ libgfortran3 liblapack3 parallel 
sudo apt-get install graphicsmagick python-pgmagick

git clone git://github.com/kennethreitz/autoenv.git ~/.autoenv
