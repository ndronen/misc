#!/bin/sh

sudo apt-get install git apt-file
sudo apt-file update
sudo apt-get install gcc g++ libgfortran3 liblapack3 parallel 
sudo apt-get install graphicsmagick python-pgmagick

git config --global user.name "Nicholas Dronen"
git config --global user.email ndronen@gmail.com

git clone git://github.com/kennethreitz/autoenv.git ~/.autoenv
