export PS1='[\u@\d \h \t \w]
$ '
export CVS_RSH=ssh
export CVSROOT=cvshost:/usr/local/cvsroot
export EDITOR=vim
export VISUAL=$EDITOR
export LESS="-erX"
export TZ=America/Denver

export PERL5LIB=~/proj
export PYTHONPATH=~/proj/Theano:~/proj/pylearn2:~/proj/Python/trunk/pykt:~/proj/jobman:~/proj/pylearnutils:~/proj/Python/trunk/pykt:~/proj/GroundHog:~/proj/pywsd

export R_HISTFILE=~/.Rhistory

export PATH=/usr/local/bin:~/proj/pylearn2/pylearn2/scripts:$PATH:/usr/local/cuda-6.5/bin
export PATH=~/proj/misc/bin:~/proj/jobman/bin:~/proj/Theano/bin:$PATH:~/proj/pylearnutils/pylearnutils/bin
export LD_LIBRARY_PATH=/usr/local/cuda-6.5/lib64

. ~/.bashrc
