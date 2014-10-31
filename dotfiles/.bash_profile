export PS1='[\u@\d \h \t \w]
$ '
export CVS_RSH=ssh
export CVSROOT=cvshost:/usr/local/cvsroot
export EDITOR=vim
export VISUAL=$EDITOR
export LESS="-erX"
export TZ=America/Denver

export PERL5LIB=~/proj
export PYTHONPATH=~/proj/Theano:~/proj/pylearn2:~/proj/Python/trunk/pykt:~/proj/jobman:~/proj/pylearnutils:~/proj/Python/trunk/pykt:~/proj/GroundHog

export R_HISTFILE=~/.Rhistory

export PATH=/usr/local/bin:~/proj/pylearn2/pylearn2/scripts:$PATH
export PATH=~/proj/misc/bin:~/proj/jobman/bin:~/proj/Theano/bin:$PATH

. ~/.bashrc
