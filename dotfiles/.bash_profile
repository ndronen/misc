export PS1='[\u@\d \h \t \w]
$ '
export EDITOR=vim
export VISUAL=$EDITOR
export LESS="-erX"
export TZ=America/Denver

export PYTHONPATH=~/proj/Theano:~/proj/pylearn2:~/proj/Python/trunk/pykt
export PYLEARN2_DATA_PATH=~/proj/pylearn2-data
export R_HISTFILE=~/.Rhistory
export PATH=/usr/local/bin:~/proj/pylearn2/pylearn2/scripts:~/proj/misc/bin:$PATH

export PYENV_ROOT=/usr/local/opt/pyenv  
eval "$(pyenv init -)"
eval "$(pyenv virtualenv init -)"

. ~/.bashrc
