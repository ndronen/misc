export PS1='[\u@\d \h \t \w]
$ '
export CVS_RSH=ssh
export EDITOR=vim
export VISUAL=$EDITOR
export LESS="-erX"
export TZ=America/Denver
export PYTHONHASHSEED=0
export R_HISTFILE=~/.Rhistory

if [ "$(uname)" == Darwin ]
then
    export PATH="/usr/local/bin:/usr/local/opt/coreutils/libexec/gnubin:$PATH"
    export PATH="$PATH:/Library/TeX/texbin"
    export MANPATH="/usr/local/man:/usr/local/opt/coreutils/libexec/gnuman:$MANPATH"
elif [ "$(uname)" == Linux ]
then
    export PATH="$PATH:/usr/local/cuda/bin"
    export LD_LIBRARY_PATH="/usr/local/cuda-7.0/lib64:$LD_LIBRARY_PATH"
fi

export PATH=/usr/local/bin:$PATH:~/proj/misc/bin
export PATH=$PATH:/usr/local/sbin

if [ -e /usr/share/virtualenvwrapper/virtualenvwrapper.sh ]
then
    export WORKON_HOME=~/proj/envs/
elif [ -e /usr/local/bin/virtualenvwrapper.sh ]
then
    export WORKON_HOME=~/proj/envs/
fi

export PYTHONUNBUFFERED=1

which nvcc >/dev/null
if [ $? -eq 0 ]
then
    export THEANORC=~/.theanorc.gpu
else
    export THEANORC=~/.theanorc.cpu
fi

. ~/.bashrc
