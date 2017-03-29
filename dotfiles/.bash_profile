export PS1='[\u@\d \h \t \w]
$ '
export CVS_RSH=ssh
export EDITOR=vim
export VISUAL=$EDITOR
export LESS="-erX"
export TZ=America/Denver
export PYTHONHASHSEED=0
export R_HISTFILE=~/.Rhistory
export OPENCV_OPENCL_RUNTIME=null

if [ -e ~/.bash_profile.sync ]
then
    . ~/.bash_profile.sync
fi

if [ "$(uname)" == Darwin ]
then
    export PATH="/usr/local/bin:/usr/local/opt/coreutils/libexec/gnubin:$PATH"
    export PATH="$PATH:/Library/TeX/texbin"
    export MANPATH="/usr/local/man:/usr/local/opt/coreutils/libexec/gnuman:$MANPATH"
    export PYTHONPATH="$HOME/proj/had_sp_segmentation"
    export PYTHONPATH="$PYTHONPATH:$HOME/proj/tfenet"
    export PYTHONPATH="$PYTHONPATH:~/proj/pyvision/src"
    export PYTHONPATH="$PYTHONPATH:/usr/local/lib/python3.5/site-packages"
elif [ "$(uname)" == Linux ]
then
    export PATH="$PATH:/usr/local/cuda/bin"
    export PYTHONPATH="$HOME/proj/had_sp_segmentation"
    export PYTHONPATH="$PYTHONPATH:$HOME/proj/tfenet"
    export PYTHONPATH="$PYTHONPATH:~/proj/pyvision/src"
    export PYTHONPATH="$PYTHONPATH:~/proj/opencv/release/lib"
fi

export PATH=/usr/local/bin:$PATH:~/proj/misc/bin
export PATH=$PATH:/usr/local/sbin:~/.local/bin

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

if [ -e ~/proj/torch/install/bin/torch-activate ]
then
    . ~/proj/torch/install/bin/torch-activate
fi
