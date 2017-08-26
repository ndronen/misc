export PS1='[\u@\d \h \t \w]
$ '
export CVS_RSH=ssh
export EDITOR=vim
export VISUAL=$EDITOR
export LESS="-erX"
export TZ=America/Denver
export PYTHONHASHSEED=0
export R_HISTFILE=~/.Rhistory
export HAD_ML_API_KEY=88znbhNZdNhqeVt3

export PATH=/usr/local/bin:$PATH:~/proj/misc/bin
export PATH=$PATH:/usr/local/sbin:~/.local/bin

if [ "$(uname)" == Darwin ]
then
    export PATH="/usr/local/opt/coreutils/libexec/gnubin:$PATH"
    export PATH="$PATH:/Library/TeX/texbin"
    export PATH="$HOME/conda/miniconda3/bin:$PATH"
    export MANPATH="/usr/local/man:/usr/local/opt/coreutils/libexec/gnuman:$MANPATH"
    export PYTHONPATH="/Users/dronen/proj/had_sp_segmentation"
elif [ "$(uname)" == Linux ]
then
    export PATH="$PATH:/usr/local/cuda/bin"
    export PATH="$HOME/conda/miniconda3/bin:$PATH"
    export PYTHONPATH="$HOME/proj/had_sp_segmentation"
    #export PYTHONPATH="$PYTHONPATH:$HOME/proj/pyvision/src"
    #export PYTHONPATH="$PYTHONPATH:$HOME/proj/opencv/release/lib/python3"
fi

export PYTHONUNBUFFERED=1

if [ $? -eq 0 ]
then
    export THEANORC=~/.theanorc.gpu
else
    export THEANORC=~/.theanorc.cpu
fi

. ~/.bashrc
