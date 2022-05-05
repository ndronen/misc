export PS1='[\u@\d \h \t \w]
$ '
export CVS_RSH=ssh
export EDITOR=vim
export VISUAL=$EDITOR
export LESS="-erX"
export TZ=America/New_York
export PYTHONHASHSEED=0
export R_HISTFILE=~/.Rhistory
export BASH_SILENCE_DEPRECATION_WARNING=1

export PATH=/usr/local/bin:$PATH:~/proj/misc/bin
export PATH=$PATH:/usr/local/sbin:~/.local/bin


export LDFLAGS="-L/usr/local/opt/openblas/lib"
export CPPFLAGS="-I/usr/local/opt/openblas/include"

export WORKSPACE_DIR=~/workspace
export LMDSK_DIR=$WORKSPACE_DIR/lm-sdk

export DYLD_LIBRARY_PATH=$LMSDK_DIR/lib:$DYLD_LIBRARY_PATH
export __DYLD_LIBRARY_PATH=$LMSDK_DIR/lib:$__DYLD_LIBRARY_PATH
export LMSDK_BLAS_LIBRARY="-lopenblas"
export LMSDK_BLAS_INCLUDE_PATH="/usr/local/opt/openblas/include"
export LMSDK_BLAS_LIBRARY_PATH="/usr/local/opt/openblas/lib"

export __LMCONF_PREFIX=$HOME/.config/lmsdk
export LMSDK_DISABLE_SWIG=true


if [ "$(uname)" == Darwin ]
then
    export PATH="/usr/local/opt/coreutils/libexec/gnubin:$PATH"
    export PATH="$PATH:/Library/TeX/texbin"
    export MANPATH="/usr/local/man:/usr/local/opt/coreutils/libexec/gnuman:$MANPATH"
elif [ "$(uname)" == Linux ]
then
    export PATH="$PATH:/usr/local/cuda/bin"
fi

export PYTHONUNBUFFERED=1

# if running bash
if [ -n "$BASH_VERSION" ]; then
    # include .bashrc if it exists
    if [ -f "$HOME/.bashrc" ]; then
        . "$HOME/.bashrc"
    fi
fi

# set PATH so it includes user's private bin if it exists
if [ -d "$HOME/bin" ] ; then
    PATH="$HOME/bin:$PATH"
fi

# set PATH so it includes user's private bin if it exists
if [ -d "$HOME/.local/bin" ] ; then
    PATH="$HOME/.local/bin:$PATH"
fi
