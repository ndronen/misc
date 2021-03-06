export PS1='[\u@\d \h \t \w]
$ '
export CVS_RSH=ssh
export EDITOR=vim
export VISUAL=$EDITOR
export LESS="-erX"
export TZ=America/New_York
export PYTHONHASHSEED=0
if [[ "$OSTYPE" == "darwin"* ]]
then
    export MAKEFLAGS="-j$(sysctl -n hw.physicalcpu)"
else
    export MAKEFLAGS="-j$(nproc)"
fi

export PATH=/usr/local/bin:$PATH:~/proj/misc/bin:~/workspace/misc/bin/
export PATH=$PATH:/usr/local/sbin:~/.local/bin:/usr/local/texlive/2019/bin/x86_64-darwin

if [[ "$(uname)" == Darwin ]]
then
    export PATH="/usr/local/opt/coreutils/libexec/gnubin:$PATH"
    export PATH="$PATH:/Library/TeX/texbin"
    export MANPATH="/usr/local/man:/usr/local/opt/coreutils/libexec/gnuman:$MANPATH"
elif [[ "$(uname)" == Linux ]]
then
    export PATH="$PATH:/usr/local/cuda/bin"
fi

export PYTHONUNBUFFERED=1

export TF_CPP_MIN_LOG_LEVEL=3

#. ~/.zshrc
