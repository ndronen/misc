# For R
export COLUMNS

if [ ! -d ~/.autoenv ]
then
    git clone git://github.com/kennethreitz/autoenv.git ~/.autoenv
fi
source ~/.autoenv/activate.sh

# If running interactively, then:
if [ "$PS1" ]
then
    set -o vi
    export PS1='[\u@\h \t \w] \n$ '

    # Only keep unique entries in .bash_history.
    export HISTCONTROL=ignoredups
    export HISTSIZE=5000

    # Some more ls aliases
    alias more=less
    alias lc='clear ; ls -l'
    alias lm='ls -l | less'
    alias screen='screen -aA'
    alias R='R --quiet'
    if [ -z $(which seq) ]
    then
        alias seq=gseq
    fi

    if [ ! -z $(which hub 2>/dev/null) ]
    then
        alias git=hub
    fi

    # Misc. aliases
    alias agent="exec ssh-agent sh -c 'ssh-add && bash --login'"
    if [ ! -z $(which xdg-open 2>/dev/null) ]
    then
        open() {
            xdg-open $@
        }
    fi
fi
