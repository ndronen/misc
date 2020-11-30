# For R
export COLUMNS

# Update PS1 if running interactively and not in a virtual environment.
if [ "$PS1" ]
then
    set -o vi

    # Only keep unique entries in .bash_history.
    export HISTCONTROL=ignoredups
    export HISTSIZE=5000

    # Some more ls aliases
    alias vi=vim
    alias more=less
    alias lc='clear ; ls -l'
    alias lm='ls -l | less'
    alias screen='screen -aA'
    alias typora='open -a Typora'
    alias R='R --quiet'
    if [ -z $(which seq) ]
    then
        alias seq=gseq
    fi

    # Misc. aliases
    alias agent="exec ssh-agent sh -c 'ssh-add && bash --login'"
    if [ ! -z $(which xdg-open 2>/dev/null) ]
    then
        open() {
            xdg-open $@
        }
    fi

    if [ -z "$CONDA_DEFAULT_ENV" ]
    then
        export PS1='[\u@\h \t \w] \n$ '
    else
        # We're in a virtual environment.  Set PS1 differently.
        export PS1="($CONDA_DEFAULT_ENV) [\\u@\\h \\t \\w] \\n\$ "
    fi
fi
