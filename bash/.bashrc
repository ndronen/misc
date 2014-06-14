# ~/.bashrc: executed by bash(1) for non-login shells.
# see /usr/share/doc/bash/examples/startup-files (in the package bash-doc)
# for examples

# For R
export COLUMNS

# If running interactively, then:
if [ "$PS1" ]
then
	set -o vi
	export PS1='[\u@\h \t \w] \n$ '

    if [ -e bin/activate ]
    then
        source bin/activate
    fi

	#sshcmd=/usr/bin/ssh-add
	#( $sshcmd -l | /bin/grep 'no identities' >/dev/null ) && $sshcmd

	# Only keep unique entries in .bash_history.
	export HISTCONTROL=ignoredups
	export HISTSIZE=5000

    vi() {
        vim $@
    }

	# Some more ls aliases
	alias more=less
	alias ll='ls -l'
	alias la='ls -A'
	alias l='ls -CF'
	alias lc='clear ; ls -l'
	alias lm='ls -l | less'
    alias R='R --quiet'
    if [ -z $(which seq) ]
    then
        alias seq=gseq
    fi
    alias ack='/opt/local/libexec/perl5.12/ack'
    alias screen='screen -aA'

	# Misc. aliases
	alias apt-grep='apt-cache search'
	#alias agent="exec ssh-agent sh -c 'ssh-add && bash --login'"
	alias chogs='( ps -ef | head -1 ; ps -ef | sed 1d | sort +3 -nr )'
	#alias ssh='ssh -A'
	alias scp='scp -oForwardAgent=yes'
	#alias dos2unix="perl -i -pe 's/\r\n/\n/' "
	#alias unix2dos="perl -i -pe 's/\n/\r\n/' "
    alias csel='ssh -X dronen@elra-01.cs.colorado.edu'
    alias csel2='ssh -X dronen@elra-02.cs.colorado.edu'
    alias csel3='ssh -X dronen@elra-03.cs.colorado.edu'
    alias csel4='ssh -X dronen@elra-04.cs.colorado.edu'
    if [ ! -z $(which xdg-open 2>/dev/null) ]
    then
        open() {
            xdg-open $@
        }
    fi
fi

lspath() {
	echo $PATH | tr ':' '\n'
	return;
}

ctand() {
    wget http://mirror.ctan.org/macros/latex/contrib/$1.zip && unzip $1.zip
}

