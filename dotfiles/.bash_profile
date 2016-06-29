export PS1='[\u@\d \h \t \w]
$ '
export CVS_RSH=ssh
export CVSROOT=cvshost:/usr/local/cvsroot
export EDITOR=vim
export VISUAL=$EDITOR
export LESS="-erX"
export TZ=America/Denver

# Old development environment variables.
export JAVA_HOME=/usr/java/default
export ANT_HOME=/opt/apache-ant
export MAVEN_HOME=/opt/apache-maven
export M2_HOME=$MAVEN_HOME
export PERL5LIB=~/lib/perl/lib/perl5/5.8.2/i686-linux:~/lib/perl/lib/perl5/site_perl/5.8.2:~/lib/perl/lib/perl5/site_perl/5.8.2/i686-linux:${LSA:-/export/home/lsa/2.9}

export PYTHONPATH=~/proj/srilm/srilm-python
export R_HISTFILE=~/.Rhistory
export R_LIBS_USER=/export/home/ndronen/R/x86_64-pc-linux-gnu-library/3.2-$(hostname)

if [ "$(uname)" == Darwin ]
then
    export PATH="/usr/local/opt/coreutils/libexec/gnubin:$PATH"
    export MANPATH="/usr/local/opt/coreutils/libexec/gnuman:$MANPATH"
elif [ "$(uname)" == Linux ]
then
    export PATH="$PATH:/usr/local/cuda/bin"
    export LD_LIBRARY_PATH="/usr/local/cuda-7.0/lib64:$LD_LIBRARY_PATH"
fi

export PATH=~/proj/srilm/lm/bin/$(~/proj/srilm/sbin/machine-type):$PATH
export PATH=$PATH:$HOME/bin
export PATH=$PATH:/usr/local/bin:/usr/local/sbin
export PATH=$PATH:$JAVA_HOME/bin:~/proj/misc/bin

if [ $(hostname) == snapper ]
then
    export PATH=~/anaconda/bin:$PATH
fi

if [ -e /usr/share/virtualenvwrapper/virtualenvwrapper.sh ]
then
    export WORKON_HOME=~/proj/envs/
fi

export PYTHONUNBUFFERED=1
export DEV64=184.72.154.89

which nvcc >/dev/null
if [ $? -eq 0 ]
then
    export THEANORC=~/.theanorc.gpu
else
    export THEANORC=~/.theanorc.cpu
fi

. ~/.bashrc
