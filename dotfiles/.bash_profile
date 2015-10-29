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

export PYTHONPATH=~/proj/Python/trunk/pykt

# For Pyro4 and gensim's distributed LSI.
export PYRO_SERIALIZERS_ACCEPTED=pickle
export PYRO_SERIALIZER=pickle

export R_HISTFILE=~/.Rhistory

export PATH=$HOME/bin:/usr/local/bin:/usr/local/sbin:/usr/local/opt/coreutils/libexec/gnubin:$PATH:/usr/local/cuda-7.0/bin:~/proj/torch/install/bin
export PATH=~/miniconda/bin:~/proj/misc/bin:~/proj/Theano/bin:$PATH
export MANPATH="/usr/local/opt/coreutils/libexec/gnuman:$MANPATH"
export LD_LIBRARY_PATH=/usr/local/cuda-7.0/lib64:~/proj/torch/install/lib # :~/proj/torch/install/lib/lua/5.1/
export BYOBU_PREFIX=$(brew --prefix)

if [ -e /export/home/ndronen/proj/torch/install/bin/torch-activate ]
then
    . /export/home/ndronen/proj/torch/install/bin/torch-activate
fi

export PYTHONUNBUFFERED=1
#export CHAINER_SEED=1

export DEV64=184.72.154.89

#export PYENV_ROOT=/usr/local/opt/pyenv  
#eval "$(pyenv init -)"

. ~/.bashrc
