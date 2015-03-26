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

export PYTHONPATH=~/proj/Theano:~/proj/pylearn2:~/proj/Python/trunk/pykt:~/proj/jobman:~/proj/pylearnutils:~/proj/Python/trunk/pykt:~/proj/GroundHog

export PYLEARN2_DATA_PATH=~/proj/pylearn2-data

# For Pyro4 and gensim's distributed LSI.
export PYRO_SERIALIZERS_ACCEPTED=pickle
export PYRO_SERIALIZER=pickle

export R_HISTFILE=~/.Rhistory

export PATH=$HOME/bin:/usr/local/bin:/usr/local/sbin:/usr/local/opt/coreutils/libexec/gnubin:~/proj/pylearn2/pylearn2/scripts:$PATH:/usr/local/cuda-6.5/bin:~/torch/bin
export PATH=~/proj/misc/bin:~/proj/jobman/bin:~/proj/Theano/bin:$PATH:~/proj/pylearnutils/pylearnutils/bin
export MANPATH="/usr/local/opt/coreutils/libexec/gnuman:$MANPATH"
export LD_LIBRARY_PATH=/usr/local/cuda-6.5/lib64:~/torch/lib
export BYOBU_PREFIX=$(brew --prefix)

#export PYENV_ROOT=/usr/local/opt/pyenv  
#eval "$(pyenv init -)"

. ~/.bashrc
