#unset INPUTRC

export PS1='[\u@\d \h \t \w]
$ '
export CVS_RSH=ssh
export CVSROOT=cvshost:/usr/local/cvsroot
export EDITOR=vim
export VISUAL=$EDITOR
export LESS="-erX"
export TZ=America/Denver
export PYTHONPATH=~/proj/Theano:~/proj/pylearn2:~/proj/Python/trunk/pykt
export PYLEARN2_DATA_PATH=~/proj/pylearn2-data

export JAVA_HOME=/usr/java/default
export CATALINA_HOME=/usr/local/tomcat
export ANT_HOME=/opt/apache-ant
export MAVEN_HOME=/opt/apache-maven
export M2_HOME=$MAVEN_HOME
export EPIC_HOME=/opt/epic5
export MULE_HOME=/usr/local/mule
export JAXWS_HOME=/export/home/khabermehl/apps/jaxws-ri
export R_HISTFILE=~/.Rhistory

export PERL5LIB=~/lib/perl/lib/perl5/5.8.2/i686-linux:~/lib/perl/lib/perl5/site_perl/5.8.2:~/lib/perl/lib/perl5/site_perl/5.8.2/i686-linux:${LSA:-/export/home/lsa/2.9}

#export PATH=/usr/local/bin:~/proj/pylearn2/pylearn2/scripts:/usr/local/texlive/2013basic/bin/x86_64-darwin:$PATH
export PATH=/usr/local/bin:~/proj/pylearn2/pylearn2/scripts:$PATH

. ~/.bashrc
