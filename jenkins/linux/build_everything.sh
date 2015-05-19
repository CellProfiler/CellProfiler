#!/bin/bash
#
# Runs as root
#
# Environment variables:
#
# PREFIX: where to build CP and its dependencies
#         $PREFIX/bin contains executables and when running CP, should
#         be on your $PATH
#         $PREFIX/lib contains .so files and should be on your $LD_LIBRARY_PATH
#
# TMPDIR: location for temporary files
# SRCDIR: location for downloaded sources (defaults to $TMPDIR/src)


set -e
set -x

yum -y update
yum -q -y install python-setuptools gcc gcc-c++ wget vim gtk2-devel git svn \
    gcc-gfortran cmake mesa-libGL mesa-libGL-devel blas atlas lapack blas-devel \
    atlas-devel lapack-devel xorg-x11-xauth* xorg-x11-xkb-utils* \
    xorg-x11-utils xorg-x11-server-Xvfb unzip tar dos2unix \
    dejavu-lgc-sans-fonts qt-devel openssl openssl-devel xclock bzip2 \
    bzip2-devel bzip2-libs libXtst make patch readline-devel \
    java-1.7.0-openjdk java-1.7.0-openjdk-devel
    
adduser cpbuild
cp=/jenkins/CellProfiler
if [ -z "${PREFIX}" ]; then
    export PREFIX=/imaging/analysis/CPCluster/CellProfiler-2.0/builds/redhat_6
fi
if [ -z "${TMPDIR}" ]; then
    export TMPDIR=/imaging/analysis/CPCluster/CellProfiler-2.0/tmp
fi
if [ -z "${SRCDIR}" ]; then
    export SRCDIR="${TMPDIR}"/src
fi
GITHOME="${PREFIX}/src/CellProfiler"
GITCOMMIT=`cd /jenkins/CellProfiler && git log -n 1 --pretty=format:%h`
GITURL=`cd /jenkins/CellProfiler && git remote show origin | grep "Fetch URL" --|sed "s/\s*Fetch URL:\s*//"`
mkdir -p $PREFIX/src
mkdir -p $TMPDIR
mkdir -p $SRCDIR
chown -R cpbuild:cpbuild $PREFIX
chown -R cpbuild:cpbuild $TMPDIR
chown -R cpbuild:cpbuild $SRCDIR

export JAVA_HOME=/usr/lib/jvm/java-1.7.0
export PATH="${PREFIX}"/bin:$PATH
export HOSTTYPE=amd64
export BLAS=/usr/lib64
export LAPACK=/usr/lib64
export LD_LIBRARY_PATH="${JAVA_HOME}"/jre/lib/"${HOSTTYPE}"/server:"${PREFIX}"/lib

su -c 'cd '$PREFIX'/src && git clone '$GITURL cpbuild
su -c 'cd '$GITHOME' && git checkout '$GITCOMMIT cpbuild
su -c 'cd '$GITHOME' && make -f Makefile.CP2 all' cpbuild
su -c 'cd '$GITHOME' && xvfb-run make -f Makefile.CP2 test' cpbuild
cd "/jenkins/CellProfiler"
tar cvzf cellprofiler.tar.gz "${PREFIX}"