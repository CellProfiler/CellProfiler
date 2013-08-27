#!/bin/bash
#
# Intended to be run from jenkins as the imgbuild user on
# imgcpbuild.broadinstitute.org. Starts a VM and a build within it.
# Then starts another VM and starts tests within it.

eval `/broad/software/dotkit/init -b`
use Python-2.6 
. ~/venv/cpbuild/bin/activate
set -ex
cd `dirname "$0"`
cp -p ~/vcloud-password.txt .
./with_vm.py -- "Build CentOS 6 x64" fab build -H
ls -l cellprofiler.tar.gz
./with_vm.py -- "Build CentOS 6 x64" fab test -H
