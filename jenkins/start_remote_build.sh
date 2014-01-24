#!/bin/bash
#
# Intended to be run from jenkins as the imgbuild user on
# imgcpbuild.broadinstitute.org. Starts a VM and a build within it.
# Then starts another VM and starts tests within it.
#
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
#
# Copyright (c) 2003-2009 Massachusetts Institute of Technology
# Copyright (c) 2009-2014 Broad Institute
# All rights reserved.
#
# Please see the AUTHORS file for credits.
#
# Website: http://www.cellprofiler.org
#

eval `/broad/software/dotkit/init -b`
use Python-2.6 
. ~/venv/cpbuild/bin/activate
set -ex
cd `dirname "$0"`
cp -p ~/vcloud-password.txt .
./with_vm.py -- "Build CentOS 6 x64" fab build -H
ls -l cellprofiler.tar.gz
./with_vm.py -- "Build CentOS 6 x64" fab test -H
