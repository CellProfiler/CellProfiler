#!/bin/bash
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

# Install tools needed to build RPMs
sudo yum -q -y install rpm-build redhat-rpm-config

rpm -i wxPython2.8-2.8.12.1-1.src.rpm
cd /root/rpmbuild/SPECS
patch < ~/wxpython_rpm.diff
rpmbuild -ba wxPython.spec

