#!/bin/bash

# Install tools needed to build RPMs
sudo yum -q -y install rpm-build redhat-rpm-config

rpm -i wxPython2.8-2.8.12.1-1.src.rpm
cd /root/rpmbuild/SPECS
patch < ~/wxpython_rpm.diff
rpmbuild -ba wxPython.spec

