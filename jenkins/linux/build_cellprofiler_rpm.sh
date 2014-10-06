#!/bin/bash
#
# Runs as root.

set -e
set -x

BUILD_NUMBER="$1"
if [ -z "$BUILD_NUMBER" ]; then
    echo >&2 Usage: $(basename "$0") BUILD-NUMBER
    exit 1
fi

yum -y update
adduser build
yum install -q -y rpm-build yum-utils git tar

cp=/jenkins/CellProfiler

rpmbuild=/home/build/rpmbuild
mkdir $rpmbuild $rpmbuild/SOURCES
tar -C /jenkins -cf $rpmbuild/SOURCES/cellprofiler.tar.gz CellProfiler
chown -R build:build $rpmbuild

spec=$cp/jenkins/linux/cellprofiler-centos6.spec
cp $cp/jenkins/linux/cellprofiler-centos6.repo /etc/yum.repos.d/cellprofiler.repo
yum makecache
(echo "%define version 0.0.0"; echo "%define release 0"; cat $spec) > /tmp/fake.spec
yum-builddep -q -y /tmp/fake.spec

describe=$(git --git-dir=$cp/.git describe --long)
#
# Tags should be in the form, "N.N.N", for release
# or "N.N.N-SNAPSHOT" for snapshots
#
# git describe --long has an output of
# <TAG>-<COMMITS-PAST-TAG>-g<GIT-HASH>
#
version=$(echo $describe | sed 's/\([0-9.]\+\)-\(SNAPSHOT\|\)-\?\([0-9]\+\)-g\([0-9a-f]\+\)/\1/')
snapshot=$(echo $describe | sed 's/\([0-9.]\+\)-\(SNAPSHOT\|\)-\?\([0-9]\+\)-g\([0-9a-f]\+\)/\2/')
commits_past_tag=$(echo $describe | sed 's/\([0-9.]\+\)-\(SNAPSHOT\|\)-\?\([0-9]\+\)-g\([0-9a-f]\+\)/\3/')
git_hash=$(echo $describe | sed 's/\([0-9.]\+\)-\(SNAPSHOT\|\)-\?\([0-9]\+\)-g\([0-9a-f]\+\)/\4/')
release=$BUILD_NUMBER.$(date +%Y%m%d).$commits_past_tag.$git_hash$snapshot
#exec /bin/bash -l
su -c 'rpmbuild -ba --define="release '$release'" --define="version '$version'" '$spec build
cp /home/build/rpmbuild/SRPMS/cellprofiler-${version}-${release}.src.rpm $cp/
cp /home/build/rpmbuild/RPMS/x86_64/cellprofiler-${version}-${release}.x86_64.rpm $cp/

