#!/bin/bash
#
# Script that is run by Jenkins on vm5a3-282 to build CellProfiler for
# Mac OS X.

# Environment variables:
#
# HUDSON_SERVER_COOKIE=fa678bfb9d635ec8
# SHELL=/bin/bash
# TMPDIR=/var/folders/sv/zv_bjd2x3xvd2kd55q4kmnph00007h/T/
# BUILD_TAG=jenkins-CellProfiler master-70
# GIT_PREVIOUS_COMMIT=680cbb94299d2f3bbb50c8330bf89cdf5aa808bf
# WORKSPACE=/Users/Shared/Jenkins/Home/workspace/CellProfiler master
# com.apple.java.jvmTask=CommandLine.java.java
# USER=jenkins
# __CF_USER_TEXT_ENCODING=0xF0:0:0
# GIT_COMMIT=81f81053beaa68b4fc4178778d326c2b89b9adfa
# JENKINS_HOME=/Users/Shared/Jenkins/Home
# PATH=/usr/bin:/bin:/usr/sbin:/sbin
# PWD=/Users/Shared/Jenkins/Home/workspace/CellProfiler master
# JOB_NAME=CellProfiler master
# BUILD_DISPLAY_NAME=#70
# com.apple.java.jvmMode=client
# JAVA_MAIN_CLASS_8268=Main
# BUILD_ID=2014-03-25_15-00-19
# HOME=/Users/Shared/Jenkins
# SHLVL=1
# GIT_BRANCH=origin/release_2.1.1
# JENKINS_SERVER_COOKIE=fa678bfb9d635ec8
# EXECUTOR_NUMBER=0
# GIT_URL=https://github.com/CellProfiler/CellProfiler.git
# NODE_LABELS=master
# LOGNAME=jenkins
# HUDSON_HOME=/Users/Shared/Jenkins/Home
# NODE_NAME=master
# BUILD_NUMBER=70
# HUDSON_COOKIE=c3d202d2-3d33-4a1b-a652-4d4e2fd7b10b
# JAVA_ARCH=x86_64
# SECURITYSESSIONID=186bb

set -e
set -x

signer=build@vmd94-150
keychain=/Users/Shared/Jenkins/Library/Keychains/cellprofiler.keychain

short_branch=$(echo "$GIT_BRANCH" | cut -d/ -f2)
hash=$(git rev-parse --short "$GIT_BRANCH")
version="${short_branch}-${hash}"

. /Users/build/homebrew/bin/activate-cpdev
rm -rf build dist imagej/jars
arch -i386 python external_dependencies.py -o
arch -i386 python CellProfiler.py --build-and-exit --do-not-fetch
#svn --trust-server-cert --non-interactive co https://svn.broadinstitute.org/CellProfiler/trunk/ExampleImages/
#svn --trust-server-cert --non-interactive co https://svn.broadinstitute.org/CellProfiler/trunk/TestImages/
export CP_EXAMPLEIMAGES=/Users/build/ExampleImages
export CP_TESTIMAGES=/Users/build/TestImages
set +e
arch -i386 python cpnose.py --with-kill-vm --noguitests --with-xunit -m \(\?\:\^\)test_\(\?\!example\).\*
set -e
arch -i386 python setup.py py2app

unsigned_zip="CellProfiler-${version}-unsigned.zip"
(cd dist; zip -r "$unsigned_zip" CellProfiler.app)
signdir="jenkins/${BUILD_TAG// /-}"
ssh ${signer} mkdir -p jenkins
scp $(dirname "$0")/sign.sh ${signer}:jenkins/sign.sh
ssh ${signer} mkdir -p "$signdir"
scp dist/"$unsigned_zip" "${signer}:${signdir}/${unsigned_zip}"
signed_zip="CellProfiler-${version}.zip"
ssh ${signer} jenkins/sign.sh "${signdir}" "${unsigned_zip}" "${signed_zip}"
scp ${signer}:"${signdir}/${signed_zip}" dist/
rm -rf dist/"$unsigned_zip" dist/CellProfiler.app
(cd dist; unzip "${signed_zip}")
mv dist/"${signed_zip}" .
#hdiutil create -ov -volname CellProfiler -srcfolder dist/CellProfiler.app CellProfiler.dmg
pkgbuild --analyze --root dist CellProfilerComponents.plist
/usr/libexec/PlistBuddy -c "Set :0:BundleHasStrictIdentifier 0" CellProfilerComponents.plist
/usr/libexec/PlistBuddy -c "Set :0:BundleIsRelocatable 0" CellProfilerComponents.plist
security unlock-keychain -p '' $keychain
rm -f *.pkg
pkgbuild --scripts jenkins/mac/scripts --install-location /Applications --keychain $keychain --root dist --component-plist CellProfilerComponents.plist --identifier org.cellprofiler.CellProfiler --version "$version" --sign 'THE BROAD INSTITUTE INC' "CellProfiler-${version}.pkg"
