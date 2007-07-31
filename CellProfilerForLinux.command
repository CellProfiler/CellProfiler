#!/bin/sh
export curdir=`dirname $0`
if [ $curdir = '.' ]; then
    curdir=`pwd`
fi
export mcr_path="$curdir/lib/MCR/v76"
export LD_LIBRARY_PATH="$mcr_path/runtime/glnx86:$mcr_path/sys/os/glnx86:$mcr_path/bin/glnx86:$mcr_path/sys/java/jre/glnx86/jre1.5.0/lib/i386/native_threads:$mcr_path/sys/java/jre/glnx86/jre1.5.0/lib/i386/client:$mcr_path/sys/java/jre/glnx86/jre1.5.0/lib/i386"
export XAPPLRESDIR="$mcr_path/X11/app-defaults"
cd $curdir/lib
./CellProfiler
