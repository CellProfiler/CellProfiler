#!/bin/bash
echo name $0
open /Applications/Utilities/X11.app;
export mcr_path=/Applications/MATLAB/MATLAB_Compiler_Runtime/v78
echo $mcr_path
export DYLD_LIBRARY_PATH="$mcr_path/runtime/maci:$mcr_path/sys/os/maci:$mcr_path/bin/maci:/System/Library/Frameworks/JavaVM.framework/JavaVM:/System/Library/Frameworks/JavaVM.framework/Libraries"
echo $DYLD_LIBRARY_PATH
export XAPPLRESDIR="$mcr_path/X11/app-defaults"
echo $XAPPLRESDIR

sysver=`sw_vers -productVersion | cut -c 1-4`
echo $sysver

if [ $sysver = 10.5 -o $sysver = 10.6 ]
then
	echo "The DISPLAY variable does not need to be set manually"
elif [ $sysver = 10.4 -o $sysver = 10.3 ]
then
	export DISPLAY=":0.0"
else
	echo "This system is too old"
fi

`dirname "$0"`/CellProfiler;
