#!/bin/bash
echo name $0
open /Applications/Utilities/X11.app;
export mcr_path=/Applications/MATLAB/MATLAB_Compiler_Runtime/v78
echo $mcr_path
export DYLD_LIBRARY_PATH="$mcr_path/runtime/maci:$mcr_path/sys/os/maci:$mcr_path/bin/maci:/System/Library/Frameworks/JavaVM.framework/JavaVM:/System/Library/Frameworks/JavaVM.framework/Libraries"
echo $DYLD_LIBRARY_PATH
export XAPPLRESDIR="$mcr_path/X11/app-defaults"
echo $XAPPLRESDIR
export DISPLAY=":0.0"
`dirname "$0"`/CellProfiler;
