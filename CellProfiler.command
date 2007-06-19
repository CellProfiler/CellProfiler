#!/bin/bash
echo name $0
open /Applications/Utilities/X11.app;
export mcr_path=`dirname "$0"`/MCR/v76
echo $mcr_path
export DYLD_LIBRARY_PATH="$mcr_path/sys/os/maci:$mcr_path/bin/maci:/System/Library/Frameworks/JavaVM.framework/JavaVM:/System/Library/Frameworks/JavaEmbedding.framework/JavaEmbedding:/System/Library/Frameworks/JavaVM.framework/Libraries"
echo $DYLD_LIBRARY_PATH
export DISPLAY=":0.0"
`dirname "$0"`/CellProfiler;