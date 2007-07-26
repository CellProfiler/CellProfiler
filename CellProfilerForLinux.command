export mcr_path=`dirname $0`/MCR/v76
export LD_LIBRARY_PATH="$mcr_path/runtime/glnx86:$mcr_path/sys/os/glnx86:$mcr_path/bin/glnx86:$mcr_path/sys/java/jre/glnx86/jre1.5.0/lib/i386/native_threads:$mcr_path/sys/java/jre/glnx86/jre1.5.0/lib/i386/client:$mcr_path/sys/java/jre/glnx86/jre1.5.0/lib/i386"
export XAPPLRESDIR="$mcr_path/x11/app-defaults"
export DISPLAY=":0.0"
`dirname $0`/CellProfiler;
