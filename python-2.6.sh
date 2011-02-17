#!/bin/bash

grep -q "SUSE" /proc/version
if [ $? -eq 0 ]
then
export DK_ROOT=/broad/tools/dotkit
source /broad/tools/dotkit/ksh/.dk_init
use -q Python-2.6
use -q Java-1.6
export MPLCONFIGDIR=/imaging/analysis/CPCluster/CellProfiler-2.0/.matplotlib
export PYTHONPATH=/imaging/analysis/CPCluster/CellProfiler-2.0/site-packages:.:/broad/tools/Linux/x86_64/pkgs/matplotlib_0.98.5.2/lib/python2.6/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$JAVA_HOME/jre/lib/amd64/server
#
# This code lists the CellProfiler checkout directory for
# the latest checkout and adds that to the end of the Python path
#
export LAST_CHECKOUT=`echo "import os;cpdir='/imaging/analysis/CPCluster/CellProfiler-2.0';print os.path.join(cpdir,str(max(*[int(x) for x in os.listdir(cpdir) if x.isdigit()])))" | python`
export PYTHONPATH=$PYTHONPATH:$LAST_CHECKOUT
else
grep -q "centos" /proc/version
if [ $? -eq 0 ]
then
. /broad/tools/scripts/useuse
reuse .python-2.6.5
reuse .cython-0.12.1-python-2.6.5
reuse .numpy-trunk-python-2.6.5
reuse .scipy-trunk-python-2.6.5
reuse .matplotlib-trunk-python-2.6.5
reuse Java-1.6
export MPLCONFIGDIR=/imaging/analysis/CPCluster/CellProfiler-2.0/.matplotlib
export LAST_CHECKOUT=`echo "import os;cpdir='/imaging/analysis/CPCluster/CellProfiler-2.0';print os.path.join(cpdir,str(max(*[int(x) for x in os.listdir(cpdir) if x.isdigit()])))" | python`
else
echo "Unknown operating system"
exit $E_NOFILE
fi
fi
if [ -n "$CELLPROFILER_USE_XVFB" ]
then
DISPLAY=:$LSB_JOBID
echo "Xvfb display = $DISPLAY"
tmp=/local/scratch/CellProfilerXVFB.$RANDOM.$RANDOM
echo "Xvfb directory = $tmp"
mkdir $tmp
Xvfb $DISPLAY -fbdir $tmp &
XVFBPID=$!
echo "Xvfb PID = $XVFBPID"
python "$@"
kill $XVFBPID
sleep 5
rmdir $tmp
else
python "$@"
fi
