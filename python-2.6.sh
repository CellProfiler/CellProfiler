#!/bin/bash
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
python "$@"

