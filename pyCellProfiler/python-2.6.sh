#!/bin/bash
export DK_ROOT=/broad/tools/dotkit
source /broad/tools/dotkit/ksh/.dk_init
use -q Python-2.6
use -q matlab
export MPLCONFIGDIR=/imaging/analysis/CPCluster/CellProfiler-2.0/.matplotlib
export PYTHONPATH=/imaging/analysis/CPCluster/CellProfiler-2.0/site-packages:.:/broad/tools/Linux/x86_64/pkgs/matplotlib_0.98.5.2/lib/python2.6/site-packages:$PYTHONPATH
python "$@"

