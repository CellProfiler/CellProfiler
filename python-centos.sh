#!/bin/bash
. /broad/tools/scripts/useuse
reuse .python-2.6.5
reuse .numpy-trunk-python-2.6.5
reuse .scipy-trunk-python-2.6.5
reuse .matplotlib-trunk-python-2.6.5
reuse Java-1.6
export MPLCONFIGDIR=/imaging/analysis/CPCluster/CellProfiler-2.0/.matplotlib
export LAST_CHECKOUT=`echo "import os;cpdir='/imaging/analysis/CPCluster/CellProfiler-2.0';print os.path.join(cpdir,str(max(*[int(x) for x in os.listdir(cpdir) if x.isdigit()])))" | python`
python "$@"
