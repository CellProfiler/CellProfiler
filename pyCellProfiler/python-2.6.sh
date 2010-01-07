#!/bin/bash
export DK_ROOT=/broad/tools/dotkit
source /broad/tools/dotkit/ksh/.dk_init
use -q Python-2.6
use -q Java-1.6
IMAGEWEB=imageweb
if test $USER = $IMAGEWEB
then
export MPLCONFIGDIR=/imaging/analysis/CPCluster/CellProfiler-2.0/.matplotlib
fi
export PYTHONPATH=/imaging/analysis/CPCluster/CellProfiler-2.0/site-packages:.:/broad/tools/Linux/x86_64/pkgs/matplotlib_0.98.5.2/lib/python2.6/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$JAVA_HOME/jre/lib/amd64/server

# Do some logging.
which python > /local/scratch/DELETEME.${LSB_JOBID}
echo "$@" >> /local/scratch/DELETEME.${LSB_JOBID}
env | sort >> /local/scratch/DELETEME.${LSB_JOBID}
python -c "import sys; print sys.path" | sed 's,\[,,g' | sed 's,\],,g' | tr ',' '\n' | sort  >> /local/scratch/DELETEME.${LSB_JOBID}
cat /local/scratch/DELETEME.${LSB_JOBID} | mail -s "[PYTHON-ERROR-QUEST] Job: ${LSB_JOBID} on Host: `hostname -s`" jbh@broadinstitute.org
rm /local/scratch/DELETEME.${LSB_JOBID}

python "$@"

