#!/bin/bash
export DK_ROOT=/broad/tools/dotkit
source /broad/tools/dotkit/ksh/.dk_init
use -q Python-2.6
export PYTHONPATH=/imaging/analysis/People/imageweb/python-packages:$PYTHONPATH
python "$@"

