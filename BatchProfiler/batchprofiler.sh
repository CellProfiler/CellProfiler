if [ -e "$HOME/.batchprofiler.sh" ]; then
. "$HOME/.batchprofiler.sh"
fi
#
# If prefix is not defined, assume this script is in the directory
# $PREFIX/src/CellProfiler/BatchProfiler
if [ -z "$PREFIX" ]; then
    BPDIR =`dirname "$0"`
    CPSRCDIR =`dirname "$BPDIR"`
    SRCDIR =`dirname "$CPSRCDIR"`
    export PREFIX=`dirname "$SRCDIR"`
else
    CPSRCDIR="$PREFIX/src/CellProfiler"
fi
. "$PREFIX/bin/cpenv.sh"

PYTHONPATH="$PWD:$CPSRCDIR" python -s "$@"
