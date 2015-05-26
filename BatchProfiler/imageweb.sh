. /broad/software/scripts/useuse
if [ "$SYSTYPE" == "redhat_5_x86_64" ]; then
#
# Use dotkit
#
reuse .toolbox-0.11.0
reuse Java-1.6
export CPCLUSTER=/imaging/analysis/CPCluster/CellProfiler-2.0
export MPLCONFIGDIR="$CPCLUSTER"/.matplotlib
export LAST_CHECKOUT="$CPCLUSTER"/release_2.1.1
else
reuse Java-1.6
export PREFIX=/imaging/analysis/CPCluster/CellProfiler-2.0/builds/redhat_6
export CPCLUSTER=/imaging/analysis/CPCluster/CellProfiler-2.0/checkouts/redhat_6
export LAST_CHECKOUT="$CPCLUSTER"/src/CellProfiler
export PATH="$PREFIX"/bin:"$PATH"
export LD_LIBRARY_PATH="$PREFIX"/lib:"$LD_LIBRARY_PATH":"$PREFIX"/lib/mysql:"$JAVA_HOME"/jre/lib/amd64/server
export LC_ALL=en_US.UTF-8
fi
python -s "$@"
