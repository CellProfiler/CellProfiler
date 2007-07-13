matlab -r BuildCPClusterSingle
cp CPClusterSingle /imaging/analysis/People/Ray/CPClusterSingle
cp CPClusterSingle.ctf /imaging/analysis/People/Ray/CPClusterSingle
cd /imaging/analysis/People/Ray/CPClusterSingle
rm -rf CPClusterSingle_mcr unpacking.txt
bsub -q priority -K -o unpacking.txt ./CPClusterSingle.sh fail
