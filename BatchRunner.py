#!/broad/tools/Linux/x86_64/pkgs/python_2.6.1/bin/python
from sys import argv,exit,path
from subprocess import Popen, PIPE, STDOUT
from scipy.io.matlab import loadmat
from os.path import exists

if len(argv) != 6:
    print "usage: %s DataDir QueueType BatchSize WriteOutData(yes/no) Timeout"%(argv[0])
    exit(1)

datadir = argv[1]
queue = argv[2]
batch_size = int(argv[3])
write_data = argv[4]
timeout = int(argv[5])

CPCluster=path[0]

# Load Batch_data and figure out the sets that need running
batch_info = loadmat("%(datadir)s/Batch_data.mat"%(locals()))
num_sets = batch_info['handles'][0,0].Current[0,0].NumberOfImageSets[0,0]

# Loop over batches, check status file, print out commands for those that need it
for start in range(1, num_sets + 1, batch_size):
    end = start + batch_size - 1
    if end > num_sets:
        end = num_sets
    status_file_name = "%(datadir)s/status/Batch_%(start)d_to_%(end)d_DONE.mat"%(locals())
    if not exists(status_file_name):
        print "bsub -q %(queue)s -o %(datadir)s/txt_output/%(start)s_to_%(end)s.txt %(CPCluster)s/CPCluster.py %(datadir)s/Batch_data.mat %(start)s %(end)s %(datadir)s/status Batch_ %(write_data)s %(timeout)d"%(locals())


#  for i in `; do
#      echo bsub -q $QueueType -o ${BatchTxtOutputDir}/${i}.txt $CPCluster/CPCluster.sh ${BatchDataDir}/${BatchPrefix}data.mat $i $BatchStatusDir $BatchPrefix
#  done
