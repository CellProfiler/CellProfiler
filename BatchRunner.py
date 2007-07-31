#!/util/bin/python
from sys import argv,exit
from subprocess import Popen, PIPE, STDOUT

if len(argv) != 6:
    print "usage: %s DataDir BatchPrefix QueueType BatchSize WriteOutData(yes/no) Timeout"%(argv[0])
    exit(1)

datadir = argv[1]
prefix = argv[2]
queue = argv[3]
batch_size = int(argv[4])
write_data = argv[5]
timeout = int(argv[6])

CPCluster='/imaging/analysis/People/Ray/CPClusterSingle2007a'

# This should probably submit to the same queuetype, but we want to make sure it works.
command = "bsub -K -q opensuse -N -oo %(datadir)s/txt_output/joblist.txt %(CPCluster)s/CPClusterSingle.py %(datadir)s/%(prefix)sdata.mat all %(batch_size)d %(datadir)s/status %(prefix)s no %(timeout)d"%(locals())
subproc = Popen(command.split(" "),stdout=PIPE,stderr=PIPE)
subproc.wait()

f = open("%(datadir)s/txt_output/joblist.txt"%(locals()))

for l in f:
    l = l.strip()
    if l == "":
        continue
    start,end = l.strip().split(" ")
    print "bsub -q %(queue)s -o %(datadir)s/txt_output/%(start)s_to_%(end)s.txt %(CPCluster)s/CPClusterSingle.py %(datadir)s/%(prefix)sdata.mat %(start)s %(end)s %(datadir)s/status %(prefix)s %(write_data)s %(timeout)d"%(locals())



#  for i in `; do
#      echo bsub -q $QueueType -o ${BatchTxtOutputDir}/${i}.txt $CPCluster/CPClusterSingle.sh ${BatchDataDir}/${BatchPrefix}data.mat $i $BatchStatusDir $BatchPrefix
#  done
