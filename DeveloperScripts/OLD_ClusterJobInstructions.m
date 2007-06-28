% -------------------------------------------------------------------------
% As an example, the following is how we run jobs on the cluster at the
% Whitehead Institute. You may need to repair the syntax of batchrun.sh to
% deal with the line-wrapping. Omit the svn update section if you are not
% using our source-control system called Subversion. Do not edit the
% instructions here; the protocol on Wikitini is more up-to-date.
% -------------------------------------------------------------------------
%
% These instructions are for writing the batch files straight to the remote
% server location. You can also write the batch files to the local computer
% and copy them over (see LOCAL ONLY steps).
% 
% 1. Put your images on the server.
% 
% Currently the following
% 
% imaging (tap2),
% carpente_ata (tap6),
% sabatini_dbw01 (tap5),
% sabatini_dbw02 (tap6), 
% sabatini1_ata (tap6)
% sabatini2_ata (tap6) 
% are all nfs mounted and accessible to the cluster (see protocols >
% Connecting to servers for how to set up network connections to above
% servers).
% 
% 2. Connect to server from your own computer
% 
% Use Go > Connect to server or use CPmount.sh script (see protocols >
% Connecting to servers) to connect. Be sure to use nfs rather than cifs
% because the connection is much faster.
% 
% 3. Log into barra
% 
% Barra is a server and a front end to submit jobs to the cluster. Log into
% barra as user=youraccount, using Terminal:
% 
% ssh youraccount@barra.wi.mit.edu
% or X Windows:
% 
% ssh -X youraccount@barra.wi.mit.edu
% 4. Create a project folder on the server
% 
% After logging into barra, make a folder for the project somewhere on gobo
% that is accessible to the cluster, and give write permission to
% ‘everyone’ for the new folder (so each cluster computer is allowed to
% write there). It may be necessary to make the folder from the command
% line in Terminal or X windows, logged in as your username rather than
% just making a folder using your local computer, because your local
% computer may not have write permission on the servers:
% 
% mkdir FOLDERNAME
% chmod a+w FOLDERNAME.
% 5. (LOCAL ONLY) Make a folder on the local computer.
% 
% 6. Modify the pipeline in CellProfiler to be used for cluster
% 
% In CellProfiler, add the module CreateBatchScripts to the end of the
% pipeline, and enter the settings (see Notes below for server naming
% issues):
% 
% CreateBatchScripts module:
% 
% What prefix ahould be added to the batch file names?	Batch_
% What is the path to the CellProfiler folder on the cluster machines?
% /home/youraccount/CellProfiler
% What is the path to the image folder on the cluster machines?	.
% What is the path to the folder where batch output should be written on
% the cluster machines?	.
% What is the path to the folder where you want to save the batch files? .
% What is the path to the folder where the batch data file will be saved on
% the cluster machines?	.
% If pathnames are specified differently between the local and cluster
% machines, enter that part of the pathname from the local machine’s
% perspective	Volumes/tap6
% If pathnames are specified differently between the local and cluster
% machines, enter that part of the pathname from the cluster machines’
% perspective	nfs/sabatini2_ata
% 
% SaveImages module:
% 
% Enter the pathname to the folder where you want to save the images:
% /nfs/sabatini2_ata/PROJECTNAME
% 
% Default image folder	/Volumes/tap6/IMAGEFOLDER
% Default output folder	/Volumes/tap6/PROJECTNAME (or LOCAL folder)
% 
% 7. Create batch files
% 
% Run the pipeline through CellProfiler, which will analyze the first cycle
% and create batch files.
% 
% 8. Move batch files onto server (FOR LOCAL ONLY)
% 
% Drag the BatchFiles folder (which now contains the Batch .m files and
% .mat file) and BatchOutputFiles folder (empty) from the local computer
% into the project folder at the remote location.
% 
% 9. Run the batch scripts on the cluster
% 
% On barra, make sure the CellProfiler code is updated at
% /home/youraccount/CellProfiler. The script below has a section which will
% automatically update your code (SVN update section) if you have set up
% your copy of CellProfiler with subversion. Any compiled functions in the
% code must be compiled for every type of architecture present in the
% cluster using the Matlab command mex (PC, Mac, Unix, 64-bit, etc).
% 
% From the command line, logged into barra, submit the jobs using a script
% like batchrun.sh as follows:
% 
%  ./batchrun.sh /FOLDERWHEREMFILESARE /FOLDERWHERETEXTLOGSSHOULDGO 
%          /FOLDERWHEREMATFILESARE BATCHPREFIXNAME  QueueType
%
% Note that FOLDERWHEREMATFILESARE is usually the same as
% FOLDERWHEREMFILESARE. This is mainly if you are trying to re-run failed
% jobs - it only runs m files if there is no corresponding mat file located
% in the FOLDERWHEREMATFILESARE. For example:
% 
%  ./batchrun.sh /nfs/sabatini2_ata/PROJECTFOLDER
%  /nfs/sabatini2_ata/PROJECTFOLDER  /nfs/sabatini2_ata/
%   PROJECTFOLDER  Batch_  sq32hp
%
%  ./batchrun.sh /nfs/sabatini_dbw01/Fly200_40x_Results
%  /nfs/sabatini_dbw01/Fly200_40x_Results/Logs /nfs/
%   sabatini_dbw01/Fly200_40x_Results Batch_sl22_ normal
% 
% The first time you run batchrun.sh or after editing it, you must often
% change the file permissions by typing this at the command line:
% 
% chmod a+w batchrun.sh
% 
% If the batchrun.sh script doesn’t exist save the following as batchrun.sh
% (using any text editor):
% 
% #!/bin/sh
% if test $# -ne 5; then
%   echo "usage: $0 M_fileDir BatchTxtOutputDir mat_fileDir BatchFilePrefix
%         QueueType" 1>&2
%   exit 1
% fi
% # svn update
% cd CellProfiler
% svn update CellProfiler.m
% cd ImageTools
% svn update
% cd ..
% cd DataTools
% svn update
% cd ..
% cd Modules
% svn update
% cd ..
% cd CPsubfunctions
% svn update
% # start to process
% BATCHDIR=$1
% BATCHTXTOUTPUTDIR=$2
% BATCHMATOUTPUTDIR=$3
% BATCHFILEPREFIX=$4
% QueueType=$5
% MATLAB=/nfs/apps/matlab701
% LICENSE_SERVER="7182@pink-panther.csail.mit.edu"
% export DISPLAY=""
% # loop through each .mat file
% for i in $BATCHDIR/$BATCHFILEPREFIX*.m; do
%    BATCHFILENAME=`basename $i .m`
%     if [ ! -e $BATCHMATOUTPUTDIR/${BATCHFILENAME}_OUT.mat ]; then
%       echo Re-running $BATCHDIR/$BATCHFILENAME
%       bsub -q $5 -o $BATCHTXTOUTPUTDIR/$BATCHFILENAME.txt -u
%           xuefang_ma@wi.mit.edu -R 'rusage  
%           [img_kit=1:duration=1]' "$MATLAB/bin/matlab -nodisplay -nojvm
%            -c $LICENSE_SERVER < $BATCHDIR/ 
% $BATCHFILENAME.m"
%    fi
% done
%
% #INSTRUCTIONS
% #From the command line, logged into barra, submit the jobs using this
% script as follows: 
% #./batchrun.sh /FOLDERWHEREMFILESARE /FOLDERWHERETEXTLOGSSHOULDGO
% /FOLDERWHEREMATFILESARE BATCHPREFIXNAME QueueType
% #Note that FOLDERWHEREMATFILESARE is usually the same as
% FOLDERWHEREMFILESARE. This is mainly if you are trying to re-run failed
% jobs - it only runs m files if there is no corresponding mat file located
% in the FOLDERWHEREMATFILESARE.
% #For example:
% #./batchrun.sh /nfs/sabatini2_ata/PROJECTFOLDER
% /nfs/sabatini2_ata/PROJECTFOLDER /nfs/sabatini2_ata/ PROJECTFOLDER Batch_
% normal
% #(currently, there is a copy of this script at /home/xma so that is the
% directory from which the script should be run. The first time I ran it, I
% had to change the permissions by doing this: chmod a+w batchrun.sh)
%
% Choosing a queue:
% 
% QUEUE_NAME      PRIORITY 
% sq32hp           50  
% sq64hp           50 
% lq32hp           50 
% lq64hp           50  
% sq32mp           40
% lq32mp           40 
% lq64lp           30  
% sq32lp           20  
% fraenkel         20  
% normal           10 
% sq = short queue jobs <20 minutes
% lq = long queue jobs > 20 minutes
% 32 & 64 indicate 32 vs 64 bit applications 
% - not important at this time as our apps run both
% lp = low priority
% mp = medium priority
% hp  = high priority
% There are fewer machines in the hp and mp queues but they have higher
% priority. If you just have a few jobs that need to be run soon the mp and
% hp queues should be ideal. If you have many jobs with a lower turn around
% requirement the lp queues have more machines available but at a lower run
% priority. Of course this is a limited resource available to all so when
% the cluster is busy everything will take bit longer. When writing to
% sabatini2_ata, use sq32hp - other computers don’t have write permission
% sometimes.
% 
% 10. Transfer data files to Oracle/SQL database
% 
% Use the import.sql script to upload all data files
% 
% Commands for monitoring jobs
% 
% To check/edit the jobs during running (bsub functions):
% 
% List all jobs:      bjobs
% Count all jobs:     bjobs | wc -l
% Count running jobs: bjobs | grep RUN | wc -l; echo jobs_running
% Count pending jobs: bjobs | grep PEND | wc -l; echo jobs_pending
% Kill all jobs:      bkill 0
% Switch job to another queue:  bswitch sq32hp 189091
%   (where 189091 in the job number)
% To check how many jobs are running and pending for a particular sample:
%   bjobs -w | grep samplename | wc -l
% To check how many jobs are running and pending in each queue:
%   echo sq32hp; bjobs | grep sq32hp | wc -l
%   echo lq32hp; bjobs | grep lq32hp | wc -l 
%   echo lq64lp; bjobs | grep lq64lp | wc -l
%   echo normal; bjobs | grep normal | wc -l
% To submit an individual job: -B sends email at beginning of the job, -N
% at the end. -q QUEUENAME allows selecting a queue.
%   bsub -B -N -u carpenter@wi.mit.edu matlab -nodisplay -r Batch_2_to_2
% 
% To see what is in batchrun.sh, in terminal type:
%   less batchrun.sh
% 
% To edit batchrun.sh use an editor like vi or emacs, or pico (which works
% only in Terminal, not X windows), be sure to change permissions on the
% script after editing (see above):
%   pico batchrun.sh 
% 
% Checking the jobs after running: To see how many lines are present in
% each log file in the directory (an indicator of successful completion):
%   wc -l *.m.txt 
% 
% To see how many files are in the directory (subtract 1, I think)
%   ls | grep *OUT.mat | wc -l
% 
% To list jobs that Exited without completing by searching the log files
% for the word ‘Exited’:
%   less /Volumes/tap5/Fly200_40x_sl03Results/*.m.txt | grep Exited
% 
% Other notes:
% 1. COPY OPTIONS:
% 
% Example 1: drag and drop to gobo/carpente or gobo/sabatini1_ata
% For some locations, it may not be permitted to create a folder using
% Mac’s functions. In these cases, it should be possible to mkdir from the
% command line when logged into barra, or chmod a+w DIRNAME when logged
% into barra as carpente.
% 
% Example 2: In Terminal, from within the folder on local computer
% containing the batch files:
% scp Batch* carpente@barra.wi.mit.edu:/home/carpente/2005_02_07BatchFiles
% 
% Example 3: (similar, to retrieve output files): From within the
% destination folder in Terminal on the local computer:
% scp
% carpente@barra.wi.mit.edu:/home/carpente/CellProfiler/ExampleFlyImages/Ba
%    tch_2_to_2_OUT.mat .
% 
% 2. SERVER NAMING:
% 
% The cluster calls gobo “nfs”, so all instances where you might normally
% use gobo should be replaced with nfs. e.g. gobo/imaging becomes
% /nfs/imaging from the cluster’s perspective.
% The local computer uses the actual address of servers to use in place of
% “gobo”. Connect to the server using cifs://gobo/DIRNAME, then in Terminal
% ssh to barra, then cd /nfs then df. This will list the actual address,
% something like: tap2.wi.mit.edu:/imaging The name tap2 is then used in
% CellProfiler. e.g. gobo/imaging becomes /Volumes/tap2 from the local
% computer’s perspective. sabatini_dbw01 is /Volumes/tap5.
%
% 3. CHECKING WHY JOBS FAILED:
% 
% We now use the script batchrun.sh to re-run failed jobs automatically,
% but these commands may help diagnose problems:
% If you are having license problems you can try using CSAIL’s licenses in
% batchrun.sh: LICENSE_SERVER=”7182@pink-panther.csail.mit.edu”.
% Alternately, the following steps will look through all the text log files
% in a directory, look for the text “exit code” within those log files to
% find the batches that did not successfully complete, and move the
% corresponding .m-file to a subdirectory. You can then run batchrun.sh on
% the subdirectory (don’t forget to copy or move Batch_data.mat into the
% subdirectory as well, or point to the parent directory which contains the
% file.)
% Make the subdirectory BatchesToRerun and change permissions so you can
% write to it. Run this line first to see a printout of the proposed moved
% files:
% 
% grep -l "exit code" *.txt | sed "s/^/mv /" | sed 's/$/BatchesToRerun/' |
%     sed "s/.txt//"
% Run the same line with | sh appended to actually move the files:
% 
% grep -l "exit code" *.txt | sed "s/^/mv /" | sed 's/$/BatchesToRerun/' |
%     sed "s/.txt//" | sh
% Start the jobs in that subdirectory using batchrun.sh.
% -------------------------------------------------------------------------