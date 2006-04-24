function GSBatchProcessing
helpdlg(help('GSBatchProcessing'))

% CellProfiler is designed to analyze images in a high-throughput manner.
% Once a pipeline has been established for a set of images, CellProfiler
% can export batches of images to be analyzed on a cluster. We have
% successfully analyzed over 130,000 images for one analysis in this
% manner. This is accomplished by breaking the entire set of images into
% separate batches, and then submitting each of these batches as individual
% jobs to a cluster. Each individual batch can be separately analyzed from
% the rest.
% 
% There are two methods of processing these batches on a cluster. The first
% is by producing small MatLab script files which specify the images to
% analyze. For this method, each cluster computer must have a MatLab
% license. The other method is by producing small MatLab .mat files which
% specify the images to analyze. This method does NOT require any MatLab
% licenses, but is more difficult to set up if the version on the website
% does not work on your cluster.
% 
% *************** SETTING UP CLUSTER WITH MATLAB **************************
% Step 1: Create a folder on your cluster for CellProfiler (e.g.
% /home/username/CellProfiler). THIS FOLDER MUST BE CONNECTED TO THE
% CLUSTER COMPUTERS NETWORK AND READABLE BY ALL. If you don't know what
% this means, please ask your IT department for help.
% 
% Step 2: Copy all CellProfiler source code files into this folder, keeping
% the file structure intact. THIS VERSION OF CELLPROFILER MUST BE THE SAME
% AS THE VERSION USED TO CREATE THE PIPELINE. SOMETIMES MODULES ARE CHANGED
% BETWEEN VERSIONS AND THIS CAN CAUSE ERRORS.
% 
% Step 3: Create batchrun.sh file. This file is how jobs are submitted to
% the cluster. THIS IS CURRENTLY DESIGNED TO WORK WITH OUR CLUSTER WHICH
% RUNS LSF SOFTWARE. THERE ARE DIFFERENT CLUSTER SOFTWARE PROGRAMS AND THIS
% FILE MUST BE EDITED TO WORK WITH YOUR SOFTWARE. PLEASE CONTACT IT FOR
% HELP WITH THIS. Open any text editor and copy the following code:
% 
% #!/bin/sh
% if test $# -ne 5; then
%   echo "usage: $0 M_fileDir BatchTxtOutputDir mat_fileDir BatchFilePrefix QueueType" 1>&2 exit 1
% fi
% # start to process
% BATCHDIR=$1
% BATCHTXTOUTPUTDIR=$2
% BATCHMATOUTPUTDIR=$3
% BATCHFILEPREFIX=$4
% QueueType=$5
% MATLAB=/SOME_PATH/MATLAB
% LICENSE_SERVER="12345@yourservers.edu"
% export DISPLAY=""
% # loop through each .mat file
% for i in $BATCHDIR/$BATCHFILEPREFIX*.m; do
%    BATCHFILENAME=`basename $i .m`
%     if [ ! -e $BATCHMATOUTPUTDIR/${BATCHFILENAME}_OUT.mat ]; then
%       echo Re-running $BATCHDIR/$BATCHFILENAME
%       bsub -q $5 -o $BATCHTXTOUTPUTDIR/$BATCHFILENAME.txt -u
%       username@wi.mit.edu -R 'rusage [img_kit=1:duration=1]' "$MATLAB/bin/matlab -nodisplay -nojvm -c $LICENSE_SERVER < $BATCHDIR/$BATCHFILENAME.m"
%    fi
% done
% #INSTRUCTIONS
% #From the command line, logged into your cluster submit the jobs using this script as follows:
% #./batchrun.sh /FOLDERWHEREMFILESARE /FOLDERWHERETEXTLOGSSHOULDGO /FOLDERWHEREMATFILESARE BATCHPREFIXNAME QueueType
% #Note that FOLDERWHEREMATFILESARE is usually the same as FOLDERWHEREMFILESARE. This is mainly
% #if you are trying to re-run failed jobs - it only runs m files if there is no corresponding
% #mat file located in the FOLDERWHEREMATFILESARE.
% #For example:
% #./batchrun.sh /nfs/sabatini2_ata/PROJECTFOLDER /nfs/sabatini2_ata/PROJECTFOLDER /nfs/sabatini2_ata/ PROJECTFOLDER Batch_ normal
% # END COPY
% 
% Save this file to any directory, usually your home directory is fine.
% Then change the following lines to fit your cluster:
% MATLAB=/SOME_PATH/MATLAB
% LICENSE_SERVER="12345@yourservers.edu"
% 
% Also, you can specify your e-mail address after the bsub command.
% 
% Step 4: Change batchrun.sh to be executable. Open a terminal, navigate to
% the folder where batchrun.sh is located, and type:
% 
% chmod a+x batchrun.sh
% 
% If you don't know what this means, please ask your IT department for
% help.
% 
% *********** END OF SETTING UP CLUSTER WITH MATLAB ***********************
% 
% 
% ************** SETTING UP CLUSTER WITHOUT MATLAB ************************
% Step 1: Download and install correct version of CPCluster. If the generic
% Linux version does not work, you may have to download the source code and
% compile it on your cluster. This requires a single MatLab license.
%
% Step 2: Create batchrun.sh file. This file is how jobs are submitted to
% the cluster. THIS IS CURRENTLY DESIGNED TO WORK WITH OUR CLUSTER WHICH
% RUNS LSF SOFTWARE. THERE ARE DIFFERENT CLUSTER SOFTWARE PROGRAMS AND THIS
% FILE MUST BE EDITED TO WORK WITH YOUR SOFTWARE. PLEASE CONTACT IT FOR
% HELP WITH THIS. Open any text editor and copy the following code:
% 
% #!/bin/sh
% if test $# -ne 5; then
%     echo "usage: $0 M_fileDir BatchTxtOutputDir mat_fileDir BatchFilePrefix QueueType" 1>&2
%     exit 1
% fi
% 
% BATCHDIR=$1
% BATCHTXTOUTPUTDIR=$2
% BATCHMATOUTPUTDIR=$3
% BATCHFILEPREFIX=$4
% QueueType=$5
% 
% echo $BATCHDIR
% echo $BATCHTXTOUTPUTDIR
% echo $BATCHMATOUTPUTDIR
% echo $BATCHFILEPREFIX
% echo $QueueType
% 
% CPCluster=/Users/username/CPCluster
% 
% for i in $BATCHDIR/$BATCHFILEPREFIX*.mat; do
%     BATCHFILENAME=`basename $i .mat`
%     if [ $BATCHFILENAME != ${BATCHFILEPREFIX}data ]; then 
%         if [ ! -e $BATCHMATOUTPUTDIR/${BATCHFILENAME}_OUT.mat ]; then
%             echo Running $BATCHDIR/$BATCHFILENAME
%             qsub -S /bin/bash -o $BATCHTXTOUTPUTDIR/$BATCHFILENAME.txt -M username@wi.mit.edu $CPCluster/CPCluster.command $BATCHDIR/${BATCHFILEPREFIX}data.mat $BATCHDIR/$BATCHFILENAME.mat
%         fi
%     fi
% done
% 
% Save this file to any directory, usually your home directory is fine.
% Then change the following lines to fit your cluster:
% CPCluster=/Users/username/CPCluster
% 
% Also, you can specify your e-mail address after the qsub command.
% 
% Step 4: Change batchrun.sh to be executable. Open a terminal, navigate to
% the folder where batchrun.sh is located, and type:
% 
% chmod a+x batchrun.sh
% 
% If you don't know what this means, please ask your IT department for
% help.
%
% ********** END OF SETTING UP CLUSTER WITHOUT MATLAB *********************
% 
% ************ SUBMITTING FILES FOR BATCH PROCESSING **********************
% Step 1:
% Create a project folder on your cluster. For high throughput analysis, it
% is a good idea to create a separate project folder for each run. In
% general, we like to name our folders with the following convention:
% 200X_XX_XX_ProjectName. Within this folder, we usually create an "images"
% folder and an "output" folder. We then transfer all of our images to the
% images folder. The output folder is where all of your data will be
% stored. THESE FOLDERS MUST BE CONNECTED TO THE CLUSTER COMPUTERS NETWORK
% AND THE OUTPUT FOLDER MUST BE WRITEABLE BY EVERYONE. The output folder
% must be writeable by everyone (or at least your cluster) because each of
% the separate cluster computers will write an output file in this folder.
% If you don't know what this means, ask your IT department for help.
% 
% Step 2: Create a pipeline for your image set. In this process, you must
% be careful to consider the worst case scenario in your images. For
% instance, some images may contain no cells. If this happens, the
% automatic thresholding algorithms will incorrectly choose a very low
% threshold, and therefore create many objects which don't exist. This can
% be overcome by setting a "minimum" threshold in IdentifyPrimAutomatic
% module.
% 
% Step 3: Add the module CreateBatchFiles to your pipeline. Please refer to
% the help for this module to choose the correct settings. If you are
% processing large batches of images, you may also consider adding
% ExportToDatabase to your pipeline, before the CreateBatchFiles module.
% This module will export your data into comma separated values (CSV), and
% also create a script to import your data into Oracle or MySQL databases.
% 
% Step 4: Analyze the first image set. After you add the CreateBatchFiles
% module, the analysis will stop after the first image set. The
% CreateBatchFiles will have exported the correct files to the output
% folder you created in Step 1. You are now ready to submit to the cluster.
% 
% Step 5: Log on to your cluster, and navigate to the directory where you
% have saved the batchrun.sh file (See "Setting Up Cluster For
% CellProfiler"). The usage of batchrun.sh is as follows:
% 
% ./batchrun.sh M_fileDir BatchTxtOutputDir mat_fileDir BatchFilePrefix QueueType
% 
% where M_fileDir is the location of the files exported by CreateBatchFiles
% in step 4, BatchTxtOutputDir is where you want to store the txt files
% which have the output of MatLab during the analysis, mat_fileDir is the
% folder where XXX_data.mat is located (this file is created in Step 4),
% BatchFilePrefix is the prefix named in CreateBatchFiles (usually Batch_),
% and QueueType is the queue for your cluster. Usually, the first three
% arguments are all the same. Here is an example of how you would submit
% all of your batch scripts to the cluster:
% 
% ./batchrun.sh /Some_Path/200X_XX_XX_ProjectName/output /Some_Path/200X_XX_XX_ProjectName/output
% /Some_Path/200X_XX_XX_ProjectName/output Batch_ normal
% 
% In this case, the output folder contains the script files, the
% XXX_data.mat file, and will also store the txt files with the MatLab
% output. The prefix is Batch_ so XXX_data.mat is actually Batch_data.mat.
% The queue type is normal, but this is specific to your cluster. Ask your
% IT department what queue's are available for your use.
% 
% Once all the jobs are submitted, the cluster will run each script
% individually and produce a separate output file in the output directory.
% Then you can decide how to view your data. In general, large analysis
% will be loaded into a database. Please refer to ExportToDatabase for
% information on how to do this.
% ********* END OF SUBMITTING FILES FOR BATCH PROCESSING ******************