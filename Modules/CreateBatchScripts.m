function handles = CreateBatchScripts(handles)

% Help for the Create Batch Scripts module:
% Category: File Processing
%
% SHORT DESCRIPTION:
% Produces script files (these are files of plain text) which allow
% individual batches of images to be processed separately on a cluster
% of computers.
% *************************************************************************
% Note: this module is beta-version and has not been thoroughly checked.
%
% This module creates a set of Matlab scripts (m-files) that can be
% submitted in parallel to a cluster for faster processing. This module
% should be placed at the end of an image processing pipeline.
% 
% Settings:
% Options include the size of each batch that the full set of images should
% be split into, a prefix to prepend to the batch filenames, and several
% pathnames describing where the batches will be processed.  For jobs that
% you do not want to split into batches but simply want to run on a
% separate computer, set the batch size to a very large number (more than
% the total number of cycles), which will create one large job.
%
% How it works: 
% After the first cycle is processed on your local computer, batch files
% are created and saved at the pathname you specify.  Each batch file is of
% the form Batch_X_to_Y.m (The prefix can be changed from Batch_ by the
% user), where X is the first cycle to be processed in the particular batch
% file, and Y is the last.  There is also a Batch_data.mat file that each
% script needs access to in order to initialize the processing.
%
% After the batch files are created, they can be submitted individually to
% the remote machines. Note that the batch files and Batch_data.mat file
% might have to be copied to the remote machines in order for them to have
% access to the data. The output files will be written in the directory
% where the batch files are running, which may or may not be the directory
% where the batch scripts are located. Details of how remote jobs will be
% started vary from location to location. Please consult your local cluster
% experts.
%
% After batch processing is complete, the output files can be merged by the
% Merge Batch Output module.  This is not recommended of course if your
% output files are huge and will result in a file that is too large to be
% opened on your computer. For the simplest behavior in merging, it is best
% to save output files to a unique and initially empty directory.
%
% If the batch processing fails for some reason, the handles structure in
% the output file will have a field BatchError, and the error will also be
% written to standard out.  Check the output from the batch processes to
% make sure all batches complete.  Batches that fail for transient reasons
% can be resubmitted.
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
% ÔeveryoneÕ for the new folder (so each cluster computer is allowed to
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
% machines, enter that part of the pathname from the local machineÕs
% perspective	Volumes/tap6
% If pathnames are specified differently between the local and cluster
% machines, enter that part of the pathname from the cluster machinesÕ
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
% If the batchrun.sh script doesnÕt exist save the following as batchrun.sh
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
% sabatini2_ata, use sq32hp - other computers donÕt have write permission
% sometimes.
% 
% 10. Transfer data files to Oracle/SQL database
% 
% Use the import.sql script to upload all data files
% 
% 11. Commands for monitoring jobs
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
% for the word ÔExitedÕ:
%   less /Volumes/tap5/Fly200_40x_sl03Results/*.m.txt | grep Exited
% 
% Other notes:
% 
% 1. COPY OPTIONS:
% 
% Example 1: drag and drop to gobo/carpente or gobo/sabatini1_ata
% For some locations, it may not be permitted to create a folder using
% MacÕs functions. In these cases, it should be possible to mkdir from the
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
% The cluster calls gobo ÒnfsÓ, so all instances where you might normally
% use gobo should be replaced with nfs. e.g. gobo/imaging becomes
% /nfs/imaging from the clusterÕs perspective.
% The local computer uses the actual address of servers to use in place of
% ÒgoboÓ. Connect to the server using cifs://gobo/DIRNAME, then in Terminal
% ssh to barra, then cd /nfs then df. This will list the actual address,
% something like: tap2.wi.mit.edu:/imaging The name tap2 is then used in
% CellProfiler. e.g. gobo/imaging becomes /Volumes/tap2 from the local
% computerÕs perspective. sabatini_dbw01 is /Volumes/tap5.
%
% 3. CHECKING WHY JOBS FAILED:
% 
% We now use the script batchrun.sh to re-run failed jobs automatically,
% but these commands may help diagnose problems:
% If you are having license problems you can try using CSAILÕs licenses in
% batchrun.sh: LICENSE_SERVER=Ó7182@pink-panther.csail.mit.eduÓ.
% Alternately, the following steps will look through all the text log files
% in a directory, look for the text Òexit codeÓ within those log files to
% find the batches that did not successfully complete, and move the
% corresponding .m-file to a subdirectory. You can then run batchrun.sh on
% the subdirectory (donÕt forget to copy or move Batch_data.mat into the
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
% See also MergeBatchOutput.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne E. Carpenter
%   Thouis Ray Jones
%   In Han Kang
%   Ola Friman
%   Steve Lowe
%   Joo Han Chang
%   Colin Clarke
%   Mike Lamprecht
%   Susan Ma
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = How many cycles should be in each batch?
%defaultVAR01 = 100
BatchSize = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,1}));

%textVAR02 = What prefix should be used to name the batch files?
%defaultVAR02 = Batch_
BatchFilePrefix = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%pathnametextVAR03 = What is the path to the CellProfiler folder on the cluster machines?  Leave a period (.) to use the default module directory.
%defaultVAR03 = .
BatchCellProfilerPath = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%pathnametextVAR04 = What is the path to the image directory on the cluster machines? Leave a period (.) to use the default image directory.
%defaultVAR04 = .
BatchImagePath = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%pathnametextVAR05 = What is the path to the directory where batch output should be written on the cluster machines? Leave a period (.) to use the default output directory.
%defaultVAR05 = .
BatchOutputPath = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%pathnametextVAR06 = What is the path to the directory where you want to save the batch files? Leave a period (.) to use the default output directory.
%defaultVAR06 = .
BatchSavePath = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%pathnametextVAR07 = What is the path to the directory where the batch data file will be saved on the cluster machines? Leave a period (.) to use the default output directory.
%defaultVAR07 = .
BatchRemotePath = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%pathnametextVAR08 = If pathnames are specified differently between the local and cluster machines, enter that part of the pathname from the local machine's perspective, omitting leading and trailing slashes. Otherwise, leave a period (.)
%defaultVAR08 = .
OldPathname = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%pathnametextVAR09 = If pathnames are specified differently between the local and cluster machines, enter that part of the pathname from the cluster machines' perspective, omitting leading and trailing slashes. Otherwise, leave a period (.)
%defaultVAR09 = .
NewPathname = char(handles.Settings.VariableValues{CurrentModuleNum,9});

%textVAR10 = Note: This module must be the last one in the analysis pipeline.

%%%VariableRevisionNumber = 6

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

if strncmp(BatchSavePath, '.',1)
    if length(BatchSavePath) == 1
        BatchSavePath = handles.Current.DefaultOutputDirectory;
    else
        BatchSavePath = fullfile(handles.Current.DefaultOutputDirectory,BatchSavePath(2:end));
    end
end

if strncmp(BatchRemotePath, '.',1)
    if length(BatchRemotePath) == 1
        BatchRemotePath = handles.Current.DefaultOutputDirectory;
    else
        BatchRemotePath = fullfile(handles.Current.DefaultOutputDirectory,BatchRemotePath(2:end));
    end
end

if strncmp(BatchImagePath, '.',1)
    if length(BatchImagePath) == 1
        BatchImagePath = handles.Current.DefaultImageDirectory;
    else
        BatchImagePath = fullfile(handles.Current.DefaultImageDirectory,BatchImagePath(2:end));
    end
end

if strncmp(BatchOutputPath, '.',1)
    if length(BatchOutputPath) == 1
        BatchOutputPath = handles.Current.DefaultOutputDirectory;
    else
        BatchOutputPath = fullfile(handles.Current.DefaultOutputDirectory,BatchOutputPath(2:end));
    end
end

if strncmp(BatchCellProfilerPath, '.',1)
    BatchCellProfilerPath = fullfile(handles.Preferences.DefaultModuleDirectory, '..');
end

%%% Checks that this is the last module in the analysis path.
if (CurrentModuleNum ~= handles.Current.NumberOfModules),
    error(['Image processing was canceled because ', ModuleName, ' must be the last module in the pipeline.']);
end;

%%% If this isn't the first cycle, we are running on the
%%% cluster, and should just continue.
if (handles.Current.SetBeingAnalyzed > 1),
    return;
end

%%% Saves a copy of the handles structure to revert back to later. The
%%% altered handles must be saved using the variable name 'handles'
%%% because the save function will not allow us to save a variable
%%% under a different name.
PreservedHandles = handles;

%%% Changes parts of several pathnames if the user has
%%% specified that parts of the pathname are named differently from
%%% the perspective of the local computer vs. the cluster
%%% machines.
if strcmp(OldPathname, '.') ~= 1
    %%% Changes pathnames in variables within this module.
    %%% BatchSavePath is not changed, because that function is carried
    %%% out on the local machine.
    %%% BatchCellProfilerPath = strrep(fullfile(NewPathname,strrep(BatchCellProfilerPath,OldPathname,'')),'\','/');
    BatchImagePath = strrep(fullfile(NewPathname,strrep(BatchImagePath,OldPathname,'')),'\','/');
    BatchOutputPath = strrep(fullfile(NewPathname,strrep(BatchOutputPath,OldPathname,'')),'\','/');
    BatchRemotePath = strrep(fullfile(NewPathname,strrep(BatchRemotePath,OldPathname,'')),'\','/');
    %%% Changes the default output and image pathnames.
    NewDefaultOutputDirectory = strrep(fullfile(NewPathname,strrep(handles.Current.DefaultOutputDirectory,OldPathname,'')),'\','/');
    handles.Current.DefaultOutputDirectory = NewDefaultOutputDirectory;
    NewDefaultImageDirectory = strrep(fullfile(NewPathname,strrep(handles.Current.DefaultImageDirectory,OldPathname,'')),'\','/');
    handles.Current.DefaultImageDirectory = NewDefaultImageDirectory;
end

%%% Makes some changes to the handles structure that will be
%%% saved and fed to the cluster machines.
%%% Rewrites the pathnames (relating to where images are stored) in
%%% the handles structure for the remote machines.
Fieldnames = fieldnames(handles.Pipeline);
PathFieldnames = Fieldnames(strncmp(Fieldnames,'Pathname',8)==1);
for i = 1:length(PathFieldnames),
    handles.Pipeline.(PathFieldnames{i}) = BatchImagePath;
end

%%% Saves the altered handles in a file which the user will feed to
%%% the remote machines.
PathAndFileName = fullfile(BatchSavePath, [BatchFilePrefix 'data.mat']);
save(PathAndFileName, 'handles', '-v6');

%%% Reverts to the preserved handles. This prevents errors from
%%% occurring as a result of the fact that we have changed the default
%%% output directory, and possibly pathnames (which, actually, I don't
%%% think is a problem).
handles = PreservedHandles;

%%% Create the individual batch files
if (BatchSize <= 0)
    BatchSize = 100;
end
for n = 2:BatchSize:handles.Current.NumberOfImageSets,
    StartImage = n;
    EndImage = min(StartImage + BatchSize - 1, handles.Current.NumberOfImageSets);
    BatchFileName = sprintf('%s%d_to_%d.m', BatchFilePrefix, StartImage, EndImage);
    BatchFile = fopen(fullfile(BatchSavePath, BatchFileName), 'wt');
    fprintf(BatchFile, 'addpath(genpath(''%s''));\n', BatchCellProfilerPath);
    fprintf(BatchFile, 'BatchFilePrefix = ''%s'';\n', BatchFilePrefix);
    fprintf(BatchFile, 'StartImage = %d;\n', StartImage);
    fprintf(BatchFile, 'EndImage = %d;\n', EndImage);
    fprintf(BatchFile, 'tic;\n');
    fprintf(BatchFile, 'load([''%s/'' BatchFilePrefix ''data.mat'']);\n', BatchRemotePath);
    fprintf(BatchFile, 'handles.Current.BatchInfo.Start = StartImage;\n');
    fprintf(BatchFile, 'handles.Current.BatchInfo.End = EndImage;\n');
    fprintf(BatchFile, 'for BatchSetBeingAnalyzed = StartImage:EndImage,\n');
    fprintf(BatchFile, '    disp(sprintf(''Analyzing set %%d'', BatchSetBeingAnalyzed));\n');
    fprintf(BatchFile, '    toc;\n');
    fprintf(BatchFile, '    handles.Current.SetBeingAnalyzed = BatchSetBeingAnalyzed;\n');
    fprintf(BatchFile, '    for SlotNumber = 1:handles.Current.NumberOfModules,\n');
    fprintf(BatchFile, '        ModuleNumberAsString = sprintf(''%%02d'', SlotNumber);\n');
    fprintf(BatchFile, '        ModuleName = char(handles.Settings.ModuleNames(SlotNumber));\n');
    fprintf(BatchFile, '        handles.Current.CurrentModuleNumber = ModuleNumberAsString;\n');
    fprintf(BatchFile, '        try\n');
    fprintf(BatchFile, '            handles = feval(ModuleName,handles);\n');
    fprintf(BatchFile, '        catch\n');
    fprintf(BatchFile, '            handles.BatchError = [ModuleName '' '' lasterr];\n');
    fprintf(BatchFile, '            disp([''Batch Error: '' ModuleName '' '' lasterr]);\n');
    fprintf(BatchFile, '            rethrow(lasterror);\n');
    fprintf(BatchFile, '            quit;\n');
    fprintf(BatchFile, '        end\n');
    fprintf(BatchFile, '    end\n');
    fprintf(BatchFile, 'end\n');
    fprintf(BatchFile, 'cd(''%s'');\n', BatchOutputPath);
    fprintf(BatchFile, 'handles.Pipeline = [];');
    fprintf(BatchFile, 'eval([''save '',sprintf(''%%s%%d_to_%%d_OUT'', BatchFilePrefix, StartImage, EndImage), '' handles;'']);\n');
    fclose(BatchFile);
end

CPhelpdlg('Batch files have been written.  This analysis pipeline will now stop.  You should submit the invidual .m scripts for processing on your cluster.', 'BatchFilesDialog');

%%% This is the first cycle, so this is the first time seeing this
%%% module.  It should cause a cancel so no further processing is done
%%% on this machine.
set(handles.timertexthandle,'string','Cancel')

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% The figure window display is unnecessary for this module, so it is
%%% closed during the starting image cycle.
if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
    ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
    if any(findobj == ThisModuleFigureNumber)
        close(ThisModuleFigureNumber)
    end
end