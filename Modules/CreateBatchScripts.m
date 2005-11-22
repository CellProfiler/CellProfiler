function handles = CreateBatchScripts(handles)

% Help for the Create Batch Scripts module:
% Category: File Processing
%
% SHORT DESCRIPTION:
% Produces text script files which allow individual batches of images to be
% processed seperately on a cluster computer.
% *************************************************************************
%
% This module writes a batch (set) of Matlab scripts (m-files) that
% can be submitted in parallel to a cluster for faster processing.
%
% This module should be placed at the end of an image processing
% pipeline.  Settings include: the size of each batch that the full set
% of images should be split into, a prefix to prepend to the batch
% filenames, and several pathnames.  For jobs that you do not want to
% split into batches but simply want to run on a separate computer,
% set the batch size to a very large number (more than the number of
% cycles), which will create one large job.
%
% After the first cycle is processed, batch files are created and
% saved at the pathname you specify.  Each batch file is of the form
% Batch_X_to_Y.m (The prefix can be changed from Batch_ by the
% user), where X is the first cycle to be processed in the particular
% batch file, and Y is the last.  There is also a Batch_data.mat file
% that each script needs access to in order to initialize the processing.
%
% After the batch files are created, they can be submitted
% individually to the remote machines. Note that the batch files and
% Batch_data.mat file might have to be copied to the remote machines
% in order for them to have access to the data. The output files will
% be written in the directory where the batch files are running, which
% may or may not be the directory where the batch scripts are located.
% Details of how remote jobs will be started vary from location to
% location. Please consult your local cluster experts.
%
% After batch processing is complete, the output files can be merged
% by the MergeBatchOutput module.  For the simplest behavior in
% merging, it is best to save output files to a unique and initially
% empty directory.
%
% If the batch processing fails for some reason, the handles structure
% in the output file will have a field BatchError, and the error will
% also be written to standard out.  Check the output from the batch
% processes to make sure all batches complete.  Batches that fail for
% transient reasons can be resubmitted.
%
% The following is a script to be run from the command line of a
% terminal shell that will submit all jobs within a given folder to a
% cluster for processing. The script is started by typing this at the
% command line within a directory that contains a copy of the
% runallbatchjobs.sh file (all typed on one line):
% ./runallbatchjobs.sh
% /PATHTOFOLDERCONTAININGBATCHM-FILESANDBATCH_DATA
% /PATHTOFOLDERWHERETEXTLOGSSHOULDGO BATCHPREFIX
%
% Here is the actual code for runallbatchjobs.sh:
% ------------------
% #!/bin/sh
% if test $# -ne 3; then
%     echo "usage: $0 BatchDir BatchOutputDir BatchFilePrefix" 1>&2
%     exit 1
% fi
%
% BATCHDIR=$1
% BATCHOUTPUTDIR=$2
% BATCHFILEPREFIX=$3
% MATLAB=/nfs/apps/matlab701
%
% export DISPLAY=""
%
% for i in $BATCHDIR/$BATCHFILEPREFIX*.m; do
%     BATCHFILENAME=`basename $i`
%     bsub -o $BATCHOUTPUTDIR/$BATCHFILENAME.txt -u carpenter@wi.mit.edu -R 'rusage[img_kit=1:duration=1]' "$MATLAB/bin/matlab -nodisplay -nojvm < $BATCHDIR/$BATCHFILENAME"
% done
% ------------------
%
% Here are instructions for running jobs on the cluster at the
% Whitehead Institute:
%
% These instructions are for writing the batch files straight to the
% remote location. You can also write the batch files to the local
% computer and copy them over (see LOCAL ONLY steps).
%
% 1. Put your images on gobo somewhere. Currently, /gobo/imaging,
% /gobo/sabatini1_ata and /gobo/sabatini2_ata are all nfs mounted and
% accessible to the cluster.
%
% 2. Connect to that location from the CP G5, using Go > Connect to
% server.  Be sure to use nfs rather than cifs because the connection
% is much faster.
%
% 3. Log into barra (a server and a front end to submit jobs to the
% cluster) as user=carpente, using Terminal or X Windows: ssh -X
% carpente@barra.wi.mit.edu
%
% 4. As carpente, logged into barra, make a directory for the project
% somewhere on gobo that is accessible to the cluster, and give write
% permission to the new directory: mkdir DIRNAME chmod a+w DIRNAME. (I
% think it is necessary to use the command line as carpente rather
% than just making a folder using the Mac, because the
% CellProfilerUser does not have write permission on sabatini_ata's, I
% think.)
%
% 5. (LOCAL ONLY) Make a folder on the local computer.
%
% 6. In CellProfiler, add the module CreateBatchScripts to the end of
% the pipeline, and enter the settings (see Notes below for server
% naming issues):
% CreateBatchScripts module:
% What prefix ahould be added to the batch file names?
% Batch_
% What is the path to the CellProfiler folder on the cluster machines?
% /home/carpente/CellProfiler
% What is the path to the image directory on the cluster machines?
% .
% What is the path to the directory where batch output should be
% written on the cluster machines?
% .
% What is the path to the directory where you want to save the batch
% files?
% .
% What is the path to the directory where the batch data file will be
% saved on the cluster machines?
% .
% If pathnames are specified differently between the local and cluster
% machines, enter that part of the pathname from the local machine's
% perspective
% Volumes/tap6
% If pathnames are specified differently between the local and cluster
% machines, enter that part of the pathname from the cluster machines'
% perspective
% nfs/sabatini2_ata
% SaveImages module:
% Enter the pathname to the directory where you want to save the
% images: (note: I am not sure if we can currently save images on each
% cycle through the pipeline)
%  /nfs/sabatini2_ata/PROJECTNAME
% Default image directory: /Volumes/tap6/IMAGEDIRECTORY
% Default output directory: /Volumes/tap6/PROJECTNAME (or LOCAL
% folder)
%
% 7. Run the pipeline through CellProfiler, which will analyze the
% first set and create batch files on the local computer.
%
% 8. (LOCAL ONLY) Drag the BatchFiles folder (which now contains the
% Batch  .m files and .mat file) and BatchOutputFiles folder (empty)
% from the local computer into the project folder at the remote
% location.
%
% 9. From the command line, logged into barra, make sure the
% CellProfiler code at  /home/carpente/CellProfiler is up to date by
% changing to /home/carpente/CellProfiler and type: cvs update.  Any
% compiled functions in the code must be compiled for every type of
% architecture present in the cluster using the matlab command mex
% (PC, Mac, Unix, 64-bit, etc).
%
% 10. From the command line, logged into barra, submit the jobs using
% the script runallbatchjobs.sh as follows: ./runallbatchjobs.sh
% /BATCHFILESFOLDER /FOLDERWHERETEXTLOGSSHOULDGO BATCHPREFIXNAME For
% example: ./runallbatchjobs.sh /nfs/sabatini2_ata/PROJECTFOLDER
% /nfs/sabatini2_ata/PROJECTFOLDER Batch_ (currently, there is a copy
% of this script at /home/carpente so that is the directory from which
% the script should be run. The first time I ran it, I had to change
% the permissions by doing this: chmod a+w runallbatchjobs.sh)
%
% 11. Certain jobs fail for transient reasons and simply need to be
% resubmitted. The following code will look through all the text log
% files in a directory, look for the text "exit code" within those log
% files to find the batches that did not successfully complete, and
% move the corresponding .m-file to a subdirectory. You can then run
% runallbatchjobs.sh on the subdirectory (don't forget to copy or move
% Batch_data.mat into the subdirectory as well, or point to the parent
% directory which contains the file.)
%  a. Make the subdirectory BatchesToRerun and change permissions so
%  you can write to it.
%  b. Run this line first to see a printout of the proposed moved files:
% grep -l "exit code" *.txt | sed "s/^/mv /" | sed 's/$/
% BatchesToRerun/' | sed "s/.txt//"
%  c. Run the same line with | sh appended to actually move the files:
% grep -l "exit code" *.txt | sed "s/^/mv /" | sed 's/$/
% BatchesToRerun/' | sed "s/.txt//" | sh
%  d. Start the jobs in that subdirectory using runallbatchjobs.
% --------------------------------------------------------------------
%
% Bsub Functions:
% List all jobs: bjobs
% Count all jobs: bjobs | wc -l
% Count running jobs: bjobs | grep RUN | wc -l
% Kill all jobs bkill 0
% Submit an individual job: copy a bsub line out of batAll.sh, like
% this:
% bsub -B -N -u carpenter@wi.mit.edu matlab -nodisplay -r Batch_2_to_2
% -B sends email at beginning of job, -N at the end.
% To see what is in batAll.sh: less batAll.sh
% To edit batAll.sh: pico batAll.sh (Works only in Terminal, not in X
% Windows).
% To show the number of lines in an output file: wc -l *_OUT.txt
%
% Other notes:
% 1. COPY OPTIONS:
% 	Example 1: drag and drop to /gobo/carpente or gobo/sabatini1_ata
% For some locations, it may not be permitted to create a folder using
% Mac's functions. In these cases, it should be possible to mkdir from
% the command line when logged into barra, or chmod a+w DIRNAME when
% logged into barra as carpente.
% 	Example 2: In Terminal, from within the folder on local computer
% 	containing the batch files:
% scp Batch*
% carpente@barra.wi.mit.edu:/home/carpente/2005_02_07BatchFiles
% 	Example 3: (similar, to retrieve output files):  From within the
% 	destination folder in Terminal on the local computer:
% scp
% carpente@barra.wi.mit.edu:/home/carpente/CellProfiler/ExampleFlyImag
% es/Test3Batch_2_to_2_OUT.mat .
% 2. SERVER NAMING:
% - The cluster calls gobo "nfs", so all instances where you might
% normally use gobo should be replaced with nfs.   e.g. gobo/imaging
% becomes /nfs/imaging from the cluster's perspective.
% - The local computer uses the actual address of servers to use in
% place of "gobo". Connect to the server using cifs://gobo/DIRNAME,
% then in Terminal ssh to barra, then df /nfs/DIRNAME, where DIRNAME
% is something like imaging or sabatini1_ata. This will list the
% actual address, something like: tap2.wi.mit.edu:/imaging   The name
% tap2 is then used in CellProfiler.    e.g. gobo/imaging becomes
% /Volumes/tap2 from the local computer's perspective.
%
% See also MERGEBATCHOUTPUT.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne Carpenter
%   Thouis Jones
%   In Han Kang
%   Ola Friman
%   Steve Lowe
%   Joo Han Chang
%   Colin Clarke
%   Mike Lamprecht
%   Susan Ma
%   Wyman Li
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

%%% Reads the current module number, because this is needed to find the
%%% variable values that the user entered.
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

%%% Saves a snapshot of handles.
handles_in = handles;

%%% Checks that this is the last module in the analysis path.
if (CurrentModuleNum ~= handles.Current.NumberOfModules),
    error([ModuleName, ' must be the last module in the pipeline.']);
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
    BatchCellProfilerPath = strrep(BatchCellProfilerPath,OldPathname,NewPathname);
    BatchImagePath = strrep(BatchImagePath,OldPathname,NewPathname);
    BatchOutputPath = strrep(BatchOutputPath,OldPathname,NewPathname);
    BatchRemotePath = strrep(BatchRemotePath,OldPathname,NewPathname);
    %%% Changes the default output and image pathnames.
    OldDefaultOutputDirectory = handles.Current.DefaultOutputDirectory;
    NewDefaultOutputDirectory = strrep(OldDefaultOutputDirectory,OldPathname,NewPathname);
    handles.Current.DefaultOutputDirectory = NewDefaultOutputDirectory;
    OldDefaultImageDirectory = handles.Current.DefaultImageDirectory;
    NewDefaultImageDirectory = strrep(OldDefaultImageDirectory,OldPathname,NewPathname);
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
    fprintf(BatchFile, '    setbeinganalyzed = handles.Current.SetBeingAnalyzed;\n');
    fprintf(BatchFile, '    for SlotNumber = 1:handles.Current.NumberOfModules,\n');
    fprintf(BatchFile, '        ModuleNumberAsString = sprintf(''%%02d'', SlotNumber);\n');
    fprintf(BatchFile, '        ModuleName = char(handles.Settings.ModuleNames(SlotNumber));\n');
    fprintf(BatchFile, '        if iscellstr(handles.Settings.ModuleNames(SlotNumber)) == 0\n');
    fprintf(BatchFile, '        else\n');
    fprintf(BatchFile, '            handles.Current.CurrentModuleNumber = ModuleNumberAsString;\n');
    fprintf(BatchFile, '            try\n');
    fprintf(BatchFile, '                eval([''handles = '',ModuleName,''(handles);''])\n');
    fprintf(BatchFile, '            catch\n');
    fprintf(BatchFile, '                handles.BatchError = [ModuleName '' '' lasterr];\n');
    fprintf(BatchFile, '                disp([''Batch Error: '' ModuleName '' '' lasterr]);\n');
    fprintf(BatchFile, '                rethrow(lasterror);\n');
    fprintf(BatchFile, '                quit;\n');
    fprintf(BatchFile, '            end\n');
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

%%% Undo the changes to handles.Pipeline, above.
handles = handles_in;

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% The figure window display is unnecessary for this module, so the figure
%%% window is closed if it was previously open.
%%% Determines the figure number.
fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
%%% If the window is open, it is closed.
if any(findobj == ThisModuleFigureNumber) == 1;
    delete(ThisModuleFigureNumber)
end