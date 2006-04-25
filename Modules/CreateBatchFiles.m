function handles = CreateBatchFiles(handles)

% Help for the Create Batch Files module:
% Category: File Processing
%
% SHORT DESCRIPTION:
% Produces script files (these are files of plain text) or mat files
% (MatLab files) which allow individual batches of images to be processed
% separately on a cluster of computers.
% *************************************************************************
%
% This module creates a set of Matlab scripts (m-files) or mat-files that
% can be submitted in parallel to a cluster for faster processing. This
% module should be placed at the end of an image processing pipeline.
%
% Before using this module, you should read Help -> Getting Started ->
% Batch Processing. Here you will learn how to set up your cluster for
% batch processing.
% 
% Settings:
% Scripts or Files: If your cluster has MatLab licenses for every node, you
% can produce script files for each batch of images. If it does not, you
% can produce mat-files which will be read by the compiled CPCluster
% program. For more information, please read Help -> Getting Started ->
% Batch Processing.
%
% Batch Size: This determines how many images will be analyzed in each set.
% If you you do not want to split a job but want to send it to the cluster,
% so it does not use a computer you might be using, you can set the batch
% size to a very large number (more than the total number of cycles) and
% this will create one large job which can be submitted to the cluster. In
% general, you do not want your batch size to be too large. If one image
% fails, the whole analysis will stop and you will have to start from the
% beginning. If you have a smaller batch size, the job that failed will not
% take as long to re-run.
%
% Batch Prefix: This determines the prefix for all the batch files.
%
% CellProfiler Path: Here you must specify the exact location of
% CellProfiler files as seen by the cluster computers.
%
% Other Paths: You can either specify the exact paths as seen by the
% cluster computers, or you can leave a period (.) to use the default image
% and output folders. The last two parameters allow you to use the default
% image and output folders but switch the beginning path. For example, when
% starting with a PC computer and going to a Linux machine, the path may be
% the same except the first notation:
%
% PC:    \\remoteserver1\cluster\project
% Linux: /remoteserver2/cluster/project
%
% In this case, for the local machine you would type "\\remoteserver1" and
% for the remote machine you would type "/remoteserver2". As of now, this
% is hardcoded to always end in Linux and Macintosh format using forward
% slashes (/).
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
% 
% See also MergeBatchOutput, GSBatchProcessing.

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
%   Wyman Li
%
% Website: http://www.cellprofiler.org
%
% $Revision: 3417 $

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = Do you want to create scripts (requires MatLab licenses) or mat-files (refer to help for more detail)?
%choiceVAR01 = Scripts
%choiceVAR01 = Files
FileChoice = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = How many cycles should be in each batch?
%defaultVAR02 = 100
BatchSize = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,2}));

%textVAR03 = What prefix should be used to name the batch files?
%defaultVAR03 = Batch_
BatchFilePrefix = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%pathnametextVAR04 = What is the path to the CellProfiler folder on the cluster machines?  Leave a period (.) to use the default module directory. (only necessary for Scripts option)
%defaultVAR04 = .
BatchCellProfilerPath = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%pathnametextVAR05 = What is the path to the image directory on the cluster machines? Leave a period (.) to use the default image directory.
%defaultVAR05 = .
BatchImagePath = char(handles.Settings.VariableValues{CurrentModuleNum,5);

%pathnametextVAR06 = What is the path to the directory where batch output should be written on the cluster machines? Leave a period (.) to use the default output directory.
%defaultVAR06 = .
BatchOutputPath = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%pathnametextVAR07 = What is the path to the directory where you want to save the batch files? Leave a period (.) to use the default output directory.
%defaultVAR07 = .
BatchSavePath = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%pathnametextVAR08 = What is the path to the directory where the batch data file will be saved on the cluster machines? Leave a period (.) to use the default output directory.
%defaultVAR08 = .
BatchRemotePath = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%pathnametextVAR09 = If pathnames are specified differently between the local and cluster machines, enter that part of the pathname from the local machine's perspective, omitting trailing slashes. Otherwise, leave a period (.)
%defaultVAR09 = .
OldPathname = char(handles.Settings.VariableValues{CurrentModuleNum,9});

%pathnametextVAR10 = If pathnames are specified differently between the local and cluster machines, enter that part of the pathname from the cluster machines' perspective, omitting trailing slashes. Otherwise, leave a period (.)
%defaultVAR10 = .
NewPathname = char(handles.Settings.VariableValues{CurrentModuleNum,10});

%textVAR11 = Note: This module must be the last one in the analysis pipeline.

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% If this isn't the first cycle, we are running on the
%%% cluster, and should just continue.
if (handles.Current.SetBeingAnalyzed > 1),
    return;
end

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
else
    handles.Current.DefaultOutputDirectory = BatchImagePath;
    handles.Current.DefaultImageDirectory = BatchOutputPath;
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
    if strcmpi(FileChoice,'Scripts')
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
    elseif strcmpi(FileChoice,'Files')
        cluster.StartImage = StartImage;
        cluster.EndImage = EndImage;
        cluster.BatchFilePrefix = BatchFilePrefix;
        cluster.OutputFolder = BatchOutputPath;
        save(fullfile(BatchSavePath, BatchFileName),'cluster','-v6');
    end
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