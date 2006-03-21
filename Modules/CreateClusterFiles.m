function handles = CreateClusterFiles(handles)

% Help for the Create Batch Scripts module:
% Category: File Processing
%
% SHORT DESCRIPTION:
% Produces mat files which allow individual batches of images to be
% processed separately on a cluster of computers.
% *************************************************************************
% Note: this module is beta-version and has not been thoroughly checked.
%
% This module is very similar to CreateBatchFiles, but is developed to work
% with a compiled version of CellProfiler made specifically for using the
% cluster. It is still beta and therefore might have bugs. Please refer to
% CreateBatchFiles for more information.
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
%   Wyman Li
%
% Website: http://www.cellprofiler.org
%
% $Revision: 3407 $

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
end;

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
    cluster.StartImage = StartImage;
    cluster.EndImage = EndImage;
    cluster.BatchFilePrefix = BatchFilePrefix;
    cluster.OutputFolder = BatchOutputPath;
    BatchFileName = sprintf('%s%d_to_%d.mat', BatchFilePrefix, StartImage, EndImage);
    save(fullfile(BatchSavePath, BatchFileName),'cluster','-v6');
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