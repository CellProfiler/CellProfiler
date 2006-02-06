function handles = MergeBatchOutput(handles)

% Help for the Merge Batch Output module:
% Category: File Processing
%
% SHORT DESCRIPTION:
% Combines output files together which were run as separate batches using
% the Create Batch Scripts module.
% *************************************************************************
% Note: this module is beta-version and has not been thoroughly checked.
%
% After a batch run has completed (using scripts created by the Create
% Batch Scripts module), the individual output files contain results from a
% subset of images and can be merged into a single output file. This module
% assumes anything matching the pattern of Prefix[0-9]*_to_[0-9]*.mat is a
% batch output file. The combined output is written to the output filename
% as specified in the lower right box of CellProfiler's main window. Once
% merged, this output file should be compatible with data tools.
%
% It does not make sense to run this module in conjunction with other
% modules.  It should be the only module in the pipeline.
%
% Sometimes output files can be quite large, so before attempting merging,
% be sure that the total size of the merged output file is of a reasonable
% size to be opened on your computer (based on the amount of memory
% available on your computer). It may be preferable instead to import data
% from individual output files directly into a database - see the
% ExportData data tool.
%
% Technical notes: The handles.Measurements field of the resulting output
% file will contain all of the merged measurement data, but
% handles.Pipeline is a snapshot of the pipeline after the first cycle
% completes.
%
% See also: CreateBatchScripts.

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
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%pathnametextVAR01 = What is the path to the folder where the batch output files were saved?  Type period (.) for default output folder.
BatchPath = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = What was the prefix of the batch output files?
%defaultVAR02 = Batch_
BatchFilePrefix = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

if strncmp(BatchPath,'.',1)
    if length(BatchPath) == 1
        BatchPath = handles.Current.DefaultOutputDirectory;
    else
        BatchPath = fullfile(handles.Current.DefaultOutputDirectory,BatchPath(2:end));
    end
end

%%% If this isn't the first cycle, we're probably running on the
%%% cluster, and should just continue.
if (handles.Current.SetBeingAnalyzed > 1),
    return;
end

%%% Load the data file
BatchData = load(fullfile(BatchPath,[BatchFilePrefix,'data.mat']));

%%% Merge into the measurements
handles.Measurements = BatchData.handles.Measurements;

%%% Also merge the pipeline after the first cycle
handles.Pipeline = BatchData.handles.Pipeline;

Fieldnames = fieldnames(handles.Measurements);

FileList = dir(BatchPath);
Matches = ~cellfun('isempty', regexp({FileList.name}, ['^' BatchFilePrefix '[0-9]+_to_[0-9]+_OUT.mat$']));
FileList = FileList(Matches);

for i = 1:length(FileList),
    SubsetData = load(fullfile(BatchPath,FileList(i).name));
    FileList(i).name

    if (isfield(SubsetData.handles, 'BatchError')),
        error(['Image processing was canceled in the ', ModuleName, ' module because there was an error merging batch file output.  File ' FileList(i).name ' encountered an error.  The error was ' SubsetData.handles.BatchError '.  Please re-run that batch file.']);
    end

    SubSetMeasurements = SubsetData.handles.Measurements;

    for fieldnum=1:length(Fieldnames)
        secondfields = fieldnames(handles.Measurements.(Fieldnames{fieldnum}));
        % Some fields should not be merged, remove these from the list of fields
        secondfields = secondfields(cellfun('isempty',strfind(secondfields,'Pathname')));   % Don't merge pathnames under handles.Measurements.GeneralInfo
        secondfields = secondfields(cellfun('isempty',strfind(secondfields,'Features')));   % Don't merge cell arrays with feature names
        for j = 1:length(secondfields)
            idxs = ~cellfun('isempty',SubSetMeasurements.(Fieldnames{fieldnum}).(secondfields{j}));
            idxs(1) = 0;
            if (fieldnum == 1),
                lo = min(find(idxs(2:end))+1);
                hi = max(find(idxs(2:end))+1);
                disp(['Merging measurements for sets ' num2str(lo) ' to ' num2str(hi) '.']);
            end
            handles.Measurements.(Fieldnames{fieldnum}).(secondfields{j})(idxs) = ...
                SubSetMeasurements.(Fieldnames{fieldnum}).(secondfields{j})(idxs);
        end
    end
end

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