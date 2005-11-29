function handles = MergeBatchOutput(handles)

% Help for the Merge Batch Output module:
% Category: File Processing
%
% SHORT DESCRIPTION:
% Combines output files together which were run as separate batches using
% the Create Batch Scripts module.
% *************************************************************************
%
% This module merges the output from several output files, each
% resulting from scripts created by the Create Batch Scripts module.
%
% After a batch run has completed, the individual output files contain
% results from a subset of images and can be merged into a single
% output file.  This module takes two arguments, the directory where
% the output files are located (by default the current default output
% directory), and the prefix for the batch files.  This module assumes
% anything matching the pattern of Prefix[0-9]*_to_[0-9]*.mat is a
% batch output file.
%
% The combined output is written to the output filename as specified
% in the lower right box of CellProfiler's main window.  (The output
% file's handles.Pipeline is a snapshot of the pipeline after the
% first cycle completes), and handles.Measurements will contain
% all of the merged measurement data.
%
% Sometimes output files can be quite large, so before attempting
% merging, be sure that the total size of the merged output file is of
% a reasonable size to be opened on your computer (based on the amount
% of RAM available). It may be preferable instead to import data from
% individual output files directly into a database.  Sabatini lab uses
% mySQL for this purpose.
%
% It does not make sense to run this module in conjunction with other
% modules.  It should be the only module in the pipeline.
%
% See also: CREATEBATCHSCRIPTS.

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


[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%pathnametextVAR01 = What is the path to the directory where the batch files were saved?  Type period (.) for default output directory.
BatchPath = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = What was the prefix of the batch files?
%defaultVAR02 = Batch_
BatchFilePrefix = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

if strncmp(BatchPath,'.',1)
    if length(BatchPath) == 1
        BatchPath = handles.Current.DefaultImageDirectory;
    else
        BatchPath = fullfile(handles.Current.DefaultImageDirectory,BatchPath(2:end));
    end
end

%%% If this isn't the first cycle, we're probably running on the
%%% cluster, and should just continue.
if (handles.Current.SetBeingAnalyzed > 1),
    return;
end

%%% Load the data file
BatchData = load(fullfile(BatchPath,BatchFilePrefix,'data.mat'));

%%% Merge into the measurements
handles.Measurements = BatchData.handles.Measurements;

%%% Also merge the pipeline after the first cycle
handles.Pipeline = BatchData.handles.Pipeline;

Fieldnames = fieldnames(handles.Measurements);

FileList = dir(BatchPath);
Matches = ~ cellfun('isempty', regexp({FileList.name}, ['^' BatchFilePrefix '[0-9]+_to_[0-9]+_OUT.mat$']));
FileList = FileList(Matches);

for i = 1:length(FileList),
    SubsetData = load(FileList(i).name);
    FileList(i).name

    if (isfield(SubsetData.handles, 'BatchError')),
        error(['Image processing was canceled in the ', ModuleName, ' module because there was an error merging batch file output.  File ' FileList(i).name ' encountered an error.  The error was ' SubsetData.handles.BatchError '.  Please re-run that batch file.']);
    end

    SubSetMeasurements = SubsetData.handles.Measurements;

    for fieldnum=1:length(Fieldnames),
        idxs = ~ cellfun('isempty', SubSetMeasurements.(Fieldnames{fieldnum}));
        if (fieldnum == 1),
            lo = min(find(idxs(2:end))+1);
            hi = max(find(idxs(2:end))+1);
            disp(['Merging measurements for sets ' num2str(lo) ' to ' num2str(hi) '.']);
        end
        handles.Measurements.(Fieldnames{fieldnum})(idxs) = ...
            SubSetMeasurements.(Fieldnames{fieldnum})(idxs);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% The figure window display is unnecessary for this module, so the figure
%%% window is closed if it was previously open.
%%% Determines the figure number.
ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
%%% If the window is open, it is closed.
if any(findobj == ThisModuleFigureNumber)
    delete(ThisModuleFigureNumber)
end