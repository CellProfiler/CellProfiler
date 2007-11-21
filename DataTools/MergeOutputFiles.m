function MergeOutputFiles(handles)

% Help for the Merge Output Files data tool:
% Category: Data Tools
%
% SHORT DESCRIPTION:
% Merges together output files produced by the Create Batch Files module
% into one regular CellProfiler output file.
% *************************************************************************
% Note: this module is beta-version and has not been thoroughly checked.
%
% After a batch run has completed (using batch files created by the Create
% Batch Files module), the individual output files contain results from a
% subset of images and can be merged into a single output file. This module
% assumes anything matching the pattern of Prefix[0-9]*_to_[0-9]*_OUT.mat is a
% batch output file. The combined output is written to the output filename
% you specify. Once merged, this output file should be compatible with data
% tools.
%
% Sometimes output files can be quite large, so before attempting merging,
% be sure that the total size of the merged output file is of a reasonable
% size to be opened on your computer (based on the amount of memory
% available on your computer). It may be preferable instead to import data
% from individual output files directly into a database - see the
% ExportDatabase data tool or the ExportToDatabase module.
%
% Technical notes: The handles.Measurements field of the resulting output
% file will contain all of the merged measurement data, but
% handles.Pipeline is a snapshot of the pipeline after the first cycle
% completes.
%
% See also: CreateBatchFiles, ExportDatabase data tool, ExportToDatabase
% module.

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
%   Peter Swire
%   Rodrigo Ipince
%   Vicky Lay
%   Jun Liu
%   Chris Gang
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%% Let the user select one output file to indicate the directory
[BatchFile, BatchPath] = CPuigetfile('*.mat', 'Select the first batch CellProfiler output file, which ends in data.mat', handles.Current.DefaultOutputDirectory);
if ~BatchPath
    return
end
if ~strfind(BatchFile,'data.mat')
    msg = CPmsgbox('You must choose the first output file, ending in data.mat');
    uiwait(msg);
    return
end

valid = 0;
while valid == 0
    Answers = inputdlg({'What is the Batch file prefix?','What do you want to call the merged output file?'},'Merge output files',1,{'Batch_','MergedOUT.mat'});
    if isempty(Answers)
        return
    end

    if length(Answers{2}) < 4 | ~strcmp(Answers{2}(end-3:end),'.mat') %#ok Ignore MLint
        msg = CPmsgbox('The filename must have a .mat extension.');
        uiwait(msg);
        continue
    elseif isempty(strfind(Answers{2},'OUT'))
        msg = CPmsgbox('The filename must contain an ''OUT'' to indicated that it is a CellProfiler file.');
        uiwait(msg);
        continue
    end

    if ~exist(fullfile(BatchPath,[Answers{1},'data.mat']),'file')
        msg = CPmsgbox(sprintf('The file %s does not exist.',[Answers{1},'data.mat']));
        uiwait(msg);
        continue
    end

    valid = 1;
end
OutputFileName = Answers{2};
BatchFilePrefix = Answers{1};

%%% We want to load the handles from the batch process
clear handles

%%% Load the data file and check that it contains handles
load(fullfile(BatchPath,[BatchFilePrefix,'data.mat']));
if ~exist('handles','var')
    CPerrordlg(sprintf('The file %s does not seem to be a CellProfiler output file.',[BatchFilePrefix,'data.mat']))
    return
end

Fieldnames = fieldnames(handles.Measurements);

FileList = dir(BatchPath);
Matches = ~cellfun('isempty', regexp({FileList.name}, ['^' BatchFilePrefix '[0-9]+_to_[0-9]+_OUT.mat$']));
FileList = FileList(Matches);
if ~any(Matches)
   CPwarndlg('MergeOutputFiles cannot find any files that match [0-9]*_to_[0-9]*_OUT.mat.  Be sure that the ''...data.mat'' file is in the same directory as the ''...OUT.mat'' files') 
end

waitbarhandle = CPwaitbar(0,['Merging ' num2str(length(FileList) + 1) ' files ...']);
for i = 1:length(FileList)
    %% Something about the loaded 'X_data.mat' file is causing this
    %% warning, perhaps a cluster vs. local named function?  Seems OK, so turning it off. David 2007.11.21
    LoadWarning = warning('off','MATLAB:dispatcher:UnresolvedFunctionHandle');
    SubsetData = load(fullfile(BatchPath,FileList(i).name));
    warning(LoadWarning)
%     disp(FileList(i).name)

    if (isfield(SubsetData.handles, 'BatchError')),
        error(['Image processing was canceled in the ', ModuleName, ' module because there was an error merging batch file output.  File ' FileList(i).name ' encountered an error.  The error was ' SubsetData.handles.BatchError '.  Please re-run that batch file.']);
    end

    SubSetMeasurements = SubsetData.handles.Measurements;

    for fieldnum=1:length(Fieldnames)
        secondfields = fieldnames(handles.Measurements.(Fieldnames{fieldnum}));
        % Some fields should not be merged, remove these from the list of fields
        secondfields = secondfields(cellfun('isempty',strfind(secondfields,'PathName')));   % Don't merge pathnames under handles.Measurements.GeneralInfo
        secondfields = secondfields(cellfun('isempty',strfind(secondfields,'Features')));   % Don't merge cell arrays with feature names
        secondfields = secondfields(cellfun('isempty',strfind(secondfields,'SubObjectFlag')));   % Don't merge cell arrays with SubObjectFlag (to play nicely with Relate module)
        for j = 1:length(secondfields)
            idxs = ~cellfun('isempty',SubSetMeasurements.(Fieldnames{fieldnum}).(secondfields{j}));
            idxs(1) = 0; %% Protects the first/main 'Batch' file
%             if (fieldnum == 1),
%                 lo = min(find(idxs(2:end))+1);
%                 hi = max(find(idxs(2:end))+1);
%                 disp(['Merging measurements for sets ' num2str(lo) ' to ' num2str(hi) '.']);
%             end
            handles.Measurements.(Fieldnames{fieldnum}).(secondfields{j})(idxs) = SubSetMeasurements.(Fieldnames{fieldnum}).(secondfields{j})(idxs);
        end
    end

    %%% Update the waitbar.
    CPwaitbar(i/length(FileList),waitbarhandle);
    drawnow
end

%%% These fields are not calculated during batch processing and should be
%%% removed.
if isfield(handles.Measurements.Image,'TimeElapsed')
    handles.Measurements.Image = rmfield(handles.Measurements.Image,'TimeElapsed');
end
handles.Measurements.Image = rmfield(handles.Measurements.Image,'ModuleError');
handles.Measurements.Image = rmfield(handles.Measurements.Image,'ModuleErrorFeatures');

CPwaitbar(1,waitbarhandle,'Saving Output...')
save(fullfile(BatchPath,OutputFileName),'handles');
close(waitbarhandle);
CPmsgbox('Merging is completed.');
