function handles = ClearData(handles)

% Help for the Clear Data tool:
% Category: Data Tools
%
% SHORT DESCRIPTION:
% Removes information/measurements from an output file.
% *************************************************************************
% Note: this tool is beta-version and has not been thoroughly checked.
%
% This tool lets the user remove a measurement or data field from a
% CellProfiler output file. The same measurement can be removed from
% several files.

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
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%% Ask the user to choose the file from which directory to extract measurements.
if exist(handles.Current.DefaultOutputDirectory, 'dir')
    Pathname = uigetdir(handles.Current.DefaultOutputDirectory,'Choose the folder that contains the output file(s) to remove data from');
else
    Pathname = uigetdir(pwd,'Choose the folder that contains the output file(s) to remove data from');
end
%%% Check if cancel button pressed
if Pathname == 0
    return
end

%%% Get all files with .mat extension in the chosen directory that contains a 'OUT' in the filename
AllFiles = dir(Pathname);                                                        % Get all file names in the chosen directory
AllFiles = {AllFiles.name};                                                      % Cell array with file names
SelectedFiles = AllFiles(~cellfun('isempty',strfind(AllFiles,'.mat')));          % Keep files that has a .mat extension
SelectedFiles = SelectedFiles(~cellfun('isempty',strfind(SelectedFiles,'OUT'))); % Keep files with an 'OUT' in the name

%%% Let the user select the files
[selection,ok] = listdlg('liststring',SelectedFiles,'name','Select output files',...
    'PromptString','Choose CellProfiler output files to remove data from. Use Ctrl+Click or Shift+Click to choose multiple files.','listsize',[300 500]);
if ~ok, return, end
SelectedFiles = SelectedFiles(selection);

%%% Load the first specified CellProfiler output file so we can choose the
%%% feature to be removed.
try
    clear handles;
    load(fullfile(Pathname, SelectedFiles{1}));
catch
    CPerrordlg([SelectedFiles{1},' is not a Matlab file'])
    return
end

%%% Quick check if it seems to be a CellProfiler file or not
if ~exist('handles','var')
    CPerrordlg([SelectedFiles{1} ,' is not a CellProfiler output file.'])
    return
end

%%% Let the user select which feature to delete
Suffix = {'Features','Text','Description'};
[ObjectTypename,FeatureType,FeatureNbr,SuffixNbr] = CPgetfeature(handles,0,Suffix);

%%% If Cancel button pressed
if isempty(ObjectTypename),return,end

%%% Ask the user if he really wants to clear the selected feature
Confirmation = CPquestdlg('Are you sure you want to delete the selected feature?','Confirmation','Yes','Cancel','Cancel');
if strcmp(Confirmation,'Cancel')
    return
end

%%% Loop over the selected files and remove the selected feature
%%% An cell array is used to indicated any errors in the processing
errors = cell(length(SelectedFiles),1);
for FileNbr = 1:length(SelectedFiles)

    %%% Load the specified CellProfiler output file
    try
        clear handles
        load(fullfile(Pathname, SelectedFiles{FileNbr}));
    catch
        errors{FileNbr} = [SelectedFiles{FileNbr},' is not a Matlab file'];
        continue
    end

    %%% Quick check if it seems to be a CellProfiler file or not
    if ~exist('handles','var')
        errors{FileNbr} = [SelectedFiles{FileNbr},' is not a CellProfiler output file'];
        continue
    end

    %%% Get the cell array of data
    data = handles.Measurements.(ObjectTypename).(FeatureType);

    %%% If there is only one feature for this feature type, we should remove the entire field
    %%% from the handles structure
    if size(data{1},2) == 1 || strcmp(Suffix{SuffixNbr},'Description')

        % Remove the data field and the associated field with suffix 'Text'/'Features'
        handles.Measurements.(ObjectTypename) = rmfield(handles.Measurements.(ObjectTypename),FeatureType);
        handles.Measurements.(ObjectTypename) = rmfield(handles.Measurements.(ObjectTypename),[FeatureType,Suffix{SuffixNbr}]);

        %%% If this was the last measurement in the ObjectTypename (e.g., Nuclei, Cells, Cytoplasm)
        %%% remove the ObjectTypename too
        if isempty(fieldnames(handles.Measurements.(ObjectTypename)))
            handles.Measurements = rmfield(handles.Measurements,ObjectTypename);
        end
        %%% Otherwise we need to loop over the image sets and remove the column indicated by
        %%% 'FeatureNbr'
    else
        %%% Loop over the image sets and remove the specified feature
        for ImageSetNbr = 1:length(data)
            data{ImageSetNbr} = cat(2,data{ImageSetNbr}(:,1:FeatureNbr-1),data{ImageSetNbr}(:,FeatureNbr+1:end));
        end
        handles.Measurements.(ObjectTypename).(FeatureType) = data;

        %%% Remove the feature from the associated description field with
        %%% suffix 'Features', 'Text', or 'Description'
        text = handles.Measurements.(ObjectTypename).([FeatureType,Suffix{SuffixNbr}]);
        text = cat(2,text(1:FeatureNbr-1),text(FeatureNbr+1:end));
        handles.Measurements.(ObjectTypename).([FeatureType,Suffix{SuffixNbr}]) = text;
    end

    %%% Save the updated CellProfiler output file
    try
        save(fullfile(Pathname, SelectedFiles{FileNbr}),'handles')
    catch
        errors{FileNbr} = ['Could not save updated ',SelectedFiles{FileNbr},' file.'];
        continue
    end
end

error_index = find(~cellfun('isempty',errors));
if isempty(error_index)
    CPmsgbox('Data successfully deleted.')
else
    %%% Show a warning dialog box for each error
    for k = 1:length(error_index)
        CPwarndlg(errors{error_index(k)},'Clear Data failure')
    end
end