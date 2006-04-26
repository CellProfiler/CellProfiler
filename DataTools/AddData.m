function AddData(handles)

% Help for the Add Data tool:
% Category: Data Tools
%
% SHORT DESCRIPTION:
% Allows adding information for each image cycle to an output file.
% *************************************************************************
% Note: this tool is beta-version and has not been thoroughly checked.
%
% Use this tool if you would like to add text information about each image
% (e.g. Gene names or sample numbers) to the output file alongside the
% measurements that have been made. Then, the text information will be
% exported with the measurements when you use the ExportData data tool,
% helping you to keep track of your samples. You can also run the LoadText
% module in your pipeline so this step happens automatically during
% processing; its function is the same.
%
% Note that the number of text entries that you load with this module must
% be identical to the number of cycles you are processing in order for
% exporting to work properly.
%
% The information to be added must be in a separate text file with the
% following syntax:
%
% IDENTIFIER <identifier>
% DESCRIPTION <description>
% <Text info for image cycle #1>
% <Text info for image cycle #2>
% <Text info for image cycle #3>
%              .
%              .
%
% <identifier> is used as field name when storing the text information in
% the Matlab structure. It must be one word. <description> is a description
% of the text information stored in the file. It can be a sentence.
%
% For example:
%
% IDENTIFIER GeneNames
% DESCRIPTION Gene names
% Gene X
% Gene Y
% Gene Z
% 
% While not thoroughly tested, most likely you can load numerical data too.
%
% See also the LoadText module.

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

%%% Select file with text information to be added
if exist(handles.Current.DefaultOutputDirectory, 'dir')
    [filename, pathname] = uigetfile(fullfile(handles.Current.DefaultOutputDirectory,'.','*.*'),'Choose the file containing the data');
else
    [filename, pathname] = uigetfile('*.*','Choose the file containing the data');
end

if filename == 0 %User canceled
    return;
end

%%% Ask the user to choose the file from which to extract measurements.
if exist(handles.Current.DefaultOutputDirectory, 'dir')
    Pathname = uigetdir(handles.Current.DefaultOutputDirectory,'Choose the folder that contains the output file(s) to add data to');
else
    Pathname = uigetdir('Choose the folder that contains the output file(s) to add data to');
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
    'PromptString','Choose CellProfiler output files to add data to. Use Ctrl+Click or Shift+Click to choose multiple files.','listsize',[300 500]);
if ~ok, return, end
SelectedFiles = SelectedFiles(selection);

FieldName = inputdlg('What name would you like to give this data (what heading)?');

%%% Loop over the selected files and add the selected feature
%%% An cell array is used to indicated any errors in the processing
errors = cell(length(SelectedFiles),1);
for FileNbr = 1:length(SelectedFiles)

    %%% Load the specified CellProfiler output file
    try
        clear handles;
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

    tempVarValues=handles.Settings.VariableValues;
    tempCurrentField = handles.Current;
    handles.Settings.VariableValues{1,1}=filename;
    handles.Settings.VariableValues{1,2}=FieldName;
    handles.Settings.VariableValues{1,3}=pathname;
    handles.Current.CurrentModuleNumber='01';
    handles.Current.SetBeingAnalyzed=1;
    handles = LoadText(handles);
    handles.Settings.VariableValues=tempVarValues;
    handles.Current=tempCurrentField;

    %%% Save the updated CellProfiler output file
    try
        save(fullfile(Pathname, SelectedFiles{FileNbr}),'handles')
    catch
        errors{FileNbr} = ['Could not save updated ',SelectedFiles{FileNbr},' file.'];
        continue
    end
end

%%% Finished, display success or warning windows if we failed for some data set
error_index = find(~cellfun('isempty',errors));
if isempty(error_index)
    CPmsgbox('Data successfully added.')
else
    %%% Show a warning dialog box for each error
    for k = 1:length(error_index)
        CPmsgbox(errors{error_index(k)},'Add Data failure')
    end
end