function AddData(handles)

% Help for the Add Data tool:
% Category: Data Tools
%
% Use this tool if you would like text information about each image to
% be recorded in the output file along with measurements (e.g. Gene
% names, accession numbers, or sample numbers). 
% The text information must be specified in a separate text file
% with the following syntax:
%
% IDENTIFIER <identfier> 
% DESCRIPTION <description>
% <Text info for image set #1>
% <Text info for image set #2>
% <Text info for image set #3>
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
%
% See also CLEARDATA VIEWDATA.
%
% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne Carpenter <carpenter@wi.mit.edu>
%   Thouis Jones   <thouis@csail.mit.edu>
%   In Han Kang    <inthek@mit.edu>
%
% $Revision$

%%% Select file with text information to be added
if exist(handles.Current.DefaultOutputDirectory, 'dir')
    [filename, pathname] = uigetfile(fullfile(handles.Current.DefaultOutputDirectory,'.','*.*'),'Pick file with text information');
else
    [filename, pathname] = uigetfile('*.*','Pick file with text information');
end

if filename == 0 %User canceled
    return;
end

%%% Ask the user to choose the file from which to extract measurements.
if exist(handles.Current.DefaultOutputDirectory, 'dir')
    Pathname = uigetdir(handles.Current.DefaultOutputDirectory,'Select directory where CellProfiler output files are located');
else
    Pathname = uigetdir('Select directory where CellProfiler output files are located');
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
    'PromptString','Select CellProfiler output files. Use Ctrl+Click or Shift+Click.','listsize',[300 500]);
if ~ok, return, end
SelectedFiles = SelectedFiles(selection);

FieldName = inputdlg('What would you like the save the data as?');

%%% Loop over the selected files and remove the selected feature
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
    tempModuleNumber = handles.Current.CurrentModuleNumber;
    handles.Settings.VariableValues{1,1}=fullfile(pathname,filename);
    handles.Settings.VariableValues{1,2}=FieldName;
    handles.Current.CurrentModuleNumber='01';
    
    handles = LoadText(handles);
    
    handles.Settings.VariableValues=tempVarValues;
    handles.Current.CurrentModuleNumber=tempModuleNumber;
    
    
    %%% Save the updated CellProfiler output file
    try
        save(fullfile(Pathname, SelectedFiles{FileNbr}),'handles')
    catch
        errors{FileNbr} = ['Could not save updated ',SelectedFiles{FileNbr},' file.'];
        continue
    end

end

%%% Finished, display success of warning windows if we failed for some data set
error_index = find(~cellfun('isempty',errors));
if isempty(error_index)
    CPmsgbox('Data successfully added.')
else
    %%% Show a warning dialog box for each error
    for k = 1:length(error_index)
        CPmsgbox(errors{error_index(k)},'Add Data failure')
    end
end




