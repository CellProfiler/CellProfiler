function handles = AddData(handles)

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
    [filename, pathname] = uigetfile(fullfile(handles.Current.DefaultOutputDirectory,'*.*'),'Pick file with text information');
else
    [filename, pathname] = uigetfile('*.*','Pick file with text information');
end

%%% Parse text file %%%
fid = fopen(fullfile(pathname,filename),'r');

% The first 10 characters must equal 'IDENTIFIER'
s = fgets(fid,10);
if ~strcmp(s,'IDENTIFIER')
    errordlg('The first line in the text information file must be of the format ''IDENTIFIER <identifiername>''.')
end
s = fgetl(fid); s = s(2:end);
if sum(isspace(s)) > 0
    errordlg('Entry after IDENTIFIER on the first line contains white spaces.')
end
FieldName = s;

% Get description
s = fgets(fid,11);
if ~strcmp(s,'DESCRIPTION')
    errordlg('The second line in the text information file must start with DESCRIPTION')
end
Description = fgetl(fid);
Description = Description(2:end);       % Remove space

% Read following lines into a cell array
Text = {};
while 1
    s = fgetl(fid);
    if ~ischar(s), break, end
    if ~isempty(s)
        Text{end+1} = s;
    end
end
fclose(fid);

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



%%% Loop over the selected files and remove the selected feature
%%% An cell array is used to indicated any errors in the processing
errors = cell(length(SelectedFiles),1);
for FileNbr = 1:length(SelectedFiles)

    %%% Load the specified CellProfiler output file
    try
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

    %%% Check that the number of processed file names equals the number of text entries
    if length(handles.Measurements.Image.FileNames) ~= length(Text)
        errors{FileNbr} = sprintf('The number of processed image sets in %s (%d) does not match the number of entries in text information file (%d)',SelectedFiles{FileNbr},length(handles.Measurements.Image.FileNames),length(Text));
        continue
    end

    %%% Add the data
    %%% If the entered field doesn't exist  (This is the convenient way of doing it. Takes time for large ouput files??)
    if ~isfield(handles.Measurements.Image,FieldName)
        handles.Measurements.Image.([FieldName,'Text']) = {Description};
        for imageset = 1:length(Text)
            handles.Measurements.Image.(FieldName){imageset} = Text(imageset);
        end
    
    %%% If the entered field already exists we have to append to this field
    else
        handles.Measurements.Image.([FieldName,'Text']) = cat(2,handles.Measurements.Image.([FieldName,'Text']),{Description});
        for imageset = 1:length(Text)
            handles.Measurements.Image.(FieldName){imageset} = cat(2,handles.Measurements.Image.(FieldName){imageset},Text(imageset));
        end
    end
    
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
    CPmsgbox('Data successfully deleted.')
else
    %%% Show a warning dialog box for each error
    for k = 1:length(error_index)
        msgbox(errors{error_index(k)},'Add Data failure')
    end
end




