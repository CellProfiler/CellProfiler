function handles = MergeOutputFiles(handles)

% Help for the MergeOutputFiles tool:
% Category: Data Tools
%
% This tool merges CellProfiler output files. The focus is on
% the measurement structure, no image data will be stored
% in the final output file.
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


%%% Clear the handles structure that is sent to this function,
%%% it is not needed. 
oldhandles = handles;
clear handles

%%% Let the user select one output file to indicate the directory
[ExampleFile, Pathname] = uigetfile('*.mat','Select one CellProfiler output file');
if ~Pathname
    handles = oldhandles;          % CellProfiler expects an output
    return
end

%%% Get all files with .mat extension in the chosen directory.
%%% If the selected file name contains an 'OUT', it is assumed
%%% that all interesting files contain an 'OUT'.
AllFiles = dir(Pathname);                                                        % Get all file names in the chosen directory
AllFiles = {AllFiles.name};                                                      % Cell array with file names
files = AllFiles(~cellfun('isempty',strfind(AllFiles,'.mat')));                  % Keep files that has a .mat extension
if strfind(ExampleFile,'OUT')
    files = files(~cellfun('isempty',strfind(files,'OUT')));                     % Keep files with an 'OUT' in the name
end

%%% Let the user select the files to be merged
[selection,ok] = listdlg('liststring',files,'name','Merge Output Files',...
    'PromptString','Select files to merge. Use Ctrl+Click or Shift+Click.','listsize',[300 500]);
if ~ok, handles = oldhandles; return, end
%if length(selection) < 2
%    errordlg('At least two files must be selected.')
%    handles = oldhandles;
%    return
%end
files = files(selection);

%%% Load the first file and check if it seems to be a CellProfiler file
load(fullfile(Pathname, files{1}));
if ~exist('handles','var')
    errordlg(sprintf('The file %s does not seem to be a CellProfiler output file.',files{1}))
    handles = oldhandles;
    return
end

%%% Create a superhandles structure, the following files must have
%%% the same structure as this superhandles structure.
superhandles.Measurements = handles.Measurements;
superhandles.Settings = handles.Settings;
superhandles.Preferences = handles.Preferences;
superhandles.Current.TimeStarted = handles.Current.TimeStarted;
supermodules = superhandles.Settings.ModuleNames;

%%% Let the user choose a name for the new output file.
%%% Repeat until a valid filename is given.
valid = 0;
while valid == 0
    answer = inputdlg({'Name for merged output file:'},'Merge output files',1,{'MergedOUT.mat'});
    if isempty(answer),handles = oldhandles;return;end
    if length(answer{1})< 4 | ~strcmp(answer{1}(end-3:end),'.mat')
        msg = CPmsgbox('The filename must have a .mat extension.');
        uiwait(msg);
    elseif isempty(strfind(answer{1},'OUT'))
        msg = CPmsgbox('The filename must contain an ''OUT'' to indicated that it is a CellProfiler file.');
        uiwait(msg);
    else
        valid = 1;
    end
end    
OutputFileName = answer{1};


%%% Loop over the selected files and add the data to the superhandles structure.
waitbarhandle = waitbar(1/length(files),'Merging files');
for fileno = 2:length(files)

    %%% Clear the handles structure before loading the next file
    clear handles

    %%% Load the file and check that it seems to be a CellProfiler file
    load(fullfile(Pathname, files{fileno}));
    if ~exist('handles','var')
        errordlg(sprintf('The file %s does not seem to be a CellProfiler output file.',files{fileno}))
        handles = oldhandles;
        return
    end

    %%% Check for inconsistencies
    %%% Compare the modules used. If they don't match, abort.
    modulenames = handles.Settings.ModuleNames;
    if length(modulenames) ~= length(supermodules)
        errordlg(sprintf('Inconsistency in file %s.',files{fileno}))
        handles = oldhandles;
        return
    end
    for j = 1:length(modulenames)
        if ~strcmp(modulenames{j},supermodules{j})
            errordlg(sprintf('Inconsistency in file %s.',files{fileno}))
            handles = oldhandles;
            return
        end
    end

    %%% OK, it looks like we can merge the files
    %%% Note that only the fields under handles.Measurements are merged
    %%% There should be two levels under handles.Measurements
    firstfields = fieldnames(handles.Measurements);                                         % The first level contains for example Image, Cells, Nuclei,...
    for i = 1:length(firstfields)
        secondfields = fieldnames(handles.Measurements.(firstfields{i}));
        
        % Some fields should not be merged, remove these from the list of fields
        secondfields = secondfields(cellfun('isempty',strfind(secondfields,'Pathname')));   % Don't merge pathnames under handles.Measurements.GeneralInfo
        secondfields = secondfields(cellfun('isempty',strfind(secondfields,'Features')));   % Don't merge cell arrays with feature names
        
        % Merge!
        for j = 1:length(secondfields)
            superhandles.Measurements.(firstfields{i}).(secondfields{j}) = ...
                cat(2,superhandles.Measurements.(firstfields{i}).(secondfields{j}),...
                handles.Measurements.(firstfields{i}).(secondfields{j}));
        end
    end
    
    %%% Update waitbar
    waitbar(fileno/length(files),waitbarhandle);drawnow
end

%%% Done! Close waitbar, save the file and restore the old handles structure
close(waitbarhandle)
CPmsgbox('Merging is completed.')
handles = superhandles;
save(fullfile(Pathname,OutputFileName),'handles');
handles = oldhandles;

