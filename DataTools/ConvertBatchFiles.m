function ConvertBatchFiles(handles)

% Help for the Convert Batch Files tool:
% Category: Data Tools
%
% SHORT DESCRIPTION:
% Converts output files produced by the Create Batch Files module into
% regular CellProfiler output files.
% *************************************************************************
% Note: this tool is beta-version and has not been thoroughly checked.
%
% CellProfiler data tools do not function on the batch output files created
% by the Create Batch Files module because they are incomplete. They are
% incomplete because each batch output file contains only the measurements
% for one batch of images.
%
% In order to access these measurements, they must be exported (using the
% ExportDatabase data tool or ExportToDatabase module), or merged together
% (using the MergeOutputFiles DataTool), or converted to regular
% CellProfiler output files using this data tool. This data tool will save
% new files with 'Converted' as a prefix in the filename. Important: note
% that the image files will be renumbered, starting with 1. For example,
% your batch output file 'Batch_102_to_201_OUT.mat' will be converted to
% 'ConvertedBatch_102_to_201_OUT.mat', but when you access the data within
% (e.g. using ViewData), image #102 will be image #1.
%
% Technical details: this data tool removes empty entries in the
% handles.Measurements structure of the output file(s) you specify.

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

%%% Let the user select one output file to indicate the directory
[ExampleFile, Pathname] = uigetfile('*.mat','Select one Batch output file');
if ~Pathname,return,end

%%% Get all files with .mat extension in the chosen directory.
%%% If the selected file name contains an 'OUT', it is assumed
%%% that all interesting files contain an 'OUT'.
AllFiles = dir(Pathname);                                                        % Get all file names in the chosen directory
AllFiles = {AllFiles.name};                                                      % Cell array with file names
files = AllFiles(~cellfun('isempty',strfind(AllFiles,'.mat')));                  % Keep files that has a .mat extension
if strfind(ExampleFile,'OUT')
    files = files(~cellfun('isempty',strfind(files,'OUT')));                     % Keep files with an 'OUT' in the name
end

%%% Let the user select the files to be converted
[selection,ok] = listdlg('liststring',files,'name','Convert Batch Files',...
    'PromptString','Select files to convert. Use Ctrl+Click or Shift+Click.','listsize',[300 500]);
if ~ok, return, end
files = files(selection);

%%% Open the files, remove empty entries in the handles.Measurements structure
%%% and store the files.
waitbarhandle = waitbar(0,'');
for fileno = 1:length(files)
    waitbar(fileno/length(files),waitbarhandle,sprintf('Converting %s.',files{fileno}));drawnow
    load(fullfile(Pathname, files{fileno}));
    firstfields = fieldnames(handles.Measurements);
    for i = 1:length(firstfields)
        secondfields = fieldnames(handles.Measurements.(firstfields{i}));
        for j = 1:length(secondfields)
            if iscell(handles.Measurements.(firstfields{i}).(secondfields{j}))
                index = ~cellfun('isempty',handles.Measurements.(firstfields{i}).(secondfields{j}));
                if sum(index==0) > 0       % There exist empty cells, remove them
                    index(1) = 0;          % First set is a dummy set
                    handles.Measurements.(firstfields{i}).(secondfields{j}) = ...
                        handles.Measurements.(firstfields{i}).(secondfields{j})(index);
                end
            end
        end
    end
    save(fullfile(Pathname,['Converted',files{fileno}]),'handles');
end
close(waitbarhandle)
CPmsgbox('Converting is completed.');