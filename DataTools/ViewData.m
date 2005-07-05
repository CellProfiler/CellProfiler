function handles = ViewData(handles)

% Help for the View Data tool:
% Category: Data Tools
%
% This tool views any text data that has been stored in a
% CellProfiler output file. It can be useful to check that
% text data added with the AddData tool is associated with
% the correct image sets.
%
% See also CLEARDATA ADDDATA.
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

%%% Ask the user to choose the file from which to extract measurements.
if exist(handles.Current.DefaultOutputDirectory, 'dir')
    [FileName, Pathname] = uigetfile(fullfile(handles.Current.DefaultOutputDirectory,'.','*.mat'),'Select the raw measurements file');
else
    [FileName, Pathname] = uigetfile('*.mat','Select the raw measurements file');
end
if FileName == 0
    return
end

%%% Load the specified CellProfiler output file
try
    load(fullfile(Pathname, FileName));
catch
    errordlg('Selected file is not a Matlab file.')
end

%%% Quick check if it seems to be a CellProfiler file or not
if ~exist('handles','var')
    errordlg('Selected file is not a CellProfiler output file.')
end

% Get text data
Fields = fieldnames(handles.Measurements.Image);
TextFields = Fields(~cellfun('isempty',strfind(Fields,'Text')));
TextData = {};
TextDataField = {};
TextDataNbr = [];
for k = 1:length(TextFields)
    TextData = cat(2,TextData,handles.Measurements.Image.(TextFields{k}));
    TextDataField = cat(2,TextDataField,repmat(TextFields(k),[1 length(handles.Measurements.Image.(TextFields{k}))]));
    TextDataNbr  = cat(2,TextDataNbr,1:length(handles.Measurements.Image.(TextFields{k})));
end

FinalOK = 0;
while FinalOK == 0
    % Let the user choose a specific text data entry
    [Selection, ok] = listdlg('ListString',TextData, 'ListSize', [300 400],...
        'Name','Select measurement',...
        'PromptString','Select text information to view',...
        'CancelString','Cancel',...
        'SelectionMode','single');
    if ok == 0, return, end                             % Should restore the previous handles structure....?

    % Get the data for the specific selection
    SelectedTextData = TextData{Selection};
    SelectedTextDataField = TextDataField{Selection}(1:end-4);
    SelectedTextDataNbr = TextDataNbr(Selection);

    % Generate a cell array with strings to display
    NbrOfImageSets = length(handles.Measurements.Image.(SelectedTextDataField));
    TextToDisply = cell(NbrOfImageSets,1);
    for ImageSet = 1:NbrOfImageSets
        TextToDisplay{ImageSet} = sprintf('Image set #%d, Filename: %s:     %s',...
            ImageSet,...
            handles.Measurements.Image.FileNames{ImageSet}{1},...
            handles.Measurements.Image.(SelectedTextDataField){ImageSet}{SelectedTextDataNbr});
    end

    % Display data in a list dialog box
    [Selection, FinalOK] = listdlg('ListString',TextToDisplay, 'ListSize', [600 400],...
        'Name',['Information for ',SelectedTextData],...
        'PromptString','Press ''Back'' to select another information entry.',...
        'CancelString','Back',...
        'SelectionMode','single');
end




