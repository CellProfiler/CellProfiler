function ViewData(handles)

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

FinalOK = 0;
while FinalOK == 0
   
    %%% Let the user select which feature to view
    Suffix = {'Features','Text'};
    [ObjectTypename,FeatureType,FeatureNbr,SuffixNbr] = CPgetfeature(handles,Suffix);
    if isempty(ObjectTypename),return,end
    
    %%% Get the description
    Description = handles.Measurements.(ObjectTypename).([FeatureType,Suffix{SuffixNbr}]){FeatureNbr};
    
    %%% Generate a cell array with strings to display
    NbrOfImageSets = length(handles.Measurements.(ObjectTypename).(FeatureType));
    TextToDisplay = cell(NbrOfImageSets,1);
    for ImageSet = 1:NbrOfImageSets
        
        % Numeric or text?
        if strcmp(Suffix{SuffixNbr},'Features')
            info = num2str(mean(handles.Measurements.(ObjectTypename).(FeatureType){ImageSet}(:,FeatureNbr)));
        elseif strcmp(Suffix{SuffixNbr},'Text')
            info = handles.Measurements.(ObjectTypename).(FeatureType){ImageSet}{FeatureNbr};
        end
        
        TextToDisplay{ImageSet} = sprintf('#%d, %s:     %s',...
            ImageSet,...
            handles.Measurements.Image.FileNames{ImageSet}{1},...
            info);
    end
    
    %%% Produce an infostring that explains what is displayed
    if strcmp(Suffix{SuffixNbr},'Text')
        InfoString = 'Image set #,  <filename>:     <text>';
    elseif strcmp(ObjectTypename,'Image')
        InfoString = 'Image set #,  <filename>:     <value>';
    else
        InfoString = 'Image set #,  <filename>:     <mean value>';
    end
    TextToDisplay = cat(2,{InfoString},{''},TextToDisplay);
 
    % Display data in a list dialog box
    [Selection, FinalOK] = listdlg('ListString',TextToDisplay, 'ListSize', [600 400],...
        'Name',['Information for ',Description],...
        'PromptString','Press ''Back'' to select another information entry.',...
        'CancelString','Back',...
        'SelectionMode','single');
end




