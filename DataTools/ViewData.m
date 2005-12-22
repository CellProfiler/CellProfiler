function ViewData(handles)

% Help for the View Data tool:
% Category: Data Tools
%
% SHORT DESCRIPTION:
% Displays data or measurements from a CellProfiler output file.
% *************************************************************************
% Note: this tool is beta-version and has not been thoroughly checked.
%
% This tool views any text data or measurements that have been stored in a
% CellProfiler output file. It can be useful to check that any text data
% added with the AddData tool is associated with the correct image sets.

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
    CPerrordlg('Selected file is not a Matlab file.')
    return
end

%%% Quick check if it seems to be a CellProfiler file or not
if ~exist('handles','var')
    CPerrordlg('Selected file is not a CellProfiler output file.')
    return
end

FinalOK = 0;
while FinalOK == 0

    %%% Let the user select which feature to view
    Suffix = {'Features','Text','Description'};
    [ObjectTypename,FeatureType,FeatureNbr,SuffixNbr] = CPgetfeature(handles,0,Suffix);
    if isempty(ObjectTypename),return,end

    %%% Get the description
    Description = handles.Measurements.(ObjectTypename).([FeatureType,Suffix{SuffixNbr}]){FeatureNbr};

    %%% Generate a cell array with strings to display
    NbrOfImageSets = length(handles.Measurements.(ObjectTypename).(FeatureType));
    TextToDisplay = cell(NbrOfImageSets,1);
    
    if strcmp(Suffix{SuffixNbr},'Description')
        if NbrOfImageSets > length(handles.Measurements.Image.FileNames)
            error('There are more text descriptions than image files. This has not yet been supported.');
        end
    end
    
    for ImageSet = 1:NbrOfImageSets

        % Numeric or text?
        if strcmp(Suffix{SuffixNbr},'Features')
            if length(handles.Measurements.(ObjectTypename).(FeatureType){ImageSet}) >= FeatureNbr
                info = num2str(mean(handles.Measurements.(ObjectTypename).(FeatureType){ImageSet}(:,FeatureNbr)));
            else
                info = 'No Objects Identified';
            end
        elseif strcmp(Suffix{SuffixNbr},'Text')
            info = handles.Measurements.(ObjectTypename).(FeatureType){ImageSet}{FeatureNbr};
        elseif strcmp(Suffix{SuffixNbr},'Description')
            info = handles.Measurements.(ObjectTypename).(FeatureType){ImageSet};
        end

        TextToDisplay{ImageSet} = sprintf('Cycle #%d, %s:     %s',...
            ImageSet,...
            handles.Measurements.Image.FileNames{ImageSet}{1},...
            info);
    end

    %%% Produce an infostring that explains what is displayed
    if strcmp(Suffix{SuffixNbr},'Text')
        InfoString = 'Cycle #,  <filename>:     <text>';
    elseif strcmp(ObjectTypename,'Image')
        InfoString = 'Cycle #,  <filename>:     <value>';
    else
        InfoString = 'Cycle #,  <filename>:     <mean value>';
    end
    TextToDisplay = cat(1,{InfoString},{''},TextToDisplay);

    % Display data in a list dialog box
    [Selection, FinalOK] = listdlg('ListString',TextToDisplay, 'ListSize', [600 200],...
        'Name',['Information for ',Description],...
        'PromptString','Press ''Back'' to select another information entry.',...
        'CancelString','Back',...
        'SelectionMode','single');
end