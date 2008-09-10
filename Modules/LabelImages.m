function handles = LabelImages(handles)

% Help for the Label Images module:
% Category: Other
%
% SHORT DESCRIPTION:
% Labels images by assigning them a row and column annotation based on a
% plate layout.
% *************************************************************************
%
% This module labels images by assigning them a row and column annotation 
% based on a plate layout. The annotation is created and stored as an image 
% measurement that is stored in the output file and can thus be exported 
% with other image data. For example, for 96 well plates, the first image
% cycle will labeled:
% PlateNumber = 1, RowNumber = 1, ColumnNumber = 1, SiteNumber = 1, 
% RowText = A, ColumnText = 01, RowAndColumnText = A01, and 
% FullLabel = Plate1_A01_site01. The second 
% well will be labeled A02 or B01, depending on your request. You can also 
% specify how many images cycles are associated per well, if there are 
% multiple fields of view per well.
%
% Features measured:     Feature Number:
% PlateNumber           |      1
% RowNumber             |      2
% ColumnNumber          |      3
% SiteNumber            |      4
% RowText               |      5
% ColumnText            |      6
% RowAndColumnText      |      7 
% FullLabelText         |      8 
%
% Settings: Most are self-explanatory.
% 
% See also DefineGrid, for labeling a grid within each image.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What would you like to call the labels that you create with this module?
%defaultVAR01 = PlateLayout
LabelName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = How many rows and columns are in your plates? (96 well plates are 8,12; 384 well plates are 16,24; 1536 well plates are 32,48)
%defaultVAR02 = 8,12
RowsCols = char(handles.Settings.VariableValues{CurrentModuleNum,2});
try
    RowsCols = str2num(RowsCols); %#ok Ignore MLint
    numRows = RowsCols(1);
    numColumns = RowsCols(2);
catch
    error(['Image processing was canceled in the ', ModuleName, ' module because there is an invalid input for the number of rows and columns.  You need two integers separated by a comma, such as "8,12".']);
end

%textVAR03 = The first image cycle will be labeled A01. What should the second well be labeled? 
%choiceVAR03 = A02
%choiceVAR03 = B01
RowOrColumn = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 = How many images cycles are associated with each well, if there are multiple fields of view per well?
%defaultVAR04 = 1
ImageCyclesPerWell = str2num(handles.Settings.VariableValues{CurrentModuleNum,4});

%%% Maybe someday allow starting from the last well (e.g. H12 for 96 well, P24 or whatever for 384 well).

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%
%%% CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines which cycle is being analyzed.
SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;

NumberOfCyclesPerPlate = numRows*numColumns*ImageCyclesPerWell;

%%% Check whether the number of image sets is divisible by the number of
%%% rows/columns, but only the first time through the module. No need to
%%% check this if we have canceled and restarted a processing cycle partway
%%% through using the Restart module.
if SetBeingAnalyzed == 1
   CPcheckplatedivisibility(handles.Current.NumberOfImageSets,NumberOfCyclesPerPlate,ModuleName);
end

%%% Calculate the data to be recorded for this image cycle, beginning with
%%% the first feature, PlateNumber.
PlateNumber = ceil(SetBeingAnalyzed/NumberOfCyclesPerPlate);

%%% Subtract previous plates to get a linear well index, which can range
%%% from 1 to NumberOfCyclesPerPlate. 
CurrentLinearWellIndex = rem(SetBeingAnalyzed-1,PlateNumber*NumberOfCyclesPerPlate)+1;

%% Get Position Numbers
SiteNumber = rem(CurrentLinearWellIndex-1,ImageCyclesPerWell)+1;

%% Transpose for 'B01' direction
if strcmp(RowOrColumn,'A02')
    ColumnNumber = rem(ceil(CurrentLinearWellIndex/ImageCyclesPerWell)-1,numColumns)+1;
    RowNumber = ceil(ceil(CurrentLinearWellIndex/ImageCyclesPerWell)/(numColumns));
elseif strcmp(RowOrColumn,'B01')    
    RowNumber = rem(ceil(CurrentLinearWellIndex/ImageCyclesPerWell)-1,numRows)+1;
    ColumnNumber = ceil(ceil(CurrentLinearWellIndex/ImageCyclesPerWell)/(numRows));
else
    CPerrordlg('RowOrColumn is invalid.')
end

%% Get Position Characters
ColumnText = CPtwodigitstring(ColumnNumber);
SiteText = CPtwodigitstring(SiteNumber);
ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
RowText = ALPHABET(RowNumber);
RowAndColumnText = [RowText ColumnText];
FullLabelText = ['Plate' num2str(PlateNumber) '_' RowText ColumnText '_site' SiteText];

%%% Make lists of the calculated values and their names for storage and
%%% display later. Note that numerical features need to be stored
%%% separately from text string features for proper exporting.
Features = {'PlateNumber' 'RowNumber' 'ColumnNumber' 'SiteNumber' ...
            'RowText' 'ColumnText' 'RowAndColumnText' 'FullLabelText'};
FeaturesValues = cellfun(@eval,Features,'UniformOutput',false);

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure('','NarrowText',ThisModuleFigureNumber)
    end
    %%% Activates the appropriate figure window.
    currentfig = CPfigure(handles,'Text',ThisModuleFigureNumber);

    %%% Places the text in the window, starting with the heading.
    uicontrol(ThisModuleFigureNumber,'style','text',...
        'units','normalized',...
        'position', [0 0.95 1 0.04],...
        'HorizontalAlignment','center',...
        'Backgroundcolor',[.7 .7 .9],...
        'fontname','Helvetica',...
        'fontsize',handles.Preferences.FontSize,'fontweight','bold','string',[LabelName, ' for cycle #',num2str(handles.Current.SetBeingAnalyzed)],'UserData',handles.Current.SetBeingAnalyzed);
    %%% There are the features to be displayed in the window.
    for idxData = 1:length(Features)
        FeatureName = Features{idxData};
        FeatureValue = num2str(FeaturesValues{idxData});
        uicontrol(currentfig,'style','text',...
            'units','normalized',...
            'fontsize',handles.Preferences.FontSize,...
            'HorizontalAlignment','left',...
            'string',[FeatureName,': ', FeatureValue],...
            'position',[.05 .85-(idxData-1)*.05 .95 .1],...
            'BackgroundColor',[.7 .7 .9])
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for idxData = 1:length(Features)
    FeatureName = Features{idxData};
    FeatureValue = num2str(FeaturesValues{idxData});
    handles = CPaddmeasurements(handles, 'Image', CPjoinstrings('Label',FeatureName), FeatureValue);
end