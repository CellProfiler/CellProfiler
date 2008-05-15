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
% cycle will labeled PlateNumber = 1, Row = A, RowNumber = 1, Column = 01, 
% ColumnNumber = 1, RowAndColumn = A01, and FullLabel = 1_A01. The second 
% well will be labeled A02 or B01, depending on your request. You can also 
% specify how many images cycles are associated per well, if there are 
% multiple fields of view per well.
%
% Features measured:     Feature Number:
% PlateNumber           |      1
% RowNumber             |      2
% Row                   |      3
% ColumnNumber          |      4
% Column                |      5
% RowAndColumn          |      6 
% FullLabel             |      7 
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
% $Revision: 5025 $

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
    Rows = RowsCols(1);
    Columns = RowsCols(2);
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

NumberOfCyclesPerPlate = Rows*Columns*ImageCyclesPerWell;

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
CurrentLinearWellIndex = SetBeingAnalyzed - PlateNumber*NumberOfCyclesPerPlate;

%%% TODO: THE FOLLOWING SIX VALUES NEED TO BE CALCULATED, BASED ON THE
%%% CurrentLinearWellIndex. I'VE INSERTED DUMMY VALUES FOR NOW. Note that
%%% we store the same number twice for Column and ColumnNumber, but one is
%%% a string and the other is a number.
RowNumber = 2;
Row = 'A';
ColumnNumber = 4;
Column = '04';
RowAndColumn = 'A01';
FullLabel = '1_A01';

%%% Make lists of the calculated values and their names for storage and
%%% display later. Note that numerical features need to be stored
%%% separately from text string features for proper exporting.
NumericalFeatureNames = {'PlateNumber' 'RowNumber' 'ColumnNumber'};
NumericalValues = [PlateNumber RowNumber ColumnNumber];
% NumericalValuesAsTextForDisplay = {num2str(PlateNumber), num2str(RowNumber), num2str(ColumnNumber)};;
TextFeatureNames = {'Row' 'Column' 'RowAndColumn' 'FullLabel'};
TextValues = {Row Column RowAndColumn FullLabel};

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
    uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0 0.95 1 0.04],...
        'HorizontalAlignment','center','Backgroundcolor',[.7 .7 .9],'fontname','Helvetica',...
        'fontsize',handles.Preferences.FontSize,'fontweight','bold','string',[LabelName, ' for cycle #',num2str(handles.Current.SetBeingAnalyzed)],'UserData',handles.Current.SetBeingAnalyzed);
    %%% There are 7 features to be displayed in the window.
    %%% Display feature 1: PlateNumber
    n = 1;
    FeatureName = NumericalFeatureNames{1};
    FeatureValue = num2str(NumericalValues(1));
    uicontrol(currentfig,'style','text','units','normalized','fontsize',handles.Preferences.FontSize,'HorizontalAlignment','left','string', [FeatureName,': ', FeatureValue],'position',[.05 .85-(n-1)*.05 .95 .1],'BackgroundColor',[.7 .7 .9])
    %%% Display feature 2: RowNumber
    n = 2;
    FeatureName = NumericalFeatureNames{2};
    FeatureValue = num2str(NumericalValues(2));
    uicontrol(currentfig,'style','text','units','normalized','fontsize',handles.Preferences.FontSize,'HorizontalAlignment','left','string', [FeatureName,': ', FeatureValue],'position',[.05 .85-(n-1)*.05 .95 .1],'BackgroundColor',[.7 .7 .9])
    %%% Display feature 3: Row
    n = 3;
    FeatureName = TextFeatureNames{1};
    FeatureValue = TextValues{1};
    uicontrol(currentfig,'style','text','units','normalized','fontsize',handles.Preferences.FontSize,'HorizontalAlignment','left','string', [FeatureName,': ', FeatureValue],'position',[.05 .85-(n-1)*.05 .95 .1],'BackgroundColor',[.7 .7 .9])
     %%% Display feature 4: ColumnNumber
    n = 4;
    FeatureName = NumericalFeatureNames{3};
    FeatureValue = num2str(NumericalValues(3));
    uicontrol(currentfig,'style','text','units','normalized','fontsize',handles.Preferences.FontSize,'HorizontalAlignment','left','string', [FeatureName,': ', FeatureValue],'position',[.05 .85-(n-1)*.05 .95 .1],'BackgroundColor',[.7 .7 .9])
    %%% Display feature 5: Column
    n = 5;
    FeatureName = TextFeatureNames{2};
    FeatureValue = TextValues{2};
    uicontrol(currentfig,'style','text','units','normalized','fontsize',handles.Preferences.FontSize,'HorizontalAlignment','left','string', [FeatureName,': ', FeatureValue],'position',[.05 .85-(n-1)*.05 .95 .1],'BackgroundColor',[.7 .7 .9])
    %%% Display feature 6: RowAndColumn
    n = 6;
    FeatureName = TextFeatureNames{3};
    FeatureValue = TextValues{3};
    uicontrol(currentfig,'style','text','units','normalized','fontsize',handles.Preferences.FontSize,'HorizontalAlignment','left','string', [FeatureName,': ', FeatureValue],'position',[.05 .85-(n-1)*.05 .95 .1],'BackgroundColor',[.7 .7 .9])    
    %%% Display feature 7: FullLabel
    n = 7;
    FeatureName = TextFeatureNames{4};
    FeatureValue = TextValues{4};
    uicontrol(currentfig,'style','text','units','normalized','fontsize',handles.Preferences.FontSize,'HorizontalAlignment','left','string', [FeatureName,': ', FeatureValue],'position',[.05 .85-(n-1)*.05 .95 .1],'BackgroundColor',[.7 .7 .9])
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% TODO: Make sure that these formats are compatible with CellProfiler
%%% Analyst. They work for exporting to Excel but I haven't checked
%%% exporting to a database, nor whether the column names are categorized
%%% correctly so that when you open them in CPA they are in nice categories in
%%% the dropdown menus. It's not often that we store text data so I am not
%%% sure that I did it properly:

handles = CPaddmeasurements(handles, 'Image', CPjoinstrings('Label','Numerical'), NumericalValues);
handles = CPaddmeasurements(handles, 'Image', CPjoinstrings('Label','Text'), TextValues);