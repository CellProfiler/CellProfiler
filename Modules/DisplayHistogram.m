function handles = DisplayHistogram(handles)

% Help for the Display Histogram module:
% Category: Other
%
% SHORT DESCRIPTION:
% Produces a histogram of measurements.
% *************************************************************************
%
% The resulting histograms can be saved using the Save Images module.
%
% Feature Number:
% The feature number specifies which feature from the Measure module will
% be used for the histogram. See each Measure module's help for the
% numbered list of the features measured by that module.
%
% Frequency counts:
% Frequency counts refers to the threshold for the leftmost and rightmost
% bins. The minimum value is the threshold at which any measurements less
% than this value will be combined into the leftmost bin. The maximum value
% is the threshold at which any measurements greater than or equal to this
% value will be combined into the rightmosot bin. 
%
% Absolute vs. Percentage
% Choose "Numbers" if you want the histogram bins to contain the actual 
% numbers of objects in the bin. Choose "Percents" if you want the 
% histogram bins to contain the percentage of objects in the bin.
%
% See also DisplayImageHistogram, MeasureObjectAreaShape,
% MeasureObjectIntensity, MeasureTexture, MeasureCorrelation,
% MeasureObjectNeighbors, CalculateRatios.

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

%textVAR01 = Which object would you like to use for the histogram, or if using a Ratio, what is the numerator object?
%choiceVAR01 = Image
%infotypeVAR01 = objectgroup
%inputtypeVAR01 = popupmenu
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = Which category of measurements would you like to use?
%inputtypeVAR02 = popupmenu category
Category = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Which feature do you want to use? (Enter the feature number - see help for details)
%defaultVAR03 = 1
%inputtypeVAR03 = popupmenu measurement
FeatureNumber = handles.Settings.VariableValues{CurrentModuleNum,3};

if isempty(FeatureNumber)
    error(['Image processing was canceled in the ', ModuleName, ' module because your entry for feature number is not valid.']);
end

%textVAR04 = For INTENSITY, AREAOCCUPIED or TEXTURE features, which image's measurements do you want to use?
%infotypeVAR04 = imagegroup
%inputtypeVAR04 = popupmenu
Image = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = For TEXTURE, RADIAL DISTRIBUTION, OR NEIGHBORS features, what previously measured size scale (TEXTURE OR NEIGHBORS) or previously used number of bins (RADIALDISTRIBUTION) do you want to use?
%defaultVAR05 = 1
%inputtypeVAR05 = popupmenu scale
SizeScale = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = What do you want to call the resulting histogram image?
%defaultVAR06 = OrigHist
%infotypeVAR06 = imagegroup indep
HistImage = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = How many histogram bins would you like to use?
%choiceVAR07 = 16
%choiceVAR07 = 2
%choiceVAR07 = 10
%choiceVAR07 = 50
%choiceVAR07 = 100
%choiceVAR07 = 256
NumberOfBins = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,7}));
%inputtypeVAR07 = popupmenu custom

if isempty(NumberOfBins)
    error(['Image processing was canceled in the ', ModuleName, ' module because your entry for number of histogram bins is not valid.']);
end

%textVAR08 = Enter the range on the X axis for frequency counts on the Y axis ('Min Max'):
%defaultVAR08 = Automatic
MinAndMax = char(handles.Settings.VariableValues{CurrentModuleNum,8});
%inputtypeVAR08 = popupmenu custom

%textVAR09 = Do you want the X axis to be log scale?
%choiceVAR09 = No
%choiceVAR09 = Yes
LogOrLinear = char(handles.Settings.VariableValues{CurrentModuleNum,9});
%inputtypeVAR09 = popupmenu

%textVAR10 = Do you want to use absolute numbers of objects or percentage of total objects?
%choiceVAR10 = Numbers
%choiceVAR10 = Percents
NumberOrPercent = char(handles.Settings.VariableValues{CurrentModuleNum,10});
%inputtypeVAR10 = popupmenu

%textVAR11 = Do you the style to be a line, bar, or area graph?
%choiceVAR11 = Bar
%choiceVAR11 = Line
%choiceVAR11 = Area
LineOrBar = char(handles.Settings.VariableValues{CurrentModuleNum,11});
%inputtypeVAR11 = popupmenu

%textVAR12 = What color do you want the graph to be?
%choiceVAR12 = Blue
%choiceVAR12 = Red
%choiceVAR12 = Green
%choiceVAR12 = Yellow
%choiceVAR12 = Magenta
%choiceVAR12 = Cyan
%choiceVAR12 = Black
%choiceVAR12 = White
%choiceVAR12 = CellProfiler background
Color = char(handles.Settings.VariableValues{CurrentModuleNum,12});
%inputtypeVAR12 = popupmenu

%%%VariableRevisionNumber = 3

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines which cycle is being analyzed.
SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;

try
    FeatureName = CPgetfeaturenamesfromnumbers(handles,ObjectName,Category,...
        FeatureNumber,Image,SizeScale);
catch
    error([lasterr '  Image processing was canceled in the ', ModuleName, ...
        ' module (#' num2str(CurrentModuleNum) ...
        ') because an error ocurred when retrieving the data.  '...
        'Likely the category of measurement you chose, ',...
        Category, ', was not available for ', ...
        ObjectName,' with feature number ' num2str(FeatureNumber) ...
        ', possibly specific to image ''' Image ''' and/or ' ...
        'Texture Scale = ' num2str(SizeScale) '.']);
end

%%% Checks that the Min and Max have valid values
%Replacement starts here
try (strread(MinAndMax));
    % Check to see if the numeric string is of the correct length.
    % Were two numberes entered? If not change to Automatic and warn user.
    if (length(strread(MinAndMax))==2)
        MinAndMax = strread(MinAndMax);
        MinHistogramValue = MinAndMax(1);
        MaxHistogramValue = MinAndMax(2);
        % Make sure that max is greater than min.
        % If not change to automatic
        if (MaxHistogramValue <= MinHistogramValue)
            MinAndMax = 'Automatic';
            MinHistogramValue = 'automatic';
            MaxHistogramValue = 'automatic';
            % Warn the users that the value is being changed.
            % Max < Min Warning Box
            if isempty(findobj('Tag',['Msgbox_' ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Max < Min']))
                CPwarndlg(['Error in module number ', ModuleName, '. The range on the X axis for frequency counts requires the second number to be greater than the first. Changed to Automatic.'],[ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Max < Min'],'replace');
            end
        end
    else
        MinHistogramValue = 'automatic';
        MaxHistogramValue = 'automatic';
        % Warn the users that the value is being changed.
        % Bad axis range Warning Box
        if isempty(findobj('Tag',['Msgbox_' ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Bad axis range']))
            CPwarndlg(['Error in module number ', ModuleName, '. The range on the X axis for frequency counts on the Y axis requires two numbers to be entered. Changed to Automatic.'],[ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Bad axis range'],'replace');
        end
    end
catch
    % This block is executed if the entry is a character string, assumed to
    % be Automatic. Check to make sure that it is.
    if ~strcmpi('Automatic',MinAndMax)
        % A string was entered that was not 'Automatic'
        MinHistogramValue = 'automatic';
        MaxHistogramValue = 'automatic';
        % Warn the users that the value is being changed.
        % Unknown Entry Warning Box
        if isempty(findobj('Tag',['Msgbox_' ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Unknown Entry']))
            CPwarndlg(['Error in module number ', ModuleName, '. The range on the X axis for frequency counts on the Y axis requires two numbers to be entered. An unknown entry was made. Changed to Automatic.'],[ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Unknown Entry'],'replace');
        end
    else
        % Automatic was entered
        MinHistogramValue='automatic';
        MaxHistogramValue='automatic';
    end
end

%Replacement ends here
%%% Get measurements
try
    Measurements = handles.Measurements.(ObjectName).(FeatureName){SetBeingAnalyzed};
catch
    error(['Image processing was canceled in the ', ModuleName, ' module because the measurements could not be found. This module must be after a measure module or no objects were identified.']);
end

%%%%%%%%%%%%%%%%%%%%%
%%% DATA ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Calculates the default bin size and range based on all the data.
% SelectedMeasurementsCellArray = Measurements;
SelectedMeasurementsMatrix = Measurements;
PotentialMaxHistogramValue = max(SelectedMeasurementsMatrix);
PotentialMinHistogramValue = min(SelectedMeasurementsMatrix);

%%% See whether the min and max histogram values were user-entered numbers or should be automatically calculated.
if strcmp(MinHistogramValue,'automatic')
    MinHistogramValue = PotentialMinHistogramValue;
end
if strcmp(MaxHistogramValue,'automatic')
    MaxHistogramValue = PotentialMaxHistogramValue;
end
% Replacement Starts here
if (((MaxHistogramValue > 0) && (MinHistogramValue > 0)) && (strcmpi(LogOrLinear, 'Yes')))
    % value are greater than zero and Log plot is requested
    MaxLog = log10(MaxHistogramValue);
    MinLog = log10(MinHistogramValue);
    HistogramRange = MaxLog - MinLog;
    if HistogramRange <= 0
        % This only occurs now if the values were automatically calculated
        % incorrectly, which would be a bad error.
        error(['Image processing was canceled in the ', ModuleName, ' module because the numbers which was calculated automatically for one of these values, results in the range being zero or less.  For example, this would occur if you entered a minimum that is greater than the maximum which you asked to be automatically calculated.']);
    end
    BinWidth = HistogramRange/NumberOfBins;
    for n = 1:(NumberOfBins+2);
        PlotBinLocations(n) = 10^(MinLog + BinWidth*(n-2)); %#ok Ignore MLint
    end
    PlotBinLocations = PlotBinLocations';
    for n = 1:length(PlotBinLocations);
        PlotBinLocations(n) = log10(PlotBinLocations(n));
    end
elseif (~((MaxHistogramValue > 0) && (MinHistogramValue > 0)) && (strcmpi(LogOrLinear, 'Yes')))
    % if a range value is less than or equal to zero and log plot is
    % requested. Warn user that plot type is being change to linear
    HistogramRange = MaxHistogramValue - MinHistogramValue;
    if HistogramRange <= 0
        % This only occurs now if the values were automatically calculated
        % incorrectly, which would be a bad error.
        error(['Image processing was canceled in the ', ModuleName, ' module because the numbers which was calculated automatically for one of these values, results in the range being zero or less.  For example, this would occur if you entered a minimum that is greater than the maximum which you asked to be automatically calculated.']);
    end
    if isempty(findobj('Tag',['Msgbox_' ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Switch to Linear Plot']))
        CPwarndlg(['Error in module ', ModuleName, '. A negative or zero range value was entered or automatically calculated which is not compatible with a log plot. Plot type changed to Linear.'],[ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Switch to Linear Plot'],'replace');
    end
    BinWidth = HistogramRange/NumberOfBins;
    for n = 1:(NumberOfBins+2);
        PlotBinLocations(n) = MinHistogramValue + BinWidth*(n-2); %#ok Ignore MLint
    end
    PlotBinLocations = PlotBinLocations';
else
    % Linear plot type was selected
    HistogramRange = MaxHistogramValue - MinHistogramValue;
    BinWidth = HistogramRange/NumberOfBins;
    for n = 1:(NumberOfBins+2);
        PlotBinLocations(n) = MinHistogramValue + BinWidth*(n-2); %#ok Ignore MLint
    end
    PlotBinLocations = PlotBinLocations';
end
% Replacement ends here
%%% Now, for histogram-calculating  (BinLocations), replace the
%%% initial and final PlotBinLocations with + or - infinity.
BinLocations = PlotBinLocations;
BinLocations(1) = -inf;
BinLocations(n+1) = +inf;
%%% Calculates the XTickLabels.
for i = 1:(length(BinLocations)-1)
    XTickLabels{i} = BinLocations(i); %#ok Ignore MLint
end
XTickLabels{1} = ['< ', num2str(BinLocations(2),3)];
XTickLabels{i} = ['>= ', num2str(BinLocations(i),3)];

HistogramData = histc(SelectedMeasurementsMatrix,BinLocations);
%%% Deletes the last value of HistogramData, which is
%%% always a zero (because it's the number of values
%%% that match + inf).
HistogramData(n+1) = [];
FinalHistogramData(:,1) = HistogramData;

if strncmpi(NumberOrPercent,'P',1)
    for i = 1: size(FinalHistogramData,2)
        SumForThatColumn = sum(FinalHistogramData(:,i));
        FinalHistogramData(:,i) = 100*FinalHistogramData(:,i)/SumForThatColumn;
    end
end

%%%%%%%%%%%%%%%
%%% DISPLAY %%%
%%%%%%%%%%%%%%%
drawnow

StdUnit = 'point';

switch Color
    case 'Blue'
        HistColor='b';
    case 'Red'
        HistColor='r';
    case 'Green'
        HistColor='g';
    case 'Yellow'
        HistColor='y';
    case 'Magenta'
        HistColor='m';
    case 'Cyan'
        HistColor='c';
    case 'Black'
        HistColor='k';
    case 'White'
        HistColor='w';
    otherwise
        HistColor=[.7 .7 .9];
end

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);

%%% Activates the appropriate figure window.
HistHandle = CPfigure(handles,'Text',ThisModuleFigureNumber);

h = subplot(1,1,1);
pos=get(h, 'Position');
set(h, 'Position', [pos(1) pos(2) pos(3) pos(4)-.1]);

if strcmpi(LineOrBar,'bar')
    Graph=bar('v6',PlotBinLocations,FinalHistogramData(:,1));
    set(Graph, 'FaceColor',HistColor);
elseif strcmpi(LineOrBar,'area')
    Graph=area('v6',PlotBinLocations,FinalHistogramData(:,1));
    set(Graph, 'FaceColor',HistColor);
else
    plot('v6',PlotBinLocations,FinalHistogramData(:,1),'LineWidth',2, 'color', HistColor);
end

% FeatureName=handles.Measurements.(ObjectName).(FeatureName){FeatureNumber};
set(h,'XTickLabel',XTickLabels)
set(h,'XTick',PlotBinLocations)
set(gca,'Tag','BarTag','ActivePositionProperty','Position')
set(get(h,'XLabel'),'String',[FeatureName,' of ',ObjectName])
set(get(h,'Title'),'String',['Histogram for ', FeatureName,' of ',ObjectName])

if strncmpi(NumberOrPercent,'N',1)
    set(get(h,'YLabel'),'String','Number of objects')
else
    set(get(h,'YLabel'),'String','Percentage of objects')
end
%%% Using "axis tight" here is ok, I think, because we are displaying
%%% data, not images.
axis tight

FontSize=handles.Preferences.FontSize;
left=30;
bottom=390;

%%% Creates text
uicontrol('Parent',HistHandle, ...
    'BackgroundColor',[.7 .7 .9], ...
    'Unit',StdUnit, ...
    'Position',[left bottom 85 15], ...
    'Units','Normalized',...
    'String','X axis labels:', ...
    'Style','text', ...
    'FontSize',FontSize);
%%% Hide every other label button.
Button1Callback = 'tempData = get(gcf,''UserData'');AxesHandles = findobj(gcf,''Tag'',''BarTag'');PlotBinLocations = get(AxesHandles(1),''XTick'');XTickLabels = get(AxesHandles(1),''XTickLabel'');if ceil(length(PlotBinLocations)/2) ~= length(PlotBinLocations)/2,PlotBinLocations(length(PlotBinLocations)) = [];XTickLabels(length(XTickLabels)) = [];end;PlotBinLocations2 = reshape(PlotBinLocations,2,[]);XTickLabels2 = reshape(XTickLabels,2,[]);set(AxesHandles,''XTick'',PlotBinLocations2(1,:));set(AxesHandles,''XTickLabel'',XTickLabels2(1,:));tempData.HideOption = 1;set(tempData.HideHandle,''UserData'',tempData);set(tempData.DecimalHandle,''UserData'',tempData);clear';
Button1 = uicontrol('Parent',HistHandle, ...
    'Unit',StdUnit, ...
    'BackgroundColor',[.7 .7 .9], ...
    'CallBack',Button1Callback, ...
    'Position',[left+100 bottom 50 22], ...
    'Units','Normalized',...
    'String','Fewer',...
    'Style','pushbutton',...
    'UserData',0,...
    'FontSize',FontSize);
%%% Decimal places Measurement axis labels.
Button2Callback = 'tempData = get(gcf,''UserData'');HideOption = tempData.HideOption;FigureSettings = tempData.FigureSettings; PlotBinLocations = FigureSettings{1};PreXTickLabels = FigureSettings{2};XTickLabels = PreXTickLabels(2:end-1); AxesHandles = findobj(gcf,''Tag'',''BarTag''); NumberOfDecimals = inputdlg(''Enter the number of decimal places to display'',''Enter the number of decimal places'',1,{''0''}); if ~isempty(NumberOfDecimals), NumberValues = cell2mat(XTickLabels); Command = [''%.'',num2str(NumberOfDecimals{1}),''f'']; NewNumberValues = num2str(NumberValues'',Command); NewNumberValuesPlusFirstLast = [PreXTickLabels(1); cellstr(NewNumberValues); PreXTickLabels(end)];if HideOption,if ceil(length(PlotBinLocations)/2) ~= length(PlotBinLocations)/2,PlotBinLocations(length(PlotBinLocations)) = [];NewNumberValuesPlusFirstLast(length(NewNumberValuesPlusFirstLast)) = [];end,PlotBinLocations2 = reshape(PlotBinLocations,2,[]);XTickLabels2 = reshape(NewNumberValuesPlusFirstLast,2,[]);set(AxesHandles,''XTickLabel'',XTickLabels2);set(AxesHandles,''XTick'',PlotBinLocations);else,set(AxesHandles,''XTickLabel'',NewNumberValuesPlusFirstLast);set(AxesHandles,''XTick'',PlotBinLocations);end,tempData.DecimalOption = 1;set(tempData.HideHandle,''UserData'',tempData);set(tempData.DecimalHandle,''UserData'',tempData);clear, drawnow, end';
Button2 = uicontrol('Parent',HistHandle, ...
    'Unit',StdUnit,...
    'BackgroundColor',[.7 .7 .9], ...
    'CallBack',Button2Callback, ...
    'Position',[left+160 bottom 50 22], ...
    'Units','Normalized',...
    'String','Decimals',...
    'Style','pushbutton',...
    'UserData',0,...
    'FontSize',FontSize);
%%% Restore original X axis labels.
Button3Callback = 'tempData = get(gcf,''UserData'');FigureSettings = tempData.FigureSettings;PlotBinLocations = FigureSettings{1}; XTickLabels = FigureSettings{2}; AxesHandles = findobj(gcf, ''Tag'', ''BarTag''); set(AxesHandles,''XTick'',PlotBinLocations); set(AxesHandles,''XTickLabel'',XTickLabels); clear';
Button3 = uicontrol('Parent',HistHandle, ...
    'Unit',StdUnit, ...
    'BackgroundColor',[.7 .7 .9], ...
    'CallBack',Button3Callback, ...
    'Position',[left+220 bottom 50 22], ...
    'Units','Normalized',...
    'String','Restore', ...
    'Style','pushbutton', ...
    'FontSize',FontSize); %#ok Ignore MLint

% %Add buttons
FigureSettings{1} = PlotBinLocations;
FigureSettings{2} = XTickLabels;
FigureSettings{3} = FinalHistogramData;
tempData.HideOption = 0;
tempData.HideHandle = Button1;
tempData.DecimalHandle = Button2;
tempData.FigureSettings = FigureSettings;
tempData.handles = rmfield(handles,'Pipeline');
tempData.Application = 'CellProfiler';
set(HistHandle,'UserData',tempData);
FigureShot = CPimcapture(HistHandle); %% using defaults of whole figure and 150 dpi
handles.Pipeline.(HistImage)=FigureShot;
