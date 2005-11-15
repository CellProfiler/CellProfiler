function handles = GetHistogram(handles)
% Help for the Get Histogram module:
% Category: Other
%
%
% Feature Number:
% The feature number is the parameter from the chosen module (AreaShape,
% Intensity, Texture) which will be used for the histogram. The following
% tables provide the feature numbers for each measurement made by the three
% modules:
%
% Area Shape:               Feature Number:
% Area                    |       1
% Eccentricity            |       2
% Solidity                |       3
% Extent                  |       4
% Euler Number            |       5
% Perimeter               |       6
% Form factor             |       7
% MajorAxisLength         |       8
% MinorAxisLength         |       9
%
% Intensity:                Feature Number:
% IntegratedIntensity     |       1
% MeanIntensity           |       2
% StdIntensity            |       3
% MinIntensity            |       4
% MaxIntensity            |       5
% IntegratedIntensityEdge |       6
% MeanIntensityEdge       |       7
% StdIntensityEdge        |       8
% MinIntensityEdge        |       9
% MaxIntensityEdge        |      10
% MassDisplacement        |      11
%
% Texture:                  Feature Number:
% AngularSecondMoment     |       1
% Contrast                |       2
% Correlation             |       3
% Variance                |       4
% InverseDifferenceMoment |       5
% SumAverage              |       6
% SumVariance             |       7
% SumEntropy              |       8
% Entropy                 |       9
% DifferenceVariance      |      10
% DifferenceEntropy       |      11
% InformationMeasure      |      12
% InformationMeasure2     |      13
% Gabor1x                 |      14
% Gabor1y                 |      15

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne Carpenter
%   Thouis Jones
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

%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.

CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);
ModuleName = char(handles.Settings.ModuleNames(CurrentModuleNum));

%textVAR01 = Which object would you like to use for the histogram (The option IMAGE currently only works with Correlation measurements)?
%choiceVAR01 = Image
%infotypeVAR01 = objectgroup
%inputtypeVAR01 = popupmenu
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = Which category of measurements would you like to use?
%choiceVAR02 = AreaShape
%choiceVAR02 = Correlation
%choiceVAR02 = Intensity
%choiceVAR02 = Neighbors
%choiceVAR02 = Texture
%inputtypeVAR02 = popupmenu custom
Measure = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Which feature do you want to use? (Enter the feature number - see HELP for explanation)
%defaultVAR03 = 1
FeatureNumber = str2double(handles.Settings.VariableValues{CurrentModuleNum,3});

if isempty(FeatureNumber)
    error('You entered an incorrect Feature Number.');
end

%textVAR04 = If using INTENSITY or TEXTURE measures, which image would you like to process?
%infotypeVAR04 = imagegroup
%inputtypeVAR04 = popupmenu
Image = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = What do you want to call the generated histograms?
%defaultVAR05 = OrigHist
%infotypeVAR05 = imagegroup indep
HistImage = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = How many bins do you want?
%choiceVAR06 = 2
%choiceVAR06 = 16
%choiceVAR06 = 256
NumberOfBins = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,6}));
%inputtypeVAR06 = popupmenu custom

if isempty(NumberOfBins)
    error('You entered an incorrect number of bins.');
end

%textVAR07 = Log transform the histogram?
%choiceVAR07 = No
%choiceVAR07 = Yes
LogOrLinear = char(handles.Settings.VariableValues{CurrentModuleNum,7});
%inputtypeVAR07 = popupmenu

%textVAR08 = Min and Max values for histogram (Min,Max) or automatic:
%defaultVAR08 = automatic
MinAndMax = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%textVAR09 = Do you want to use numbers of percents?
%choiceVAR09 = Numbers
%choiceVAR09 = Percents
NumberOrPercent = char(handles.Settings.VariableValues{CurrentModuleNum,9});
%inputtypeVAR09 = popupmenu

%textVAR10 = Do you want a line, bar, or area graph?
%choiceVAR10 = Line
%choiceVAR10 = Bar
%choiceVAR10 = Area
LineOrBar = char(handles.Settings.VariableValues{CurrentModuleNum,10});
%inputtypeVAR10 = popupmenu

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines which image set is being analyzed.
SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;
NumberOfImageSets = handles.Current.NumberOfImageSets;

if strcmp(Measure,'Intensity') || strcmp(Measure,'Texture')
    Measure = [Measure, '_',Image];
end

%%% Checks that the Min and Max have valid values
index = strfind(MinAndMax,',');
if isempty(index)
    if strcmpi(MinAndMax,'automatic')
        MinHistogramValue = 'automatic';
        MaxHistogramValue = 'automatic';
    else
        error(['The Min and Max size entry in the ', ModuleName, ' module is invalid.'])
    end
else
    MinHistogramValue = str2double(MinAndMax(1:index-1));
    MaxHistogramValue = str2double(MinAndMax(index+1:end));
    if isempty(MinHistogramValue) || isempty(MaxHistogramValue)
        error('You did not enter numbers for the min and max histogram values');
    end
end

try
        Measurements = handles.Measurements.(ObjectName).(Measure){SetBeingAnalyzed}(:,FeatureNumber);
catch
    error(['The measurements for ',ModuleName,' could not be found. This module must be after a measure module or no objects were identified.']);
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

if strcmpi(LogOrLinear,'Yes')
    MaxLog = log10(MaxHistogramValue);
    MinLog = log10(MinHistogramValue);
    HistogramRange = MaxLog - MinLog;
    if HistogramRange <= 0
        error('The numbers you entered for the minimum or maximum, or the number which was calculated automatically for one of these values, results in the range being zero or less.  For example, this would occur if you entered a minimum that is greater than the maximum which you asked to be automatically calculated.');
    end
    BinWidth = HistogramRange/NumberOfBins;
    for n = 1:(NumberOfBins+2);
        PlotBinLocations(n) = 10^(MinLog + BinWidth*(n-2));
    end
else
    %%% Determine plot bin locations.
    HistogramRange = MaxHistogramValue - MinHistogramValue;
    if HistogramRange <= 0
        error('The numbers you entered for the minimum or maximum, or the number which was calculated automatically for one of these values, results in the range being zero or less.  For example, this would occur if you entered a minimum that is greater than the maximum which you asked to be automatically calculated.');
    end
    BinWidth = HistogramRange/NumberOfBins;
    for n = 1:(NumberOfBins+2);
        PlotBinLocations(n) = MinHistogramValue + BinWidth*(n-2);
    end
end

%%% Now, for histogram-calculating bins (BinLocations), replace the
%%% initial and final PlotBinLocations with + or - infinity.
PlotBinLocations = PlotBinLocations';
BinLocations = PlotBinLocations;
BinLocations(1) = -inf;
BinLocations(n+1) = +inf;
%%% Calculates the XTickLabels.
for i = 1:(length(BinLocations)-1)
    XTickLabels{i} = BinLocations(i);
end
XTickLabels{1} = ['< ', num2str(BinLocations(2),3)];
XTickLabels{i} = ['>= ', num2str(BinLocations(i),3)];

if strcmpi(LogOrLinear,'Yes')
    for n = 1:length(PlotBinLocations);
        PlotBinLocations(n) = log10(PlotBinLocations(n));
    end
end

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

fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);

StdUnit = 'point';
StdColor = get(0,'DefaultUIcontrolBackgroundColor');
PointsPerPixel = 72/get(0,'ScreenPixelsPerInch');
%%% Creates the display window.
HistHandle = CPfigure(handles,ThisModuleFigureNumber);

h = subplot(1,1,1);

if strcmpi(LineOrBar,'bar')
    bar('v6',PlotBinLocations,FinalHistogramData(:,1))
elseif strcmpi(LineOrBar,'area')
    area('v6',PlotBinLocations,FinalHistogramData(:,1))
else
    plot('v6',PlotBinLocations,FinalHistogramData(:,1),'LineWidth',2)
end

set(h,'XTickLabel',XTickLabels)
set(h,'XTick',PlotBinLocations)
set(gca,'Tag','BarTag','ActivePositionProperty','Position')

if strncmpi(NumberOrPercent,'N',1)
    set(get(h,'YLabel'),'String','Number of objects')
else
    set(get(h,'YLabel'),'String','Percentage of objects')
end
axis tight

OneFrame = getframe(HistHandle);
handles.Pipeline.(HistImage)=OneFrame.cdata;