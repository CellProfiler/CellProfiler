function handles = MeasureRatios(handles)

% Help for the Measure Ratios module:
% Category: Measurement
%
% SHORT DESCRIPTION:
% Measures the ratio between any measurements already taken
% (e.g. Intensity/Area)
% *************************************************************************
%
% This module can take any measurement produced by previous modules and
% calculate a ratio. Ratios can also be used to calculate other ratios and
% be used in ClassifyObjects.
%
% Feature Number:
% The feature number is the parameter from the chosen module (AreaShape,
% Intensity, Texture) which will be used for the ratio. The following
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
% Gabor2x                 |      16
% Gabor2y                 |      17
% Gabor3x                 |      18
% Gabor3y                 |      19
%
% See also MEASUREIMAGEAREAOCCUPIED, MEASUREIMAGEINTENSITY,
% MEASUREOBJECTINTENSITY, MEASUREOBJECTAREASHAPE,
% MEASUREOBJECTTEXTURE, MEASURECORRELATION

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
% $Revision: 1843 $

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

%%% Reads the current module number, because this is needed to find the
%%% variable values that the user entered.
[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = Which object would you like to use for the numerator (The option IMAGE currently only works with Correlation measurements)?
%choiceVAR01 = Image
%infotypeVAR01 = objectgroup
%inputtypeVAR01 = popupmenu
NumObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = Which category of measurements would you like to use?
%choiceVAR02 = AreaShape
%choiceVAR02 = Correlation
%choiceVAR02 = Intensity
%choiceVAR02 = Neighbors
%choiceVAR02 = Texture
%inputtypeVAR02 = popupmenu custom
NumMeasure = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Which feature do you want to use? (Enter the feature number - see HELP for explanation)
%defaultVAR03 = 1
NumFeatureNumber = str2double(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = If using INTENSITY or TEXTURE measures, which image would you like to process?
%infotypeVAR04 = imagegroup
%inputtypeVAR04 = popupmenu
NumImage = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = Which object would you like to use for the denominator (The option IMAGE currently only works with Correlation measurements)?
%choiceVAR05 = Image
%infotypeVAR05 = objectgroup
%inputtypeVAR05 = popupmenu
DenomObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = Which category of measurements would you like to use?
%choiceVAR06 = AreaShape
%choiceVAR06 = Correlation
%choiceVAR06 = Intensity
%choiceVAR06 = Neighbors
%choiceVAR06 = Texture
%inputtypeVAR06 = popupmenu custom
DenomMeasure = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = Which feature do you want to use? (Enter the feature number - see HELP for explanation)
%defaultVAR07 = 1
DenomFeatureNumber = str2double(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = If using INTENSITY or TEXTURE measures, which image would you like to process?
%infotypeVAR08 = imagegroup
%inputtypeVAR08 = popupmenu
DenomImage = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%textVAR09 = Do you want the log of the ratio?
%choiceVAR09 = No
%choiceVAR09 = Yes
%inputtypeVAR09 = popupmenu
LogChoice = char(handles.Settings.VariableValues{CurrentModuleNum,9});

%%%VariableRevisionNumber = 4

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% MAKE MEASUREMENTS & SAVE TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;

if strcmp(NumMeasure,'Intensity') || strcmp(NumMeasure,'Texture')
    NumMeasure = [NumMeasure, '_', NumImage];
end

if strcmp(DenomMeasure,'Intensity') || strcmp(DenomMeasure,'Texture')
    DenomMeasure = [DenomMeasure, '_', DenomImage];
end

% Get measurements
NumeratorMeasurements = handles.Measurements.(NumObjectName).(NumMeasure){SetBeingAnalyzed};
NumeratorMeasurements = NumeratorMeasurements(:,NumFeatureNumber);
DenominatorMeasurements = handles.Measurements.(DenomObjectName).(DenomMeasure){SetBeingAnalyzed};
DenominatorMeasurements = DenominatorMeasurements(:,DenomFeatureNumber);

if length(NumeratorMeasurements) ~= length(DenominatorMeasurements)
    error(['Image processing was canceled in the ', ModuleName, ' module because the specified object names ',NumObjectName,' and ',DenomObjectName,' do not have the same object count.']);
end

try
    NewFieldName = [NumObjectName,'_',NumMeasure(1),'_',num2str(NumFeatureNumber),'_dividedby_',DenomObjectName,'_',DenomMeasure(1),'_',num2str(DenomFeatureNumber)];
    if isfield(handles.Measurements.(NumObjectName),'Ratios')
        OldPos = strmatch(NewFieldName,handles.Measurements.(NumObjectName).RatiosFeatures,'exact');
        if isempty(OldPos)
            handles.Measurements.(NumObjectName).RatiosFeatures(end+1) = {NewFieldName};
            if strcmp(LogChoice,'No')
                handles.Measurements.(NumObjectName).Ratios{SetBeingAnalyzed}(:,end+1) = NumeratorMeasurements./DenominatorMeasurements;
            else
                handles.Measurements.(NumObjectName).Ratios{SetBeingAnalyzed}(:,end+1) = log10(NumeratorMeasurements./DenominatorMeasurements);
            end
        else
            if strcmp(LogChoice,'No')
                handles.Measurements.(NumObjectName).Ratios{SetBeingAnalyzed}(:,OldPos) = NumeratorMeasurements./DenominatorMeasurements;
            else
                handles.Measurements.(NumObjectName).Ratios{SetBeingAnalyzed}(:,OldPos) = log10(NumeratorMeasurements./DenominatorMeasurements);
            end
        end
    else
        handles.Measurements.(NumObjectName).RatiosFeatures = {NewFieldName};
        if strcmp(LogChoice,'No')
            handles.Measurements.(NumObjectName).Ratios{SetBeingAnalyzed}(:,1) = NumeratorMeasurements./DenominatorMeasurements;
        else
            handles.Measurements.(NumObjectName).Ratios{SetBeingAnalyzed}(:,1) = log10(NumeratorMeasurements./DenominatorMeasurements);
        end
    end
catch
    error(['Image processing was canceled in the ', ModuleName, ' module because storing the measurements failed for some reason.']);
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% The figure window display is unnecessary for this module, so the figure
%%% window is closed the first time through the module.
fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
if any(findobj == ThisModuleFigureNumber)
    close(ThisModuleFigureNumber);
end