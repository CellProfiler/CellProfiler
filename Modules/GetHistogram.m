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
FeatureNumber = str2num(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = If using INTENSITY or TEXTURE measures, which image would you like to process?
%infotypeVAR04 = imagegroup
%inputtypeVAR04 = popupmenu
Image = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = What do you want to call the generated histograms?
%defaultVAR05 = OrigHist
%infotypeVAR05 = imagegroup indep
HistImage = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = How many bins do you want?
%choiceVAR06 = Automatic
%choiceVAR06 = 2
%choiceVAR06 = 16
%choiceVAR06 = 256
NumBins = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%inputtypeVAR06 = popupmenu custom

%textVAR07 = Log transform the histogram?
%choiceVAR07 = No
%choiceVAR07 = Yes
LogOption = char(handles.Settings.VariableValues{CurrentModuleNum,7});
%inputtypeVAR07 = popupmenu

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% FIRST IMAGE SET FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

if strcmp(LogOption,'Yes')
    OrigImage(OrigImage == 0) = min(OrigImage(OrigImage > 0));
    OrigImage = log(OrigImage);
    hi = max(OrigImage(:));
    lo = min(OrigImage(:));
    OrigImage = (OrigImage - lo)/(hi - lo);
end

HistHandle = CPfigure(handles,ThisModuleFigureNumber);

drawnow

fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
OneFrame = getframe(HistHandle);
handles.Pipeline.(HistImage)=OneFrame.cdata;