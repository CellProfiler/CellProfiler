function handles = GetPlotMeasurement(handles)

% Help for the Get Plot Measurement module:
% Category: Other
%
% SHORT DESCRIPTION:
% Plot measured data in several formats.
% *************************************************************************
%
% Feature Number:
% The feature number specifies which feature from the Measure module will
% be used for plotting. See each Measure module's help for the numbered
% list of the features measured by that module.
%
% See also MEASUREOBJECTINTENSITY, MEASUREOBJECTAREASHAPE,
% MEASUREOBJECTTEXTURE, MEASURECORRELATION

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
% $Revision: 2606 $

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What type of plot do you want?
%choiceVAR01 = Bar
%choiceVAR01 = Line
%choiceVAR01 = Scatter 1
%choiceVAR01 = Scatter 2
PlotType = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = Which object would you like to use for the plots (The option IMAGE currently only works with Correlation measurements)?
%choiceVAR02 = Image
%infotypeVAR02 = objectgroup
%inputtypeVAR02 = popupmenu
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Which category of measurements would you like to use?
%choiceVAR03 = AreaShape
%choiceVAR03 = Correlation
%choiceVAR03 = Intensity
%choiceVAR03 = Neighbors
%choiceVAR03 = Texture
%inputtypeVAR03 = popupmenu custom
FeatureType = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = Which feature do you want to use? (Enter the feature number - see HELP for explanation)
%defaultVAR04 = 1
FeatureNo = str2double(handles.Settings.VariableValues{CurrentModuleNum,4});

if isempty(FeatureNo)
    error('You entered an incorrect Feature Number.');
end

%textVAR05 = If using INTENSITY or TEXTURE measures, which image would you like to process?
%infotypeVAR05 = imagegroup
%inputtypeVAR05 = popupmenu
Image = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = What do you want to call the generated plots?
%defaultVAR06 = OrigPlot
%infotypeVAR06 = imagegroup indep
PlotImage = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = ONLY ENTER THE FOLLOWING INFORMATION IF USING SCATTER PLOT WITH TWO MEASUREMENTS!

%textVAR08 = Which object would you like for the second scatter plot measurement (The option IMAGE currently only works with Correlation measurements)?
%choiceVAR08 = Image
%infotypeVAR08 = objectgroup
%inputtypeVAR08 = popupmenu
ObjectName2 = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%textVAR09 = Which category of measurements would you like to use?
%choiceVAR09 = AreaShape
%choiceVAR09 = Correlation
%choiceVAR09 = Intensity
%choiceVAR09 = Neighbors
%choiceVAR09 = Texture
%inputtypeVAR09 = popupmenu custom
FeatureType2 = char(handles.Settings.VariableValues{CurrentModuleNum,9});

%textVAR10 = Which feature do you want to use? (Enter the feature number - see HELP for explanation)
%defaultVAR10 = 1
FeatureNo2 = str2double(handles.Settings.VariableValues{CurrentModuleNum,10});

if isempty(FeatureNo2)
    error(['Image processing was canceled in the ', ModuleName, ' module because you entered an incorrect Feature Number.']);
end

%textVAR11 = If using INTENSITY or TEXTURE measures, which image would you like to process?
%infotypeVAR11 = imagegroup
%inputtypeVAR11 = popupmenu
Image2 = char(handles.Settings.VariableValues{CurrentModuleNum,11});

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines which cycle is being analyzed.
SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;
NumberOfImageSets = handles.Current.NumberOfImageSets;

if strcmp(FeatureType,'Intensity') || strcmp(FeatureType,'Texture')
    FeatureType = [FeatureType, '_',Image];
end

if strcmp(FeatureType2,'Intensity') || strcmp(FeatureType2,'Texture')
    FeatureType2 = [FeatureType2, '_',Image2];
end

%%%%%%%%%%%%%%%%%%%%%
%%% DATA ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow

if strcmp(PlotType,'Bar')
    PlotType = 1;
    %%% Line chart
elseif strcmp(PlotType,'Line')
    PlotType = 2;
    %%% Scatter plot, 1 measurement
elseif strcmp(PlotType,'Scatter 1')
    PlotType = 3;
    %%% Scatter plot, 2 measurements
elseif strcmp(PlotType,'Scatter 2')
    PlotType = 4;
end

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
FigHandle = CPfigure(handles,ThisModuleFigureNumber);

if PlotType == 4
    CPplotmeasurement(handles,FigHandle,PlotType,1,ObjectName,FeatureType,FeatureNo,ObjectName2,FeatureType2,FeatureNo2);
else
    CPplotmeasurement(handles,FigHandle,PlotType,1,ObjectName,FeatureType,FeatureNo);
end

%%%%%%%%%%%%%%%
%%% DISPLAY %%%
%%%%%%%%%%%%%%%
drawnow

OneFrame = getframe(FigHandle);
handles.Pipeline.(PlotImage)=OneFrame.cdata;