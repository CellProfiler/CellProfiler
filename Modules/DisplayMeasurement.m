function handles = DisplayMeasurement(handles)

% Help for the Display Measurement module:
% Category: Other
%
% SHORT DESCRIPTION:
% Plots measured data in several formats.
% *************************************************************************
%
% The DisplayMeasurement module allows data generated from the previous
% modules to be displayed on a plot.  In the Settings, the type of the plot
% can be specified.  The data can be displayed in a bar, line, or scatter
% plot.  The user must choose the category of the data set to plot or, the
% user may choose to plot a ratio of two data sets.  The scatterplot
% requires additional information about the second set of measurements
% used.
%
% The resulting plots can be saved using the Save Images module.
%
% Feature Number:
% The feature number specifies which feature from the Measure module will
% be used for plotting. See each Measure module's help for the numbered
% list of the features measured by that module.
%
% See also MeasureObjectAreaShape, MeasureObjectIntensity, MeasureTexture,
% MeasureCorrelation, MeasureObjectNeighbors, CalculateRatios.

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

%textVAR01 = What type of plot do you want?
%choiceVAR01 = Bar
%choiceVAR01 = Line
%choiceVAR01 = Scatter 1
%choiceVAR01 = Scatter 2
PlotType = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = Which object would you like to use for the plots, or if using a Ratio, what is the numerator object (The option IMAGE currently only works with Correlation measurements)?
%choiceVAR02 = Image
%infotypeVAR02 = objectgroup
%inputtypeVAR02 = popupmenu
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Which category of measurements would you like to use?
%choiceVAR03 = AreaOccupied
%choiceVAR03 = AreaShape
%choiceVAR03 = Children
%choiceVAR03 = Parent
%choiceVAR03 = Correlation
%choiceVAR03 = Intensity
%choiceVAR03 = Neighbors
%choiceVAR03 = Ratio
%choiceVAR03 = Texture
%choiceVAR03 = RadialDistribution
%choiceVAR03 = Granularity
%inputtypeVAR03 = popupmenu custom
MeasureChoice = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = Which feature do you want to use? (Enter the feature name or number - see HELP for explanation)
%defaultVAR04 = 1
FeatureNo = handles.Settings.VariableValues{CurrentModuleNum,4};

if isempty(FeatureNo)
    error(['Image processing was canceled in the ', ModuleName, ' module because your entry for the Feature Number is invalid.']);
end

%textVAR05 = For INTENSITY or TEXTURE features, which image would you like to process?
%infotypeVAR05 = imagegroup
%inputtypeVAR05 = popupmenu
Image = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = For TEXTURE or RADIALDISTRIBUTION features, what previously measured texture scale (TEXTURE) or previously used number of bins (RADIALDISTRIBUTION) do you want to use?
%defaultVAR06 = 1
UserSpecifiedNumber = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = What do you want to call the generated plots?
%defaultVAR07 = OrigPlot
%infotypeVAR07 = imagegroup indep
PlotImage = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = ONLY ENTER THE FOLLOWING INFORMATION IF USING SCATTER PLOT WITH TWO MEASUREMENTS!

%textVAR09 = Which object would you like for the second scatter plot measurement, or if using a Ratio, what is the numerator object (The option IMAGE currently only works with Correlation measurements)?
%choiceVAR09 = Image
%infotypeVAR09 = objectgroup
%inputtypeVAR09 = popupmenu
ObjectName2 = char(handles.Settings.VariableValues{CurrentModuleNum,9});

%textVAR10 = Which category of measurements would you like to use?
%choiceVAR10 = AreaOccupied
%choiceVAR10 = AreaShape
%choiceVAR10 = Children
%choiceVAR10 = Parent
%choiceVAR10 = Correlation
%choiceVAR10 = Intensity
%choiceVAR10 = Neighbors
%choiceVAR10 = Ratio
%choiceVAR10 = Texture
%choiceVAR10 = RadialDistribution
%choiceVAR10 = Granularity
%inputtypeVAR10 = popupmenu custom
MeasureChoice2 = char(handles.Settings.VariableValues{CurrentModuleNum,10});

%textVAR11 = Which feature do you want to use? (Enter the feature name or number - see HELP for explanation)
%defaultVAR11 = 1
FeatureNo2 = handles.Settings.VariableValues{CurrentModuleNum,11};

if isempty(FeatureNo2)
    error(['Image processing was canceled in the ', ModuleName, ' module because you entered an incorrect Feature Number.']);
end

%textVAR12 = For INTENSITY or TEXTURE features, which image would you like to process?
%infotypeVAR12 = imagegroup
%inputtypeVAR12 = popupmenu
Image2 = char(handles.Settings.VariableValues{CurrentModuleNum,12});

%textVAR13 = For TEXTURE or RADIALDISTRIBUTION features, what previously measured texture scale (TEXTURE) or previously used number of bins (RADIALDISTRIBUTION) do you want to use?
%defaultVAR13 = 1
UserSpecifiedNumber2 = char(handles.Settings.VariableValues{CurrentModuleNum,13});

%%%VariableRevisionNumber = 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines which cycle is being analyzed.
% SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;
% NumberOfImageSets = handles.Current.NumberOfImageSets;

%%% Get the correct fieldname where measurements are located
try
    switch lower(MeasureChoice)
        case {'areaoccupied','intensity','granularity','imagequality','radialdistribution'}
            FeatureName = CPgetfeaturenamesfromnumbers(handles, ObjectName, MeasureChoice, FeatureNo, Image);
        case {'areashape','neighbors','ratio'}
            FeatureName = CPgetfeaturenamesfromnumbers(handles, ObjectName, MeasureChoice, FeatureNo);
        case {'texture','radialdistribution'}
            FeatureName = CPgetfeaturenamesfromnumbers(handles, ObjectName, MeasureChoice, FeatureNo, Image, UserSpecifiedNumber);
    end
catch
    error(['Image processing was canceled in the ', ModuleName, ' module because the category of measurement you chose, ', MeasureChoice, ', was not available for ', ObjectName]);
end

if strcmp(PlotType,'Scatter 2')
    %%% Get the correct fieldname where measurements are located for second
    %%% scatterplot measures
    try
        switch lower(MeasureChoice2)
            case {'areaoccupied','intensity','granularity','imagequality','radialdistribution'}
                FeatureName2 = CPgetfeaturenamesfromnumbers(handles, ObjectName2, MeasureChoice2, FeatureNo2, Image2);
            case {'areashape','neighbors','ratio'}
                FeatureName2 = CPgetfeaturenamesfromnumbers(handles, ObjectName2, MeasureChoice2, FeatureNo2);
            case {'texture','radialdistribution'}
                FeatureName2 = CPgetfeaturenamesfromnumbers(handles, ObjectName2, MeasureChoice2, FeatureNo2, Image2, UserSpecifiedNumber2);
        end
    catch
        error(['Image processing was canceled in the ', ModuleName, ' module because the category of measurement you chose, ', MeasureChoice, ', was not available for ', ObjectName]);
    end
end

%%%%%%%%%%%%%%%%%%%%%
%%% DATA ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow

if strcmp(PlotType,'Bar')           %%%% Bar chart
    PlotType = 1;
elseif strcmp(PlotType,'Line')      %%% Line chart
    PlotType = 2;
elseif strcmp(PlotType,'Scatter 1')     %%% Scatter plot, 1 measurement
    PlotType = 3;
elseif strcmp(PlotType,'Scatter 2')    %%% Scatter plot, 2 measurements
    PlotType = 4;
end

%%% Activates the appropriate figure window.
ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
drawnow
FigHandle = CPfigure(handles,'Image',ThisModuleFigureNumber);

if PlotType == 4
%     CPplotmeasurement(handles,PlotType,FigHandle,1,ObjectName,MeasureChoice,FeatureNo,ObjectName2,MeasureChoice2,FeatureNo2);
    CPplotmeasurement(handles,PlotType,FigHandle,1,ObjectName,FeatureName,ObjectName2,FeatureName2);
else
    CPplotmeasurement(handles,PlotType,FigHandle,1,ObjectName,FeatureName);

%                     handles,PlotType,FigHandle,ModuleFlag,Object,Feature,Object2,Feature2
end

%%%%%%%%%%%%%%%
%%% DISPLAY %%%
%%%%%%%%%%%%%%%
drawnow

FigureShot = CPimcapture(FigHandle); %% using defaults of whole figure and 150 dpi
handles.Pipeline.(PlotImage)=FigureShot;