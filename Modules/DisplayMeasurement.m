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
ObjectName{1} = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Which category of measurements would you like to use?
%inputtypeVAR03 = popupmenu category
Category{1} = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = Which feature do you want to use? (Enter the feature name or number - see HELP for explanation)
%defaultVAR04 = 1
%inputtypeVAR04 = popupmenu measurement
FeatureNbr{1} = handles.Settings.VariableValues{CurrentModuleNum,4};

if isempty(FeatureNbr{1})
    error(['Image processing was canceled in the ', ModuleName, ' module because your entry for the Feature Number is invalid.']);
end

%textVAR05 = For INTENSITY, AREAOCCUPIED or TEXTURE features, which image would you like to process?
%infotypeVAR05 = imagegroup
%inputtypeVAR05 = popupmenu custom
ImageName{1} = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = For TEXTURE, RADIAL DISTRIBUTION, OR NEIGHBORS features, what previously measured size scale (TEXTURE OR NEIGHBORS) or previously used number of bins (RADIALDISTRIBUTION) do you want to use?
%defaultVAR06 = 1
%inputtypeVAR06 = popupmenu scale
SizeScale{1} = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = What do you want to call the generated plots?
%defaultVAR07 = OrigPlot
%infotypeVAR07 = imagegroup indep
PlotImage = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = ONLY ENTER THE FOLLOWING INFORMATION IF USING SCATTER PLOT WITH TWO MEASUREMENTS!

%textVAR09 = Which object would you like for the second scatter plot measurement, or if using a Ratio, what is the numerator object (The option IMAGE currently only works with Correlation measurements)?
%choiceVAR09 = Image
%infotypeVAR09 = objectgroup
%inputtypeVAR09 = popupmenu
ObjectName{2} = char(handles.Settings.VariableValues{CurrentModuleNum,9});

%textVAR10 = Which category of measurements would you like to use?
%inputtypeVAR10 = popupmenu category
Category{2} = char(handles.Settings.VariableValues{CurrentModuleNum,10});

%textVAR11 = Which feature do you want to use? (Enter the feature name or number - see HELP for explanation)
%defaultVAR11 = 1
%inputtypeVAR11 = popupmenu measurement
FeatureNbr{2} = handles.Settings.VariableValues{CurrentModuleNum,11};

if isempty(FeatureNbr{2})
    error(['Image processing was canceled in the ', ModuleName, ' module because you entered an incorrect Feature Number.']);
end

%textVAR12 = For INTENSITY, AREAOCCUPIED or TEXTURE features, which image would you like to process?
%infotypeVAR12 = imagegroup
%inputtypeVAR12 = popupmenu custom
ImageName{2} = char(handles.Settings.VariableValues{CurrentModuleNum,12});

%textVAR13 = For TEXTURE, RADIAL DISTRIBUTION, OR NEIGHBORS features, what previously measured size scale (TEXTURE OR NEIGHBORS) or previously used number of bins (RADIALDISTRIBUTION) do you want to use?
%defaultVAR13 = 1
%inputtypeVAR13 = popupmenu scale
SizeScale{2} = char(handles.Settings.VariableValues{CurrentModuleNum,13});

%%%VariableRevisionNumber = 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

for idx = 1:2
    if idx == 2 && ~strcmp(PlotType,'Scatter 2'), break, end
    try
        FeatureName{idx} = CPgetfeaturenamesfromnumbers(handles, ObjectName{idx}, ...
            Category{idx}, FeatureNbr{idx}, ImageName{idx},SizeScale{idx});

    catch
        error([lasterr '  Image processing was canceled in the ', ModuleName, ...
            ' module (#' num2str(CurrentModuleNum) ...
            ') because an error ocurred when retrieving the data.  '...
            'Likely the category of measurement you chose, ',...
            Category{idx}, ', was not available for ', ...
            ObjectName{idx},' with feature number ' FeatureNbr{idx} ...
            ', possibly specific to image ''' ImageName{idx} ''' and/or ' ...
            'Texture Scale = ' num2str(SizeScale{idx}) '.']);
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
    CPplotmeasurement(handles,PlotType,FigHandle,1,ObjectName{1},FeatureName{1},ObjectName{2},FeatureName{2});
else
    CPplotmeasurement(handles,PlotType,FigHandle,1,char(ObjectName{1}),char(FeatureName{1}));
end

%%%%%%%%%%%%%%%
%%% DISPLAY %%%
%%%%%%%%%%%%%%%
drawnow
%% Handle the case where Cancel button was pushed.  This results in the
%% Renderer being set to 'None' and will throw an error in CPimcapture
%% if not handled here.
if ~strcmp(get(FigHandle,'renderer'),'None')
    FigureShot = CPimcapture(FigHandle); %% using defaults of whole figure and 150 dpi
    handles.Pipeline.(PlotImage)=FigureShot;
end