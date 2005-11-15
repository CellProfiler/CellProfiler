function handles = GetPlotMeasurement(handles)
% Help for the Get Histogram module:
% Category: Other
%
%
% Feature Number:
% The feature number is the parameter from the chosen module (AreaShape,
% Intensity, Texture) which will be used for the plotted measurement. The
% following tables provide the feature numbers for each measurement made by
% the three modules:
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
% $Revision: 2606 $

%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.

CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);
ModuleName = char(handles.Settings.ModuleNames(CurrentModuleNum));

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
    error('You entered an incorrect Feature Number.');
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

%%% Determines which image set is being analyzed.
SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;
NumberOfImageSets = handles.Current.NumberOfImageSets;

if strcmp(FeatureType,'Intensity') || strcmp(FeatureType,'Texture')
    FeatureType = [FeatureType, '_',Image];
end

if strcmp(FeatureType2,'Intensity') || strcmp(FeatureType2,'Texture')
    FeatureType2 = [FeatureType2, '_',Image];
end

%%%%%%%%%%%%%%%%%%%%%
%%% DATA ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow

FontSize = handles.Preferences.FontSize;

fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);

if strcmp(PlotType,'Bar')

    %%% Extract the measurement and calculate mean and standard deviation
    Measurements = handles.Measurements.(ObjectName).(FeatureType);
    MeasurementMean = zeros(length(Measurements),1);
    MeasurementStd = zeros(length(Measurements),1);
    for k = 1:length(Measurements)
        if ~isempty(Measurements{k})
            MeasurementsMean(k) = mean(Measurements{k}(:,FeatureNo));
            MeasurementsStd(k)  = std(Measurements{k}(:,FeatureNo));
        end
    end

    HistHandle = CPfigure(handles,ThisModuleFigureNumber);
    %%% Do the plotting
    bar(MeasurementsMean);
    hold on
    colormap([.7 .7 .7])
    shading flat
    for k = 1:length(MeasurementsMean)
        plot([k k],[MeasurementsMean(k)-MeasurementsStd(k),MeasurementsMean(k)+MeasurementsStd(k)],'k','linewidth',1)
        plot([k-0.075,k+0.075],[MeasurementsMean(k)-MeasurementsStd(k),MeasurementsMean(k)-MeasurementsStd(k)],'k','linewidth',1)
        plot([k-0.075,k+0.075],[MeasurementsMean(k)+MeasurementsStd(k),MeasurementsMean(k)+MeasurementsStd(k)],'k','linewidth',1)
    end
    hold off
    str = handles.Measurements.(ObjectName).([FeatureType,'Features']){FeatureNo};
    xlabel(gca,'Image number','Fontname','times','fontsize',FontSize+2)
    ylabel(gca,[str,', mean +/- standard deviation'],'fontname','times','fontsize',FontSize+2)
    axis([0 length(Measurements)+1 ylim])
    set(gca,'xtick',1:length(Measurements))
    titlestr = [str,' of ', ObjectName];

    %%% Line chart
elseif strcmp(PlotType,'Line')

    %%% Extract the measurement and calculate mean and standard deviation
    Measurements = handles.Measurements.(ObjectName).(FeatureType);
    MeasurementMean = zeros(length(Measurements),1);
    MeasurementStd = zeros(length(Measurements),1);
    for k = 1:length(Measurements)
        if ~isempty(Measurements{k})
            MeasurementsMean(k) = mean(Measurements{k}(:,FeatureNo));
            MeasurementsStd(k)  = std(Measurements{k}(:,FeatureNo));
        end
    end

    %%% Plots a line chart, where the X dimensions are incremented
    %%% from 1 to the number of measurements to be PlotTypeed, and Y is
    %%% the measurement of interest.
    HistHandle = CPfigure(handles,ThisModuleFigureNumber);
    hold on
    plot(1:length(MeasurementsMean), MeasurementsMean,'Color',[0 0 0],'LineWidth',1);

    %%% Plots the Standard deviations as lines, too.
    plot(1:length(MeasurementsMean), MeasurementsMean-MeasurementsStd,':','Color',[0.7 0.7 0.7]);
    plot(1:length(MeasurementsMean), MeasurementsMean+MeasurementsStd,':','Color',[0.7 0.7 0.7]);
    hold off
    axis([0 length(Measurements)+1 ylim])
    str = handles.Measurements.(ObjectName).([FeatureType,'Features']){FeatureNo};
    xlabel(gca,'Image number','Fontname','times','fontsize',FontSize+2)
    ylabel(gca,[str,', mean +/- standard deviation'],'fontname','times','fontsize',FontSize+2)
    set(gca,'xtick',1:length(Measurements))
    titlestr = [str,' of ', ObjectName];

    %%% Scatter plot, 1 measurement
elseif strcmp(PlotType,'Scatter 1')

    %%% Extract the measurements
    Measurements = handles.Measurements.(ObjectName).(FeatureType);

    HistHandle = CPfigure(handles,ThisModuleFigureNumber);
    %%% Plot
    hold on
    for k = 1:length(Measurements)
        if ~isempty(Measurements{k})
            plot(k*ones(length(Measurements{k}(:,FeatureNo))),Measurements{k}(:,FeatureNo),'.k')
            plot(k,mean(Measurements{k}(:,FeatureNo)),'.r','Markersize',20)
        end
    end
    hold off
    axis([0 length(Measurements)+1 ylim])
    str = handles.Measurements.(ObjectName).([FeatureType,'Features']){FeatureNo};
    xlabel(gca,'Image number','Fontname','times','fontsize',FontSize+2)
    ylabel(gca,str,'fontname','times','fontsize',FontSize+2)
    set(gca,'xtick',1:length(Measurements))
    titlestr = [handles.Measurements.(ObjectName).([FeatureType,'Features']){FeatureNo},' of ', ObjectName];

    %%% Scatter plot, 2 measurements
elseif strcmp(PlotType,'Scatter 2')

    %%% Extract the measurements
    Measurements1 = handles.Measurements.(ObjectName).(FeatureType);
    Measurements2 = handles.Measurements.(ObjectName2).(FeatureType2);

    HistHandle = CPfigure(handles,ThisModuleFigureNumber);
    %%% Plot
    hold on
    for k = 1:length(Measurements1)
        if size(Measurements1{k},1) ~= size(Measurements2{k})
            errordlg('The number object for the chosen measurements does not match.')
            return
        end
        if ~isempty(Measurements1{k})
            plot(Measurements1{k}(:,FeatureNo),Measurements2{k}(:,FeatureNo2),'.k')
        end
    end
    hold off
    str1 = [handles.Measurements.(ObjectName).([FeatureType,'Features']){FeatureNo},' of ', ObjectName];
    str2 = [handles.Measurements.(ObjectName2).([FeatureType2,'Features']){FeatureNo2},' of ', ObjectName2];
    xlabel(gca,str1,'fontsize',FontSize+2,'fontname','times')
    ylabel(gca,str2,'fontsize',FontSize+2,'fontname','times')
    titlestr = [str1,' vs. ',str2];
end

%%%%%%%%%%%%%%%
%%% DISPLAY %%%
%%%%%%%%%%%%%%%
drawnow

OneFrame = getframe(HistHandle);
handles.Pipeline.(PlotImage)=OneFrame.cdata;