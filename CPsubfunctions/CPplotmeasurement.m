function CPplotmeasurement(handles,FigHandle,PlotType,ModuleFlag,Object,Feature,FeatureNo,Object2,Feature2,FeatureNo2)

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
%
% Website: http://www.cellprofiler.org
%
% $Revision: 2802 $

try FontSize = handles.Preferences.FontSize;
    %%% We used to store the font size in Current, so this line makes old
    %%% output files compatible. Shouldn't be necessary with any files made
    %%% after November 15th, 2006.
catch
    FontSize = handles.Current.FontSize;
end

if (length(Feature) > 10) & strncmp(Feature,'Intensity_',10)
    str = [handles.Measurements.(Object).([Feature,'Features']){FeatureNo},' in ',Feature(11:end)];
elseif (length(Feature) > 8) & strncmp(Feature,'Texture_',8)
    str = [handles.Measurements.(Object).([Feature,'Features']){FeatureNo},' in ',Feature(9:end)];
else
    str = handles.Measurements.(Object).([Feature,'Features']){FeatureNo};
end

% Bar chart
if PlotType == 1

    %%% Extract the measurement and calculate mean and standard deviation
    Measurements = handles.Measurements.(Object).(Feature);
    MeasurementMean = zeros(length(Measurements),1);
    MeasurementStd = zeros(length(Measurements),1);
    for k = 1:length(Measurements)
        if ~isempty(Measurements{k})
            MeasurementsMean(k) = mean(Measurements{k}(:,FeatureNo));
            MeasurementsStd(k)  = std(Measurements{k}(:,FeatureNo));
        end
    end

    CPfigure(FigHandle);
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

    xlabel(gca,'Image number','Fontname','Helvetica','fontsize',FontSize+2)
    ylabel(gca,[str,', mean +/- standard deviation'],'fontname','Helvetica','fontsize',FontSize+2)
    axis([0 length(Measurements)+1 ylim])
    set(gca,'xtick',1:length(Measurements))
    titlestr = [str,' of ', Object];

    %%% Line chart
elseif PlotType == 2

    %%% Extract the measurement and calculate mean and standard deviation
    Measurements = handles.Measurements.(Object).(Feature);
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
    CPfigure(FigHandle);
    hold on
    plot(1:length(MeasurementsMean), MeasurementsMean,'Color',[0 0 0],'LineWidth',1);

    %%% Plots the Standard deviations as lines, too.
    plot(1:length(MeasurementsMean), MeasurementsMean-MeasurementsStd,':','Color',[0.7 0.7 0.7]);
    plot(1:length(MeasurementsMean), MeasurementsMean+MeasurementsStd,':','Color',[0.7 0.7 0.7]);
    hold off
    axis([0 length(Measurements)+1 ylim])

    xlabel(gca,'Image number','Fontname','Helvetica','fontsize',FontSize+2)
    ylabel(gca,[str,', mean +/- standard deviation'],'fontname','Helvetica','fontsize',FontSize+2)
    set(gca,'xtick',1:length(Measurements))
    titlestr = [str,' of ', Object];

    %%% Scatter plot, 1 measurement
elseif PlotType == 3

    %%% Extract the measurements
    Measurements = handles.Measurements.(Object).(Feature);

    CPfigure(FigHandle);
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

    xlabel(gca,'Image number','Fontname','Helvetica','fontsize',FontSize+2)
    ylabel(gca,str,'fontname','Helvetica','fontsize',FontSize+2)
    set(gca,'xtick',1:length(Measurements))
    titlestr = [handles.Measurements.(Object).([Feature,'Features']){FeatureNo},' of ', Object];

    %%% Scatter plot, 2 measurements
elseif PlotType == 4

    %%% Extract the measurements
    Measurements1 = handles.Measurements.(Object).(Feature);
    Measurements2 = handles.Measurements.(Object2).(Feature2);

    if ModuleFlag == 0
        %%% Calculates some values for the next dialog box.
        TotalNumberImageSets = length(Measurements1);
        TextTotalNumberImageSets = num2str(TotalNumberImageSets);
        %%% Ask the user to specify scatter plot settings.
        Prompts{1} = 'Enter the first image set to use for the scatter plot';
        Prompts{2} = ['Enter the last last image set to use for scatter plot (the total number of image sets with data in the file is ',TextTotalNumberImageSets,').'];
        Defaults{1} = '1';
        Defaults{2} = TextTotalNumberImageSets;
        Answers = inputdlg(Prompts(1:2),'Choose scatter plot settings',1,Defaults(1:2),'on');
        FirstImage = str2num(Answers{1});
        LastImage = str2num(Answers{2});

        CPfigure(FigHandle);
        %%% Plot
        hold on
        for k = FirstImage:LastImage
            if size(Measurements1{k},1) ~= size(Measurements2{k})
                CPerrordlg('The number of objects for the chosen measurements does not match.')
                return
            end
            if ~isempty(Measurements1{k})
                plot(Measurements1{k}(:,FeatureNo),Measurements2{k}(:,FeatureNo2),'.k')
            end
        end
        hold off
    else
        hold on
        for k = 1:length(Measurements1)
            if size(Measurements1{k},1) ~= size(Measurements2{k})
                CPerrordlg('The number of objects for the chosen measurements does not match.')
                return
            end
            if ~isempty(Measurements1{k})
                plot(Measurements1{k}(:,FeatureNo),Measurements2{k}(:,FeatureNo2),'.k')
            end
        end
        hold off
    end

    if (length(Feature2) > 10) & strncmp(Feature2,'Intensity_',10)
        str2 = [handles.Measurements.(Object2).([Feature2,'Features']){FeatureNo2},' of ', Object2, ' in ',Feature2(11:end)];
    elseif (length(Feature2) > 8) & strncmp(Feature2,'Texture_',8)
        str2 = [handles.Measurements.(Object2).([Feature2,'Features']){FeatureNo2},' of ', Object2, ' in ',Feature2(9:end)];
    else
        str2 = [handles.Measurements.(Object2).([Feature2,'Features']){FeatureNo2},' of ', Object2];
    end

    xlabel(gca,str,'fontsize',FontSize+2,'fontname','Helvetica')
    ylabel(gca,str2,'fontsize',FontSize+2,'fontname','Helvetica')
    titlestr = [str,' vs. ',str2];
end

% Set some general figure and axes properties
set(gca,'fontname','Helvetica','fontsize',FontSize)
title(titlestr,'Fontname','Helvetica','fontsize',FontSize+2)