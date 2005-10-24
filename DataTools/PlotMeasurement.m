function PlotMeasurement(handles)

% Help for the Plot Measurement tool:
% Category: Data Tools
%
% This tool allows the user to plot either a bar chart or a line chart of
% the mean and standard deviation of a measurement.
% As prompted, select a CellProfiler output file containing the measurements,
% choose the measurement parameter to be displayed, and choose the display
% type.
%
% See also HISTOGRAM.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne Carpenter <carpenter@wi.mit.edu>
%   Thouis Jones   <thouis@csail.mit.edu>
%   In Han Kang    <inthek@mit.edu>
%   Ola Friman     <friman@bwh.harvard.edu>
%   Steve Lowe     <stevelowe@alum.mit.edu>
%   Joo Han Chang  <joohan.chang@gmail.com>
%   Colin Clarke   <colinc@mit.edu>
%   Mike Lamprecht <mrl@wi.mit.edu>
%   Susan Ma       <xuefang_ma@wi.mit.edu>
%
% $Revision$

%%% Ask the user to choose the file from which to extract
%%% measurements. The window opens in the default output directory.
[RawFileName, RawPathname] = uigetfile(fullfile(handles.Current.DefaultOutputDirectory,'.','*.mat'),'Select the raw measurements file');
%%% Allows canceling.
if RawFileName == 0
    return
end
load(fullfile(RawPathname, RawFileName));

PlotType = listdlg('Name','Choose the plot type','SelectionMode','single','ListSize',[200 200],... 
    'ListString',{'Bar chart','Line chart','Scatter plot, 1 measurement','Scatter plot, 2 measurements'});
if isempty(PlotType), return,end


% Open figure
fig = CPfigure;
set(gcf,'Color',[1 1 1])
FontSize = handles.Current.FontSize;


% Bar chart
if PlotType == 1

    %%% Get the feature type
    [ObjectTypename,FeatureType,FeatureNo] = CPgetfeature(handles);
    if isempty(ObjectTypename),return,end

    %%% Extract the measurement and calculate mean and standard deviation
    Measurements = handles.Measurements.(ObjectTypename).(FeatureType);
    MeasurementMean = zeros(length(Measurements),1);
    MeasurementStd = zeros(length(Measurements),1);
    for k = 1:length(Measurements)
        if ~isempty(Measurements{k})
            MeasurementsMean(k) = mean(Measurements{k}(:,FeatureNo));
            MeasurementsStd(k)  = std(Measurements{k}(:,FeatureNo));
        end
    end

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
    str = handles.Measurements.(ObjectTypename).([FeatureType,'Features']){FeatureNo};
    xlabel(gca,'Image number','Fontname','times','fontsize',FontSize+2)
    ylabel(gca,[str,', mean +/- standard deviation'],'fontname','times','fontsize',FontSize+2)
    axis([0 length(Measurements)+1 ylim])
    set(gca,'xtick',1:length(Measurements))
    titlestr = [str,' of ', ObjectTypename];



%%% Line chart
elseif PlotType == 2
    
    %%% Get the feature type
    [ObjectTypename,FeatureType,FeatureNo] = CPgetfeature(handles);
    if isempty(ObjectTypename),return,end

    %%% Extract the measurement and calculate mean and standard deviation
    Measurements = handles.Measurements.(ObjectTypename).(FeatureType);
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
    hold on
    plot(1:length(MeasurementsMean), MeasurementsMean,'Color',[0 0 0],'LineWidth',1);

    %%% Plots the Standard deviations as lines, too.
    plot(1:length(MeasurementsMean), MeasurementsMean-MeasurementsStd,':','Color',[0.7 0.7 0.7]);
    plot(1:length(MeasurementsMean), MeasurementsMean+MeasurementsStd,':','Color',[0.7 0.7 0.7]);
    hold off
    axis([0 length(Measurements)+1 ylim])
    str = handles.Measurements.(ObjectTypename).([FeatureType,'Features']){FeatureNo};
    xlabel(gca,'Image number','Fontname','times','fontsize',FontSize+2)
    ylabel(gca,[str,', mean +/- standard deviation'],'fontname','times','fontsize',FontSize+2)
    set(gca,'xtick',1:length(Measurements))
    titlestr = [str,' of ', ObjectTypename];



%%% Scatter plot, 1 measurement
elseif PlotType == 3

    %%% Get the feature type
    [ObjectTypename,FeatureType,FeatureNo] = CPgetfeature(handles);
    if isempty(ObjectTypename),return,end

    %%% Extract the measurements 
    Measurements = handles.Measurements.(ObjectTypename).(FeatureType);

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
    str = handles.Measurements.(ObjectTypename).([FeatureType,'Features']){FeatureNo};
    xlabel(gca,'Image number','Fontname','times','fontsize',FontSize+2)
    ylabel(gca,str,'fontname','times','fontsize',FontSize+2)
    set(gca,'xtick',1:length(Measurements))
    titlestr = [handles.Measurements.(ObjectTypename).([FeatureType,'Features']){FeatureNo},' of ', ObjectTypename];



%%% Scatter plot, 2 measurements
elseif PlotType == 4
    
    %%% Get the feature type 1
    [ObjectTypename1,FeatureType1,FeatureNo1] = CPgetfeature(handles);
    if isempty(ObjectTypename1),return,end
    
    %%% Get the feature type 2
    [ObjectTypename2,FeatureType2,FeatureNo2] = CPgetfeature(handles);
    if isempty(ObjectTypename2),return,end
    
    %%% Extract the measurements
    Measurements1 = handles.Measurements.(ObjectTypename1).(FeatureType1);
    Measurements2 = handles.Measurements.(ObjectTypename2).(FeatureType2);

    %%% Plot
    hold on
    for k = 1:length(Measurements1)
        if size(Measurements1{k},1) ~= size(Measurements2{k})
            errordlg('The number object for the chosen measurements does not match.')
            return
        end
        if ~isempty(Measurements1{k})
            plot(Measurements1{k}(:,FeatureNo1),Measurements2{k}(:,FeatureNo2),'.k')
        end
    end
    hold off
    str1 = [handles.Measurements.(ObjectTypename1).([FeatureType1,'Features']){FeatureNo1},' of ', ObjectTypename1];
    str2 = [handles.Measurements.(ObjectTypename2).([FeatureType2,'Features']){FeatureNo2},' of ', ObjectTypename2];
    xlabel(gca,str1,'fontsize',FontSize+2,'fontname','times')
    ylabel(gca,str2,'fontsize',FontSize+2,'fontname','times')
    titlestr = [str1,' vs. ',str2];
end

% Set some general figure and axes properties
set(gca,'fontname','times','fontsize',FontSize)
title(titlestr,'Fontname','times','fontsize',FontSize+2)
set(fig,'Numbertitle','off','name',['Plot Measurement: ',titlestr])
