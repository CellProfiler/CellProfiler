function handles = PlotMeasurement(handles)

% Help for the Plot Measurements tool:
% Category: Data Tools
%
% This tool makes a bar plot of the mean and standard deviation of a measurement.
% As prompted, select a CellProfiler output file containing the measurements,
% then choose the measurement parameter to be displayed.
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

%%% Call the function blabla(), which opens a series of list dialogs and
%%% lets the user choose a feature. The feature can be identified via 'ObjectTypename',
%%% 'FeatureType' and 'FeatureNo'.
[ObjectTypename,FeatureType,FeatureNo] = CPgetfeature(handles);
if isempty(ObjectTypename),return,end

%%% Allows the user to choose how to display the standard deviation.
Display = CPquestdlg('Choose the display type','Display options','Bar chart','Line chart','Cancel','Line chart');
if strcmp(Display,'Cancel') == 1, return, end

%%% Extract the measurement and calculate mean and standard deviation
tmp = handles.Measurements.(ObjectTypename).(FeatureType);
MeasurementMean = zeros(length(tmp),1);
MeasurementStd = zeros(length(tmp),1);
for k = 1:length(tmp)
    if ~isempty(tmp{k})
        MeasurementsMean(k) = mean(tmp{k}(:,FeatureNo));
        MeasurementsStd(k)  = std(tmp{k}(:,FeatureNo));
    end
end

%%% Do the plotting
fig = figure;
titlestr = [handles.Measurements.(ObjectTypename).([FeatureType,'Features']){FeatureNo},' of ', ObjectTypename];
set(fig,'Numbertitle','off','name',['Plot Measurement: ',titlestr])
set(gcf,'Color',[1 1 1])
hold on

if strcmp(Display,'Bar chart') == 1
    bar(MeasurementsMean);
    colormap([.7 .7 .7])
    shading flat
    for k = 1:length(MeasurementsMean)
        plot([k k],[MeasurementsMean(k)-MeasurementsStd(k),MeasurementsMean(k)+MeasurementsStd(k)],'k','linewidth',1)
        plot([k-0.075,k+0.075],[MeasurementsMean(k)-MeasurementsStd(k),MeasurementsMean(k)-MeasurementsStd(k)],'k','linewidth',1)
        plot([k-0.075,k+0.075],[MeasurementsMean(k)+MeasurementsStd(k),MeasurementsMean(k)+MeasurementsStd(k)],'k','linewidth',1)
    end
    hold off
elseif strcmp(Display,'Line chart') == 1
    %%% Plots a line chart, where the X dimensions are incremented
    %%% from 1 to the number of measurements to be displayed, and Y is
    %%% the measurement of interest.
    hold on
    plot(1:1:length(MeasurementsMean), MeasurementsMean,'Color',[0 0 0],'LineWidth',1);
    %%% Plots the Standard deviations as lines, too.
    plot(1:1:length(MeasurementsMean), MeasurementsMean-MeasurementsStd,'Color',[0.7 0.7 0.7]);
    plot(1:1:length(MeasurementsMean), MeasurementsMean+MeasurementsStd,'Color',[0.7 0.7 0.7]);
    hold off
end

set(gca,'xtick',[0:100:length(MeasurementsMean)])
FontSize = get(0,'UserData');
set(gca,'fontname','times','fontsize',FontSize)
xlabel(gca,'Image number','Fontname','times','fontsize',FontSize+2)
ylabel(gca,'Mean +/- standard deviation','fontname','times','fontsize',FontSize+2)
title(titlestr,'Fontname','times','fontsize',FontSize+2)
axis([0 length(MeasurementsMean)+1 ylim])


