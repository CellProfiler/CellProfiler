% Help for the Plot Measurement tool:
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

[ObjectTypename2,FeatureType2,FeatureNo2] = CPgetfeature(handles);
if isempty(ObjectTypename2),return,end

[ObjectTypename3,FeatureType3,FeatureNo3] = CPgetfeature(handles);
if isempty(ObjectTypename3),return,end

%%% Allows the user to choose how to display the standard deviation.
Display = questdlg('Choose the display type','Display options','Bar chart','Line chart','Cancel','Line chart');
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

tmp2 = handles.Measurements.(ObjectTypename2).(FeatureType2);
MeasurementMean2 = zeros(length(tmp2),1);
MeasurementStd2 = zeros(length(tmp2),1);
for k = 1:length(tmp2)
    if ~isempty(tmp2{k})
        MeasurementsMean2(k) = mean(tmp2{k}(:,FeatureNo2));
        MeasurementsStd2(k)  = std(tmp2{k}(:,FeatureNo2));
    end
end

tmp3 = handles.Measurements.(ObjectTypename3).(FeatureType3);
MeasurementMean3 = zeros(length(tmp3),1);
MeasurementStd3 = zeros(length(tmp3),1);
for k = 1:length(tmp3)
    if ~isempty(tmp3{k})
        MeasurementsMean3(k) = mean(tmp3{k}(:,FeatureNo3));
        MeasurementsStd3(k)  = std(tmp3{k}(:,FeatureNo3));
    end
end

%%% Do the plotting

titlestr = [handles.Measurements.(ObjectTypename).([FeatureType,'Features']){FeatureNo},' of ', ObjectTypename];
titlestr2 = [handles.Measurements.(ObjectTypename2).([FeatureType2,'Features']){FeatureNo2},' of ', ObjectTypename2];
titlestr3 = [handles.Measurements.(ObjectTypename3).([FeatureType3,'Features']){FeatureNo3},' of ', ObjectTypename3];

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

    for l = 1:1467
        
        figure('Position',[1 500 792 813],'visible','off')
        
        subplot('position',[0 0.75 1 0.25])
        imshow(mov(l).cdata)
        
        subplot('position',[0.1 0.55 .8 0.18])
        hold on
        plot(1:1:length(MeasurementsMean), MeasurementsMean,'Color',[0 0 0],'LineWidth',1);
        %%% Plots the Standard deviations as lines, too.
        plot(1:1:length(MeasurementsMean), MeasurementsMean-MeasurementsStd,'Color',[0.7 0.7 0.7]);
        plot(1:1:length(MeasurementsMean), MeasurementsMean+MeasurementsStd,'Color',[0.7 0.7 0.7]);
        plot(l, MeasurementsMean(l),'rV');
        hold off
        

        set(gca,'xtick',[0:100:length(MeasurementsMean)])
        FontSize = 10;
        set(gca,'fontname','times','fontsize',FontSize)
        xlabel(gca,'Image number','Fontname','times','fontsize',FontSize+2)
        ylabel(gca,'Mean +/- standard deviation','fontname','times','fontsize',FontSize+2)
        title(titlestr,'Fontname','times','fontsize',FontSize+2)
        axis([0 length(MeasurementsMean)+1 0 600])
        
        subplot('position',[0.1 0.3 .8 0.18])
        hold on
        plot(1:1:length(MeasurementsMean2), MeasurementsMean2,'Color',[0 0 0],'LineWidth',1);
        %%% Plots the Standard deviations as lines, too.
        plot(1:1:length(MeasurementsMean2), MeasurementsMean2-MeasurementsStd2,'Color',[0.7 0.7 0.7]);
        plot(1:1:length(MeasurementsMean2), MeasurementsMean2+MeasurementsStd2,'Color',[0.7 0.7 0.7]);
        plot(l, MeasurementsMean2(l),'rV');
        hold off
        
        
        set(gca,'xtick',[0:100:length(MeasurementsMean2)])
        FontSize = 10;
        set(gca,'fontname','times','fontsize',FontSize)
        xlabel(gca,'Image number','Fontname','times','fontsize',FontSize+2)
        ylabel(gca,'Mean +/- standard deviation','fontname','times','fontsize',FontSize+2)
        title(titlestr2,'Fontname','times','fontsize',FontSize+2)
        axis([0 length(MeasurementsMean2)+1 0 3500])
        
        
        subplot('position',[.1 .05 .8 0.18])
        hold on
        plot(1:1:length(MeasurementsMean3), MeasurementsMean3,'Color',[0 0 0],'LineWidth',1);
        %%% Plots the Standard deviations as lines, too.
        plot(1:1:length(MeasurementsMean3), MeasurementsMean3-MeasurementsStd3,'Color',[0.7 0.7 0.7]);
        plot(1:1:length(MeasurementsMean3), MeasurementsMean3+MeasurementsStd3,'Color',[0.7 0.7 0.7]);
        plot(l, MeasurementsMean3(l),'rV');
        hold off

        set(gca,'xtick',[0:100:length(MeasurementsMean3)])
        FontSize = 10;
        set(gca,'fontname','times','fontsize',FontSize)
        xlabel(gca,'Image number','Fontname','times','fontsize',FontSize+2)
        ylabel(gca,'Mean +/- standard deviation','fontname','times','fontsize',FontSize+2)
        title(titlestr3,'Fontname','times','fontsize',FontSize+2)
        axis([0 length(MeasurementsMean3)+1 0 1.5])
        
        set(gcf,'Color','w')

        if l==1
            Xmo=avifile('CPMovie.avi');
        end
        Xmo=addframe(Xmo,gcf);
        l
        close
    end
end

Xmo=close(Xmo);