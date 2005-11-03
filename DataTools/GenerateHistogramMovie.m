function GenerateHistogramMovie(handles)

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
% $Revision: 1011 $

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

titlestr = [handles.Measurements.(ObjectTypename).([FeatureType,'Features']){FeatureNo},' of ', ObjectTypename];

%%% Plots a line chart, where the X dimensions are incremented
%%% from 1 to the number of measurements to be displayed, and Y is
%%% the measurement of interest.

for l = 1:length(MeasurementMean)

    h = figure('Position',[1 500 792 813],'visible','off');

    %subplot('position',[0.1 0.55 .8 0.18])
    hold on
    plot(1:1:length(MeasurementsMean), MeasurementsMean,'Color',[0 0 0],'LineWidth',1);
    %%% Plots the Standard deviations as lines, too.
    plot(1:1:length(MeasurementsMean), MeasurementsMean-MeasurementsStd,'Color',[0.7 0.7 0.7]);
    plot(1:1:length(MeasurementsMean), MeasurementsMean+MeasurementsStd,'Color',[0.7 0.7 0.7]);
    plot(l,MeasurementsMean(l),'rV');
    hold off


    set(gca,'xtick',[0:100:length(MeasurementsMean)])
    FontSize = 10;
    set(gca,'fontname','times','fontsize',FontSize)
    xlabel(gca,'Image number','Fontname','times','fontsize',FontSize+2)
    ylabel(gca,'Mean +/- standard deviation','fontname','times','fontsize',FontSize+2)
    title(titlestr,'Fontname','times','fontsize',FontSize+2)
    %axis([0 length(MeasurementsMean)+1 0 600])

    set(gcf,'Color','w')

    if l==1
        [filename,pathname] = uiputfile('*.avi');
        Xmo=avifile(fullfile(pathname,filename));
    end
    Xmo=addframe(Xmo,h);
    l
    close
end

Xmo=close(Xmo);