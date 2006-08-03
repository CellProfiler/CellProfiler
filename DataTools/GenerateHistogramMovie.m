function GenerateHistogramMovie(handles)

% Help for the Generate Histogram Movie tool:
% Category: Data Tools
%
% SHORT DESCRIPTION:
% Creates a movie of the histogram of any measurement.
% *************************************************************************
% Note: this tool is beta-version and has not been thoroughly checked.

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
%   Peter Swire
%   Rodrigo Ipince
%   Vicky Lay
%   Jun Liu
%   Chris Gang
%
% Website: http://www.cellprofiler.org
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

%%% Call the function CPgetfeature, which opens a series of list dialogs
%%% and lets the user choose a feature. The feature can be identified via
%%% 'ObjectTypename', 'FeatureType' and 'FeatureNo'.
try
    [ObjectTypename,FeatureType,FeatureNo] = CPgetfeature(handles);
catch
    ErrorMessage = lasterr;
    CPerrordlg(['An error occurred in the GenerateHistogramMovie Data Tool. ' ErrorMessage(30:end)]);
    return
end
if isempty(ObjectTypename),return,end

%%% Extract the measurement and calculate mean and standard deviation
tmp = handles.Measurements.(ObjectTypename).(FeatureType);
MeasurementsMean = zeros(length(tmp),1);
MeasurementsStd = zeros(length(tmp),1);
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

for l = 1:length(MeasurementsMean)

    FigureHandle = CPfigure('Position',[1 500 792 813],'visible','off');

    %subplot('position',[0.1 0.55 .8 0.18])
    hold on
    plot(1:1:length(MeasurementsMean), MeasurementsMean,'Color',[0 0 0],'LineWidth',1);
    %%% Plots the Standard deviations as lines, too.
    plot(1:1:length(MeasurementsMean), MeasurementsMean-MeasurementsStd,'Color',[0.7 0.7 0.7]);
    plot(1:1:length(MeasurementsMean), MeasurementsMean+MeasurementsStd,'Color',[0.7 0.7 0.7]);
    plot(l,MeasurementsMean(l),'rV');
    hold off


    set(gca,'XTick',0:100:length(MeasurementsMean))
    FontSize = 10;
    set(gca,'fontname','Helvetica','fontsize',FontSize)
    xlabel(gca,'Image number','Fontname','Helvetica','fontsize',FontSize+2)
    ylabel(gca,'Mean +/- standard deviation','fontname','Helvetica','fontsize',FontSize+2)
    title(titlestr,'Fontname','Helvetica','fontsize',FontSize+2)
    %axis([0 length(MeasurementsMean)+1 0 600])

    set(FigureHandle,'Color','w')

    if l==1
        [filename,pathname] = uiputfile('*.avi');
        Xmo=avifile(fullfile(pathname,filename));
    end
    Xmo=addframe(Xmo,FigureHandle);
    l %#ok Ignore MLint We want to see which frame we are at.
    close
end

close(Xmo);