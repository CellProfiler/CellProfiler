function GenerateHistogramMovie(handles)

% Help for the Generate Histogram Movie tool:
% Category: Data Tools
%
% SHORT DESCRIPTION:
% Creates a movie of the histogram of any measurement. This will be done
% after specifying which output file the measurements exist in and where to
% write the resulting .avi file.
% *************************************************************************
% Note: this tool is beta-version and has not been thoroughly checked.

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

% Ask the user to choose the file from which to extract
% measurements. The window opens in the default output directory.
[RawFileName, RawPathname] = CPuigetfile('*.mat', 'Select the raw measurements file',handles.Current.DefaultOutputDirectory);
% Allows canceling.
if RawFileName == 0
    return
end
load(fullfile(RawPathname, RawFileName));

% Try to convert old measurements
handles = CP_convert_old_measurements(handles);

% Call the function CPgetfeature, which opens a series of list dialogs
% and lets the user choose a feature. The feature can be identified via
% 'ObjectTypename', 'FeatureType' and 'FeatureNo'.
try
    [ObjectTypename,FeatureType] = CPgetfeature(handles);
catch
    ErrorMessage = lasterr;
    CPerrordlg(['An error occurred in the GenerateHistogramMovie Data Tool. ' ErrorMessage(30:end)]);
    return
end
if isempty(ObjectTypename),return,end

% Extract the measurement and calculate mean and standard deviation
tmp = handles.Measurements.(ObjectTypename).(FeatureType);
MeasurementsMean = zeros(length(tmp),1);
MeasurementsStd = zeros(length(tmp),1);
for k = 1:length(tmp)
    if ~isempty(tmp{k})
        MeasurementsMean(k) = mean(tmp{k});
        MeasurementsStd(k)  = std(tmp{k});
    end
end

% Do the plotting
titlestr = [FeatureType,' of ', ObjectTypename];

% Plots a line chart, where the X dimensions are incremented from 1 to the 
% number of measurements to be displayed, and Y is the measurement of
% interest.
for l = 1:length(MeasurementsMean)
    if l == 1,
        FigureHandle = CPfigure('Position',[1 500 792 813],'visible','off');
        
        % Plots the line chart and the standard deviations as lines, too
        plot(1:1:length(MeasurementsMean), MeasurementsMean,'Color',[0 0 0],'LineWidth',1);
        AxisHandle = findobj(FigureHandle,'type','axes');
        hold(AxisHandle,'on');
        plot(1:1:length(MeasurementsMean), MeasurementsMean-MeasurementsStd,'Color',[0.7 0.7 0.7]);
        plot(1:1:length(MeasurementsMean), MeasurementsMean+MeasurementsStd,'Color',[0.7 0.7 0.7]);
        h = plot(l,MeasurementsMean(l),'rV');
        hold(AxisHandle,'off');
    
        FontSize = 10;
        set(AxisHandle,'fontname','Helvetica','fontsize',FontSize)
        xlabel(AxisHandle,'Image number','Fontname','Helvetica','fontsize',FontSize+2)
        ylabel(AxisHandle,'Mean +/- standard deviation','fontname','Helvetica','fontsize',FontSize+2)
        title(titlestr,'Fontname','Helvetica','fontsize',FontSize+2)

        set(FigureHandle,'Color','w')
        
        % Create the figure
        [filename,pathname] = CPuiputfile('*.avi', 'Save Movie As...',handles.Current.DefaultOutputDirectory);
        Xmo = avifile(fullfile(pathname,filename));
    else
        set(h,'xdata',l,'ydata',MeasurementsMean(l));
    end

    Xmo = addframe(Xmo,FigureHandle);
end

Xmo = close(Xmo);