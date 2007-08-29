function PlotMeasurement(handles)

% Help for the Plot Measurement tool:
% Category: Data Tools
%
% SHORT DESCRIPTION:
% Plots measured data in bar charts, line charts, or scatterplots.
% *************************************************************************
% Note: this tool is beta-version and has not been thoroughly checked.
%
% Bar charts, line charts and one dimensional scatter plots show the mean
% and standard deviation of a measurement. Two dimensional scatter plots
% allow plotting one measurement against another. As prompted, select a
% CellProfiler output file containing the measurements, choose the
% measurement parameter to be displayed, and choose the display type.
%
% See also Histogram data tool.

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
% $Revision$

%%% Ask the user to choose the file from which to extract
%%% measurements. The window opens in the default output directory.
[FileName, Pathname] = CPuigetfile('*.mat', 'Select the raw measurements file',handles.Current.DefaultOutputDirectory);

%%% Allows canceling.
if FileName == 0
    return
end

%%% Load the specified CellProfiler output file
try
    load(fullfile(Pathname, FileName));
catch
    CPerrordlg('Selected file is not a CellProfiler or MATLAB file (it does not have the extension .mat).')
    return
end

PlotType = listdlg('Name','Choose the plot type','SelectionMode','single','ListSize',[200 200],...
    'ListString',{'Bar chart','Line chart','Scatter plot, 1 measurement','Scatter plot, 2 measurements'});
if isempty(PlotType), return,end

if PlotType == 4
    %%% Get the feature types
    try
        msg=CPmsgbox('In the following dialog, please choose measurements for the Y axis.');
        uiwait(msg);
        [Object2,Feature2,FeatureNo2] = CPgetfeature(handles);
        
        msg=CPmsgbox('In the following dialog, please choose measurements for the X axis.');
        uiwait(msg);
        [Object,Feature,FeatureNo] = CPgetfeature(handles);
        
    catch
        ErrorMessage = lasterr;
        CPerrordlg(['An error occurred in the PlotMeasurement Data Tool. ' ErrorMessage(30:end)]);
        return
    end
    if isempty(Object),return,end
    if isempty(Object2),return,end
    try
        CPplotmeasurement(handles,PlotType,[],0,Object,Feature,FeatureNo,Object2,Feature2,FeatureNo2)
    catch
        ErrorMessage = lasterr;
        CPerrordlg(['An error occurred in the PlotMeasurement Data Tool. ' ErrorMessage(35:end)]);
        return
    end
else
    %%% Get the feature type
    try
        [Object,Feature,FeatureNo] = CPgetfeature(handles);
    catch
        ErrorMessage = lasterr;
        CPerrordlg(['An error occurred in the PlotMeasurement Data Tool. ' ErrorMessage(30:end)]);
        return
    end
    if isempty(Object),return,end
    try
       CPplotmeasurement(handles,PlotType,[],0,Object,Feature,FeatureNo)
    catch
        ErrorMessage = lasterr;
        CPerrordlg(['An error occurred in the PlotMeasurement Data Tool. ' ErrorMessage(35:end)]);
        return
    end
end