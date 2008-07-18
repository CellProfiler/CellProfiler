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
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% $Revision$

[FileName, Pathname] = CPuigetfile('*.mat', 'Select the raw measurements file',handles.Current.DefaultOutputDirectory);
if FileName == 0
    return
end
fn = fullfile(Pathname, FileName);
try
    temp = load(fn);
    handles = CP_convert_old_measurements(temp.handles);
catch
    CPerrordlg('Selected file is not a CellProfiler or MATLAB file (it does not have the extension .mat).')
    return
end

PlotType = listdlg('Name','Choose the plot type','SelectionMode','single','ListSize',[200 200],...
    'ListString',{'Bar chart','Line chart','Scatter plot, 1 measurement','Scatter plot, 2 measurements'});
if isempty(PlotType)
    return
end

if PlotType == 4
    %%% Get the feature types
    msg=CPmsgbox('In the following dialog, please choose measurements for the Y axis.');
    uiwait(msg);
    [Object2,Feature2] = CPgetfeature(handles, 1);
        
    msg=CPmsgbox('In the following dialog, please choose measurements for the X axis.');
    uiwait(msg);
    [Object,Feature] = CPgetfeature(handles, 1);
    if isempty(Object) || isempty(Object2)
	return
    end
    CPplotmeasurement(handles,PlotType,[],0,Object,Feature,Object2,Feature2)
else
    %%% Get the feature type
    [Object,Feature] = CPgetfeature(handles, 1);
    if isempty(Object)
	return
    end
    CPplotmeasurement(handles,PlotType,[],0,Object,Feature)
end