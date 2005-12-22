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
% allow plotting one measuerement versus another. As prompted, select a
% CellProfiler output file containing the measurements, choose the
% measurement parameter to be displayed, and choose the display type.

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
%   Susan Ma
%   Wyman Li
%
% Website: http://www.cellprofiler.org
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

FigHandle = CPfigure;

if PlotType == 4
    %%% Get the feature type 1
    [Object,Feature,FeatureNo] = CPgetfeature(handles);
    if isempty(Object),return,end
    %%% Get the feature type 2
    [Object2,Feature2,FeatureNo2] = CPgetfeature(handles);
    if isempty(Object2)
        return
    end
    CPplotmeasurement(handles,FigHandle,PlotType,0,Object,Feature,FeatureNo,Object2,Feature2,FeatureNo2)
else
    %%% Get the feature type
    [Object,Feature,FeatureNo] = CPgetfeature(handles);
    if isempty(Object)
        return
    end
    CPplotmeasurement(handles,FigHandle,PlotType,0,Object,Feature,FeatureNo)
end
set(FigHandle,'Numbertitle','off','name',['Plot Measurement: ',get(get(gca,'title'),'string')])