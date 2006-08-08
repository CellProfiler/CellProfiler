function DataLayout(handles)

% Help for the Data Layout tool:
% Category: Data Tools
%
% SHORT DESCRIPTION:
% Shows mean measurements for each image in a specified spatial layout.
% *************************************************************************
% Note: this tool is beta-version and has not been thoroughly checked.
%
% When images are collected in a particular spatial layout, it is
% sometimes useful to view measurements collected from the images in the
% same spatial layout to look for patterns (e.g. edge effects). The mean
% measurement for each image is shown in the plot that is produced.

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

%%% Ask the user to choose the file from which to extract measurements.
if exist(handles.Current.DefaultOutputDirectory, 'dir')
    [RawFileName, RawPathname] = uigetfile(fullfile(handles.Current.DefaultOutputDirectory,'.','*.mat'),'Select the raw measurements file');
else
    [RawFileName, RawPathname] = uigetfile('*.mat','Select the raw measurements file');
end
if RawFileName == 0
    return
end
load(fullfile(RawPathname, RawFileName));

% Ask the user for the feature
try
    [ObjectTypename,FeatureType,FeatureNo] = CPgetfeature(handles);
catch
    ErrorMessage = lasterr;
    CPerrordlg(['An error occurred in the DataLayout Data Tool. ' ErrorMessage(30:end)]);
    return
end
if isempty(ObjectTypename),return,end

% Get the measurements cell array
CellArray = handles.Measurements.(ObjectTypename).(FeatureType);

% Extract the selected feature and calculate the mean
Measurements = zeros(length(CellArray),1);
try
    for k = 1:length(CellArray)
        Measurements(k) = mean(CellArray{k}(:,FeatureNo));
    end
catch
     CPerrordlg('use the data tool MergeOutputFiles or ConvertBatchFiles to convert the data first');
     return;
end

% Ask for the dimensions of the image
Prompts = {'Enter the number of rows','Enter the number of columns'};
Defaults = {'24','16'};
Answers = inputdlg(Prompts,'Describe Array/Slide Format',1,Defaults);
if isempty(Answers)
    return
end

% Pad or remove measurements to fit the entered image size
NumberRows = str2double(Answers{1});
NumberColumns = str2double(Answers{2});
TotalSamplesToBeGridded = NumberRows*NumberColumns;
NumberSamplesImported = length(Measurements);
if TotalSamplesToBeGridded > NumberSamplesImported
    h = CPwarndlg(['You have specified a layout of ', num2str(TotalSamplesToBeGridded), ' samples in the layout, but only ', num2str(NumberSamplesImported), ' measurements were imported. The remaining spaces in the layout will be filled in with the value of the last sample.']);
    waitfor(h)
    Measurements(NumberSamplesImported+1:TotalSamplesToBeGridded) = Measurements(NumberSamplesImported);
elseif TotalSamplesToBeGridded < NumberSamplesImported
    h = CPwarndlg(['You have specified a layout of ', num2str(TotalSamplesToBeGridded), ' samples in the layout, but ', num2str(NumberSamplesImported), ' measurements were imported. The imported measurements at the end will be ignored.']);
    waitfor(h)
    Measurements(TotalSamplesToBeGridded+1:NumberSamplesImported) = [];
end

% Produce the image
MeanImage = reshape(Measurements,NumberRows,NumberColumns);

%%% Shows the results.
TitleString = sprintf('Objects: %s, Feauture classification: %s, Feature: %s',ObjectTypename, FeatureType, handles.Measurements.(ObjectTypename).([FeatureType ,'Features']){FeatureNo});
CPfigure, 
CPimagesc(MeanImage,handles), 
title(TitleString), 
colorbar
