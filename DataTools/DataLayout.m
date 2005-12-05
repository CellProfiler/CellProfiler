function DataLayout(handles)

% Help for the Data Layout tool:
% Category: Data Tools
%
% This module produces an image of mean values
% of an feature.
%
% NORMALIZATION:
% If there is some sort of artifact that causes a systematic shift in
% your measurements depending on where the sample was physically
% located in the layout (e.g. certain parts of the slide were more
% brightly illuminated), you can calculate normalization factors to
% correct your data using this button.  You will be prompted first to
% copy the measurements to the clipboard.  This should be a single
% column of data from a set of images. Then, use Matlab's import
% wizard to import your data and press 'Finish' before you click the
% OK button in the tiny prompt window. You will then be asked about
% the physical layout of the samples (rows and columns). The
% normalization assumes that most of the samples should in reality
% equal values, with only a few being real hits. Therefore, you can
% enter the percentile (range = 0 to 1) below and above which values
% will be excluded from the normalization correction.  This should
% basically exclude the percentage of samples that are likely to be
% real hits.
%
% The results of the calculation are displayed in an output window.
% The imported data is shown on the left: the range of colors
% indicates the range of values, as indicated by the bar immediately
% to the right of the plot.  The central plot shows the ignored
% samples. Usually the ignored samples are shown as blue (these are
% the samples that were in the top or bottom percentile that you
% specified, and were not used to fit the smooth correction function).
% The plot on the right, with its corresponding color bar, shows the
% smoothened, calculated correction factors.  You can change the order
% of the polynomial which is used to fit to the data by opening the
% main CellProfiler program file (CellProfiler.m) and adjusting the
% calculation towards the end of the subfunction called
% 'NormalizationButton'. (Save a backup elsewhere first, and do not
% change the name of the main CellProfiler program file or problems
% will ensue.)
%
% The correction factors (or, normalization factors) are placed onto
% the clipboard for you to paste next to the original data (e.g. in
% Excel). You can then divide the input values by these correction
% factors to yield the normalized data. After clicking OK, your
% original data is then placed on the clipboard so that, if desired,
% you can paste it next to the original data (e.g. in Excel) just to
% doublecheck that the proper numbers were used. Both sets of data are
% also displayed in the Matlab command window to copy and paste.
%
% See also <nothing relevant>.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehexad Institute for Biomedical Research.
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
[ObjectTypename,FeatureType,FeatureNo] = CPgetfeature(handles);

% Get the measurements cell array
CellArray = handles.Measurements.(ObjectTypename).(FeatureType);

% Extract the selected feature and calculate the mean
Measurements = zeros(length(CellArray),1);
for k = 1:length(CellArray)
    Measurements(k) = mean(CellArray{k}(:,FeatureNo));
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
title(TitleString,'fontsize',handles.Preferences.FontSize), 
colorbar