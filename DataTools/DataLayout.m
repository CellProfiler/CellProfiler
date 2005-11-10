function DataLayout(handles)

% Help for the Data Layout tool:
% Category: Data Tools
%
% This module produces an image of mean values
% of an feature. 
%
% See also <nothing relevant>.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehexad Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne Carpenter
%   Thouis Jones
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
[ObjectTypename,FeatureType,FeatureNo] = CPgetfeature(handles,0);

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
figure, imagesc(MeanImage), title(TitleString,'fontsize',handles.Current.FontSize), colorbar






% % --- Executes on button press in DataLayoutButton.
% %%% THIS WAS A VERY SPECIALIZED VERSION OUR LAB USED TO NORMALIZE OUR
% %%% DATA>>>>
% function DataLayoutButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
% h = CPmsgbox('Copy your data to the clipboard then press OK');
% waitfor(h)
%
% uiimport('-pastespecial');
% h = CPmsgbox('After importing your data and pressing "Finish", click OK');
% waitfor(h)
% if exist('clipboarddata','var') == 0
%     return
% end
% IncomingData = clipboarddata;
%
% Prompts = {'Enter the number of rows','Enter the number of columns','Enter the percentile below which values will be excluded from fitting the normalization function.','Enter the percentile above which values will be excluded from fitting the normalization function.'};
% Defaults = {'24','16','.05','.95'};
% Answers = inputdlg(Prompts,'Describe Array/Slide Format',1,Defaults);
% if isempty(Answers)
%     return
% end
% NumberRows = str2double(Answers{1});
% NumberColumns = str2double(Answers{2});
% LowPercentile = str2double(Answers{3});
% HighPercentile = str2double(Answers{4});
% TotalSamplesToBeGridded = NumberRows*NumberColumns;
% NumberSamplesImported = length(IncomingData);
% if TotalSamplesToBeGridded > NumberSamplesImported
%     h = CPwarndlg(['You have specified a layout of ', num2str(TotalSamplesToBeGridded), ' samples in the layout, but only ', num2str(NumberSamplesImported), ' measurements were imported. The remaining spaces in the layout will be filled in with the value of the last sample.']);
%     waitfor(h)
%     IncomingData(NumberSamplesImported+1:TotalSamplesToBeGridded) = IncomingData(NumberSamplesImported);
% elseif TotalSamplesToBeGridded < NumberSamplesImported
%     h = CPwarndlg(['You have specified a layout of ', num2str(TotalSamplesToBeGridded), ' samples in the layout, but ', num2str(NumberSamplesImported), ' measurements were imported. The imported measurements at the end will be ignored.']);
%     waitfor(h)
%     IncomingData(TotalSamplesToBeGridded+1:NumberSamplesImported) = [];
% end
%
% %%% The data is shaped into the appropriate grid.
% MeanImage = reshape(IncomingData,NumberRows,NumberColumns);
%
% %%% The data are listed in ascending order.
% AscendingData = sort(IncomingData);
%
% %%% The percentiles are calculated. (Statistics Toolbox has a percentile
% %%% function, but many users may not have that function.)
% %%% The values to be ignored are set to zero in the mask.
% mask = MeanImage;
% if LowPercentile ~= 0
% RankOrderOfLowThreshold = floor(LowPercentile*NumberSamplesImported);
% LowThreshold = AscendingData(RankOrderOfLowThreshold);
% mask(mask <= LowThreshold) = 0;
% end
% if HighPercentile ~= 1
% RankOrderOfHighThreshold = ceil(HighPercentile*NumberSamplesImported);
% HighThreshold = AscendingData(RankOrderOfHighThreshold);
% mask(mask >= HighThreshold) = 0;
% end
% ThrownOutDataForDisplay = mask;
% ThrownOutDataForDisplay(mask > 0) = 1;
%
% %%% Fits the data to a third-dimensional polynomial
% % [x,y] = meshgrid(1:size(MeanImage,2), 1:size(MeanImage,1));
% % x2 = x.*x;
% % y2 = y.*y;
% % xy = x.*y;
% % x3 = x2.*x;
% % x2y = x2.*y;
% % xy2 = y2.*x;
% % y3 = y2.*y;
% % o = ones(size(MeanImage));
% % ind = find((MeanImage > 0) & (mask > 0));
% % coeffs = [x3(ind) x2y(ind) xy2(ind) y3(ind) x2(ind) y2(ind) xy(ind) x(ind) y(ind) o(ind)] \ double(MeanImage(ind));
% % IlluminationImage = reshape([x3(:) x2y(:) xy2(:) y3(:) x2(:) y2(:) xy(:) x(:) y(:) o(:)] * coeffs, size(MeanImage));
% %
%
% %%% Fits the data to a fourth-dimensional polynomial
% [x,y] = meshgrid(1:size(MeanImage,2), 1:size(MeanImage,1));
% x2 = x.*x;
% y2 = y.*y;
% xy = x.*y;
% x3 = x2.*x;
% x2y = x2.*y;
% xy2 = y2.*x;
% y3 = y2.*y;
% x4 = x2.*x2;
% y4 = y2.*y2;
% x3y = x3.*y;
% x2y2 = x2.*y2;
% xy3 = x.*y3;
% o = ones(size(MeanImage));
% ind = find((MeanImage > 0) & (mask > 0));
% coeffs = [x4(ind) x3y(ind) x2y2(ind) xy3(ind) y4(ind) ...
%           x3(ind) x2y(ind) xy2(ind) y3(ind) ...
%           x2(ind) xy(ind) y2(ind)  ...
%           x(ind) y(ind) ...
%           o(ind)] \ double(MeanImage(ind));
% IlluminationImage = reshape([x4(:) x3y(:) x2y2(:) xy3(:) y4(:) ...
%           x3(:) x2y(:) xy2(:) y3(:) ...
%           x2(:) xy(:) y2(:)  ...
%           x(:) y(:) ...
%           o(:)] * coeffs, size(MeanImage));
% CorrFactorsRaw = reshape(IlluminationImage,TotalSamplesToBeGridded,1);
% IlluminationImage2 = IlluminationImage ./ mean(CorrFactorsRaw);
%
% %%% Shows the results.
% figure, subplot(1,3,1), imagesc(MeanImage), title('Imported Data'), colorbar
% subplot(1,3,2), imagesc(ThrownOutDataForDisplay), title('Ignored Samples'),
% subplot(1,3,3), imagesc(IlluminationImage2), title('Correction Factors'), colorbar
%
% %%% Puts the results in a column and displays in the main Matlab window.
% OrigData = reshape(MeanImage,TotalSamplesToBeGridded,1) %#ok We want to ignore MLint error checking for this line.
% CorrFactors = reshape(IlluminationImage2,TotalSamplesToBeGridded,1);
% CorrectedData = OrigData./CorrFactors %#ok We want to ignore MLint error checking for this line.
%
% CPmsgbox('The original data and the corrected data are now displayed in the Matlab window. You can cut and paste from there.')
%
% % %%% Exports the results to the clipboard.
% % clipboard('copy',CorrFactors);
% % h = CPmsgbox('The correction factors are now on the clipboard. Paste them where desired and press OK.  The data is also displayed in column format in the main Matlab window, so you can copy and paste from there as well.');
% % waitfor(h)
% % clipboard('copy',OrigData);
% % h = CPmsgbox('The original data used to generate those normalization factors is now on the clipboard. Paste them where desired (if desired) and press OK.  The data is also displayed in column format in the main Matlab window, so you can copy and paste from there as well.');
% % waitfor(h)

% Help for NORMALIZATION:
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