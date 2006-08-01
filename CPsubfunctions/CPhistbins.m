function [BinLocations,PlotBinLocations,XTickLabels,YData] = CPhistbins(Measurements,NumBins,LeftBin,LeftVal,RightBin,RightVal,Log,CountOption)

% This function will calculate a histogram based on measurements, bin
% numbers, minimum value, and maximum value. The x-axis can be in log
% scale. The histogram can either be a count or normal values.
%
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
%
% Website: http://www.cellprofiler.org
%
% $Revision: 2802 $

if nargin <= 5
    CountOption = 'Normal';
end

%%% If measurements are in cell array, convert to matrix.
if iscell(Measurements)
    SelectedMeasurementsMatrix = cell2mat(Measurements(:));
else
    SelectedMeasurementsMatrix = Measurements(:);
end

PotentialMaxHistogramValue = max(SelectedMeasurementsMatrix);
PotentialMinHistogramValue = min(SelectedMeasurementsMatrix);

%%% See whether the min and max histogram values were user-entered numbers
%%% or should be automatically calculated.
if strcmp(LeftBin,'Min value found') == 1
    MinHistogramValue = PotentialMinHistogramValue;
else
    MinHistogramValue = LeftVal; %#ok
end


if strcmp(RightBin,'Max value found') == 1  %#ok
    MaxHistogramValue = PotentialMaxHistogramValue;
else
    MaxHistogramValue = RightVal; %#ok
end
    
if strcmp(Log,'Yes') == 1
    MaxLog = log10(MaxHistogramValue);
    if MinHistogramValue == 0
        if PotentialMaxHistogramValue > 1
            MinLog = 0;
        else
            A=SelectedMeasurementsMatrix(SelectedMeasurementsMatrix~=0);
            MinLog = log10(min(A));
        end
    else
        MinLog = log10(MinHistogramValue);
    end
    HistogramRange = MaxLog - MinLog;
    if HistogramRange <= 0
        error('The numbers you entered for the leftmost or rightmost bin thresholds, or the number which was calculated automatically for one of these values, results in the range being zero or less.  For example, this would occur if you entered a leftmost bin threshold value that is greater than the rightmost bin threshold value which you asked to be automatically calculated.');
    elseif imag(HistogramRange) ~= 0
        error('The numbers you entered for the leftmost or rightmost bin thresholds, or the number which was calculated automatically for one of these values, results in an imaginary number. This might happen if you chose log scale for negative numbers.');
    end
    BinWidth = HistogramRange/NumBins;
    for n = 1:(NumBins+2);
        PlotBinLocations(n) = 10^(MinLog + BinWidth*(n-2));
    end
else
    %%% Determine plot bin locations.
    HistogramRange = MaxHistogramValue - MinHistogramValue;
    if HistogramRange <= 0
        error('The numbers you entered for the minimum or maximum, or the number which was calculated automatically for one of these values, results in the range being zero or less.  For example, this would occur if you entered a minimum that is greater than the maximum which you asked to be automatically calculated.');
    end
    BinWidth = HistogramRange/NumBins;
    for n = 1:(NumBins+2);
        PlotBinLocations(n) = MinHistogramValue + BinWidth*(n-2);
    end
end

%%% Now, for histogram-calculating bins (BinLocations), replace the
%%% initial and final PlotBinLocations with + or - infinity.
PlotBinLocations = PlotBinLocations';
BinLocations = PlotBinLocations;

if strcmp(upper(CountOption(1)),'C')
    BinLocations(1) = -inf;
    BinLocations(n+1) = +inf;
    %%% Calculates the XTickLabels.
    for i = 1:(length(BinLocations)-1)
        XTickLabels{i} = BinLocations(i);
    end
else
    %%% Calculates the XTickLabels.
    for i = 1:(length(BinLocations))
        XTickLabels{i} = BinLocations(i);
    end
end

XTickLabels{1} = ['< ', num2str(BinLocations(2),3)];
XTickLabels{i} = ['>= ', num2str(BinLocations(i),3)];

if strcmp(Log,'Yes')
    for n = 1:length(PlotBinLocations);
        PlotBinLocations(n) = log10(PlotBinLocations(n));
    end
end

if strcmp(upper(CountOption(1)),'C')
    YData = histc(SelectedMeasurementsMatrix,real(BinLocations));
else
    YData = hist(SelectedMeasurementsMatrix,real(BinLocations));
end