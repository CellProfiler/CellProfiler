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
%   Peter Swire
%   Rodrigo Ipince
%   Vicky Lay
%   Jun Liu
%   Chris Gang
%
% Website: http://www.cellprofiler.org
%
% $Revision$

if nargin <= 7
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
% Jason change begin
% situations for log plot request
%   max > 0 and min > 0
%       GOOD
%   max > 0 and min < 0
%       set minLog = 0 and use maxLog
%   max < 0 and min < 0
%       cannot use log plot therefore use linear

if (strcmp(Log,'Yes')) && (MaxHistogramValue > 0)
    % Check here that the Max value is > 0, if this is false
    % then a linear plot is performed which is the else of
    % this if/else block.
    MaxLog = log10(MaxHistogramValue);
    if MinHistogramValue <= 0
        if PotentialMaxHistogramValue > 1
            if (findobj('Tag','Msgbox_CPhistbins: Log left bin < 0, changed to 0'))
                %                 % This warning dialog is already open
            else
                %                 % This warning dialog is NOT open
                a=CPwarndlg('The value you have entered for the the leftmost bin threshold is a negative number which would result in an imaginary number in this log scale mode. To prevent an error from occurring, CellProfiler has set this number to be 0 and will graph the histogram with this setting.','CPhistbins: Log left bin < 0, changed to 0','replace');
            end
            MinLog = 0;
        else
            A=SelectedMeasurementsMatrix(SelectedMeasurementsMatrix~=0);
            MinLog = log10(min(A));
        end
    else
        MinLog = log10(MinHistogramValue);
    end
    HistogramRange = MaxLog - MinLog;
    if HistogramRange <= 0 % Bad positive values entered
        % change this to a warning and set the bin values to the automatic
        % values
        % The entered values are both grater than zero now, but left >
        % right. Use the automatic values and than check to make sure that
        % they are both > 0.
        %        CPwarndlg('The numbers you entered for the leftmost or rightmost bin thresholds results in the range being zero or less. For example, this would occur if you entered a leftmost bin threshold value that is greater than the rightmost bin threshold value. The values are being changed to automatically calculated values.');
        MaxHistogramValue = PotentialMaxHistogramValue;
        MinHistogramValue = PotentialMinHistogramValue;
        if ((MaxHistogramValue > 0) && (MinHistogramValue>0))
            % Everything is good with automatic
            if (findobj('Tag','Msgbox_CPhistbins: Range values changed to automatic'))
            else
                CPwarndlg('The numbers you entered for the leftmost or rightmost bin thresholds results in the range being zero or less. For example, this would occur if you entered a leftmost bin threshold value that is greater than the rightmost bin threshold value. The values are being changed to automatically calculated values.','CPhistbins: Range values changed to automatic','replace');
            end
            MinLog = log10(MinHistogramValue);
            MaxLog = log10(MaxHistogramValue);
            HistogramRange = MaxLog - MinLog;
            BinWidth = HistogramRange/NumBins;
            for n = 1:(NumBins+2);
                PlotBinLocations(n) = 10^(MinLog + BinWidth*(n-2));
            end
        else %switch to linear plotting and warn user
            HistogramRange = MaxHistogramValue - MinHistogramValue;
            % values are now automatically calculated so if the
            % HistogramRange is less than zero, there is a data problem not
            % a problem with incorrectly entered limits.
            if HistogramRange <= 0
                error('The numbers calculated automatically for one of these values, results in the range being zero or less. ');
            end
            Log='No';
            if (findobj('Tag','Msgbox_CPhistbins: Changing to linear plot'))
            else
                CPwarndlg('The numbers you entered for the leftmost or rightmost bin thresholds results in the range being zero or less. For example, this would occur if you entered a leftmost bin threshold value that is greater than the rightmost bin threshold value. One or both of the automatically calculated values are less than or equal to zero; therefore, plotting is changed to linear.','CPhistbins: Changing to linear plot','replace');
            end
            BinWidth = HistogramRange/NumBins;
            for n = 1:(NumBins+2);
                PlotBinLocations(n) = MinHistogramValue + BinWidth*(n-2);
            end
        end
    else
        BinWidth = HistogramRange/NumBins;
        for n = 1:(NumBins+2);
            PlotBinLocations(n) = 10^(MinLog + BinWidth*(n-2));
        end
    end
elseif (strcmp(Log,'Yes')) % catch the case when Log plot is selected and Max value <= 0
    % switch to automatic values and try, otherwise switch to linear.
    %    CPwarndlg('The numbers you entered for the leftmost or rightmost bin thresholds results in the range being zero or less. For example, this would occur if you entered a leftmost bin threshold value that is greater than the rightmost bin threshold value. The values are being changed to automatically calculated values.');
    MaxHistogramValue = PotentialMaxHistogramValue;
    MinHistogramValue = PotentialMinHistogramValue;
    if ((MaxHistogramValue > 0) && (MinHistogramValue>0))
        % Everything is good with automatic
        MinLog = log10(MinHistogramValue);
        MaxLog = log10(MaxHistogramValue);
        HistogramRange = MaxLog - MinLog;
        BinWidth = HistogramRange/NumBins;
        for n = 1:(NumBins+2);
            PlotBinLocations(n) = 10^(MinLog + BinWidth*(n-2));
        end
        if (findobj('Tag','Msgbox_CPhistbins: Bad range values, changed to automatic'))
        else
            CPwarndlg('The numbers you entered for the leftmost or rightmost bin thresholds results in the range being zero or less. For example, this would occur if you entered a leftmost bin threshold value that is greater than the rightmost bin threshold value. The values are being changed to automatically calculated values.','CPhistbins: Bad range values, changed to automatic','replace');
        end
    else %switch to linear plotting and warn user
        HistogramRange = MaxHistogramValue - MinHistogramValue;
        % values are now automatically calculated so if the
        % HistogramRange is less than zero, there is a data problem not
        % a problem with incorrectly entered limits.
        if HistogramRange <= 0
            error('The numbers calculated automatically for one of these values, results in the range being zero or less. ');
        end
        Log='No';
        if (findobj('Tab','Msgbox_CPhistbins: Changing from Log to linear plot'))
        else
            CPwarndlg('The numbers you entered for the leftmost or rightmost bin thresholds results in the range being zero or less. For example, this would occur if you entered a leftmost bin threshold value that is greater than the rightmost bin threshold value. One or both of the automatically calculated values are less than or equal to zero; therefore, plotting is changed to linear.','CPhistbins: Changing from Log to linear plot','replace');
        end
        BinWidth = HistogramRange/NumBins;
        for n = 1:(NumBins+2);
            PlotBinLocations(n) = MinHistogramValue + BinWidth*(n-2);
        end
    end
elseif (~strcmp(Log,'Yes')) % Linear plotting was selected
    %%% Determine plot bin locations.
    HistogramRange = MaxHistogramValue - MinHistogramValue;
    if HistogramRange <= 0
        %error('The numbers you entered for the minimum or maximum, or the number which was calculated automatically for one of these values, results in the range being zero or less.  For example, this would occur if you entered a minimum that is greater than the maximum which you asked to be automatically calculated.');
        MaxHistogramValue = PotentialMaxHistogramValue;
        MinHistogramValue = PotentialMinHistogramValue;
        HistogramRange = MaxHistogramValue - MinHistogramValue;
        if HistogramRange <= 0
            error('The numbers you entered for the minimum or maximum results in the range being zero or less.  For example, this would occur if you entered a minimum that is greater than the maximum. The values were switched to automatically calculated values and the problem still exists.');
        end
        if (findobj('Tag','Msgbox_CPhistbins: Changing Linear plot to automaic range values'))
        else
            CPwarndlg('The numbers you entered for the minimum or maximum results in the range being zero or less.  For example, this would occur if you entered a minimum that is greater than the maximum. The values are being switched to atumatically calculated values.','CPhistbins: Changing Linear plot to automaic range values','replace');
        end
    end
    BinWidth = HistogramRange/NumBins;
    for n = 1:(NumBins+2);
        PlotBinLocations(n) = MinHistogramValue + BinWidth*(n-2);
    end
end

%%% Now, for histogram-calculating bins (BinLocations), replace the
%%% initial and final PlotBinLocations with + or - infinity.
% This will need to be put up into the above cases
PlotBinLocations = PlotBinLocations';
BinLocations = PlotBinLocations;

if strcmp(Log,'Yes')&& (MaxHistogramValue > 0)
    for n = 1:length(PlotBinLocations);
        PlotBinLocations(n) = log10(PlotBinLocations(n));
    end
    for m = 1:length(BinLocations);
        BinLocations(m) = log10(BinLocations(m));
    end
end
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
    for m = 1:length(BinLocations);
        BinLocations(m) = 10^BinLocations(m);
    end
end

if strcmp(upper(CountOption(1)),'C')
    YData = histc(SelectedMeasurementsMatrix,real(BinLocations));
else
    YData = hist(SelectedMeasurementsMatrix,real(BinLocations));
end

