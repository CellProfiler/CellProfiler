function [BinLocations,PlotBinLocations,XTickLabels] = CPhistbins(Measurements,NumberOfBins,MinVal,MaxVal,PlotLog)

SelectedMeasurementsMatrix = Measurements(:);
PotentialMaxHistogramValue = max(SelectedMeasurementsMatrix);
PotentialMinHistogramValue = min(SelectedMeasurementsMatrix);

%%% See whether the min and max histogram values were user-entered numbers or should be automatically calculated.
if isempty(str2num(MinVal)) %#ok
    if strcmp(MinVal,'automatic')
        MinHistogramValue = PotentialMinHistogramValue;
    else
        CPerrordlg('The value entered for the minimum histogram value must be either a number or the word ''automatic''.')
    end
else
    MinHistogramValue = str2num(MinVal); %#ok
end

if isempty(str2num(MaxVal)) %#ok
    if strcmp(MaxVal,'automatic')
        MaxHistogramValue = PotentialMaxHistogramValue;
    else
        CPerrordlg('The value entered for the maximum histogram value must be either a number or the word ''automatic''.')
    end
else
    MaxHistogramValue = str2num(MaxVal); %#ok
end

if strcmpi(PlotLog,'Yes')
    MaxLog = log10(MaxHistogramValue);
    MinLog = log10(MinHistogramValue);
    HistogramRange = MaxLog - MinLog;
    if HistogramRange <= 0
        CPerrordlg('The numbers you entered for the minimum or maximum, or the number which was calculated automatically for one of these values, results in the range being zero or less.  For example, this would occur if you entered a minimum that is greater than the maximum which you asked to be automatically calculated.')
    end
    BinWidth = HistogramRange/NumberOfBins;
    for n = 1:(NumberOfBins+2);
        PlotBinLocations(n) = 10^(MinLog + BinWidth*(n-2));
    end
else
    %%% Determine plot bin locations.
    HistogramRange = MaxHistogramValue - MinHistogramValue;
    if HistogramRange <= 0
        CPerrordlg('The numbers you entered for the minimum or maximum, or the number which was calculated automatically for one of these values, results in the range being zero or less.  For example, this would occur if you entered a minimum that is greater than the maximum which you asked to be automatically calculated.')
    end
    BinWidth = HistogramRange/NumberOfBins;
    for n = 1:(NumberOfBins+2);
        PlotBinLocations(n) = MinHistogramValue + BinWidth*(n-2);
    end
end

%%% Now, for histogram-calculating bins (BinLocations), replace the
%%% initial and final PlotBinLocations with + or - infinity.
PlotBinLocations = PlotBinLocations';
BinLocations = PlotBinLocations;
% BinLocations(1) = -inf;
% BinLocations(n+1) = +inf;
%%% Calculates the XTickLabels.
for i = 1:(length(BinLocations)-1)
    XTickLabels{i} = BinLocations(i);
end
XTickLabels{1} = ['< ', num2str(BinLocations(2),3)];
XTickLabels{i} = ['>= ', num2str(BinLocations(i),3)];

if strcmpi(PlotLog,'Yes')
    for n = 1:length(PlotBinLocations);
        PlotBinLocations(n) = log10(PlotBinLocations(n));
    end
end