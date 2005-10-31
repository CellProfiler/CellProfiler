function Histogram(handles)

% Help for the Histogram tool:
% Category: Data Tools
%
% The individual object measurements can be displayed in histogram
% format using this tool.  As prompted, select the output file
% containing the measurements, then choose the measurement parameter
% to be displayed, and the sample information label.
%
% You may then choose which images' measurements to display or export
% (To display data from only one image, enter that image's number as
% both the first and last sample); the number of bins to be used; the
% minimum and maximum values to be used for the histogram (on the X
% axis); whether you want all the objects' data to be displayed in a
% single (cumulative) histogram or in separate histograms; whether you
% want to calculate histogram data only for objects meeting a
% threshold in a measurement (you will be asked later which
% measurement to use for this thresholding); whether you
% want the Y axis (number of cells) to be absolute (the same for all
% histograms) or relative (scaled to fit the maximum value for that
% sample); whether you want to display the results as a compressed
% histogram (heatmap) rather than a conventional histogram; whether
% you want to export the data (tab-delimited format, which can be
% opened in Excel); whether you want each row in the exported histogram
% to contain an image or a bin; and whether you want to display the
% histograms (Impractical when exporting large amounts of data). It may
% take some time to then process the data.
%
% X axis labels for histograms: Typically, the X axis labels will be
% too crowded.  This default state is shown because you might want to
% know the exact values that were used for the histogram bins.  The
% actual numbers can be viewed by clicking the 'This window' button
% under 'Change plots' and looking at the numbers listed under
% 'Labels'.  To change the X axis labels, you can click 'Fewer' in the
% main histogram window, or you can click a button under 'Change
% plots' and either change the font size on the 'Style' tab, or check
% the boxes marked 'Auto' for 'Ticks' and 'Labels' on the 'X axis'
% tab. Be sure to check both boxes, or the labels will not be
% accurate. To revert to the original labels, click 'Restore' in the
% main histogram window, but beware that this function does not work
% when more than one histogram window is open at once, because the
% most recently produced histogram's labels will be used for everything.
%
% Change plots/change bars buttons: These buttons allow you to change
% properties of the plots or the bars within the plots for either
% every plot in the window ('This window'), the current plot only
% ('Current'), or every plot inevery open window ('All windows').
% This include colors, axis limits and other properties.
%
% Other notes about histograms: (1) Data outside the range you
% specified to calculate histogram bins are added together and
% displayed in the first and last bars of the histogram. (2) Only the
% display can be changed in this window, including axis limits.  The
% histogram bins themselves cannot be changed here because the data
% must be recalculated. (3) If a change you make using the 'Change
% display' buttons does not seem to take effect in all of the desired
% windows, try pressing enter several times within that box, or look
% in the bottom of the Property Editor window that opens when you
% first press one of those buttons.  There may be a message describing
% why.  For example, you may need to deselect 'Auto' before changing
% the limits of the axes. (4) The labels for each bar specify the low
% bound for that bin.  In other words, each bar includes data equal to
% or greater than the label, but less than the label on the bar to its
% right.
%
% POTENTIAL IMPROVEMENT:
% - You will then have the option of loading names for
% each image so that each histogram you make will be labeled with
% those names (if the measurement file does not already have names
% embedded). If you choose to import sample names here, you will need
% to select a text file that contains one name for every sample, even
% if you only plan to view a subset of the image data as histograms.
%
% See also PLOTSINGLEMEASUREMENTS.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne Carpenter <carpenter@wi.mit.edu>
%   Thouis Jones   <thouis@csail.mit.edu>
%   In Han Kang    <inthek@mit.edu>
%   Ola Friman     <friman@bwh.harvard.edu>
%   Steve Lowe     <stevelowe@alum.mit.edu>
%   Joo Han Chang  <joohan.chang@gmail.com>
%   Colin Clarke   <colinc@mit.edu>
%   Mike Lamprecht <mrl@wi.mit.edu>
%   Susan Ma       <xuefang_ma@wi.mit.edu>
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


%%% Call the function CPgetfeature(), which opens a series of list dialogs and
%%% lets the user choose a feature. The feature can be identified via 'ObjectTypename',
%%% 'FeatureType' and 'FeatureNo'.
[ObjectTypename,FeatureType,FeatureNo] = CPgetfeature(handles,0);
if isempty(ObjectTypename),return,end
MeasurementToExtract = [handles.Measurements.(ObjectTypename).([FeatureType,'Features']){FeatureNo},' of ', ObjectTypename];

%%% Put the measurements for this feature in a cell array, one
%%% cell for each image set.
tmp = handles.Measurements.(ObjectTypename).(FeatureType);
Measurements = cell(length(tmp),1);
for k = 1:length(tmp)
    Measurements{k} = tmp{k}(:,FeatureNo);
end

%%% Determines whether any sample info has been loaded.  If sample
%%% info is present, the fieldnames for those are extracted.
ImportedFieldnames = fieldnames(handles.Measurements.Image);
ImportedFieldnames = ImportedFieldnames(strncmp(ImportedFieldnames,'Imported',8) == 1 | strncmp(ImportedFieldnames,'Filename',8) == 1);
if ~isempty(ImportedFieldnames)
    %%% Allows the user to select a heading from the list.
    [Selection, ok] = listdlg('ListString',ImportedFieldnames, 'ListSize', [300 400],...
        'Name','Select sample info',...
        'PromptString','Choose the sample info with which to label each histogram.','CancelString','Cancel',...
        'SelectionMode','single');
    if ok ~= 0
        HeadingName = char(ImportedFieldnames(Selection));
        try SampleNames = handles.Measurements.Image.(HeadingName);
        catch SampleNames = handles.Pipeline.(HeadingName);
        end
    else
        return
    end
end


%%% Calculates some values for the next dialog box.
TotalNumberImageSets = length(Measurements);
TextTotalNumberImageSets = num2str(TotalNumberImageSets);
%%% Ask the user to specify histogram settings.
Prompts{1} = 'Enter the first image number to show or export';
Prompts{2} = ['Enter the last sample number to show or export (the total number of image sets with data in the file is ', TextTotalNumberImageSets, ').'];
Prompts{3} = 'Enter the number of bins you want for the histogram(s). This number should not include the first and last bins, which will contain anything outside the specified range.';
Prompts{4} = 'Any measurements less than this value will be combined in the leftmost bin. Enter automatic to determine this value automatically.';
Prompts{5} = 'Any measurements greater than or equal to this value will be combined in the rightmost bin. Enter automatic to determine this value automatically.';
Prompts{6} = 'Do you want to calculate one histogram for all the images you selected? (The alternative is to calculate a separate histogram for each image?';
Prompts{7} = 'If you want to calculate histogram data only for objects meeting a threshold in a measurement, enter >, >=, =, <=, <, and enter the threshold in the box below, or leave the letter A to include all objects';
Prompts{8} = 'Threshold value, if applicable';
Prompts{9} = 'Do you want the Y-axis (number of objects) to be absolute or relative?';
Prompts{10} = 'Display as a compressed histogram (heatmap)?';
Prompts{11} = 'Do you want the histogram data to be exported? To export the histogram data, enter a filename (with ".xls" to open easily in Excel), or type "no" if you do not want to export the data.';
Prompts{12} = 'If exporting histograms or displaying as a compressed histogram (heatmap), is each row to be one image or one histogram bin? Enter "image" or "bin".';
Prompts{13} = 'Do you want the histograms to be displayed? (Impractical when exporting large amounts of data)';
Prompts{14} = 'Do you want the histograms bins to contain the actual numbers of objects in the bin (N) or the percentage of objects in the bin (P)?';

AcceptableAnswers = 0;
global Answers
while AcceptableAnswers == 0
    Defaults{1} = '1';
    Defaults{2} = TextTotalNumberImageSets;
    Defaults{3} = '20';
    Defaults{4} = 'automatic';
    Defaults{5} = 'automatic';
    Defaults{6} = 'no';
    Defaults{7} = 'A';
    Defaults{8} = '1';
    Defaults{9} = 'relative';
    Defaults{10} = 'no';
    Defaults{11} = 'no';
    Defaults{12} = 'image';
    Defaults{13} = 'yes';
    Defaults{14} = 'N';
    %%% Loads the Defaults from the global workspace (i.e. from the
    %%% previous time the user ran this tool) if possible.
    for i = 1: length(Prompts)
        try Defaults{i} = Answers{i};
        end
    end

    %%% Creates the dialog box for user input.
    % Replaced: Answers = inputdlg(Prompts,'Choose histogram settings',1,Defaults,'on');
    % Workaround because input dialog box is too tall. Break questions up
    % into two sets. We could create a custom version of inputdlg with a
    % vertical slider, but that would be more complicated.
    Answers1 = inputdlg(Prompts(1:7),'Choose histogram settings - page 1',1,Defaults(1:7),'on');

    %%% If user clicks cancel button Answers1 will be empty.
    if isempty(Answers1)
        return
    end
    Answers2 = inputdlg(Prompts(8:14),'Choose histogram settings - page 2',1,Defaults(8:14),'on');

    %%% If user clicks cancel button Answers2 will be empty.
    if isempty(Answers2)
        return
    end
    %%% If both sets were non-empty, concatenate into Answers.
    Answers = { Answers1{:} Answers2{:} };
    clear Answers1 Answers2;
    %%% TO DO: reset the width of the dialog box: it's too narrow!

    %%% Error checking for individual answers being empty.
    ErrorFlag = 0;
    for i = 1:length(Prompts)
        if isempty(Answers{i}) == 1
            errordlg(['The box was left empty in response to the question: ', Prompts{i},'.'])
            ErrorFlag = 1;
            break
        end
    end
    if ErrorFlag == 1
        continue
    end

    try FirstImage = str2double(Answers{1});
    catch errordlg(['You must enter a number in answer to the question: ', Prompts{1}, '.']);
        continue
    end

    try LastImage = str2double(Answers{2});
    catch errordlg(['You must enter a number in answer to the question: ', Prompts{2}, '.']);
        continue
    end
    NumberOfImages = LastImage - FirstImage + 1;
    if NumberOfImages == 0
        NumberOfImages = TotalNumberImageSets;
    elseif NumberOfImages > TotalNumberImageSets
        errordlg(['There are only ', TextTotalNumberImageSets, ' image sets total, but you specified that you wanted to view image set number ', num2str(LastImage),'.'])
        continue
    end
    try NumberOfBins = str2double(Answers{3});
    catch errordlg(['You must enter a number in answer to the question: ', Prompts{3}, '.']);
        continue
    end
    MinHistogramValue = Answers{4};
    try str2double(MinHistogramValue);
    catch errordlg(['You must enter a number in answer to the question: ', Prompts{4}, '.']);
        continue
    end

    MaxHistogramValue = Answers{5};
    try str2double(MaxHistogramValue);
    catch errordlg(['You must enter a number in answer to the question: ', Prompts{5}, '.']);
        continue
    end
    CumulativeHistogram = Answers{6};
    GreaterOrLessThan = Answers{7};
    try ThresholdValue = str2double(Answers{8});
    catch errordlg(['You must enter a number in answer to the question: ', Prompts{8}, '.']);
        continue
    end

    %%% Error checking for the Y Axis Scale question.
    try YAxisScale = lower(Answers{9});
    catch errordlg(['You must enter "absolute" or "relative" in answer to the question: ', Prompts{9}, '.']);
        continue
    end
    if strcmpi(YAxisScale, 'relative') ~= 1 && strcmpi(YAxisScale, 'absolute') ~= 1
        errordlg(['You must enter "absolute" or "relative" in answer to the question: ', Prompts{9}, '.']);
        continue
    end
    CompressedHistogram = Answers{10};
    if strncmpi(CompressedHistogram,'Y',1) ~= 1 && strncmpi(CompressedHistogram,'N',1) ~= 1
        errordlg(['You must enter "yes" or "no" in answer to the question: ', Prompts{10}, '.']);
        continue
    end
    SaveData = Answers{11};
    if isempty(SaveData)
        errordlg(['You must enter "no", or a filename, in answer to the question: ', Prompts{11}, '.']);
        continue
    end
    %%% Adds the full pathname to the filename.
    if strcmpi(SaveData,'No') ~= 1
        SaveData = fullfile(RawPathname,SaveData);
        OutputFileOverwrite = exist(SaveData,'file');
        if OutputFileOverwrite ~= 0
            Answer = CPquestdlg('A file with that name already exists in the directory containing the raw measurements file. Do you wish to overwrite?','Confirm overwrite','Yes','No','No');
            if strcmp(Answer, 'No') == 1
                continue
            end
        end
    end
    RowImageOrBin = Answers{12};
    if ~strncmpi(RowImageOrBin,'I',1) && ~strncmpi(RowImageOrBin,'B',1)
        errordlg(['You must enter "image" or "bin" in answer to the question: ', Prompts{12}, '.']);
        continue
    end

    ShowDisplay = Answers{13};
    if ~strncmpi(ShowDisplay,'N',1) && ~strncmpi(ShowDisplay,'Y',1)
        errordlg(['You must enter "yes" or "no" in answer to the question: ', Prompts{13}, '.']);
        continue
    end

    NumberOrPercent = Answers{14};
    if ~strncmpi(NumberOrPercent,'N',1) && ~strncmpi(NumberOrPercent,'P',1)
        errordlg(['You must enter "N" or "P" in answer to the question: ', Prompts{14}, '.']);
        continue
    end

    %%% If the user selected A for all, the measurements are not thresholded on some other measurement.
    if ~strcmpi(GreaterOrLessThan,'A')
        [ObjectTypename,FeatureType,FeatureNo] = CPgetfeature(handles,0);
        MeasurementToThresholdValueOnName = handles.Measurements.(ObjectTypename).([FeatureType,'Features'])(FeatureNo);
        tmp = handles.Measurements.(ObjectTypename).(FeatureType);
        MeasurementToThresholdValueOn = cell(length(tmp),1);
        for k = 1:length(tmp)
            MeasurementToThresholdValueOn{k} = tmp{k}(:,FeatureNo);
        end
    end

    %%% Calculates the default bin size and range based on all the data.
    SelectedMeasurementsCellArray = Measurements(FirstImage:LastImage);
    SelectedMeasurementsMatrix = cell2mat(SelectedMeasurementsCellArray(:));
    PotentialMaxHistogramValue = max(SelectedMeasurementsMatrix);
    PotentialMinHistogramValue = min(SelectedMeasurementsMatrix);
    %%% See whether the min and max histogram values were user-entered numbers or should be automatically calculated.
    if isempty(str2num(MinHistogramValue)) %#ok
        if strcmp(MinHistogramValue,'automatic') == 1
            MinHistogramValue = PotentialMinHistogramValue;
        else
            errordlg('The value entered for the minimum histogram value must be either a number or the word ''automatic''.')
            continue
        end
    else MinHistogramValue = str2num(MinHistogramValue); %#ok
    end
    if isempty(str2num(MaxHistogramValue)) %#ok
        if strcmp(MaxHistogramValue,'automatic') == 1
            MaxHistogramValue = PotentialMaxHistogramValue;
        else
            errordlg('The value entered for the maximum histogram value must be either a number or the word ''automatic''.')
            continue
        end
    else MaxHistogramValue = str2num(MaxHistogramValue); %#ok
    end
    %%% Determine plot bin locations.
    HistogramRange = MaxHistogramValue - MinHistogramValue;
    if HistogramRange <= 0
        errordlg('The numbers you entered for the minimum or maximum, or the number which was calculated automatically for one of these values, results in the range being zero or less.  For example, this would occur if you entered a minimum that is greater than the maximum which you asked to be automatically calculated.')
        continue
    end
    BinWidth = HistogramRange/NumberOfBins;
    for n = 1:(NumberOfBins+2);
        PlotBinLocations(n) = MinHistogramValue + BinWidth*(n-2);
    end
    %%% Now, for histogram-calculating bins (BinLocations), replace the
    %%% initial and final PlotBinLocations with + or - infinity.
    PlotBinLocations = PlotBinLocations';
    BinLocations = PlotBinLocations;
    BinLocations(1) = -inf;
    BinLocations(n+1) = +inf;
    %%% Calculates the XTickLabels.
    for i = 1:(length(BinLocations)-1), XTickLabels{i} = BinLocations(i); end
    XTickLabels{1} = ['< ', num2str(BinLocations(2))];
    XTickLabels{i} = ['>= ', num2str(BinLocations(i))];
    %%% Saves this info in a variable, FigureSettings, which
    %%% will be stored later with the figure.
    FigureSettings{1} = PlotBinLocations;
    FigureSettings{2} = XTickLabels;
    %%% If we have gotten this far, the answers must have been
    %%% acceptable, so we can now exit this while loop.
    AcceptableAnswers = 1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Calculates histogram data for cumulative histogram %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strncmpi(CumulativeHistogram, 'Y',1) == 1
    OutputMeasurements{1,1} = SelectedMeasurementsMatrix;
    %%% Retrieves the measurements to threshold on, if requested.
    if strcmpi(GreaterOrLessThan,'A') ~= 1
        SelectMeasurementsCellArray = MeasurementToThresholdValueOn(FirstImage:LastImage);
        OutputMeasurements{1,2} = cell2mat(SelectMeasurementsCellArray(:));
        AdditionalInfoForTitle = [' for objects where ', MeasurementToThresholdValueOnName, GreaterOrLessThan, num2str(ThresholdValue)];
    else AdditionalInfoForTitle = [];
    end
    %%% Applies the specified ThresholdValue and gives a cell
    %%% array as output.
    if strcmpi(GreaterOrLessThan,'A') == 1
        %%% If the user selected A for all, the measurements are not
        %%% altered.
    elseif strcmpi(GreaterOrLessThan,'>') == 1
        OutputMeasurements{1,1} = OutputMeasurements{1,1}(OutputMeasurements{1,2} > ThresholdValue);
    elseif strcmpi(GreaterOrLessThan,'>=') == 1
        OutputMeasurements{1,1} = OutputMeasurements{1,1}(OutputMeasurements{1,2} >= ThresholdValue);
    elseif strcmpi(GreaterOrLessThan,'<') == 1
        OutputMeasurements{1,1} = OutputMeasurements{1,1}(OutputMeasurements{1,2} < ThresholdValue);
    elseif strcmpi(GreaterOrLessThan,'<=') == 1
        OutputMeasurements{1,1} = OutputMeasurements{1,1}(OutputMeasurements{1,2} <= ThresholdValue);
    elseif strcmpi(GreaterOrLessThan,'=') == 1
        OutputMeasurements{1,1} = OutputMeasurements{1,1}(OutputMeasurements{1,2} == ThresholdValue);
    else error(['The value you entered for the method to threshold ', GreaterOrLessThan, ' was not valid.  Acceptable entries are >, >=, =, <=, <.']);
    end

    if isempty(OutputMeasurements{1,1}) == 1
        HistogramData = [];
    else HistogramData = histc(OutputMeasurements{1,1},BinLocations);
    end

    %%% Deletes the last value of HistogramData, which is
    %%% always a zero (because it's the number of values
    %%% that match + inf).
    HistogramData(n+1) = [];
    FinalHistogramData(:,1) = HistogramData;
    HistogramTitles{1} = ['Histogram of data from Image #', num2str(FirstImage), ' to #', num2str(LastImage)];
    FirstImage = 1;
    LastImage = 1;
    NumberOfImages = 1;


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Calculates histogram data for non-cumulative histogram %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
else
    %%% Preallocates the variable ListOfMeasurements.
    ListOfMeasurements{NumberOfImages,1} = Measurements{LastImage};
    if strcmpi(GreaterOrLessThan,'A') ~= 1
        ListOfMeasurements{NumberOfImages,2} = MeasurementToThresholdValueOn{LastImage};
        AdditionalInfoForTitle = [' for objects where ', MeasurementToThresholdValueOnName, GreaterOrLessThan, num2str(ThresholdValue)];
    else AdditionalInfoForTitle = [];
    end
    CompressedImageNumber = 1;
    OutputMeasurements = cell(size(NumberOfImages,1),1);
    FinalHistogramData = [];
    for ImageNumber = FirstImage:LastImage
        ListOfMeasurements{CompressedImageNumber,1} = Measurements{ImageNumber};
        if strcmpi(GreaterOrLessThan,'A') ~= 1
            ListOfMeasurements{CompressedImageNumber,2} = MeasurementToThresholdValueOn{ImageNumber};
        end
        %%% Applies the specified ThresholdValue and gives a cell
        %%% array as output.
        if strcmpi(GreaterOrLessThan,'A') == 1
            %%% If the user selected A for all, the measurements are not
            %%% altered.
            OutputMeasurements = ListOfMeasurements;
        elseif strcmpi(GreaterOrLessThan,'>') == 1
            OutputMeasurements{CompressedImageNumber,1} = ListOfMeasurements{CompressedImageNumber,1}(ListOfMeasurements{CompressedImageNumber,2} > ThresholdValue);
        elseif strcmpi(GreaterOrLessThan,'>=') == 1
            OutputMeasurements{CompressedImageNumber,1} = ListOfMeasurements{CompressedImageNumber,1}(ListOfMeasurements{CompressedImageNumber,2} >= ThresholdValue);
        elseif strcmpi(GreaterOrLessThan,'<') == 1
            OutputMeasurements{CompressedImageNumber,1} = ListOfMeasurements{CompressedImageNumber,1}(ListOfMeasurements{CompressedImageNumber,2} < ThresholdValue);
        elseif strcmpi(GreaterOrLessThan,'<=') == 1
            OutputMeasurements{CompressedImageNumber,1} = ListOfMeasurements{CompressedImageNumber,1}(ListOfMeasurements{CompressedImageNumber,2} <= ThresholdValue);
        elseif strcmpi(GreaterOrLessThan,'=') == 1
            OutputMeasurements{CompressedImageNumber,1} = ListOfMeasurements{CompressedImageNumber,1}(ListOfMeasurements{CompressedImageNumber,2} == ThresholdValue);
        else error(['The value you entered for the method to threshold ', GreaterOrLessThan, ' was not valid.  Acceptable entries are >, >=, =, <=, <.']);
        end
        if isempty(OutputMeasurements{CompressedImageNumber}) == 1
            HistogramData = [];
        else HistogramData = histc(OutputMeasurements{CompressedImageNumber},BinLocations);
        end

        %%% Deletes the last value of HistogramData, which
        %%% is always a zero (because it's the number of values that match
        %%% + inf).
        HistogramData(n+1) = [];
        FinalHistogramData(:,ImageNumber) = HistogramData;
        if exist('SampleNames','var') == 1
            SampleName = SampleNames{ImageNumber};
            HistogramTitles{ImageNumber} = ['#', num2str(ImageNumber), ': ' , SampleName];
        else HistogramTitles{ImageNumber} = ['Image #', num2str(ImageNumber)];
        end
        %%% Increments the CompressedImageNumber.
        CompressedImageNumber = CompressedImageNumber + 1;
    end
end


if strncmpi(NumberOrPercent,'P',1)
    for i = 1: size(FinalHistogramData,2)
        SumForThatColumn = sum(FinalHistogramData(:,i));
        FinalHistogramData(:,i) = FinalHistogramData(:,i)/SumForThatColumn;
    end
end


%%% Saves this info in a variable, FigureSettings, which
%%% will be stored later with the figure.
FigureSettings{3} = FinalHistogramData;
if strcmpi(GreaterOrLessThan,'A') ~= 1
    AnswerFileName = inputdlg({'Name the file'},'Name the file in which to save the subset of measurements',1,{'temp.mat'},'on');
    try
        save(fullfile(handles.DefaultOutputDirectory,AnswerFileName{1}),'OutputMeasurements')
    catch errordlg('Saving did not work.')
    end
end

%%% Saves the data to an excel file if desired.
if strcmpi(SaveData,'No') ~= 1
    WriteHistToExcel(SaveData, FirstImage, LastImage, XTickLabels,...
        FinalHistogramData, MeasurementToExtract, AdditionalInfoForTitle,...
        HistogramTitles, RowImageOrBin);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Displays histogram data for non-compressed histograms %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

VersionCheck = version;

if strcmp(CompressedHistogram,'no') == 1 && strncmpi(ShowDisplay,'Y',1) == 1
    %%% Calculates the square root in order to determine the dimensions for the
    %%% display window.
    SquareRoot = sqrt(NumberOfImages);
    %%% Converts the result to an integer.
    NumberDisplayRows = fix(SquareRoot);
    NumberDisplayColumns = ceil((NumberOfImages)/NumberDisplayRows);
    %%% Acquires basic screen info for making buttons in the
    %%% display window.
    StdUnit = 'point';
    StdColor = get(0,'DefaultUIcontrolBackgroundColor');
    PointsPerPixel = 72/get(0,'ScreenPixelsPerInch');
    %%% Creates the display window.
    FigureHandle = CPfigure;
    set(FigureHandle, 'Name',MeasurementToExtract);

    Increment = 0;
    for ImageNumber = FirstImage:LastImage
        Increment = Increment + 1;
        h = subplot(NumberDisplayRows,NumberDisplayColumns,Increment);
        bar('v6',PlotBinLocations,FinalHistogramData(:,ImageNumber))
        axis tight
        set(get(h,'XLabel'),'String',{MeasurementToExtract;AdditionalInfoForTitle})
        set(h,'XTickLabel',XTickLabels)
        set(h,'XTick',PlotBinLocations)
        set(gca,'UserData',FigureSettings)
        % Fix underscores in HistogramTitles
        AdjustedHistogramTitle = strrep(HistogramTitles{ImageNumber},'_','\_');
        title(AdjustedHistogramTitle)
        if Increment == 1
            if strncmpi(NumberOrPercent,'N',1)
                set(get(h,'YLabel'),'String','Number of objects')
            else
                set(get(h,'YLabel'),'String','Percentage of objects')
            end
        end
    end
    %%% Sets the Y axis scale to be absolute or relative.
    AxesHandles = findobj('Parent', FigureHandle, 'Type', 'axes');
    if strcmp(YAxisScale, 'relative') == 1
        %%% Automatically stretches the x data to fill the plot
        %%% area.
        axis(AxesHandles, 'tight')
        %%% Automatically stretches the y data to fill the plot
        %%% area, except that "auto" leaves a bit more buffer
        %%% white space around the data.
        axis(AxesHandles, 'auto y')
    elseif strcmp(YAxisScale, 'absolute') == 1
        YLimits = get(AxesHandles, 'YLim');
        YLimits2 = cell2mat(YLimits);
        Ymin = min(YLimits2(:,1));
        Ymax = 1.05*max(YLimits2(:,2));
        XLimits = get(AxesHandles, 'XLim');
        XLimits2 = cell2mat(XLimits);
        Xmin = min(XLimits2(:,1));
        Xmax = max(XLimits2(:,2));
        %%% Sets the axis limits as calculated.
        axis(AxesHandles, [Xmin Xmax Ymin Ymax])
    end

    %%% Adds buttons to the figure window.
    %%% Resizes the figure window to make room for the buttons.
    %%% The axis units are changed
    %%% to a non-normalized unit (pixels) prior to resizing so
    %%% that the axes don't resize to fill up the entire figure
    %%% window. Then after the figure is resized, the axes are
    %%% set back to normalized so they scale appropriately if
    %%% the user resizes the window.
    FigurePosition = get(FigureHandle, 'Position');
    ScreenSize = get(0,'ScreenSize');
    NewFigurePosition = [ScreenSize(3)/4 ScreenSize(4)-(FigurePosition(4)+170) 600 FigurePosition(4)];
    set(FigureHandle,'Position',NewFigurePosition)
    set(AxesHandles,'Units', 'pixels');
    NewHeight = FigurePosition(4)+80;
    NewFigurePosition = [NewFigurePosition(1) NewFigurePosition(2) NewFigurePosition(3) NewHeight];
    set(FigureHandle,'Position',NewFigurePosition)
    set(AxesHandles,'Units','normalized');
    
    %%% Creates text 1
    uicontrol('Parent',FigureHandle, ...
        'BackgroundColor',[.7 .7 .9], ...
        'Unit',StdUnit, ...
        'Position',PointsPerPixel*[5 NewHeight-30 85 22], ...
        'Units','Normalized',...
        'String','Change plots:', ...
        'Style','text', ...
        'FontSize',handles.Current.FontSize);
    %%% Creates text 2
    uicontrol('Parent',FigureHandle, ...
        'BackgroundColor',[.7 .7 .9], ...
        'Unit',StdUnit, ...
        'Position',PointsPerPixel*[5 NewHeight-60 85 22], ...
        'Units','Normalized',...
        'String','Change bars:', ...
        'Style','text', ...
        'FontSize',handles.Current.FontSize);
    %%% Creates text 3
    uicontrol('Parent',FigureHandle, ...
        'BackgroundColor',[.7 .7 .9], ...
        'Unit',StdUnit, ...
        'Position',PointsPerPixel*[445 NewHeight-30 85 22], ...
        'Units','Normalized',...
        'String','X axis labels:', ...
        'Style','text', ...
        'FontSize',handles.Current.FontSize);
    %%% These callbacks control what happens when display
    %%% buttons are pressed within the histogram display
    %%% window.
    if strcmp(computer,'MAC') && str2num(VersionCheck(1:3)) < 7.1
        Button1Callback = 'FigureHandle = gcf; AxesHandles = findobj(''Parent'', FigureHandle, ''Type'', ''axes''); drawnow';
    else
        Button1Callback = 'FigureHandle = gcf; AxesHandles = findobj(''Parent'', FigureHandle, ''Type'', ''axes''); try, propedit(AxesHandles,''v6''), catch, CPmsgbox(''A bug in Matlab is preventing this function from working. Service Request #1-RR6M1''), end; drawnow';
    end
    uicontrol('Parent',FigureHandle, ...
        'Unit',StdUnit, ...
        'BackgroundColor',[0.65 0.65 0.85], ...
        'CallBack',Button1Callback, ...
        'Position',PointsPerPixel*[95 NewHeight-30 85 22], ...
        'Units','Normalized',...
        'String','This window', ...
        'Style','pushbutton', ...
        'FontSize',handles.Current.FontSize);
    if ispc || str2num(VersionCheck(1:3)) >= 7.1
        Button2Callback = 'propedit(gca,''v6''); drawnow';
        uicontrol('Parent',FigureHandle, ...
            'Unit',StdUnit, ...
            'BackgroundColor',[0.65 0.65 0.85], ...
            'CallBack',Button2Callback, ...
            'Position',PointsPerPixel*[185 NewHeight-30 70 22], ...
            'Units','Normalized',...
            'String','Current', ...
            'Style','pushbutton', ...
            'FontSize',handles.Current.FontSize);
    end
    if strcmp(computer,'MAC') && str2num(VersionCheck(1:3)) < 7.1
        Button3Callback = 'FigureHandles = findobj(''Parent'', 0); AxesHandles = findobj(FigureHandles, ''Type'', ''axes''); axis(AxesHandles, ''manual''); drawnow';
    else
        Button3Callback = 'FigureHandles = findobj(''Parent'', 0); AxesHandles = findobj(FigureHandles, ''Type'', ''axes''); axis(AxesHandles, ''manual''); try, propedit(AxesHandles,''v6''), catch, CPmsgbox(''A bug in Matlab is preventing this function from working. Service Request #1-RR6M1''), end, drawnow';
    end
    uicontrol('Parent',FigureHandle, ...
        'Unit',StdUnit, ...
        'BackgroundColor',[0.65 0.65 0.85], ...
        'CallBack', Button3Callback, ...
        'Position',PointsPerPixel*[260 NewHeight-30 85 22], ...
        'Units','Normalized',...
        'String','All windows', ...
        'Style','pushbutton', ...
        'FontSize',handles.Current.FontSize);
    if strcmp(computer,'MAC') && str2num(VersionCheck(1:3)) < 7.1
        Button4Callback = 'FigureHandle = gcf; PatchHandles = findobj(FigureHandle, ''Type'', ''patch''); drawnow';
    else
        Button4Callback = 'FigureHandle = gcf; PatchHandles = findobj(FigureHandle, ''Type'', ''patch''); try, propedit(PatchHandles,''v6''), catch, CPmsgbox(''A bug in Matlab is preventing this function from working. Service Request #1-RR6M1''), end; drawnow';
    end
    uicontrol('Parent',FigureHandle, ...
        'Unit',StdUnit, ...
        'BackgroundColor',[0.65 0.65 0.85], ...
        'CallBack', Button4Callback, ...
        'Position',PointsPerPixel*[95 NewHeight-60 85 22], ...
        'Units','Normalized',...
        'String','This window', ...
        'Style','pushbutton', ...
        'FontSize',handles.Current.FontSize);
    if strcmp(computer,'MAC') && str2num(VersionCheck(1:3)) < 7.1
        Button5Callback = 'AxisHandle = gca; PatchHandles = findobj(''Parent'', AxisHandle, ''Type'', ''patch''); drawnow';
    else
        Button5Callback = 'AxisHandle = gca; PatchHandles = findobj(''Parent'', AxisHandle, ''Type'', ''patch''); try, propedit(PatchHandles,''v6''), catch, CPmsgbox(''A bug in Matlab is preventing this function from working. Service Request #1-RR6M1''), end; drawnow';
    end
    uicontrol('Parent',FigureHandle, ...
        'Unit',StdUnit, ...
        'BackgroundColor',[0.65 0.65 0.85], ...
        'CallBack', Button5Callback, ...
        'Position',PointsPerPixel*[185 NewHeight-60 70 22], ...
        'Units','Normalized',...
        'String','Current', ...
        'Style','pushbutton', ...
        'FontSize',handles.Current.FontSize);
    if strcmp(computer,'MAC') && str2num(VersionCheck(1:3)) < 7.1
        Button6Callback = 'FigureHandles = findobj(''Parent'', 0); PatchHandles = findobj(FigureHandles, ''Type'', ''patch''); drawnow';
    else
        Button6Callback = 'FigureHandles = findobj(''Parent'', 0); PatchHandles = findobj(FigureHandles, ''Type'', ''patch''); try, propedit(PatchHandles,''v6''), catch, CPmsgbox(''A bug in Matlab is preventing this function from working. Service Request #1-RR6M1''), end; drawnow';
    end
    uicontrol('Parent',FigureHandle, ...
        'Unit',StdUnit, ...
        'BackgroundColor',[0.65 0.65 0.85], ...
        'CallBack', Button6Callback, ...
        'Position',PointsPerPixel*[260 NewHeight-60 85 22], ...
        'Units','Normalized',...
        'String','All windows', ...
        'Style','pushbutton', ...
        'FontSize',handles.Current.FontSize);
    Button7Callback = 'CPmsgbox(''Histogram display info: (1) Data outside the range you specified to calculate histogram bins are added together and displayed in the first and last bars of the histogram.  (2) Only the display can be changed in this window, including axis limits.  The histogram bins themselves cannot be changed here because the data must be recalculated. (3) If a change you make using the "Change display" buttons does not seem to take effect in all of the desired windows, try pressing enter several times within that box, or look in the bottom of the Property Editor window that opens when you first press one of those buttons.  There may be a message describing why.  For example, you may need to deselect "Auto" before changing the limits of the axes. (4) The labels for each bar specify the low bound for that bin.  In other words, each bar includes data equal to or greater than the label, but less than the label on the bar to its right. (5) If the tick mark labels are overlapping each other on the X axis, click a "Change display" button and either change the font size on the "Style" tab, or check the boxes marked "Auto" for "Ticks" and "Labels" on the "X axis" tab. Be sure to check both boxes, or the labels will not be accurate.  Changing the labels to "Auto" cannot be undone, and you will lose the detailed info about what values were actually used for the histogram bins.'')';
    uicontrol('Parent',FigureHandle, ...
        'Unit',StdUnit, ...
        'BackgroundColor',[0.65 0.65 0.85], ...
        'CallBack', Button7Callback, ...
        'Position',PointsPerPixel*[1 NewHeight-25 13 22], ...
        'Units','Normalized',...
        'String','?', ...
        'Style','pushbutton', ...
        'FontSize',handles.Current.FontSize);
    %%% Hide every other label button.
    Button8Callback = 'tempData = get(gcbo,''UserData'');PlotBinLocations = get(gca,''XTick''); AxesHandles = findobj(gcf, ''Type'', ''axes'');XTickLabels = get(gca,''XTickLabel'');if ceil(length(PlotBinLocations)/2) ~= length(PlotBinLocations)/2, PlotBinLocations(length(PlotBinLocations)) = []; XTickLabels(length(XTickLabels)) = []; end; PlotBinLocations2 = reshape(PlotBinLocations,2,[]); XTickLabels2 = reshape(XTickLabels,2,[]); set(AxesHandles,''XTick'',PlotBinLocations2(1,:)); set(AxesHandles,''XTickLabel'',XTickLabels2(1,:));tempData.HideOption = 1;set(tempData.HideHandle,''UserData'',tempData);set(tempData.DecimalHandle,''UserData'',tempData);clear';
    Button8 = uicontrol('Parent',FigureHandle, ...
        'Unit',StdUnit, ...
        'BackgroundColor',[0.65 0.65 0.85], ...
        'CallBack',Button8Callback, ...
        'Position',PointsPerPixel*[395 NewHeight-60 50 22], ...
        'Units','Normalized',...
        'String','Fewer',...
        'Style','pushbutton',...
        'UserData',0,...
        'FontSize',handles.Current.FontSize);
    %%% Decimal places X axis labels.
    Button9Callback = 'tempData = get(gcbo,''UserData'');HideOption = tempData.HideOption;FigureSettings = get(gca,''UserData''); PlotBinLocations = FigureSettings{1};PreXTickLabels = FigureSettings{2};XTickLabels = PreXTickLabels(2:end-1); AxesHandles = findobj(gcf, ''Type'', ''axes''); NumberOfDecimals = inputdlg(''Enter the number of decimal places to display'',''Enter the number of decimal places'',1,{''0''}); NumberValues = cell2mat(XTickLabels); Command = [''%.'',num2str(NumberOfDecimals{1}),''f'']; NewNumberValues = num2str(NumberValues'',Command); NewNumberValuesPlusFirstLast = [PreXTickLabels(1); cellstr(NewNumberValues); PreXTickLabels(end)];if HideOption,if ceil(length(PlotBinLocations)/2) ~= length(PlotBinLocations)/2,PlotBinLocations(length(PlotBinLocations)) = [];NewNumberValuesPlusFirstLast(length(NewNumberValuesPlusFirstLast)) = [];end,PlotBinLocations2 = reshape(PlotBinLocations,2,[]);XTickLabels2 = reshape(NewNumberValuesPlusFirstLast,2,[]);set(AxesHandles,''XTickLabel'',XTickLabels2);set(AxesHandles,''XTick'',PlotBinLocations);else,set(AxesHandles,''XTickLabel'',NewNumberValuesPlusFirstLast);set(AxesHandles,''XTick'',PlotBinLocations);end,tempData.DecimalOption = 1;set(tempData.HideHandle,''UserData'',tempData);set(tempData.DecimalHandle,''UserData'',tempData);clear, drawnow';
    Button9 = uicontrol('Parent',FigureHandle, ...
        'Unit',StdUnit,...
        'BackgroundColor',[0.65 0.65 0.85], ...
        'CallBack',Button9Callback, ...
        'Position',PointsPerPixel*[450 NewHeight-60 50 22], ...
        'Units','Normalized',...
        'String','Decimals',...
        'Style','pushbutton',...
        'UserData',0,...
        'FontSize',handles.Current.FontSize);
    %%% Restore original X axis labels.
    Button10Callback = 'FigureSettings = get(gca,''UserData'');  PlotBinLocations = FigureSettings{1}; XTickLabels = FigureSettings{2}; AxesHandles = findobj(gcf, ''Type'', ''axes''); set(AxesHandles,''XTick'',PlotBinLocations); set(AxesHandles,''XTickLabel'',XTickLabels); clear';
    Button10 = uicontrol('Parent',FigureHandle, ...
        'Unit',StdUnit, ...
        'BackgroundColor',[0.65 0.65 0.85], ...
        'CallBack',Button10Callback, ...
        'Position',PointsPerPixel*[505 NewHeight-60 50 22], ...
        'Units','Normalized',...
        'String','Restore', ...
        'Style','pushbutton', ...
        'FontSize',handles.Current.FontSize);
    %%% This code sets the UserData for Button8, Button9, and Button10
    tempData.HideOption = 0;
    tempData.DecimalOption = 0;
    tempData.HideHandle = Button8;
    tempData.DecimalHandle = Button9;
    set(Button8,'UserData',tempData);
    set(Button9,'UserData',tempData);
    set(Button10,'UserData',tempData);
    %%% Puts the menu and tool bar in the figure window.
    set(FigureHandle,'toolbar','figure')

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Displays histogram data for compressed histograms %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif strcmp(CompressedHistogram,'yes') == 1 && strncmpi(ShowDisplay,'Y',1) == 1
    FinalHistogramData = FinalHistogramData';
    FigureHandle = CPfigure;
    set(FigureHandle,'Color',[1 1 1])
    if strcmpi(RowImageOrBin,'image') == 1
        imagesc(FinalHistogramData),
        AxisHandle = gca;
        set(get(AxisHandle,'XLabel'),'String',MeasurementToExtract)
        set(AxisHandle,'XTickLabel',XTickLabels)
        NewPlotBinLocations = 1:2:length(FinalHistogramData');
        set(AxisHandle,'XTick',NewPlotBinLocations)
    elseif strcmpi(RowImageOrBin,'bin') == 1
        imagesc(flipud(FinalHistogramData')),
        AxisHandle = gca;
        EveryNthLabel = 3;
        XTickLabels = flipud(XTickLabels');
        %%% Checks the spread of the data to decide whether to round
        %%% it off.
        if XTickLabels{2} - XTickLabels{end-1} > NumberOfBins
            YTickLabels{1} = XTickLabels{1};
            YTickLabels{length(XTickLabels)} = XTickLabels{end};
            for n = 2:length(XTickLabels) - 1
                YTickLabels{n} = round(XTickLabels{n});
            end
        else YTickLabels = XTickLabels;
        end
        YTickLabels = YTickLabels(1:EveryNthLabel:length(YTickLabels));
        set(AxisHandle,'YTickLabel',YTickLabels)
        NewPlotBinLocations = 1:EveryNthLabel:length(flipud(XTickLabels'));
        set(AxisHandle,'YTick',NewPlotBinLocations)
        set(AxisHandle,'XTick',0:100:size(FinalHistogramData,1))
    end
    NewColormap = 1 - colormap(gray);
    colormap(NewColormap),
    ColorbarHandle = colorbar,
    %%% Labels the colorbar's units.
    if strncmpi(NumberOrPercent,'P',1) == 1
        ylabel(ColorbarHandle, ['Percentage of ', ObjectTypename, ' in each image'])
    else ylabel(ColorbarHandle, ['Number of ', ObjectTypename])
    end
    set(FigureHandle,'UserData',FigureSettings)
    FontSize = handles.Current.FontSize;
    set(gca,'fontname','times','fontsize',FontSize)
    set(get(ColorbarHandle,'title'),'fontname','times','fontsize',FontSize+2)
    xlabel(gca,'Image number','Fontname','times','fontsize',FontSize+2)
    ylabel(gca,'Histogram bins','fontname','times','fontsize',FontSize+2)
    title(MeasurementToExtract,'Fontname','times','fontsize',FontSize+2)
end

function WriteHistToExcel(FileName, FirstImage, LastImage, XTickLabels,...
    FinalHistogramData, MeasurementToExtract, AdditionalInfoForTitle,...
    HistogramTitles, RowImageOrBin)
%{
Function to write histogram data to tab-separated-value (.tsv) file
for input to Excel.
FileName        Name of file to write
FirstImage      Index of first image (second index to FinalHistogramData)
LastImage       Index of first image (second index to FinalHistogramData)
XTickLabels     Cell array of labels for Histogram; 1st, last are text,
rest are numeric. Must have dimension no larger than first
dimension of FinalHistogramData (and should be the same).
FinalHistogramData  Histogram data, rows=bins, cols=images.
MeasurementToExtract    Text name of measurement, used in title
AdditionalInfoForTitle  Text, used in title
HistogramTitles Cell array of text image labels.
RowImageOrBin   String "image" or "bin" to indicate orientation for output:
image => each row is one image, bin => each row is one bin.
(Actually, first letter only is checked, case insensitively.)
%}
%%% Open the file and name it appropriately.

fid = fopen(FileName, 'wt');
if fid < 0
    h = errordlg(['Unable to open output file ',FileName,'.']);
    waitfor(h);
    return;
end
try
    if strncmpi(RowImageOrBin,'I',1)
        %%% Each row is an image
        %%% Write "Bins used" as the title of the first column.
        fwrite(fid, ['Bins used for ', MeasurementToExtract,AdditionalInfoForTitle,' ->'], 'char');
        %%% Should check that XTickLabels matches FinalHistogramData.
        %%% TODO: Should we trust this next statement?
        NumberBinsToWrite = size(XTickLabels(:),1);
        fwrite(fid, sprintf('\t'), 'char');
        fwrite(fid, XTickLabels{1}, 'char');
        for BinNum = 2:NumberBinsToWrite-1
            fwrite(fid, sprintf('\t%g',XTickLabels{BinNum}), 'char');
        end
        fwrite(fid, sprintf('\t'), 'char');
        fwrite(fid, XTickLabels{NumberBinsToWrite}, 'char');
        fwrite(fid, sprintf('\n'), 'char');

        WaitbarHandle = waitbar(0,'Writing the histogram data file...');
        %%% Cycles through the images, one per row
        NumImages = LastImage-FirstImage+1;
        for ImageNumber = FirstImage:LastImage
            %%% Write the HistogramTitle as a heading for the column.
            fwrite(fid, char(HistogramTitles{ImageNumber}), 'char');
            for BinNum = 1:NumberBinsToWrite
                fwrite(fid, sprintf('\t'), 'char');
                fwrite(fid, sprintf('\t%g', FinalHistogramData(BinNum,ImageNumber)), 'char');
            end
            fwrite(fid, sprintf('\n'), 'char');
            waitbar(ImageNumber/NumImages);
        end
        close(WaitbarHandle)
    elseif strncmpi(RowImageOrBin, 'B', 1)
        %%% Each row is an bin
        %%% Write "Bins used" as the title of the first column.
        fwrite(fid, ['Bins used for ', MeasurementToExtract,AdditionalInfoForTitle], 'char');
        %%% Tab to the second column.
        fwrite(fid, sprintf('\t'), 'char');

        %%% Cycles through the remaining columns, one column per
        %%% image.
        for ImageNumber = FirstImage:LastImage
            %%% Write the HistogramTitle as a heading for the column.
            fwrite(fid, char(HistogramTitles{ImageNumber}), 'char');
            %%% Tab to the next column.
            fwrite(fid, sprintf('\t'), 'char');
        end
        %%% Return, to the next row.
        fwrite(fid, sprintf('\n'), 'char');

        WaitbarHandle = waitbar(0,'Writing the histogram data file...');
        NumberBinsToWrite = size(XTickLabels(:),1);
        %%% Writes the first X Tick Label (which is a string) in the first
        %%% column.
        fwrite(fid, XTickLabels{1}, 'char');
        %%% Tab to the second column.
        fwrite(fid, sprintf('\t'), 'char');
        for ImageNumber = FirstImage:LastImage
            %%% Writes the first FinalHistogramData in the second column and tab.
            fwrite(fid, sprintf('%g\t', FinalHistogramData(1,ImageNumber)), 'char');
        end
        %%% Return to the next row.
        fwrite(fid, sprintf('\n'), 'char');

        %%% Writes all the middle values.
        for i = 2:NumberBinsToWrite-1
            %%% Writes the XTickLabels (which are numbers) in
            %%% the remaining columns.
            fwrite(fid, sprintf('%g\t', XTickLabels{i}), 'char');
            for ImageNumber = FirstImage:LastImage
                %%% Writes the FinalHistogramData in the remaining
                %%% columns, tabbing after each one.
                fwrite(fid, sprintf('%g\t', FinalHistogramData(i,ImageNumber)), 'char');
            end
            %%% Return, to the next row.
            fwrite(fid, sprintf('\n'), 'char');
            waitbar(i/NumberBinsToWrite)
        end
        %%% Writes the last value.
        if NumberBinsToWrite ~= 1
            %%% Writes the last PlotBinLocations value (which is a string) in the first
            %%% column.
            fwrite(fid, XTickLabels{NumberBinsToWrite}, 'char');
            %%% Tab to the second column.
            fwrite(fid, sprintf('\t'), 'char');
            %%% Writes the last FinalHistogramData in the remaining columns.
            for ImageNumber = FirstImage:LastImage
                %%% Writes the FinalHistogramData in the remaining
                %%% columns, tabbing after each one.
                fwrite(fid, sprintf('%g\t', FinalHistogramData(NumberBinsToWrite,ImageNumber)), 'char');
            end
            %%% Return, to the next row.
            fwrite(fid, sprintf('\n'), 'char');
        end
        close(WaitbarHandle)
    else
        h = errordlg('Neither "image" nor "bin" selected, ',FileName,' will be empty.');
        waitfor(h);
    end
catch
    h = errordlg(['Problem occurred while writing to ',FileName,'. File is incomplete.']);
    waitfor(h);
end
%%% Close the file
try
    fclose(fid);
    h = CPhelpdlg(['The file ', FileName, ' has been written to the directory where the raw measurements file is located.']);
    waitfor(h)
catch
    h = errordlg(['Unable to close file ',FileName,'.']);
    waitfor(h);
    return;
end