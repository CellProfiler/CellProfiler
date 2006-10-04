function Histogram(handles)

% Help for the Histogram tool:
% Category: Data Tools
%
% SHORT DESCRIPTION:
% Displays a histogram of individual object measurements.
% *************************************************************************
% Note: this tool is beta-version and has not been thoroughly checked.
%
% The object measurements can be displayed in histogram format using this
% tool.  As prompted, select the output file containing the measurements,
% then choose the measurement parameter to be displayed, and the sample
% information label. It may take some time to then process the data.
%
% SETTINGS:
%
% * Which images' measurements to display or export - To display data from
% only one image, enter that image's number as both the first and last
% sample)
%
% * The number of bins to be used
%
% * Whether you want the histogram bins to contain the actual numbers of
% objects in the bin or the percentage of objects in the bin
%
% * How to determine the threshold values for the leftmost and rightmost
% bins - on the Measurement axis (e.g. Area of Nuclei). For the leftmost
% bin, any measurements less than the threshold will be combined in the
% leftmost bin. For the rightmost bin, any measurements greater than or
% equal to the thresholdd will be combined in the rightmost bin. Choosing
% "Min/Max value found" will instruct CellProfiler to determine the
% threshold values. Choosing "Other" will allow you to enter your custom
% threshold values.
%
% * Whether you want to calculate histogram data only for objects meeting a
% threshold in a measurement - If you choose other than "None", you can
% specify the type of threshold to use, and the threshold value.
%
% * Whether you want to combine all the objects' data to be displayed in a
% single (cumulative) histogram or in separate histograms
%
% * Whether the X axis will be the "Measurements" axis (e.g. Area of
% Nuclei) or the "Number of objects in bin" axis. The default for the X
% axis is "Measurements". By choosing "Number of objects in bin", you are
% essentially flipping the axes. Flipping is possible for both bar and
% line graphs, but not area graphs because there is no function that will
% work. If you attempt to flip an area graph, you will get a warning
% message, and the display will be a normal unflipped area graph.
%
% * For multiple histograms, whether you want the "Number of objects" axis
% to be absolute (the same for all histograms) or relative (scaled to fit
% the maximum value for that sample)
%
% * Whether you want the axis to be log scale
%
% * The style of the graph: bar, line, area, or heatmap
%
% * The color that the inital plot should be
%
% * Whether you want to display the histograms (Impractical when exporting
% large amounts of data).
%
% * Whether you want to export the data - tab-delimited format, which can
% be opened in Excel. When entering the filename, use the extension ".xls"
% so it can be opened easily in Excel.
%
% * Whether you want each row in the exported histogram or heatmap to
% contain an image or a bin
%
%
% NOTES:
%
% Measurement axis labels for histograms: Typically, the measurement axis
% labels will be too crowded.  This default state is shown because you
% might want to know the exact values that were used for the histogram
% bins.  The actual numbers can be viewed by clicking the 'This window'
% button under 'Change plots' and looking at the numbers listed under
% 'Labels'.  To change the measurement axis labels, you can click 'Fewer'
% in the main histogram window, or you can click a button under 'Change
% plots' and either change the font size on the 'Style' tab, or check
% the boxes marked 'Auto' for 'Ticks' and 'Labels' on the 'X (or Y) axis'
% tab. Be sure to check both boxes, or the labels will not be
% accurate. To revert to the original labels, click 'Restore' in the
% main histogram window, but beware that this function does not work
% when more than one histogram window is open at once, because the
% most recently produced histogram's labels will be used for everything.
%
% Change plots/change bars buttons: These buttons allow you to change
% properties of the plots or the bars within the plots for either
% every plot in the window ('This window'), the current plot only
% ('Current'), or every plot in every open window ('All windows').
% This includes colors, axis limits and other properties.
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
% See also PlotMeasurement data tool and DisplayHistogram and
% DisplayImageHistogram modules.


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

try
    %%% Ask the user to choose the file from which to extract
    %%% measurements. The window opens in the default output directory.
    [RawFileName, RawPathname] = uigetfile(fullfile(handles.Current.DefaultOutputDirectory,'.','*.mat'),'Select the raw measurements file');
    %%% Allows canceling.
    if RawFileName == 0
        return
    end
    load(fullfile(RawPathname, RawFileName));

    try FontSize = handles.Preferences.FontSize;
        %%% We used to store the font size in Current, so this line makes old
        %%% output files compatible. Shouldn't be necessary with any files made
        %%% after November 15th, 2006.
    catch FontSize = handles.Current.FontSize;
    end

    %%% Call the function CPgetfeature(), which opens a series of list dialogs and
    %%% lets the user choose a feature. The feature can be identified via 'ObjectTypename',
    %%% 'FeatureType' and 'FeatureNo'.
    try
        [ObjectTypename,FeatureType,FeatureNo] = CPgetfeature(handles);
    catch
        ErrorMessage = lasterr;
        CPerrordlg(['An error occurred in the Histogram Data Tool. ' ErrorMessage(30:end)]);
        return
    end
    if isempty(ObjectTypename),return,end
    MeasurementToExtract = [handles.Measurements.(ObjectTypename).([FeatureType,'Features']){FeatureNo},' of ', ObjectTypename];

    %%% Put the measurements for this feature in a cell array, one
    %%% cell for each cycle.
    tmp = handles.Measurements.(ObjectTypename).(FeatureType);
    ImageSets = length(tmp);
    Measurements = cell(length(tmp),1);
    for k = 1:ImageSets
        if length(tmp{k}) >= FeatureNo
            Measurements{k} = tmp{k}(:,FeatureNo);
        end
    end


    %%% Determines whether any sample info has been loaded.  If sample
    %%% info is present, the fieldnames for those are extracted.
    ImportedFieldnames = fieldnames(handles.Measurements.Image);

    %%% Finds fields in ImportedFieldnames that contain the string 'Description'
    %%% which serves as a tag (created by LoadText.m) that the file was imported
    %%% using the LoadText module or the AddData data tool. For each field containing
    %%% 'Description' (e.g. 'filenameDescription'), the substring preceding 'Description'
    %%% is the actual file name (e.g. 'filename'). This file name is also itself
    %%% a separate field in ImportedFieldnames. testmat is populated with these
    %%% imported file names.
    testmat=[];
    for index=1:length(ImportedFieldnames)
        str=ImportedFieldnames{index};
        indexD=strfind(str, 'Description');
        if ~isempty(indexD)
            strtest=str(1:indexD-1);
            equalstr=strtest;
            %%% The testmat matrix cannot be populated unless all strings are
            %%% the same length. All strings added are made to be 50 char long
            %%% by appending trailing spaces to the original file name string.
            for strind=1:50-length(strtest)
                equalstr=[equalstr, ' '];
            end
            testmat=[testmat;equalstr];
        end
    end

    %%% testmat is converted to a cell array of strings to be the same format
    %%% as ImportedFieldnames, removing all trailing spaces in each string
    if ~isempty(testmat)
        testmat=cellstr(testmat);
    end

    %%% Creates a boolean matrix Importedmat in which the 1's mark the location
    %%% in ImportedFieldnames where an imported file name exists.
    Importedmat=[];
    for index1=1:length(ImportedFieldnames)
        fieldmatch=0;
        for index2=1:length(testmat)
            if strcmp(ImportedFieldnames{index1},testmat{index2})
                fieldmatch=1;
            end
        end
        if fieldmatch
            Importedmat=[Importedmat;1];
        else
            Importedmat=[Importedmat;0];
        end
    end


    ImportedFieldnames = ImportedFieldnames(Importedmat == 1 | strcmp(ImportedFieldnames,'FileNames') == 1);
    if ~isempty(ImportedFieldnames)
        %%% Allows the user to select a heading from the list.
        [Selection, ok] = CPlistdlg('ListString',ImportedFieldnames, 'ListSize', [300 400],...
            'Name','Select sample info',...
            'PromptString','Choose the sample info with which to label each histogram.','CancelString','Cancel',...
            'SelectionMode','single');
        if ok ~= 0
            HeadingName = char(ImportedFieldnames(Selection));
            try SampleNames = handles.Measurements.Image.(HeadingName);
                if iscell(SampleNames{1})
                    CellArray=SampleNames;
                    %%% prevents displaying entire cell on histogram title,
                    %%% only displays first entry in each cell
                    for count=1:length(CellArray)
                        SampleNames{count}=CellArray{count}{1};
                    end
                end
            catch SampleNames = handles.Pipeline.(HeadingName);
            end
        else
            return
        end
    end

    %%% Opens a window that lets the user choose histogram settings
    %%% This function returns a UserInput structure with the
    %%% information required to carry out the calculations.
    global UserInput
    try UserInput = UserInputWindow(handles,RawFileName,UserInput);
    catch CPerrordlg(lasterr)
        return
    end

    % If Cancel button pressed, return
    if ~isfield(UserInput,'FirstSample')
        return
    end

    NumberOfImages = UserInput.LastSample - UserInput.FirstSample + 1;

    switch UserInput.Color
        case 'Blue'
            GraphColor='b';
        case 'Red'
            GraphColor='r';
        case 'Green'
            GraphColor='g';
        case 'Yellow'
            GraphColor='y';
        case 'Magenta'
            GraphColor='m';
        case 'Cyan'
            GraphColor='c';
        case 'Black'
            GraphColor='k';
        case 'White'
            GraphColor='w';
        otherwise
            GraphColor=[.7 .7 .9];
    end


    %%% If the user selected to threshold histogram data, the measurements are thresholded on some other measurement.
    if strcmp(UserInput.Logical,'None') ~= 1
        [ObjectTypename,FeatureType,FeatureNo] = CPgetfeature(handles);
        % There's no need to nest the call to CPgetfeature again, because
        % the only errors that can occur in CPgetfeature happen when
        % handles is faulty, but CPgetfeature was called before and handles
        % hasn't been modified. However, an empty check for ObjectTypename
        % is needed. This would happen if the user clicked Cancel.
        if isempty(ObjectTypename),return,end
        MeasurementToThresholdValueOnName = handles.Measurements.(ObjectTypename).([FeatureType,'Features'])(FeatureNo);
        tmp = handles.Measurements.(ObjectTypename).(FeatureType);
        MeasurementToThresholdValueOn = cell(length(tmp),1);
        for k = 1:length(tmp)
            MeasurementToThresholdValueOn{k} = tmp{k}(:,FeatureNo);
        end
    end

    %%% Calculates the default bin size and range based on all the data.
    SelectedMeasurementsCellArray = Measurements(UserInput.FirstSample:UserInput.LastSample);
    SelectedMeasurementsMatrix = cell2mat(SelectedMeasurementsCellArray(:));

    try
        [BinLocations,PlotBinLocations,XTickLabels] = CPhistbins(SelectedMeasurementsMatrix,UserInput.NumBins,UserInput.LeftBin,UserInput.LeftVal,UserInput.RightBin,UserInput.RightVal,UserInput.Log,'Count');
    catch
        ErrorMessage = lasterr;
        CPerrordlg(['An error occurred in the Histogram Data Tool. ' ErrorMessage(28:end)])
        return
    end

    % Saves this info in a variable, FigureSettings, which
    % will be stored later with the figure.
    FigureSettings{1} = PlotBinLocations;
    FigureSettings{2} = XTickLabels;


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Calculates histogram data for cumulative histogram %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if strcmp(UserInput.Combine, 'Yes') == 1
        OutputMeasurements{1,1} = SelectedMeasurementsMatrix;
        %%% Retrieves the measurements to threshold on, if requested.
        if strcmp(UserInput.Logical,'None') ~= 1
            SelectMeasurementsCellArray = MeasurementToThresholdValueOn(UserInput.FirstSample:UserInput.LastSample);
            OutputMeasurements{1,2} = cell2mat(SelectMeasurementsCellArray(:));
            AdditionalInfoForTitle = [' for objects where ', MeasurementToThresholdValueOnName, UserInput.Logical, num2str(UserInput.ThresholdVal)];
        else AdditionalInfoForTitle = [];
        end
        %%% Applies the specified ThresholdValue and gives a cell
        %%% array as output.
        if strcmp(UserInput.Logical,'None') == 1
            %%% If the user selected None, the measurements are not
            %%% altered.
        elseif strcmp(UserInput.Logical,'>') == 1
            OutputMeasurements{1,1} = OutputMeasurements{1,1}(OutputMeasurements{1,2} > UserInput.ThresholdVal);
        elseif strcmp(UserInput.Logical,'>=') == 1
            OutputMeasurements{1,1} = OutputMeasurements{1,1}(OutputMeasurements{1,2} >= UserInput.ThresholdVal);
        elseif strcmp(UserInput.Logical,'<') == 1
            OutputMeasurements{1,1} = OutputMeasurements{1,1}(OutputMeasurements{1,2} < UserInput.ThresholdVal);
        elseif strcmp(UserInput.Logical,'<=') == 1
            OutputMeasurements{1,1} = OutputMeasurements{1,1}(OutputMeasurements{1,2} <= UserInput.ThresholdVal);
        else
            OutputMeasurements{1,1} = OutputMeasurements{1,1}(OutputMeasurements{1,2} == UserInput.ThresholdVal);
        end

        if isempty(OutputMeasurements{1,1}) == 1
            HistogramData = [];
        else HistogramData = histc(OutputMeasurements{1,1},BinLocations);
        end

        %%% Deletes the last value of HistogramData, which is
        %%% always a zero (because it's the number of values
        %%% that match + inf).
        HistogramData(end) = [];
        FinalHistogramData(:,1) = HistogramData;
        HistogramTitles{1} = ['Histogram of data from Image #', num2str(UserInput.FirstSample), ' to #', num2str(UserInput.LastSample)];
        UserInput.FirstSample = 1;
        UserInput.LastSample = 1;
        NumberOfImages = 1;
        
        %%% Might make this optional by adding a Yes/No to the User window.
        CalcStatistics = 'Yes';
        if strcmpi(CalcStatistics,'Yes')
            %%% Calculates a variety of statistics for the population.
            Mean = mean(OutputMeasurements{1,1});
            Median = median(OutputMeasurements{1,1});
            %%% pObject is the percentage of image covered by objects. We will
            %%% set to 50% for now
            pObject = '50';
            [handles,Otsu] = CPthreshold(handles,'Otsu Global',pObject,-Inf,Inf,1,OutputMeasurements{1,1},'HistogramData','Histogram Data tool');
            [handles,MoG] = CPthreshold(handles,'MoG Global',pObject,-Inf,Inf,1,OutputMeasurements{1,1},'HistogramData','Histogram Data tool');

            %%% Display the results.
            StatsFigHandle = CPfigure;
            CPresizefigure([],'NarrowText',StatsFigHandle)
            %%% Removes RGB buttons from the CPfigure.
            delete(findobj('parent',StatsFigHandle,'string','R'));
            delete(findobj('parent',StatsFigHandle,'string','G'));
            delete(findobj('parent',StatsFigHandle,'string','B'));
            %%% Prints out the text in the display window.
            FontSize = handles.Preferences.FontSize;
            DistanceBetweenTextEntries = .1;
            CurrentEntryPosition = .92;
            
            uicontrol(StatsFigHandle,'style','text','units','normalized',...
                'position', [0.02 CurrentEntryPosition 1 0.07], 'HorizontalAlignment','left',...
                'Backgroundcolor',[.7 .7 .9],'fontname','Helvetica',...
                'fontsize',FontSize,'fontweight','bold',...
                'string', 'Statistics for the histogrammed population');
            CurrentEntryPosition = CurrentEntryPosition - DistanceBetweenTextEntries;
            uicontrol(StatsFigHandle,'style','text','units','normalized',...
                'position', [0.02 CurrentEntryPosition 1 0.06], 'HorizontalAlignment','left',...
                'Backgroundcolor',[.7 .7 .9],'fontname','Helvetica',...
                'fontsize',FontSize,'string', sprintf('Mean: %g',Mean));
            CurrentEntryPosition = CurrentEntryPosition - DistanceBetweenTextEntries;
            uicontrol(StatsFigHandle,'style','text','units','normalized',...
                'position', [0.02 CurrentEntryPosition 1 0.06], 'HorizontalAlignment','left',...
                'Backgroundcolor',[.7 .7 .9],'fontname','Helvetica',...
                'fontsize',FontSize,'string', sprintf('Median: %g',Median));
            CurrentEntryPosition = CurrentEntryPosition - DistanceBetweenTextEntries;
            uicontrol(StatsFigHandle,'style','text','units','normalized',...
                'position', [0.02 CurrentEntryPosition 1 0.06], 'HorizontalAlignment','left',...
                'Backgroundcolor',[.7 .7 .9],'fontname','Helvetica',...
                'fontsize',FontSize,'string', sprintf('Otsu threshold: %g',Otsu));
            CurrentEntryPosition = CurrentEntryPosition - DistanceBetweenTextEntries;
            uicontrol(StatsFigHandle,'style','text','units','normalized',...
                'position', [0.02 CurrentEntryPosition 1 0.06], 'HorizontalAlignment','left',...
                'Backgroundcolor',[.7 .7 .9],'fontname','Helvetica',...
                'fontsize',FontSize,'string', sprintf('MoG: %g',MoG));
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% Calculates histogram data for non-cumulative histogram %%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    else
        %%% Preallocates the variable ListOfMeasurements.
        ListOfMeasurements{NumberOfImages,1} = Measurements{UserInput.LastSample};
        if strcmpi(UserInput.Logical,'None') ~= 1
            ListOfMeasurements{NumberOfImages,2} = MeasurementToThresholdValueOn{UserInput.LastSample};
            AdditionalInfoForTitle = [' for objects where ', MeasurementToThresholdValueOnName, UserInput.Logical, num2str(UserInput.ThresholdVal)];
        else AdditionalInfoForTitle = [];
        end
        CompressedImageNumber = 1;
        OutputMeasurements = cell(size(NumberOfImages,1),1);
        FinalHistogramData = [];
        for ImageNumber = UserInput.FirstSample:UserInput.LastSample
            ListOfMeasurements{CompressedImageNumber,1} = Measurements{ImageNumber};
            if strcmp(UserInput.Logical,'None') ~= 1
                ListOfMeasurements{CompressedImageNumber,2} = MeasurementToThresholdValueOn{ImageNumber};
            end
            %%% Applies the specified ThresholdValue and gives a cell
            %%% array as output.
            if strcmpi(UserInput.Logical,'None') == 1
                %%% If the user selected None, the measurements are not
                %%% altered.
                OutputMeasurements = ListOfMeasurements;
            elseif strcmp(UserInput.Logical,'>') == 1
                OutputMeasurements{CompressedImageNumber,1} = ListOfMeasurements{CompressedImageNumber,1}(ListOfMeasurements{CompressedImageNumber,2} > UserInput.ThresholdVal);
            elseif strcmp(UserInput.Logical,'>=') == 1
                OutputMeasurements{CompressedImageNumber,1} = ListOfMeasurements{CompressedImageNumber,1}(ListOfMeasurements{CompressedImageNumber,2} >= UserInput.ThresholdVal);
            elseif strcmp(UserInput.Logical,'<') == 1
                OutputMeasurements{CompressedImageNumber,1} = ListOfMeasurements{CompressedImageNumber,1}(ListOfMeasurements{CompressedImageNumber,2} < UserInput.ThresholdVal);
            elseif strcmp(UserInput.Logical,'<=') == 1
                OutputMeasurements{CompressedImageNumber,1} = ListOfMeasurements{CompressedImageNumber,1}(ListOfMeasurements{CompressedImageNumber,2} <= UserInput.ThresholdVal);
            else
                OutputMeasurements{CompressedImageNumber,1} = ListOfMeasurements{CompressedImageNumber,1}(ListOfMeasurements{CompressedImageNumber,2} == UserInput.ThresholdVal);
            end
            if isempty(OutputMeasurements{CompressedImageNumber}) == 1
                HistogramData = [];
            else HistogramData = histc(OutputMeasurements{CompressedImageNumber},BinLocations);
            end

            %%% Deletes the last value of HistogramData, which
            %%% is always a zero (because it's the number of values that match
            %%% + inf).
            if ~isempty(HistogramData)
                HistogramData(end) = [];
            end
            if ~isempty(HistogramData)
                FinalHistogramData(:,ImageNumber) = HistogramData;
            end
            if exist('SampleNames','var') == 1
                try %#ok Ignore MLint
                    SampleName = SampleNames{ImageNumber};
                end;
                HistogramTitles{ImageNumber} = ['#', num2str(ImageNumber), ': ' , SampleName];
            else HistogramTitles{ImageNumber} = ['Image #', num2str(ImageNumber)];
            end
            %%% Increments the CompressedImageNumber.
            CompressedImageNumber = CompressedImageNumber + 1;
        end
    end

    if strcmp(UserInput.BinVar,'Percentages')
        for i = 1: size(FinalHistogramData,2)
            SumForThatColumn = sum(FinalHistogramData(:,i));
            FinalHistogramData(:,i) = 100*FinalHistogramData(:,i)/SumForThatColumn;
        end
    end

    %%% Saves this info in a variable, FigureSettings, which
    %%% will be stored later with the figure.
    FigureSettings{3} = FinalHistogramData;


    %%% Saves the data to an excel file if desired.
    if strcmp(UserInput.ExportHist,'Yes') == 1
        WriteHistToExcel([RawPathname,UserInput.ExportFile], UserInput.FirstSample, UserInput.LastSample, XTickLabels,...
            FinalHistogramData, MeasurementToExtract, AdditionalInfoForTitle,...
            HistogramTitles, UserInput.EachRow);
    end



    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Displays histogram data for non-heatmap graphs %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    VersionCheck = version;


    if ~strcmp(UserInput.Style,'Heatmap') && strcmp(UserInput.Display,'Yes')
        %%% Calculates the square root in order to determine the dimensions for the
        %%% display window.
        SquareRoot = sqrt(NumberOfImages);
        %%% Converts the result to an integer.
        NumberDisplayRows = fix(SquareRoot);
        NumberDisplayColumns = ceil((NumberOfImages)/NumberDisplayRows);
        %%% Acquires basic screen info for making buttons in the
        %%% display window.
        StdUnit = 'point';
        PointsPerPixel = 72/get(0,'ScreenPixelsPerInch');
        %%% Creates the display window.
        FigureHandle = CPfigure;
        set(FigureHandle, 'Name',[UserInput.Style,' graph for ',MeasurementToExtract]);

        Increment = 0;
        for ImageNumber = UserInput.FirstSample:UserInput.LastSample
            Increment = Increment + 1;
            h = subplot(NumberDisplayRows,NumberDisplayColumns,Increment);
            if strcmp(UserInput.Style,'Bar')
                if ~isempty(FinalHistogramData) && strcmp(UserInput.Xaxis,'Number of objects in bin')
                    k=barh('v6',PlotBinLocations,FinalHistogramData(:,ImageNumber), 1);
                elseif ~isempty(FinalHistogramData) && strcmp(UserInput.Xaxis,'Measurements')
                    k=bar('v6',PlotBinLocations,FinalHistogramData(:,ImageNumber), 1);
                end
                set(k,'FaceColor',GraphColor);
            elseif strcmp(UserInput.Style,'Area')
                if ~isempty(FinalHistogramData) %Since there is no function to flip axes of area plot, will display normal graph regardless of whether user chooses to flip axes
                    k=area('v6',PlotBinLocations,FinalHistogramData(:,ImageNumber));
                end
                set(k,'FaceColor',GraphColor);
            else
                if ~isempty(FinalHistogramData) && strcmp(UserInput.Xaxis,'Number of objects in bin')
                    plot('v6',FinalHistogramData(:,ImageNumber),PlotBinLocations,'LineWidth',2,'color',GraphColor);
                elseif ~isempty(FinalHistogramData) && strcmp(UserInput.Xaxis,'Measurements')
                    plot('v6',PlotBinLocations,FinalHistogramData(:,ImageNumber),'LineWidth',2,'color',GraphColor);
                end
            end
            if strcmp(UserInput.Xaxis,'Number of objects in bin') && ~strcmp(UserInput.Style,'Area')
                if Increment == 1
                    set(get(h,'YLabel'),'String',cat(2,MeasurementToExtract,AdditionalInfoForTitle))
                end
                set(h,'YTickLabel',XTickLabels)
                set(h,'YTick',PlotBinLocations)
                set(gcf, 'Tag', 'AxesFlipped')
                set(gca,'Tag','BarTag','ActivePositionProperty','Position')
                % Fix underscores in HistogramTitles
                AdjustedHistogramTitle = strrep(HistogramTitles{ImageNumber},'_','\_');
                title(AdjustedHistogramTitle)
                if strcmp(UserInput.BinVar,'Actual numbers')
                    set(get(h,'XLabel'),'String','Number of objects')
                else
                    set(get(h,'XLabel'),'String','Percentage of objects')
                end
                axis tight
            else
                set(get(h,'XLabel'),'String',cat(2,MeasurementToExtract,AdditionalInfoForTitle))
                set(h,'XTickLabel',XTickLabels)
                set(h,'XTick',PlotBinLocations)
                set(gcf, 'Tag', 'AxesNotFlipped')
                set(gca,'Tag','BarTag','ActivePositionProperty','Position')
                % Fix underscores in HistogramTitles
                AdjustedHistogramTitle = strrep(HistogramTitles{ImageNumber},'_','\_');
                title(AdjustedHistogramTitle)
                if Increment == 1
                    if strcmp(UserInput.BinVar,'Actual numbers')
                        set(get(h,'YLabel'),'String','Number of objects')
                    else
                        set(get(h,'YLabel'),'String','Percentage of objects')
                    end
                end
                axis tight
            end
        end

        %%% Sets the Y axis scale to be absolute or relative.
        AxesHandles = findobj('Parent', FigureHandle,'Type','axes');
        set(AxesHandles,'UserData',FigureSettings)
        if strcmp(UserInput.RelAbs, 'Relative number of objects') == 1
            %%% Automatically stretches the x data to fill the plot
            %%% area.
            axis(AxesHandles, 'tight')
            %%% Automatically stretches the y data to fill the plot
            %%% area, except that "auto" leaves a bit more buffer
            %%% white space around the data.
            axis(AxesHandles, 'auto y')
        elseif strcmp(UserInput.RelAbs, 'Absolute number of objects') == 1
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
        NewHeight = FigurePosition(4)+30;
        NewFigurePosition = [NewFigurePosition(1) NewFigurePosition(2) NewFigurePosition(3) NewHeight];
        set(FigureHandle,'Position',NewFigurePosition)
        set(AxesHandles,'Units','normalized');

        %%% These callbacks control what happens when display
        %%% buttons are pressed within the histogram display
        %%% window.
        if strcmp(computer,'MAC') && str2double(VersionCheck(1:3)) < 7.1
            Button1Callback = 'FigureHandle = gcf; AxesHandles = findobj(''Parent'', FigureHandle, ''Type'', ''axes''); drawnow;';
            Button2Callback = '';
            Button3Callback = 'FigureHandles = findobj(''Parent'', 0); AxesHandles = findobj(FigureHandles, ''Type'', ''axes''); axis(AxesHandles, ''manual''); drawnow;';
            Button4Callback = 'FigureHandle = gcf; PatchHandles = findobj(FigureHandle, ''Type'', ''patch''); drawnow;';
            Button5Callback = 'AxisHandle = gca; PatchHandles = findobj(''Parent'', AxisHandle, ''Type'', ''patch''); drawnow;';
            Button6Callback = 'FigureHandles = findobj(''Parent'', 0); PatchHandles = findobj(FigureHandles, ''Type'', ''patch''); drawnow;';
        else
            Button1Callback = 'FigureHandle = gcf; AxesHandles = findobj(''Parent'', FigureHandle, ''Type'', ''axes''); try, propedit(AxesHandles,''v6''), catch, CPmsgbox(''A bug in MATLAB is preventing this function from working. Service Request #1-RR6M1''), end; drawnow;';
            Button2Callback = 'propedit(gca,''v6''); drawnow;';
            Button3Callback = 'FigureHandles = findobj(''Parent'', 0); AxesHandles = findobj(FigureHandles, ''Type'', ''axes''); axis(AxesHandles, ''manual''); try, propedit(AxesHandles,''v6''), catch, CPmsgbox(''A bug in MATLAB is preventing this function from working. Service Request #1-RR6M1''), end, drawnow;';
            if strcmp(UserInput.Style,'Line')
                Button4Callback = 'FigureHandle = gcf; PatchHandles = findobj(FigureHandle, ''Type'', ''line''); try, propedit(PatchHandles,''v6''), catch, CPmsgbox(''A bug in MATLAB is preventing this function from working. Service Request #1-RR6M1''), end; drawnow;';
                Button5Callback = 'AxisHandle = gca; PatchHandles = findobj(''Parent'', AxisHandle, ''Type'', ''line''); try, propedit(PatchHandles,''v6''), catch, CPmsgbox(''A bug in MATLAB is preventing this function from working. Service Request #1-RR6M1''), end; drawnow;';
                Button6Callback = 'FigureHandles = findobj(''Parent'', 0); PatchHandles = findobj(FigureHandles, ''Type'', ''line''); try, propedit(PatchHandles,''v6''), catch, CPmsgbox(''A bug in MATLAB is preventing this function from working. Service Request #1-RR6M1''), end; drawnow;';
            else
                Button4Callback = 'FigureHandle = gcf; PatchHandles = findobj(FigureHandle, ''Type'', ''patch''); try, propedit(PatchHandles,''v6''), catch, CPmsgbox(''A bug in MATLAB is preventing this function from working. Service Request #1-RR6M1''), end; drawnow;';
                Button5Callback = 'AxisHandle = gca; PatchHandles = findobj(''Parent'', AxisHandle, ''Type'', ''patch''); try, propedit(PatchHandles,''v6''), catch, CPmsgbox(''A bug in MATLAB is preventing this function from working. Service Request #1-RR6M1''), end; drawnow;';
                Button6Callback = 'FigureHandles = findobj(''Parent'', 0); PatchHandles = findobj(FigureHandles, ''Type'', ''patch''); try, propedit(PatchHandles,''v6''), catch, CPmsgbox(''A bug in MATLAB is preventing this function from working. Service Request #1-RR6M1''), end; drawnow;';
            end
        end

        if strcmp(UserInput.Xaxis,'Number of objects in bin') && ~strcmp(UserInput.Style,'Area')
            Button7Callback = 'tempData = get(gcf,''UserData'');AxesHandles = findobj(gcf,''Tag'',''BarTag'');PlotBinLocations = get(AxesHandles(1),''YTick'');XTickLabels = get(AxesHandles(1),''YTickLabel'');if ceil(length(PlotBinLocations)/2) ~= length(PlotBinLocations)/2,PlotBinLocations(length(PlotBinLocations)) = [];XTickLabels(length(XTickLabels)) = [];end;PlotBinLocations2 = reshape(PlotBinLocations,2,[]);XTickLabels2 = reshape(XTickLabels,2,[]);set(AxesHandles,''YTick'',PlotBinLocations2(1,:));set(AxesHandles,''YTickLabel'',XTickLabels2(1,:));tempData.HideOption = 1;set(gcf,''UserData'',tempData);';
            Button8Callback = 'tempData = get(gcf,''UserData'');HideOption = tempData.HideOption;FigureSettings = tempData.FigureSettings; PlotBinLocations = FigureSettings{1};PreXTickLabels = FigureSettings{2};XTickLabels = PreXTickLabels(2:end-1); AxesHandles = findobj(gcf,''Tag'',''BarTag''); NumberOfDecimals = inputdlg(''Enter the number of decimal places to display'',''Enter the number of decimal places'',1,{''0''}); if ~isempty(NumberOfDecimals), NumberValues = cell2mat(XTickLabels); Command = [''%.'',num2str(NumberOfDecimals{1}),''f'']; NewNumberValues = num2str(NumberValues'',Command); NewNumberValuesPlusFirstLast = [PreXTickLabels(1); cellstr(NewNumberValues); PreXTickLabels(end)];if HideOption,if ceil(length(PlotBinLocations)/2) ~= length(PlotBinLocations)/2,PlotBinLocations(length(PlotBinLocations)) = [];NewNumberValuesPlusFirstLast(length(NewNumberValuesPlusFirstLast)) = [];end,PlotBinLocations2 = reshape(PlotBinLocations,2,[]);XTickLabels2 = reshape(NewNumberValuesPlusFirstLast,2,[]);set(AxesHandles,''YTickLabel'',XTickLabels2);set(AxesHandles,''YTick'',PlotBinLocations);else,set(AxesHandles,''YTickLabel'',NewNumberValuesPlusFirstLast);set(AxesHandles,''YTick'',PlotBinLocations);end,tempData.DecimalOption = 1;set(gcf,''UserData'',tempData);drawnow, end;';
            Button9Callback = 'tempData = get(gcf,''UserData'');FigureSettings = tempData.FigureSettings;PlotBinLocations = FigureSettings{1}; XTickLabels = FigureSettings{2}; AxesHandles = findobj(gcf, ''Tag'', ''BarTag''); set(AxesHandles,''YTick'',PlotBinLocations); set(AxesHandles,''YTickLabel'',XTickLabels);';
        else
            Button7Callback = 'tempData = get(gcf,''UserData'');AxesHandles = findobj(gcf,''Tag'',''BarTag'');PlotBinLocations = get(AxesHandles(1),''XTick'');XTickLabels = get(AxesHandles(1),''XTickLabel'');if ceil(length(PlotBinLocations)/2) ~= length(PlotBinLocations)/2,PlotBinLocations(length(PlotBinLocations)) = [];XTickLabels(length(XTickLabels)) = [];end;PlotBinLocations2 = reshape(PlotBinLocations,2,[]);XTickLabels2 = reshape(XTickLabels,2,[]);set(AxesHandles,''XTick'',PlotBinLocations2(1,:));set(AxesHandles,''XTickLabel'',XTickLabels2(1,:));tempData.HideOption = 1;set(gcf,''UserData'',tempData);';
            Button8Callback = 'tempData = get(gcf,''UserData'');HideOption = tempData.HideOption;FigureSettings = tempData.FigureSettings; PlotBinLocations = FigureSettings{1};PreXTickLabels = FigureSettings{2};XTickLabels = PreXTickLabels(2:end-1); AxesHandles = findobj(gcf,''Tag'',''BarTag''); NumberOfDecimals = inputdlg(''Enter the number of decimal places to display'',''Enter the number of decimal places'',1,{''0''}); if ~isempty(NumberOfDecimals), NumberValues = cell2mat(XTickLabels); Command = [''%.'',num2str(NumberOfDecimals{1}),''f'']; NewNumberValues = num2str(NumberValues'',Command); NewNumberValuesPlusFirstLast = [PreXTickLabels(1); cellstr(NewNumberValues); PreXTickLabels(end)];if HideOption,if ceil(length(PlotBinLocations)/2) ~= length(PlotBinLocations)/2,PlotBinLocations(length(PlotBinLocations)) = [];NewNumberValuesPlusFirstLast(length(NewNumberValuesPlusFirstLast)) = [];end,PlotBinLocations2 = reshape(PlotBinLocations,2,[]);XTickLabels2 = reshape(NewNumberValuesPlusFirstLast,2,[]);set(AxesHandles,''XTickLabel'',XTickLabels2);set(AxesHandles,''XTick'',PlotBinLocations);else,set(AxesHandles,''XTickLabel'',NewNumberValuesPlusFirstLast);set(AxesHandles,''XTick'',PlotBinLocations);end,tempData.DecimalOption = 1;set(gcf,''UserData'',tempData);drawnow, end;';
            Button9Callback = 'tempData = get(gcf,''UserData'');FigureSettings = tempData.FigureSettings;PlotBinLocations = FigureSettings{1}; XTickLabels = FigureSettings{2}; AxesHandles = findobj(gcf, ''Tag'', ''BarTag''); set(AxesHandles,''XTick'',PlotBinLocations); set(AxesHandles,''XTickLabel'',XTickLabels);';
        end

        if strcmp(UserInput.Style,'Line')
            FunctionString = {'CHANGE DISPLAY:' 'Plots - This Window' 'Plots - Current' 'Plots - All Windows' 'Lines - This Window' 'Lines - Current' 'Lines - All Windows' 'Axis Labels - Fewer' 'Axis Labels - Decimals' 'Axis Labels - Restore'};
        else
            FunctionString = {'CHANGE DISPLAY:' 'Plots - This Window' 'Plots - Current' 'Plots - All Windows' 'Bars - This Window' 'Bars - Current' 'Bars - All Windows' 'Axis Labels - Fewer' 'Axis Labels - Decimals' 'Axis Labels - Restore'};
        end

        MyButtonCallback = ['Value=get(gcbo,''value'');switch Value,case 2,',Button1Callback,'case 3,',Button2Callback,'case 4,',Button3Callback,'case 5,',Button4Callback,'case 6,',Button5Callback,'case 7,',Button6Callback,'case 8,',Button7Callback,'case 9,',Button8Callback,'case 10,',Button9Callback,'end;clear AxesHandles AxisHandle FigureHandle FigureHandles PatchHandles Value Command FigureSettings HideOption NewNumberValues NewNumberValuesPlusFirstLast NumberOfDecimals NumberValues PlotBinLocations PlotBinLocations2 PreXTickLabels XTickLabels XTickLabels2 tempData;'];
        uicontrol('Parent',FigureHandle, ...
            'Unit',StdUnit, ...
            'BackgroundColor',[.7 .7 .9], ...
            'CallBack', MyButtonCallback, ...
            'Position',PointsPerPixel*[100 NewHeight-30 200 25], ...
            'Units','Normalized',...
            'String',FunctionString, ...
            'Style','popupmenu', ...
            'FontSize',FontSize);

        Button10Callback = 'CPmsgbox(''Histogram display info: (1) Data outside the range you specified to calculate histogram bins are added together and displayed in the first and last bars of the histogram.  (2) Only the display can be changed in this window, including axis limits.  The histogram bins themselves cannot be changed here because the data must be recalculated. (3) If a change you make using the "Change display" buttons does not seem to take effect in all of the desired windows, try pressing enter several times within that box, or look in the bottom of the Property Editor window that opens when you first press one of those buttons.  There may be a message describing why.  For example, you may need to deselect "Auto" before changing the limits of the axes. (4) The labels for each bar specify the low bound for that bin.  In other words, each bar includes data equal to or greater than the label, but less than the label on the bar to its right. (5) If the tick mark labels are overlapping each other on the X axis, click a "Change display" button and either change the font size on the "Style" tab, or check the boxes marked "Auto" for "Ticks" and "Labels" on the "X axis" tab. Be sure to check both boxes, or the labels will not be accurate.  Changing the labels to "Auto" cannot be undone, and you will lose the detailed info about what values were actually used for the histogram bins.'')';
        uicontrol('Parent',FigureHandle, ...
            'Unit',StdUnit, ...
            'BackgroundColor',[.7 .7 .9], ...
            'CallBack', Button10Callback, ...
            'Position',PointsPerPixel*[.5 NewHeight-25 11 22], ...
            'Units','Normalized',...
            'String','?', ...
            'Style','pushbutton', ...
            'FontSize',FontSize);

        %%% Button for adding control graph
        Button11Callback = [...
            'try,',...
            'CPcontrolhistogram(get(gcbo,''parent''),get(gcf,''UserData''),get(gcf,''Tag''));',...
            'catch,',...
            'ErrorMessage = lasterr;',...
            'CPerrordlg([''An error occurred in the Histogram Data Tool. '' ErrorMessage(36:end)]);',...
            'return;',...
            'end;'];
        %%% Button for exporting data to MATLAB
        if ~isdeployed      %button should only be shown in developer's version
            FunctionString = {'DATA:' 'Add Control Histogram' 'Export Data to MATLAB'};
            if strcmp(UserInput.Style,'Bar') || strcmp(UserInput.Style,'Area')
                Button12Callback = 'Bins = get(findobj(gcf,''type'',''patch''),''XData'');Data = get(findobj(gcf,''type'',''patch''),''YData''); CPmsgbox(''The data is now saved as the variables Bins and Data in the Matlab workspace.'');';
            else
                Button12Callback = 'Bins = get(findobj(gcf,''type'',''line''),''XData'');Data = get(findobj(gcf,''type'',''line''),''YData''); CPmsgbox(''The data is now saved as the variables Bins and Data in the Matlab workspace.'');';
            end
            MyButtonCallback = ['Value=get(gcbo,''value'');switch Value,case 2,',Button11Callback,'case 3,',Button12Callback,'end;clear Value;'];
        else
            FunctionString = {'DATA:' 'Add Control Histogram'};
            MyButtonCallback = ['Value=get(gcbo,''value'');switch Value,case 2,',Button11Callback,'end;clear Value;'];
        end

        uicontrol('Parent',FigureHandle, ...
            'Unit',StdUnit, ...
            'BackgroundColor',[.7 .7 .9], ...
            'CallBack', MyButtonCallback, ...
            'Position',PointsPerPixel*[310 NewHeight-30 200 25], ...
            'Units','Normalized',...
            'String',FunctionString, ...
            'Style','popupmenu', ...
            'FontSize',FontSize);

        %%% This code sets the UserData for the figure
        tempData.HideOption = 0;
        tempData.DecimalOption = 0;
        tempData.FigureSettings = FigureSettings;
        tempData.Logical = UserInput.Logical;
        tempData.ThresholdVal = UserInput.ThresholdVal;
        tempData.BinVar = UserInput.BinVar;
        tempData.BinLocations = BinLocations;
        tempData.handles = rmfield(handles,'Pipeline');
        tempData.Application = 'CellProfiler';
        set(FigureHandle,'UserData',tempData);
        %%% Puts the menu and tool bar in the figure window.
        set(FigureHandle,'toolbar','figure')

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% Displays histogram data for heatmaps %%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    elseif strcmp(UserInput.Style,'Heatmap') == 1 && strcmp(UserInput.Display,'Yes') == 1
        FinalHistogramData = FinalHistogramData';
        FigureHandle = CPfigure;
        set(FigureHandle,'Name', [UserInput.Style,' for ',MeasurementToExtract],'Color',[.7 .7 .9])
        heatmp=subplot(1,1,1);
        pos=get(heatmp, 'Position');
        set(heatmp, 'Position', [pos(1) pos(2) pos(3) pos(4)-.1]);
        if  XTickLabels{end-1} - XTickLabels{2} > UserInput.NumBins
            for n = 2:length(XTickLabels) - 1
                XTickLabels{n} = XTickLabels{n};
            end

        end

        if strcmp(UserInput.EachRow,'Image') == 1
            CPimagesc(FinalHistogramData,handles);
            set(heatmp,'XTickLabel',XTickLabels)
            NewPlotBinLocations = 1:length(FinalHistogramData');
            set(heatmp,'XTick',NewPlotBinLocations)

            NewColormap = 1 - colormap(pink);
            colormap(NewColormap),
            ColorbarHandle = colorbar;
            %%% Labels the colorbar's units.
            if strcmp(UserInput.BinVar,'Percentages') == 1
                ylabel(ColorbarHandle, ['Percentage of ', ObjectTypename, ' in each image'])
            else ylabel(ColorbarHandle, ['Number of ', ObjectTypename])
            end
            set(gca,'fontname','Helvetica','fontsize',FontSize)
            set(get(ColorbarHandle,'title'),'fontname','Helvetica','fontsize',FontSize+2)
            xlabel(gca,'Histogram bins','Fontname','Helvetica','fontsize',FontSize+2)
            ylabel(gca,'Image number','fontname','Helvetica','fontsize',FontSize+2)
            title(MeasurementToExtract,'Fontname','Helvetica','fontsize',FontSize+2)
        else
            CPimagesc(flipud(FinalHistogramData'),handles);
            XTickLabels=fliplr(XTickLabels);
            set(heatmp,'YTickLabel',XTickLabels);
            NewPlotBinLocations = 1:length(FinalHistogramData');
            set(heatmp,'YTick',NewPlotBinLocations)

            NewColormap = 1 - colormap(pink);
            colormap(NewColormap);
            ColorbarHandle = colorbar;
            %%% Labels the colorbar's units.
            if strcmp(UserInput.BinVar,'Percentages') == 1
                ylabel(ColorbarHandle, ['Percentage of ', ObjectTypename, ' in each image']);
            else ylabel(ColorbarHandle, ['Number of ', ObjectTypename]);
            end
            set(gca,'fontname','Helvetica','fontsize',FontSize);
            set(get(ColorbarHandle,'title'),'fontname','Helvetica','fontsize',FontSize+2);
            xlabel(gca,'Image number','Fontname','Helvetica','fontsize',FontSize+2);
            ylabel(gca,'Histogram bins','fontname','Helvetica','fontsize',FontSize+2);
            title(MeasurementToExtract,'Fontname','Helvetica','fontsize',FontSize+2);
        end

        set(gca,'Tag','BarTag','ActivePositionProperty','Position')

        left=30;
        bottom=370;
        StdUnit = 'point';

        %%% Creates text
        uicontrol('Parent',FigureHandle, ...
            'BackgroundColor',[.7 .7 .9], ...
            'Unit',StdUnit, ...
            'Position',[left bottom 60 25], ...
            'Units','Normalized',...
            'String','Histogram bins labels:', ...
            'Style','text', ...
            'FontSize',FontSize);

        left=left+70;

        %%% Hide every other label button.
        if strcmp(UserInput.EachRow,'Bin')
            Button1Callback = 'tempData = get(gcf,''UserData'');AxesHandles = findobj(gcf,''Tag'',''BarTag'');PlotBinLocations = get(AxesHandles(1),''YTick'');XTickLabels = get(AxesHandles(1),''YTickLabel'');if ceil(length(PlotBinLocations)/2) ~= length(PlotBinLocations)/2,PlotBinLocations(length(PlotBinLocations)) = [];XTickLabels(length(XTickLabels)) = [];end;PlotBinLocations2 = reshape(PlotBinLocations,2,[]);XTickLabels2 = reshape(XTickLabels,2,[]);set(AxesHandles,''YTick'',PlotBinLocations2(1,:));set(AxesHandles,''YTickLabel'',XTickLabels2(1,:));tempData.HideOption = 1;set(tempData.HideHandle,''UserData'',tempData);set(tempData.DecimalHandle,''UserData'',tempData);clear';
        else
            Button1Callback = 'tempData = get(gcf,''UserData'');AxesHandles = findobj(gcf,''Tag'',''BarTag'');PlotBinLocations = get(AxesHandles(1),''XTick'');XTickLabels = get(AxesHandles(1),''XTickLabel'');if ceil(length(PlotBinLocations)/2) ~= length(PlotBinLocations)/2,PlotBinLocations(length(PlotBinLocations)) = [];XTickLabels(length(XTickLabels)) = [];end;PlotBinLocations2 = reshape(PlotBinLocations,2,[]);XTickLabels2 = reshape(XTickLabels,2,[]);set(AxesHandles,''XTick'',PlotBinLocations2(1,:));set(AxesHandles,''XTickLabel'',XTickLabels2(1,:));tempData.HideOption = 1;set(tempData.HideHandle,''UserData'',tempData);set(tempData.DecimalHandle,''UserData'',tempData);clear';
        end
        Button1 = uicontrol('Parent',FigureHandle, ...
            'Unit',StdUnit, ...
            'BackgroundColor',[.7 .7 .9], ...
            'CallBack',Button1Callback, ...
            'Position',[left bottom 55 22], ...
            'Units','Normalized',...
            'String','Fewer',...
            'Style','pushbutton',...
            'UserData',0,...
            'FontSize',FontSize);

        left=left+60;

        %%% Decimal places Measurement axis labels.
        if strcmp(UserInput.EachRow,'Bin')
            Button2Callback = 'tempData = get(gcf,''UserData'');HideOption = tempData.HideOption;FigureSettings = tempData.FigureSettings; PlotBinLocations = FigureSettings{1};PreXTickLabels = FigureSettings{2};XTickLabels = PreXTickLabels(2:end-1); AxesHandles = findobj(gcf,''Tag'',''BarTag''); NumberOfDecimals = inputdlg(''Enter the number of decimal places to display'',''Enter the number of decimal places'',1,{''0''}); if ~isempty(NumberOfDecimals), NumberValues = cell2mat(XTickLabels); Command = [''%.'',num2str(NumberOfDecimals{1}),''f'']; NewNumberValues = num2str(NumberValues'',Command); NewNumberValuesPlusFirstLast = [PreXTickLabels(1); cellstr(NewNumberValues); PreXTickLabels(end)];if HideOption,if ceil(length(PlotBinLocations)/2) ~= length(PlotBinLocations)/2,PlotBinLocations(length(PlotBinLocations)) = [];NewNumberValuesPlusFirstLast(length(NewNumberValuesPlusFirstLast)) = [];end,PlotBinLocations2 = reshape(PlotBinLocations,2,[]);XTickLabels2 = reshape(NewNumberValuesPlusFirstLast,2,[]);set(AxesHandles,''YTickLabel'',XTickLabels2);set(AxesHandles,''YTick'',PlotBinLocations);else,set(AxesHandles,''YTickLabel'',NewNumberValuesPlusFirstLast);set(AxesHandles,''YTick'',PlotBinLocations);end,tempData.DecimalOption = 1;set(tempData.HideHandle,''UserData'',tempData);set(tempData.DecimalHandle,''UserData'',tempData);clear, drawnow, end';
        else
            Button2Callback = 'tempData = get(gcf,''UserData'');HideOption = tempData.HideOption;FigureSettings = tempData.FigureSettings; PlotBinLocations = FigureSettings{1};PreXTickLabels = FigureSettings{2};XTickLabels = PreXTickLabels(2:end-1); AxesHandles = findobj(gcf,''Tag'',''BarTag''); NumberOfDecimals = inputdlg(''Enter the number of decimal places to display'',''Enter the number of decimal places'',1,{''0''}); if ~isempty(NumberOfDecimals), NumberValues = cell2mat(XTickLabels); Command = [''%.'',num2str(NumberOfDecimals{1}),''f'']; NewNumberValues = num2str(NumberValues'',Command); NewNumberValuesPlusFirstLast = [PreXTickLabels(1); cellstr(NewNumberValues); PreXTickLabels(end)];if HideOption,if ceil(length(PlotBinLocations)/2) ~= length(PlotBinLocations)/2,PlotBinLocations(length(PlotBinLocations)) = [];NewNumberValuesPlusFirstLast(length(NewNumberValuesPlusFirstLast)) = [];end,PlotBinLocations2 = reshape(PlotBinLocations,2,[]);XTickLabels2 = reshape(NewNumberValuesPlusFirstLast,2,[]);set(AxesHandles,''XTickLabel'',XTickLabels2);set(AxesHandles,''XTick'',PlotBinLocations);else,set(AxesHandles,''XTickLabel'',NewNumberValuesPlusFirstLast);set(AxesHandles,''XTick'',PlotBinLocations);end,tempData.DecimalOption = 1;set(tempData.HideHandle,''UserData'',tempData);set(tempData.DecimalHandle,''UserData'',tempData);clear, drawnow, end';
        end
        Button2 = uicontrol('Parent',FigureHandle, ...
            'Unit',StdUnit,...
            'BackgroundColor',[.7 .7 .9], ...
            'CallBack',Button2Callback, ...
            'Position',[left bottom 55 22], ...
            'Units','Normalized',...
            'String','Decimals',...
            'Style','pushbutton',...
            'UserData',0,...
            'FontSize',FontSize);

        left=left+60;

        %%% Restore original X axis labels.
        if strcmp(UserInput.EachRow,'Bin')
            Button3Callback = 'tempData = get(gcf,''UserData'');FigureSettings = tempData.FigureSettings;PlotBinLocations = FigureSettings{1}; XTickLabels = FigureSettings{2}; AxesHandles = findobj(gcf, ''Tag'', ''BarTag''); set(AxesHandles,''YTick'',PlotBinLocations); set(AxesHandles,''YTickLabel'',XTickLabels); clear';
        else
            Button3Callback = 'tempData = get(gcf,''UserData'');FigureSettings = tempData.FigureSettings;PlotBinLocations = FigureSettings{1}; XTickLabels = FigureSettings{2}; AxesHandles = findobj(gcf, ''Tag'', ''BarTag''); set(AxesHandles,''XTick'',PlotBinLocations); set(AxesHandles,''XTickLabel'',XTickLabels); clear';
        end
        Button3 = uicontrol('Parent',FigureHandle, ...
            'Unit',StdUnit, ...
            'BackgroundColor',[.7 .7 .9], ...
            'CallBack',Button3Callback, ...
            'Position',[left bottom 55 22], ...
            'Units','Normalized',...
            'String','Restore', ...
            'Style','pushbutton', ...
            'FontSize',FontSize);

        left=left+160;

        Button4Callback = 'cmap=colormap; colormap(1-colormap(cmap));';

        Button4 = uicontrol('Parent',FigureHandle, ...
            'Unit',StdUnit, ...
            'BackgroundColor',[.7 .7 .9], ...
            'CallBack',Button4Callback, ...
            'Position',[left bottom 90 22], ...
            'Units','Normalized',...
            'String','Invert colormap', ...
            'Style','pushbutton', ...
            'FontSize',FontSize);

        left=left+92;

        % Help button
        Help_Callback = 'CPhelpdlg(''Inverting the colormap subtracts each intensity value of the colormap from 1. To change the colormap, click on the heatmap graph which will display the image tool.'')';

        uicontrol('Parent', FigureHandle, ...
            'Unit', StdUnit, ...
            'BackgroundColor', [.7 .7 .9], ...
            'CallBack', Help_Callback, ...
            'Position', [left bottom 15 22], ...
            'Units', 'Normalized', ...
            'String', '?', ...
            'Style', 'pushbutton', ...
            'FontSize', FontSize);


        %Add buttons
        FigureSettings{1} = NewPlotBinLocations;
        FigureSettings{2} = XTickLabels;
        FigureSettings{3} = FinalHistogramData;
        tempData.HideOption = 0;
        tempData.HideHandle = Button1;
        tempData.DecimalHandle = Button2;
        tempData.FigureSettings = FigureSettings;
        tempData.handles = rmfield(handles,'Pipeline');
        tempData.Application = 'CellProfiler';
        set(FigureHandle,'UserData',tempData);

    end

catch
    ErrorMessage = lasterr;
    CPerrordlg(['An error occurred in the Histogram Data Tool. ' ErrorMessage(29:end)]);
    return
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
RowImageOrBin   String "Image" or "Bin" to indicate orientation for output:
image => each row is one image, bin => each row is one bin.
(Actually, first letter only is checked, case insensitively.)
%}
%%% Open the file and name it appropriately.

fid = fopen(FileName, 'wt');
if fid < 0
    h = uiwait(CPerrordlg(['Unable to open output file ',FileName,'.']));
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

        WaitbarHandle = CPwaitbar(0,'Writing the histogram data file...');
        %%% Cycles through the images, one per row
        NumImages = LastImage-FirstImage+1;
        for ImageNumber = FirstImage:LastImage
            %%% Write the HistogramTitle as a heading for the column.
            fwrite(fid, char(HistogramTitles{ImageNumber}), 'char');
            for BinNum = 1:NumberBinsToWrite
                fwrite(fid, sprintf('\t%g',FinalHistogramData(BinNum,ImageNumber)), 'char');
            end
            fwrite(fid, sprintf('\n'), 'char');
            CPwaitbar(ImageNumber/NumImages,WaitbarHandle);
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
        h = uiwait(CPerrordlg('Neither "image" nor "bin" selected, ',FileName,' will be empty.'));
        waitfor(h);
    end
catch
    h = uiwait(CPerrordlg(['Problem occurred while writing to ',FileName,'. File is incomplete.']));
    waitfor(h);
end
%%% Close the file
try
    fclose(fid);
    h = CPhelpdlg(['The file ', FileName, ' has been written to the directory where the raw measurements file is located.']);
    waitfor(h)
catch
    h = uiwait(CPerrordlg(['Unable to close file ',FileName,'.']));
    waitfor(h);
    return;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function UserInput = UserInputWindow(handles,RawFileName,UserInput)
% This function displays a window so that lets the user choose which
% measurements to export. If the return variable 'UserInput' is empty
% it means that either no measurements were found or the user pressed
% the Cancel button (or the window was closed).


% Store font size
FontSize = handles.Preferences.FontSize;

% Create Dialog window
FigHandle = CPfigure;
set(FigHandle,'units','pixels','resize','on','menubar','none','toolbar','none','numbertitle','off','Name','Choose histogram settings');
Height = 625;
Width  = 475;
ScreenSize = get(0,'ScreenSize');
LeftPos = (ScreenSize(3)-Width)/2;
BottomPos = (ScreenSize(4)-Height)/2;
set(FigHandle,'position',[LeftPos BottomPos Width Height]);

NumMat=(1:handles.Current.SetBeingAnalyzed)';
ReverseNumMat=flipud(NumMat);

uiheight=25;
ypos=Height-uiheight-15;

% Dialog user input
uicontrol(FigHandle,'style','text','String','First sample number to show or export:','FontSize',FontSize,'FontWeight', 'bold',...
    'HorizontalAlignment','left','units','pixels','position',[10 ypos 300 uiheight],'BackgroundColor',get(FigHandle,'color'));
FirstSample = uicontrol(FigHandle,'style','popupmenu','String',{NumMat},'FontSize',FontSize,...
    'HorizontalAlignment','left','units','pixels','position',[325 ypos+5 125 uiheight],'BackgroundColor',get(FigHandle, 'color'));

ypos = ypos - uiheight - 5;

uicontrol(FigHandle,'style','text','String','Last sample number to show or export:','FontSize',FontSize,'FontWeight', 'bold',...
    'HorizontalAlignment','left','units','pixels','position',[10 ypos 300 uiheight],'BackgroundColor',get(FigHandle,'color'));
LastSample = uicontrol(FigHandle,'style','popupmenu','String',{ReverseNumMat},'FontSize',FontSize,...
    'HorizontalAlignment','left','units','pixels','position',[325 ypos+5 125 uiheight],'BackgroundColor',get(FigHandle, 'color'));

%Help button
LastSample_Help_Callback = 'CPhelpdlg(''To display data from only one image, choose the image number as both the first and last sample number.'')';
uicontrol(FigHandle,'style','pushbutton','String','?','FontSize',FontSize,...
    'HorizontalAlignment','center','units','pixels','position',[455 ypos+8 15 uiheight],'BackgroundColor',get(FigHandle,'color'),'FontWeight', 'bold',...
    'Callback', LastSample_Help_Callback);

ypos = ypos - uiheight - 5;

uicontrol(FigHandle,'style','text','String','Number of histogram bins:','FontSize',FontSize,'FontWeight', 'bold',...
    'HorizontalAlignment','left','units','pixels','position',[10 ypos 300 uiheight],'BackgroundColor',get(FigHandle,'color'));
NumBins = uicontrol(FigHandle,'style','edit','String','20','FontSize',FontSize,...
    'HorizontalAlignment','center','units','pixels','position',[325 ypos+5 125 uiheight],'BackgroundColor',[1 1 1]);

%Help button
NumBins_Help_Callback = 'CPhelpdlg(''This number should not include the first and last bins, which will contain anything outside the specified range.'')';
uicontrol(FigHandle,'style','pushbutton','String','?','FontSize',FontSize,...
    'HorizontalAlignment','center','units','pixels','position',[455 ypos+5 15 uiheight],'BackgroundColor',get(FigHandle,'color'),'FontWeight', 'bold',...
    'Callback', NumBins_Help_Callback);

ypos = ypos - uiheight - 5;

uicontrol(FigHandle,'style','text','String','Each histogram bin contains the objects'':','FontSize',FontSize,'FontWeight', 'bold',...
    'HorizontalAlignment','left','units','pixels','position',[10 ypos 300 uiheight],'BackgroundColor',get(FigHandle,'color'));
BinVar = uicontrol(FigHandle,'style','popupmenu','String',{'Actual numbers','Percentages'},'FontSize',FontSize,...
    'HorizontalAlignment','left','units','pixels','position',[325 ypos+5 125 uiheight],'BackgroundColor',get(FigHandle, 'color'));

ypos = ypos - uiheight - 5;

uicontrol(FigHandle,'style','text','String','Method to threshold leftmost bin:','FontSize',FontSize,'FontWeight', 'bold',...
    'HorizontalAlignment','left','units','pixels','position',[10 ypos-.5*uiheight 125 1.5*uiheight],'BackgroundColor',get(FigHandle,'color'));
LeftBin = uicontrol(FigHandle,'style','popupmenu','String',{'Min value found','Other'},'FontSize',FontSize,...
    'HorizontalAlignment','left','units','pixels','position',[130 ypos 110 uiheight],'BackgroundColor',get(FigHandle, 'color'));

uicontrol(FigHandle,'style','text','String','If other, enter threshold value:','FontSize',FontSize,'FontWeight', 'bold',...
    'HorizontalAlignment','left','units','pixels','position',[250 ypos 150 uiheight],'BackgroundColor',get(FigHandle,'color'));
LeftVal = uicontrol(FigHandle,'style','edit','String','','FontSize',FontSize,...
    'HorizontalAlignment','center','units','pixels','position',[400 ypos 50 uiheight],'BackgroundColor',[1 1 1]);

%Help button
LeftVal_Help_Callback = 'CPhelpdlg(''Any measurements less than this value will be combined in the leftmost bin.'')';
uicontrol(FigHandle,'style','pushbutton','String','?','FontSize',FontSize,...
    'HorizontalAlignment','center','units','pixels','position',[455 ypos 15 uiheight],'BackgroundColor',get(FigHandle,'color'),'FontWeight', 'bold',...
    'Callback', LeftVal_Help_Callback);

ypos = ypos - uiheight*2;

uicontrol(FigHandle,'style','text','String','Method to threshold rightmost bin:','FontSize',FontSize,'FontWeight', 'bold',...
    'HorizontalAlignment','left','units','pixels','position',[10 ypos-.5*uiheight 125 1.5*uiheight],'BackgroundColor',get(FigHandle,'color'));
RightBin = uicontrol(FigHandle,'style','popupmenu','String',{'Max value found','Other'},'FontSize',FontSize,...
    'HorizontalAlignment','left','units','pixels','position',[130 ypos 110 uiheight],'BackgroundColor',get(FigHandle, 'color'));

uicontrol(FigHandle,'style','text','String','If other, enter threshold value:','FontSize',FontSize,'FontWeight', 'bold',...
    'HorizontalAlignment','left','units','pixels','position',[250 ypos 150 uiheight],'BackgroundColor',get(FigHandle,'color'));
RightVal = uicontrol(FigHandle,'style','edit','String','','FontSize',FontSize,...
    'HorizontalAlignment','center','units','pixels','position',[400 ypos 50 uiheight],'BackgroundColor',[1 1 1]);

%Help button
RightVal_Help_Callback = 'CPhelpdlg(''Any measurements greater than or equal to this value will be combined in the rightmost bin.'')';
uicontrol(FigHandle,'style','pushbutton','String','?','FontSize',FontSize,...
    'HorizontalAlignment','center','units','pixels','position',[455 ypos 15 uiheight],'BackgroundColor',get(FigHandle,'color'),'FontWeight', 'bold',...
    'Callback', RightVal_Help_Callback);

ypos = ypos - uiheight*2;

uicontrol(FigHandle,'style','text','String','Threshold applied to data:','FontSize',FontSize,'FontWeight', 'bold',...
    'HorizontalAlignment','left','units','pixels','position',[10 ypos 125 uiheight],'BackgroundColor',get(FigHandle,'color'));
Logical = uicontrol(FigHandle,'style','popupmenu','String',{'None','>','>=','=','<=','<'},'FontSize',FontSize,...
    'HorizontalAlignment','left','units','pixels','position',[130 ypos 110 uiheight],'BackgroundColor',get(FigHandle, 'color'));

uicontrol(FigHandle,'style','text','String','If other than none, enter threshold value:','FontSize',FontSize,'FontWeight', 'bold',...
    'HorizontalAlignment','left','units','pixels','position',[250 ypos 150 uiheight],'BackgroundColor',get(FigHandle,'color'));
ThresholdVal = uicontrol(FigHandle,'style','edit','String','','FontSize',FontSize,...
    'HorizontalAlignment','center','units','pixels','position',[400 ypos 50 uiheight],'BackgroundColor',[1 1 1]);

%Help button
ThresholdVal_Help_Callback = 'CPhelpdlg(''Use this option if you want to calculate histogram data only for objects meeting a threshold in a measurement.'')';
uicontrol(FigHandle,'style','pushbutton','String','?','FontSize',FontSize,...
    'HorizontalAlignment','center','units','pixels','position',[455 ypos 15 uiheight],'BackgroundColor',get(FigHandle,'color'),'FontWeight', 'bold',...
    'Callback', ThresholdVal_Help_Callback);

ypos = ypos - uiheight*2;

uicontrol(FigHandle,'style','text','String','Combine all data into one histogram?','FontSize',FontSize,'FontWeight', 'bold',...
    'HorizontalAlignment','left','units','pixels','position',[10 ypos 300 uiheight],'BackgroundColor',get(FigHandle,'color'));
Combine = uicontrol(FigHandle,'style','popupmenu','String',{'No','Yes'},'FontSize',FontSize,...
    'HorizontalAlignment','left','units','pixels','position',[325 ypos+5 125 uiheight],'BackgroundColor',get(FigHandle, 'color'));

%Help button
Combine_Help_Callback = 'CPhelpdlg(''Choose "Yes" if you want to calculate one histogram for all the images you selected. Choose "No" if you want to calculate a separate histogram for each image.'')';
uicontrol(FigHandle,'style','pushbutton','String','?','FontSize',FontSize,...
    'HorizontalAlignment','center','units','pixels','position',[455 ypos+8 15 uiheight],'BackgroundColor',get(FigHandle,'color'),'FontWeight', 'bold',...
    'Callback', Combine_Help_Callback);

ypos = ypos - uiheight - 5;

uicontrol(FigHandle,'style','text','String','X axis is:','FontSize',FontSize,'FontWeight', 'bold',...
    'HorizontalAlignment','left','units','pixels','position',[10 ypos 300 uiheight],'BackgroundColor',get(FigHandle,'color'));
Xaxis = uicontrol(FigHandle,'style','popupmenu','String',{'Measurements','Number of objects in bin'},'FontSize',FontSize,...
    'HorizontalAlignment','left','units','pixels','position',[325 ypos+5 125 uiheight],'BackgroundColor',get(FigHandle, 'color'));

%Help button
Xaxis_Help_Callback = 'CPhelpdlg(''The default for the X axis is "Measurements". By choosing "Number of objects in bin", you are essentially flipping the axes. Flipping is possible for both bar and line graphs, but not area graphs because there is no function that will work.'')';
uicontrol(FigHandle,'style','pushbutton','String','?','FontSize',FontSize,...
    'HorizontalAlignment','center','units','pixels','position',[455 ypos+8 15 uiheight],'BackgroundColor',get(FigHandle,'color'),'FontWeight', 'bold',...
    'Callback', Xaxis_Help_Callback);

ypos = ypos - uiheight - 5;

uicontrol(FigHandle,'style','text','String','For multiple histograms, the axes should show:','FontSize',FontSize,'FontWeight', 'bold',...
    'HorizontalAlignment','left','units','pixels','position',[10 ypos 300 uiheight],'BackgroundColor',get(FigHandle,'color'));
RelAbs = uicontrol(FigHandle,'style','popupmenu','String',{'Relative number of objects','Absolute number of objects'},'FontSize',FontSize,...
    'HorizontalAlignment','left','units','pixels','position',[325 ypos+5 125 uiheight],'BackgroundColor',get(FigHandle, 'color'));

%Help button
IndepAxis_Help_Callback = 'CPhelpdlg(''Choosing "Relative" will scale the "Number of objects" axis to fit the maximum value for that sample. Choosing "Absolute" will make the "Number of objects" axis the same for all histograms.'')';
uicontrol(FigHandle,'style','pushbutton','String','?','FontSize',FontSize,...
    'HorizontalAlignment','center','units','pixels','position',[455 ypos+8 15 uiheight],'BackgroundColor',get(FigHandle,'color'),'FontWeight', 'bold',...
    'Callback', IndepAxis_Help_Callback);

ypos = ypos - uiheight - 5;

uicontrol(FigHandle,'style','text','String','Do you want the measurement axis to be log scale?','FontSize',FontSize,'FontWeight', 'bold',...
    'HorizontalAlignment','left','units','pixels','position',[10 ypos 300 uiheight],'BackgroundColor',get(FigHandle,'color'));
Log = uicontrol(FigHandle,'style','popupmenu','String',{'No','Yes'},'FontSize',FontSize,...
    'HorizontalAlignment','left','units','pixels','position',[325 ypos+5 125 uiheight],'BackgroundColor',get(FigHandle, 'color'));

ypos = ypos - uiheight - 5;

uicontrol(FigHandle,'style','text','String','Style of graph:','FontSize',FontSize,'FontWeight', 'bold',...
    'HorizontalAlignment','left','units','pixels','position',[10 ypos 300 uiheight],'BackgroundColor',get(FigHandle,'color'));
Style = uicontrol(FigHandle,'style','popupmenu','String',{'Bar','Line','Area','Heatmap'},'FontSize',FontSize,...
    'HorizontalAlignment','left','units','pixels','position',[325 ypos+5 125 uiheight],'BackgroundColor',get(FigHandle, 'color'));

ypos = ypos - uiheight - 5;

uicontrol(FigHandle,'style','text','String','Color of the initial plot:','FontSize',FontSize,'FontWeight', 'bold',...
    'HorizontalAlignment','left','units','pixels','position',[10 ypos 300 uiheight],'BackgroundColor',get(FigHandle,'color'));
Color = uicontrol(FigHandle,'style','popupmenu','String',{'Blue','Red','Green','Yellow','Magenta','Cyan','Black','White','CellProfiler background'},'FontSize',FontSize,...
    'HorizontalAlignment','left','units','pixels','position',[325 ypos+5 125 uiheight],'BackgroundColor',get(FigHandle, 'color'));

ypos = ypos - uiheight - 5;

uicontrol(FigHandle,'style','text','String','Do you want the histograms to be displayed?','FontSize',FontSize,'FontWeight', 'bold',...
    'HorizontalAlignment','left','units','pixels','position',[10 ypos 300 uiheight],'BackgroundColor',get(FigHandle,'color'));
Display = uicontrol(FigHandle,'style','popupmenu','String',{'Yes','No'},'FontSize',FontSize,...
    'HorizontalAlignment','left','units','pixels','position',[325 ypos+5 125 uiheight],'BackgroundColor',get(FigHandle, 'color'));

%Help button
Display_Help_Callback = 'CPhelpdlg(''Displaying histograms is impractical when exporting large amounts of data.'')';
uicontrol(FigHandle,'style','pushbutton','String','?','FontSize',FontSize,...
    'HorizontalAlignment','center','units','pixels','position',[475 ypos+10 15 uiheight],'BackgroundColor',get(FigHandle,'color'),'FontWeight', 'bold',...
    'Callback', Display_Help_Callback);

ypos = ypos - uiheight - 5;

uicontrol(FigHandle,'style','text','String','Export histogram data?','FontSize',FontSize,'FontWeight', 'bold',...
    'HorizontalAlignment','left','units','pixels','position',[10 ypos 150 uiheight],'BackgroundColor',get(FigHandle,'color'));
ExportHist = uicontrol(FigHandle,'style','popupmenu','String',{'No','Yes'},'FontSize',FontSize,...
    'HorizontalAlignment','left','units','pixels','position',[160 ypos 50 uiheight],'BackgroundColor',get(FigHandle, 'color'));

uicontrol(FigHandle,'style','text','String','If yes, enter a filename:','FontSize',FontSize,'FontWeight', 'bold',...
    'HorizontalAlignment','left','units','pixels','position',[230 ypos 90 uiheight],'BackgroundColor',get(FigHandle,'color'));
ExportFile = uicontrol(FigHandle,'style','edit','String',[RawFileName(1:end-4),'_Histogram.xls'],'FontSize',FontSize,...
    'HorizontalAlignment','center','units','pixels','position',[325 ypos 125 uiheight],'BackgroundColor',[1 1 1]);

%Help button
ExportFile_Help_Callback = 'CPhelpdlg(''Enter the filename with the extension ".xls" so it can be opened easily in Excel.'')';
uicontrol(FigHandle,'style','pushbutton','String','?','FontSize',FontSize,...
    'HorizontalAlignment','center','units','pixels','position',[455 ypos 15 uiheight],'BackgroundColor',get(FigHandle,'color'),'FontWeight', 'bold',...
    'Callback', ExportFile_Help_Callback);

ypos = ypos - uiheight*1.5;

uicontrol(FigHandle,'style','text','String','If exporting histograms or using heatmap format, each row is one:','FontSize',FontSize,'FontWeight', 'bold',...
    'HorizontalAlignment','left','units','pixels','position',[10 ypos 300 uiheight],'BackgroundColor',get(FigHandle,'color'));
EachRow = uicontrol(FigHandle,'style','popupmenu','String',{'Image','Bin'},'FontSize',FontSize,...
    'HorizontalAlignment','left','units','pixels','position',[325 ypos+5 125 uiheight],'BackgroundColor',get(FigHandle, 'color'));

%%% OK AND CANCEL BUTTONS
posx = (Width - 200)/2;               % Centers buttons horizontally
okbutton = uicontrol(FigHandle,'style','pushbutton','String','OK','Fontweight','bold','FontSize',FontSize,'units','pixels',...
    'position',[posx 10 75 30],'BackgroundColor',[.7 .7 .9],'Callback','[cobj,cfig] = gcbo;set(cobj,''UserData'',1);uiresume(cfig);clear cobj cfig;','BackgroundColor',[.7 .7 .9]);
cancelbutton = uicontrol(FigHandle,'style','pushbutton','String','Cancel','Fontweight','bold','FontSize',FontSize,'units','pixels',...
    'position',[posx+100 10 75 30],'Callback','close(gcf)','BackgroundColor',[.7 .7 .9]); %#ok Ignore MLint

if ~isempty(UserInput)
    try %#ok Ignore MLint
        set(FirstSample,'value',UserInput.FirstSample);
        set(LastSample,'value',ReverseNumMat(UserInput.LastSample));
        set(NumBins,'string',num2str(UserInput.NumBins));

        if strcmp(UserInput.BinVar,'Actual numbers')
            set(BinVar, 'value',1);
        else
            set(BinVar, 'value',2);
        end

        if strcmp(UserInput.LeftBin,'Min value found')
            set(LeftBin,'value',1);
        else
            set(LeftBin,'value',2);
        end

        LeftValNumber=num2str(UserInput.LeftVal);
        if strcmp(LeftValNumber,'NaN')
            LeftValNumber='';
        end
        set(LeftVal,'string',LeftValNumber);

        if strcmp(UserInput.RightBin,'Max value found')
            set(RightBin,'value',1);
        else
            set(RightBin,'value',2);
        end

        RightValNumber=num2str(UserInput.RightVal);
        if strcmp(RightValNumber,'NaN')
            RightValNumber='';
        end
        set(RightVal,'string',RightValNumber);

        if strcmp(UserInput.Logical,'None')
            set(Logical,'value',1);
        elseif strcmp(UserInput.Logical,'>')
            set(Logical,'value',2);
        elseif strcmp(UserInput.Logical,'>=')
            set(Logical,'value',3);
        elseif strcmp(UserInput.Logical,'=')
            set(Logical,'value',4);
        elseif strcmp(UserInput.Logical,'<=')
            set(Logical,'value',5);
        elseif strcmp(UserInput.Logical,'<')
            set(Logical,'value',6);
        end

        ThresholdValNumber=num2str(UserInput.ThresholdVal);
        if strcmp(ThresholdValNumber,'NaN')
            ThresholdValNumber='';
        end
        set(ThresholdVal,'string',ThresholdValNumber);

        if strcmp(UserInput.Combine,'No')
            set(Combine,'value',1);
        else
            set(Combine,'value',2);
        end

        if strcmp(UserInput.Xaxis,'Measurements')
            set(Xaxis,'value',1);
        else
            set(Xaxis,'value',2);
        end

        if strcmp(UserInput.RelAbs,'Relative number of objects')
            set(RelAbs,'value',1);
        else
            set(RelAbs,'value',2);
        end

        if strcmp(UserInput.Log,'No')
            set(Log,'value',1);
        else
            set(Log,'value',2);
        end

        if strcmp(UserInput.Style,'Bar')
            set(Style,'value',1);
        elseif strcmp(UserInput.Style,'Line')
            set(Style,'value',2);
        elseif strcmp(UserInput.Style,'Area')
            set(Style,'value',3);
        elseif strcmp(UserInput.Style,'Heatmap')
            set(Style,'value',4);
        end

        if strcmp(UserInput.Color,'Blue')
            set(Color,'value',1);
        elseif strcmp(UserInput.Color,'Red')
            set(Color,'value',2);
        elseif strcmp(UserInput.Color,'Green')
            set(Color,'value',3);
        elseif strcmp(UserInput.Color,'Yellow')
            set(Color,'value',4);
        elseif strcmp(UserInput.Color,'Magenta')
            set(Color,'value',5);
        elseif strcmp(UserInput.Color,'Cyan')
            set(Color,'value',6);
        elseif strcmp(UserInput.Color,'Black')
            set(Color,'value',7);
        elseif strcmp(UserInput.Color,'White')
            set(Color,'value',8);
        elseif strcmp(UserInput.Color,'CellProfiler background')
            set(Color,'value',9);
        end

        if strcmp(UserInput.Display,'Yes')
            set(Display,'value',1);
        else
            set(Display,'value',2);
        end

        if strcmp(UserInput.ExportHist,'No')
            set(ExportHist,'value',1);
        else
            set(ExportHist,'value',2);
        end

        set(ExportFile,'string',UserInput.ExportFile);

        if strcmp(UserInput.EachRow,'Image')
            set(EachRow,'value',1);
        else
            set(EachRow,'value',2);
        end
    end
end

% Repeat until valid input has been entered or the window is destroyed
while 1

    % Wait until window is destroyed or uiresume() is called
    uiwait(FigHandle)

    % Action depending on user input
    if ishandle(okbutton)               % The OK button pressed
        %UserInput = get(Dialog,'UserData');

        % Populate structure array
        UserInput.FirstSample = get(FirstSample,'value');
        UserInput.LastSample = ReverseNumMat(get(LastSample,'value'));
        UserInput.NumBins = str2double(get(NumBins,'string'));
        UserInput.BinVar = get(BinVar, 'value');
        UserInput.LeftBin = get(LeftBin,'value');
        UserInput.LeftVal = str2double(get(LeftVal,'string'));
        UserInput.RightBin = get(RightBin,'value');
        UserInput.RightVal = str2double(get(RightVal,'string'));
        UserInput.Logical = get(Logical,'value');
        UserInput.ThresholdVal = str2double(get(ThresholdVal,'string'));
        UserInput.Combine = get(Combine,'value');
        UserInput.Xaxis = get(Xaxis,'value');
        UserInput.RelAbs = get(RelAbs,'value');
        UserInput.Log = get(Log,'value');
        UserInput.Style = get(Style,'value');
        UserInput.Color = get(Color,'value');
        UserInput.Display = get(Display,'value');
        UserInput.ExportHist = get(ExportHist,'value');
        UserInput.ExportFile = get(ExportFile,'string');
        UserInput.EachRow = get(EachRow,'value');


        if UserInput.FirstSample > UserInput.LastSample         % Error check for sample numbers
            warnfig=CPwarndlg('Please make the first sample number less than or equal to the last sample number! Please try again.');
            uiwait(warnfig);
            set(okbutton,'UserData',[]);
        elseif isempty(UserInput.NumBins) || ~isnumeric(UserInput.NumBins)    % Error check for number of bins
            warnfig=CPwarndlg('You did not enter a valid number for "Number of bins". Please try again.');
            uiwait(warnfig);
            set(okbutton,'UserData',[]);
        elseif UserInput.NumBins<1 || floor(UserInput.NumBins) ~= UserInput.NumBins
            warnfig=CPwarndlg('You did not enter a valid number for "Number of bins". The number must be an integer greater than or equal to 1. Please try again.');
            uiwait(warnfig);
            set(okbutton,'UserData',[]);
        elseif UserInput.LeftBin == 2 && isempty(UserInput.LeftVal)                     % Error check for thresholding leftmost bin
            warnfig=CPwarndlg('You chose "Other" for "Method to threshold leftmost bin" but did not enter a valid threshold number. Please try again.');
            uiwait(warnfig);
            set(okbutton,'UserData',[]);
        elseif UserInput.RightBin == 2 && isempty(UserInput.RightVal)                     % Error check for thresholding rightmost bin
            warnfig=CPwarndlg('You chose "Other" for "Method to threshold rightmost bin" but did not enter a valid threshold number. Please try again.');
            uiwait(warnfig);
            set(okbutton,'UserData',[]);
        elseif UserInput.Logical ~= 1 && isempty(UserInput.ThresholdVal)                     % Error check for thresholding histogram data
            warnfig=CPwarndlg('You chose an option other than "None" for "Threshold applied to histogram data" but did not enter a threshold value. Please try again.');
            uiwait(warnfig);
            set(okbutton,'UserData',[]);
        elseif UserInput.Logical ~= 1 && isempty(UserInput.ThresholdVal)
            warnfig=CPwarndlg('You chose an option other than "None" for "Threshold applied to histogram data" but did not enter a valid number. Please try again.');
            uiwait(warnfig);
            set(okbutton,'UserData',[]);
        elseif UserInput.ExportHist == 2 && isempty(UserInput.ExportFile)   % Error check for exporting histogram
            warnfig=CPwarndlg('You chose to export the histogram but did not enter a filename. Please try again.');
            uiwait(warnfig);
            set(okbutton,'UserData',[]);
        else                                                        % User input is valid
            SaveData = fullfile(handles.Current.DefaultOutputDirectory, UserInput.ExportFile);
            OutputFileOverwrite = exist(SaveData,'file');
            if UserInput.ExportHist == 2 && OutputFileOverwrite ~= 0                             % File already exists
                Answer=CPquestdlg('A file with that name already exists in the directory containing the raw measurements file. Do you wish to overwrite?','Confirm overwrite','Yes','No','No');
                if strcmp(Answer,'No') == 1
                    warnfig=CPwarndlg('You chose not to overwrite the file. Please enter another file name and try again.');
                    uiwait(warnfig);
                    set(okbutton,'UserData',[]);
                    continue
                end
            end

            if UserInput.Xaxis == 2 && UserInput.Style == 3      % Check axes of area graph
                Answer=CPquestdlg('You chose "Number of objects in bin" for the X axis of an Area graph, but there is no function that can produce a graph with these settings. If you choose these settings, the default area graph will be displayed with "Measurements" as the X axis. Do you wish to continue?','Confirm Area graph settings','Yes','No','No');
                if strcmp(Answer, 'No') == 1
                    set(okbutton,'UserData',[]);
                    continue
                end
            end

            if UserInput.BinVar == 1                        % Assign string values
                UserInput.BinVar='Actual numbers';
            else
                UserInput.BinVar='Percentages';
            end

            if UserInput.LeftBin == 1
                UserInput.LeftBin='Min value found';
            else
                UserInput.LeftBin='Other';
            end

            if UserInput.RightBin == 1
                UserInput.RightBin='Max value found';
            else
                UserInput.RightBin='Other';
            end

            switch UserInput.Logical
                case 1
                    UserInput.Logical='None';
                case 2
                    UserInput.Logical='>';
                case 3
                    UserInput.Logical='>=';
                case 4
                    UserInput.Logical='=';
                case 5
                    UserInput.Logical='<=';
                otherwise
                    UserInput.Logical='<';
            end

            if UserInput.Combine == 1
                UserInput.Combine='No';
            else
                UserInput.Combine='Yes';
            end

            if UserInput.Xaxis == 1
                UserInput.Xaxis='Measurements';
            else
                UserInput.Xaxis='Number of objects in bin';
            end

            if UserInput.RelAbs == 1
                UserInput.RelAbs='Relative number of objects';
            else
                UserInput.RelAbs='Absolute number of objects';
            end

            if UserInput.Log == 1
                UserInput.Log='No';
            else
                UserInput.Log='Yes';
            end

            switch UserInput.Style
                case 1
                    UserInput.Style='Bar';
                case 2
                    UserInput.Style='Line';
                case 3
                    UserInput.Style='Area';
                otherwise
                    UserInput.Style='Heatmap';
            end

            switch UserInput.Color
                case 1
                    UserInput.Color='Blue';
                case 2
                    UserInput.Color='Red';
                case 3
                    UserInput.Color='Green';
                case 4
                    UserInput.Color='Yellow';
                case 5
                    UserInput.Color='Magenta';
                case 6
                    UserInput.Color='Cyan';
                case 7
                    UserInput.Color='Black';
                case 8
                    UserInput.Color='White';
                otherwise
                    UserInput.Color='CellProfiler background';
            end

            if UserInput.Display == 1
                UserInput.Display='Yes';
            else
                UserInput.Display='No';
            end

            if UserInput.ExportHist == 1
                UserInput.ExportHist='No';
            else
                UserInput.ExportHist='Yes';
            end

            if UserInput.EachRow == 1
                UserInput.EachRow='Image';
            else
                UserInput.EachRow='Bin';
            end

            delete(FigHandle);
            return
        end
    else                                % The user pressed the cancel button or closed the window
        UserInput = [];
        if ishandle(FigHandle),delete(FigHandle);end
        return
    end
end