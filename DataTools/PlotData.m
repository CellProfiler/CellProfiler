function handles = PlotData(handles)

% Help for the Plot Data tool:
% Category: Data Tools
%
% This module has not yet been documented.
%
% See also <nothing relevant>.

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
%
% $Revision$

cd(handles.Current.DefaultOutputDirectory)
%%% Ask the user to choose the file from which to extract measurements.
[RawFileName, RawPathname] = uigetfile('*.mat','Select the raw measurements file');
if RawFileName == 0
    cd(handles.Current.StartupDirectory);
    return
end
load(fullfile(RawPathname, RawFileName));

%%% Checks if the user wants to plot a set of histograms or a single
%%% measurement per image.
Answer = questdlg('Do you want to plot histograms of cell populations, or a single measurement per image?', 'Type of Plot', 'Histograms', 'Single Measurement', 'Histograms');

if (strcmp(Answer, 'Single Measurement') == 1),
    Fieldnames = fieldnames(handles.Measurements);
    MeasFieldnames = Fieldnames(strncmp(Fieldnames,'Image',5)==1);
    %%% Error detection.
    if isempty(MeasFieldnames)
        errordlg('No measurements were found in the file you selected.  They would be found within the output file''s handles.Measurements structure preceded by ''Image''.')
        cd(handles.Current.StartupDirectory);
        return
    end
    %%% Removes the 'Image' prefix from each name for display purposes.
    for Number = 1:length(MeasFieldnames)
        EditedMeasFieldnames{Number} = MeasFieldnames{Number}(6:end);
    end
    %%% Allows the user to select a measurement from the list.
    [Selection, ok] = listdlg('ListString',EditedMeasFieldnames, 'ListSize', [300 600],...
        'Name','Select measurement',...
        'PromptString','Choose a measurement to display as histograms','CancelString','Cancel',...
        'SelectionMode','single');
    if ok ~= 0,
        EditedMeasurementToExtract = char(EditedMeasFieldnames(Selection));
        MeasurementToExtract = ['Image', EditedMeasurementToExtract];
        figure;
        h = bar(cell2mat(handles.Measurements.(MeasurementToExtract)));
        axis tight;
        set(get(h, 'Children'), 'EdgeAlpha', 0);
        title(EditedMeasurementToExtract);
    end
    cd(handles.Current.StartupDirectory);
    return;
end

%%% Extract the fieldnames of measurements from the handles structure.
Fieldnames = fieldnames(handles.Measurements);
MeasFieldnames = Fieldnames(strncmp(Fieldnames,'Object',6)==1);
%%% Error detection.
if isempty(MeasFieldnames)
    errordlg('No measurements were found in the file you selected.  They would be found within the output file''s handles.Measurements structure preceded by ''Object''.')
    cd(handles.Current.StartupDirectory);
    return
end
%%% Removes the 'Object' prefix from each name for display purposes.
for Number = 1:length(MeasFieldnames)
    EditedMeasFieldnames{Number} = MeasFieldnames{Number}(7:end);
end
%%% Allows the user to select a measurement from the list.
[Selection, ok] = listdlg('ListString',EditedMeasFieldnames, 'ListSize', [300 600],...
    'Name','Select measurement',...
    'PromptString','Choose a measurement to display as histograms','CancelString','Cancel',...
    'SelectionMode','single');
if ok ~= 0
    EditedMeasurementToExtract = char(EditedMeasFieldnames(Selection));
    MeasurementToExtract = ['Object', EditedMeasurementToExtract];
    %%% Determines whether any sample info has been loaded.  If sample
    %%% info is present, the fieldnames for those are extracted.
    ImportedFieldnames = Fieldnames(strncmp(Fieldnames,'Imported',8) == 1 | strncmp(Fieldnames,'Filename',8) == 1);
    if isempty(ImportedFieldnames) == 0
        %%% Allows the user to select a heading from the list.
        [Selection, ok] = listdlg('ListString',ImportedFieldnames, 'ListSize', [300 600],...
            'Name','Select sample info',...
            'PromptString','Choose the sample info with which to label each histogram.','CancelString','Cancel',...
            'SelectionMode','single');
        if ok ~= 0
            HeadingName = char(ImportedFieldnames(Selection));
            try SampleNames = handles.Measurements.(HeadingName);
            catch SampleNames = handles.Pipeline.(HeadingName);
            end
        else cd(handles.Current.StartupDirectory);
            return
        end
    end
    %%% Asks the user whether a histogram should be shown for all image
    %%% sets or just a few.
    TotalNumberImageSets = length(handles.Measurements.(MeasurementToExtract));
    TextTotalNumberImageSets = num2str(TotalNumberImageSets);
    %%% Ask the user to specify which image sets to display.
    Prompts = {'Enter the first sample number to display','Enter the last sample number to display'};
    Defaults = {'1',TextTotalNumberImageSets};
    Answers = inputdlg(Prompts,'Choose samples to display',1,Defaults);
    if isempty(Answers) ~= 1
        FirstImage = str2double(Answers{1});
        LastImage = str2double(Answers{2});
        if isempty(FirstImage)
            errordlg('No number was entered for the first sample number to display.')
            cd(handles.Current.StartupDirectory);
            return
        end
        if isempty(LastImage)
            errordlg('No number was entered for the last sample number to display.')
            cd(handles.Current.StartupDirectory);
            return
        end
        NumberOfImages = LastImage - FirstImage + 1;
        if NumberOfImages == 0
            NumberOfImages = TotalNumberImageSets;
        elseif NumberOfImages > TotalNumberImageSets
            errordlg(['There are only ', TextTotalNumberImageSets, ' image sets total.'])
            cd(handles.Current.StartupDirectory);
            return
        end

        %%% Ask the user to specify histogram settings.
        Prompts = {'Enter the number of bins you want to be displayed in the histogram','Enter the minimum value to display', 'Enter the maximum value to display', 'Do you want to calculate one histogram for all of the specified data?', 'Do you want the Y-axis (number of cells) to be absolute or relative?','Display as a compressed histogram?','To save the histogram data, enter a filename (with ".xls" to open easily in Excel).'};
        Defaults = {'20','automatic','automatic','no','relative','no','no'};
        Answers = inputdlg(Prompts,'Choose histogram settings',1,Defaults);
        %%% Error checking/canceling.
        if isempty(Answers)
            cd(handles.Current.StartupDirectory);
            return
        end
        try NumberOfBins = str2double(Answers{1});
        catch errordlg('The text entered for the question "Enter the number of bins you want to be displayed in the histogram" was not a number.')
            cd(handles.Current.StartupDirectory);
            return
        end
        if isempty(NumberOfBins) ==1
            errordlg('No text was entered for "Enter the number of bins you want to be displayed in the histogram".')
            cd(handles.Current.StartupDirectory);
            return
        end
        MinHistogramValue = Answers{2};
        if isempty(MinHistogramValue) ==1
            errordlg('No text was entered for "Enter the minimum value to display".')
            cd(handles.Current.StartupDirectory);
            return
        end
        MaxHistogramValue = Answers{3};
        if isempty(MaxHistogramValue) ==1
            errordlg('No text was entered for "Enter the maximum value to display".')
            cd(handles.Current.StartupDirectory);
            return
        end
        CumulativeHistogram = Answers{4};
        %%% Error checking for the Y Axis Scale question.
        try YAxisScale = lower(Answers{5});
        catch errordlg('The text you entered for ''Do you want the Y-axis (number of cells) to be absolute or relative?'' was not recognized.');
            cd(handles.Current.StartupDirectory);
            return
        end
        if strcmp(YAxisScale, 'relative') ~= 1 && strcmp(YAxisScale, 'absolute') ~= 1
            errordlg('The text you entered for ''Do you want the Y-axis (number of cells) to be absolute or relative?'' was not recognized.');
            cd(handles.Current.StartupDirectory);
            return
        end
        CompressedHistogram = Answers{6};
        if strcmp(CompressedHistogram,'yes') ~= 1 && strcmp(CompressedHistogram,'no') ~= 1
            errordlg('You must enter "yes" or "no" for displaying the histograms in compressed format.');
            cd(handles.Current.StartupDirectory);
            return
        end
        SaveData = Answers{7};
        if isempty(SaveData)
            errordlg('You must enter "no", or a filename, in answer to the question about saving the data.');
            cd(handles.Current.StartupDirectory);
            return
        end
        OutputFileOverwrite = exist([cd,'/',SaveData],'file'); %%% TODO: Fix filename construction.
        if OutputFileOverwrite ~= 0
            Answer = questdlg('A file with that name already exists in the directory containing the raw measurements file. Do you wish to overwrite?','Confirm overwrite','Yes','No','No');
            if strcmp(Answer, 'No') == 1
                cd(handles.Current.StartupDirectory);
                return
            end
        end

        %%% Calculates the default bin size and range based on all
        %%% the data.
        AllMeasurementsCellArray = handles.Measurements.(MeasurementToExtract);
        SelectedMeasurementsCellArray = AllMeasurementsCellArray(:,FirstImage:LastImage);
        SelectedMeasurementsMatrix = cell2mat(SelectedMeasurementsCellArray(:));
        PotentialMaxHistogramValue = max(SelectedMeasurementsMatrix);
        PotentialMinHistogramValue = min(SelectedMeasurementsMatrix);
        %%% See whether the min and max histogram values were user-entered numbers or should be automatically calculated.
        if isempty(str2num(MinHistogramValue)) %#ok
            if strcmp(MinHistogramValue,'automatic') == 1
                MinHistogramValue = PotentialMinHistogramValue;
            else
                errordlg('The value entered for the minimum histogram value must be either a number or the word ''automatic''.')
                cd(handles.Current.StartupDirectory);
                return
            end
        else MinHistogramValue = str2num(MinHistogramValue); %#ok
        end
        if isempty(str2num(MaxHistogramValue)) %#ok
            if strcmp(MaxHistogramValue,'automatic') == 1
                MaxHistogramValue = PotentialMaxHistogramValue;
            else
                errordlg('The value entered for the maximum histogram value must be either a number or the word ''automatic''.')
                cd(handles.Current.StartupDirectory);
                return
            end
        else MaxHistogramValue = str2num(MaxHistogramValue); %#ok
        end
        %%% Determine plot bin locations.
        HistogramRange = MaxHistogramValue - MinHistogramValue;
        if HistogramRange <= 0
            errordlg('The numbers you entered for the minimum or maximum, or the number which was calculated automatically for one of these values, results in the range being zero or less.  For example, this would occur if you entered a minimum that is greater than the maximum which you asked to be automatically calculated.')
            cd(handles.Current.StartupDirectory);
            return
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

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% Calculates histogram data for cumulative histogram %%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if strcmpi(CumulativeHistogram, 'no') ~= 1
            HistogramData = histc(SelectedMeasurementsMatrix,BinLocations);
            %%% Deletes the last value of HistogramData, which is
            %%% always a zero (because it's the number of values
            %%% that match + inf).
            HistogramData(n+1) = [];
            FinalHistogramData(:,1) = HistogramData;
            %%% Saves this info in a variable, FigureSettings, which
            %%% will be stored later with the figure.
            FigureSettings{3} = FinalHistogramData;
            HistogramTitles{1} = ['Histogram of data from Image #', num2str(FirstImage), ' to #', num2str(LastImage)];
            FirstImage = 1;
            LastImage = 1;
            NumberOfImages = 1;
            %%% Saves the data to an excel file if desired.
            if strcmp(SaveData,'no') ~= 1
                %%% Open the file and name it appropriately.
                fid = fopen(SaveData, 'wt');
                %%% Write "Bins used" as the title of the first column.
                fwrite(fid, ['Bins used for ', MeasurementToExtract], 'char');
                %%% Tab to the second column.
                fwrite(fid, sprintf('\t'), 'char');
                %%% Write the HistogramTitle as a heading for the second column.
                fwrite(fid, char(HistogramTitles{1}), 'char');
                %%% Return, to the second row.
                fwrite(fid, sprintf('\n'), 'char');
                %%% Write the histogram data.
                WaitbarHandle = waitbar(0,'Writing the histogram data file...');
                NumberToWrite = size(PlotBinLocations,1);

                %%% Writes the first XTickLabel (which is a string) in the first
                %%% column.
                fwrite(fid, XTickLabels{1}, 'char');
                %%% Tab to the second column.
                fwrite(fid, sprintf('\t'), 'char');
                %%% Writes the first FinalHistogramData in the second column.
                fwrite(fid, sprintf('%g\t', FinalHistogramData(1,1)), 'char');
                %%% Return, to the next row.
                fwrite(fid, sprintf('\n'), 'char');
                %%% Writes all the middle values.
                for i = 2:NumberToWrite-1
                    %%% Writes the XTickLabel (which is a number) in
                    %%% the first column.
                    fwrite(fid, sprintf('%g\t', XTickLabels{i}), 'char');
                    %%% Writes the FinalHistogramData in the second column.
                    fwrite(fid, sprintf('%g\t', FinalHistogramData(i,1)), 'char');
                    %%% Return, to the next row.
                    fwrite(fid, sprintf('\n'), 'char');
                    waitbar(i/NumberToWrite)
                end
                %%% Writes the last value.
                if NumberToWrite ~= 1
                    %%% Writes the last XTickLabel (which is a string) in the first
                    %%% column.
                    fwrite(fid, XTickLabels{NumberToWrite}, 'char');
                    %%% Tab to the second column.
                    fwrite(fid, sprintf('\t'), 'char');
                    %%% Writes the first FinalHistogramData in the second column.
                    fwrite(fid, sprintf('%g\t', FinalHistogramData(NumberToWrite,1)), 'char');
                    %%% Return, to the next row.
                    fwrite(fid, sprintf('\n'), 'char');
                end
                close(WaitbarHandle)
                %%% Close the file
                fclose(fid);
                h = helpdlg(['The file ', SaveData, ' has been written to the directory where the raw measurements file is located.']);
                waitfor(h)
            end

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%% Calculates histogram data for non-cumulative histogram %%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        else
            %%% Preallocates the variable ListOfMeasurements.
            ListOfMeasurements{LastImage} = handles.Measurements.(MeasurementToExtract){LastImage};
            for ImageNumber = FirstImage:LastImage
                ListOfMeasurements{ImageNumber} = handles.Measurements.(MeasurementToExtract){ImageNumber};
                HistogramData = histc(ListOfMeasurements{ImageNumber},BinLocations);
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
            end
            %%% Saves this info in a variable, FigureSettings, which
            %%% will be stored later with the figure.
            FigureSettings{3} = FinalHistogramData;

            %%% Saves the data to an excel file if desired.
            if strcmp(SaveData,'no') ~= 1
                %%% Open the file and name it appropriately.
                fid = fopen(SaveData, 'wt');
                %%% Write "Bins used" as the title of the first column.
                fwrite(fid, ['Bins used for ', MeasurementToExtract], 'char');
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
                NumberBinsToWrite = size(PlotBinLocations,1);
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
                %%% Close the file
                fclose(fid);
                h = helpdlg(['The file ', SaveData, ' has been written to the directory where the raw measurements file is located.']);
                waitfor(h)
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% Displays histogram data for non-compressed histograms %%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        if strcmp(CompressedHistogram,'no') == 1
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
            FigureHandle = figure;
            set(FigureHandle, 'Name', EditedMeasurementToExtract);

            Increment = 0;
            for ImageNumber = FirstImage:LastImage
                Increment = Increment + 1;
                h = subplot(NumberDisplayRows,NumberDisplayColumns,Increment);
                bar('v6',PlotBinLocations,FinalHistogramData(:,ImageNumber))
                axis tight
                set(get(h,'XLabel'),'String',EditedMeasurementToExtract)
                set(h,'XTickLabel',XTickLabels)
                set(h,'XTick',PlotBinLocations)
                set(gca,'UserData',FigureSettings)
                title(HistogramTitles{ImageNumber})
                if Increment == 1
                    set(get(h,'YLabel'),'String','Number of objects')
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
            set(AxesHandles,'Units', 'pixels');
            NewFigurePosition = FigurePosition;
            NewHeight = FigurePosition(4) + 60;
            NewFigurePosition(4) = NewHeight;
            set(FigureHandle, 'Position', NewFigurePosition)
            set(AxesHandles,'Units', 'normalized');

            %%% Creates the frames and buttons and text for the "Change display"
            %%% buttons.
            %%% LeftFrame
            uicontrol('Parent',FigureHandle, ...
                'Unit',StdUnit, ...
                'BackgroundColor',[0.8 0.8 0.8], ...
                'Position',PointsPerPixel*[0 NewHeight-52 0.5*NewFigurePosition(3) 60], ...
                'Units','Normalized',...
                'Style','frame');
            %%% RightFrame
            uicontrol('Parent',FigureHandle, ...
                'Unit',StdUnit, ...
                'BackgroundColor',[0.8 0.8 0.8], ...
                'Position',PointsPerPixel*[0.5*NewFigurePosition(3) NewHeight-52 0.5*NewFigurePosition(3) 60], ...
                'Units','Normalized',...
                'Style','frame');
            %%% MiddleFrame
            uicontrol('Parent',FigureHandle, ...
                'Unit',StdUnit, ...
                'BackgroundColor',[0.8 0.8 0.8], ...
                'Position',PointsPerPixel*[100 NewHeight-26 240 30], ...
                'Units','Normalized',...
                'Style','frame');
            %%% Creates text 1
            uicontrol('Parent',FigureHandle, ...
                'BackgroundColor',get(FigureHandle,'Color'), ...
                'Unit',StdUnit, ...
                'Position',PointsPerPixel*[5 NewHeight-30 85 22], ...
                'Units','Normalized',...
                'String','Change plots:', ...
                'Style','text');
            %%% Creates text 2
            uicontrol('Parent',FigureHandle, ...
                'BackgroundColor',get(FigureHandle,'Color'), ...
                'Unit',StdUnit, ...
                'Position',PointsPerPixel*[0.5*NewFigurePosition(3)+85 NewHeight-30 85 22], ...
                'Units','Normalized',...
                'String','Change bars:', ...
                'Style','text');
            %%% Creates text 3
            uicontrol('Parent',FigureHandle, ...
                'BackgroundColor',get(FigureHandle,'Color'), ...
                'Unit',StdUnit, ...
                'Position',PointsPerPixel*[103 NewHeight-20 70 16], ...
                'Units','Normalized',...
                'String','X axis labels:', ...
                'Style','text');
            %%% These callbacks control what happens when display
            %%% buttons are pressed within the histogram display
            %%% window.
            Button1Callback = 'FigureHandle = gcf; AxesHandles = findobj(''Parent'', FigureHandle, ''Type'', ''axes''); try, propedit(AxesHandles,''v6''), catch, msgbox(''A bug in Matlab is preventing this function from working. Service Request #1-RR6M1''), end; drawnow';
            uicontrol('Parent',FigureHandle, ...
                'Unit',StdUnit, ...
                'BackgroundColor',StdColor, ...
                'CallBack',Button1Callback, ...
                'Position',PointsPerPixel*[25 NewHeight-48 85 22], ...
                'Units','Normalized',...
                'String','This window', ...
                'Style','pushbutton');
            Button2Callback = 'propedit(gca,''v6''); drawnow';
            uicontrol('Parent',FigureHandle, ...
                'Unit',StdUnit, ...
                'BackgroundColor',StdColor, ...
                'CallBack',Button2Callback, ...
                'Position',PointsPerPixel*[115 NewHeight-48 70 22], ...
                'Units','Normalized',...
                'String','Current', ...
                'Style','pushbutton');
            Button3Callback = 'FigureHandles = findobj(''Parent'', 0); AxesHandles = findobj(FigureHandles, ''Type'', ''axes''); axis(AxesHandles, ''manual''); try, propedit(AxesHandles,''v6''), catch, msgbox(''A bug in Matlab is preventing this function from working. Service Request #1-RR6M1''), end, drawnow';
            uicontrol('Parent',FigureHandle, ...
                'Unit',StdUnit, ...
                'BackgroundColor',StdColor, ...
                'CallBack', Button3Callback, ...
                'Position',PointsPerPixel*[190 NewHeight-48 85 22], ...
                'Units','Normalized',...
                'String','All windows', ...
                'Style','pushbutton');
            Button4Callback = 'FigureHandle = gcf; PatchHandles = findobj(FigureHandle, ''Type'', ''patch''); try, propedit(PatchHandles,''v6''), catch, msgbox(''A bug in Matlab is preventing this function from working. Service Request #1-RR6M1''), end; drawnow';
            uicontrol('Parent',FigureHandle, ...
                'Unit',StdUnit, ...
                'BackgroundColor',StdColor, ...
                'CallBack', Button4Callback, ...
                'Position',PointsPerPixel*[0.5*NewFigurePosition(3)+5 NewHeight-48 85 22], ...
                'Units','Normalized',...
                'String','This window', ...
                'Style','pushbutton');
            Button5Callback = 'AxisHandle = gca; PatchHandles = findobj(''Parent'', AxisHandle, ''Type'', ''patch''); try, propedit(PatchHandles,''v6''), catch, msgbox(''A bug in Matlab is preventing this function from working. Service Request #1-RR6M1''), end; drawnow';
            uicontrol('Parent',FigureHandle, ...
                'Unit',StdUnit, ...
                'BackgroundColor',StdColor, ...
                'CallBack', Button5Callback, ...
                'Position',PointsPerPixel*[0.5*NewFigurePosition(3)+95 NewHeight-48 70 22], ...
                'Units','Normalized',...
                'String','Current', ...
                'Style','pushbutton');
            Button6Callback = 'FigureHandles = findobj(''Parent'', 0); PatchHandles = findobj(FigureHandles, ''Type'', ''patch''); try, propedit(PatchHandles,''v6''), catch, msgbox(''A bug in Matlab is preventing this function from working. Service Request #1-RR6M1''), end; drawnow';
            uicontrol('Parent',FigureHandle, ...
                'Unit',StdUnit, ...
                'BackgroundColor',StdColor, ...
                'CallBack', Button6Callback, ...
                'Position',PointsPerPixel*[0.5*NewFigurePosition(3)+170 NewHeight-48 85 22], ...
                'Units','Normalized',...
                'String','All windows', ...
                'Style','pushbutton');
            Button7Callback = 'msgbox(''Histogram display info: (1) Data outside the range you specified to calculate histogram bins are added together and displayed in the first and last bars of the histogram.  (2) Only the display can be changed in this window, including axis limits.  The histogram bins themselves cannot be changed here because the data must be recalculated. (3) If a change you make using the "Change display" buttons does not seem to take effect in all of the desired windows, try pressing enter several times within that box, or look in the bottom of the Property Editor window that opens when you first press one of those buttons.  There may be a message describing why.  For example, you may need to deselect "Auto" before changing the limits of the axes. (4) The labels for each bar specify the low bound for that bin.  In other words, each bar includes data equal to or greater than the label, but less than the label on the bar to its right. (5) If the tick mark labels are overlapping each other on the X axis, click a "Change display" button and either change the font size on the "Style" tab, or check the boxes marked "Auto" for "Ticks" and "Labels" on the "X axis" tab. Be sure to check both boxes, or the labels will not be accurate.  Changing the labels to "Auto" cannot be undone, and you will lose the detailed info about what values were actually used for the histogram bins.'')';
            uicontrol('Parent',FigureHandle, ...
                'Unit',StdUnit, ...
                'BackgroundColor',StdColor, ...
                'CallBack', Button7Callback, ...
                'Position',PointsPerPixel*[5 NewHeight-48 15 22], ...
                'Units','Normalized',...
                'String','?', ...
                'Style','pushbutton');
            %%% Hide every other label button.
            Button8Callback = 'FigureSettings = get(gca,''UserData'');  PlotBinLocations = FigureSettings{1}; XTickLabels = FigureSettings{2}; AxesHandles = findobj(gcf, ''Type'', ''axes''); if ceil(length(PlotBinLocations)/2) ~= length(PlotBinLocations)/2, PlotBinLocations(length(PlotBinLocations)) = []; XTickLabels(length(XTickLabels)) = []; end; PlotBinLocations2 = reshape(PlotBinLocations,2,[]); XTickLabels2 = reshape(XTickLabels,2,[]); set(AxesHandles,''XTick'',PlotBinLocations2(1,:)); set(AxesHandles,''XTickLabel'',XTickLabels2(1,:)); clear';
            uicontrol('Parent',FigureHandle, ...
                'Unit',StdUnit, ...
                'BackgroundColor',StdColor, ...
                'CallBack',Button8Callback, ...
                'Position',PointsPerPixel*[177 NewHeight-22 45 22], ...
                'Units','Normalized',...
                'String','Fewer', ...
                'Style','pushbutton');
            %%% Decimal places X axis labels.
            Button9Callback = 'FigureSettings = get(gca,''UserData''); PlotBinLocations = FigureSettings{1}; PreXTickLabels = FigureSettings{2}; XTickLabels = PreXTickLabels(2:end-1); AxesHandles = findobj(gcf, ''Type'', ''axes''); set(AxesHandles,''XTick'',PlotBinLocations); NumberOfDecimals = inputdlg(''Enter the number of decimal places to display'',''Enter the number of decimal places'',1,{''0''}); NumberValues = cell2mat(XTickLabels); Command = [''%.'',num2str(NumberOfDecimals{1}),''f'']; NewNumberValues = num2str(NumberValues'',Command); NewNumberValuesPlusFirstLast = [PreXTickLabels(1); cellstr(NewNumberValues); PreXTickLabels(end)]; set(AxesHandles,''XTickLabel'',NewNumberValuesPlusFirstLast); clear, drawnow';
            uicontrol('Parent',FigureHandle, ...
                'Unit',StdUnit, ...
                'BackgroundColor',StdColor, ...
                'CallBack',Button9Callback, ...
                'Position',PointsPerPixel*[227 NewHeight-22 50 22], ...
                'Units','Normalized',...
                'String','Decimals', ...
                'Style','pushbutton');
            %%% Restore original X axis labels.
            Button10Callback = 'FigureSettings = get(gca,''UserData'');  PlotBinLocations = FigureSettings{1}; XTickLabels = FigureSettings{2}; AxesHandles = findobj(gcf, ''Type'', ''axes''); set(AxesHandles,''XTick'',PlotBinLocations); set(AxesHandles,''XTickLabel'',XTickLabels); clear';
            uicontrol('Parent',FigureHandle, ...
                'Unit',StdUnit, ...
                'BackgroundColor',StdColor, ...
                'CallBack',Button10Callback, ...
                'Position',PointsPerPixel*[282 NewHeight-22 50 22], ...
                'Units','Normalized',...
                'String','Restore', ...
                'Style','pushbutton');
            %%% Puts the menu and tool bar in the figure window.
            set(FigureHandle,'toolbar', 'figure')

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%% Displays histogram data for compressed histograms %%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        elseif strcmp(CompressedHistogram,'yes') == 1
            FigureHandle = figure;
            imagesc(FinalHistogramData'),
            colormap(gray), colorbar,
            %title(['Title goes here'])
            AxisHandle = gca;
            set(get(AxisHandle,'XLabel'),'String',EditedMeasurementToExtract)
            set(AxisHandle,'XTickLabel',XTickLabels)
            NewPlotBinLocations = 1:length(FinalHistogramData');
            set(AxisHandle,'XTick',NewPlotBinLocations)
            set(FigureHandle,'UserData',FigureSettings)
        else errordlg('In answering the question of whether to display a compressed histogram, you must type "yes" or "no".');
            cd(handles.Current.StartupDirectory);
            return
        end
    end % Goes with cancel button when selecting the measurement to display.
end
cd(handles.Current.StartupDirectory);