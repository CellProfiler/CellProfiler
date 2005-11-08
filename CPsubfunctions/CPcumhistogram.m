function CPcumhistogram(h_Parent,tempData)

GreaterOrLessThan = tempData.GreaterOrLessThan;
ThresholdValue = tempData.ThresholdValue;
NumberOrPercent = tempData.NumberOrPercent;
FigureSettings = tempData.FigureSettings;
BinLocations = tempData.BinLocations;
handles = tempData.handles;

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
[ObjectTypename,FeatureType,FeatureNo] = CPgetfeature(handles);
if isempty(ObjectTypename),return,end
MeasurementToExtract = [handles.Measurements.(ObjectTypename).([FeatureType,'Features']){FeatureNo},' of ', ObjectTypename];

%%% Put the measurements for this feature in a cell array, one
%%% cell for each image set.
tmp = handles.Measurements.(ObjectTypename).(FeatureType);
%%% Calculates some values for the next dialog box.
TotalNumberImageSets = length(tmp);
TextTotalNumberImageSets = num2str(TotalNumberImageSets);
%%% Ask the user to specify histogram settings.
Prompts{1} = 'Enter the first image set to use for Histogram';
Prompts{2} = ['Enter the last last image set to use for Histogram (the total number of image sets with data in the file is ',TextTotalNumberImageSets,').'];
Prompts{3} = 'What color do you want this histogram to be?';
Defaults{1} = '1';
Defaults{2} = TextTotalNumberImageSets;
Defaults{3} = 'red';
Answers = inputdlg(Prompts(1:3),'Choose histogram settings',1,Defaults(1:3),'on');
FirstImage = str2num(Answers{1});
LastImage = str2num(Answers{2});
LineColor = Answers{3};
Measurements = cell((FirstImage-LastImage),1);
i = 1;
for k = FirstImage:LastImage
    Measurements{i} = tmp{k}(:,FeatureNo);
    i = i+1;
end

if ~strcmpi(GreaterOrLessThan,'A')
    [ObjectTypename,FeatureType,FeatureNo] = CPgetfeature(handles);
    MeasurementToThresholdValueOnName = handles.Measurements.(ObjectTypename).([FeatureType,'Features'])(FeatureNo);
    tmp = handles.Measurements.(ObjectTypename).(FeatureType);
    MeasurementToThresholdValueOn = cell(length(tmp),1);
    for k = 1:length(tmp)
        MeasurementToThresholdValueOn{k} = tmp{k}(:,FeatureNo);
    end
end

SelectedMeasurementsCellArray = Measurements(1:end);
SelectedMeasurementsMatrix = cell2mat(SelectedMeasurementsCellArray(:));

OutputMeasurements{1,1} = SelectedMeasurementsMatrix;
%%% Retrieves the measurements to threshold on, if requested.
if ~strcmpi(GreaterOrLessThan,'A')
    SelectMeasurementsCellArray = MeasurementToThresholdValueOn(1:end);
    OutputMeasurements{1,2} = cell2mat(SelectMeasurementsCellArray(:));
    AdditionalInfoForTitle = [' for objects where ', MeasurementToThresholdValueOnName, GreaterOrLessThan, num2str(ThresholdValue)];
else AdditionalInfoForTitle = [];
end
%%% Applies the specified ThresholdValue and gives a cell
%%% array as output.
if strcmpi(GreaterOrLessThan,'A')
    %%% If the user selected A for all, the measurements are not
    %%% altered.
elseif strcmpi(GreaterOrLessThan,'>')
    OutputMeasurements{1,1} = OutputMeasurements{1,1}(OutputMeasurements{1,2} > ThresholdValue);
elseif strcmpi(GreaterOrLessThan,'>=')
    OutputMeasurements{1,1} = OutputMeasurements{1,1}(OutputMeasurements{1,2} >= ThresholdValue);
elseif strcmpi(GreaterOrLessThan,'<')
    OutputMeasurements{1,1} = OutputMeasurements{1,1}(OutputMeasurements{1,2} < ThresholdValue);
elseif strcmpi(GreaterOrLessThan,'<=')
    OutputMeasurements{1,1} = OutputMeasurements{1,1}(OutputMeasurements{1,2} <= ThresholdValue);
elseif strcmpi(GreaterOrLessThan,'=')
    OutputMeasurements{1,1} = OutputMeasurements{1,1}(OutputMeasurements{1,2} == ThresholdValue);
else error(['The value you entered for the method to threshold ', GreaterOrLessThan, ' was not valid.  Acceptable entries are >, >=, =, <=, <.']);
end

if isempty(OutputMeasurements{1,1}) == 1
    HistogramData = [];
else
    HistogramData = histc(OutputMeasurements{1,1},BinLocations);
end

%%% Deletes the last value of HistogramData, which is
%%% always a zero (because it's the number of values
%%% that match + inf).
HistogramData(end) = [];
FinalHistogramData(:,1) = HistogramData;

if strncmpi(NumberOrPercent,'P',1)
    for i = 1: size(FinalHistogramData,2)
        SumForThatColumn = sum(FinalHistogramData(:,i));
        FinalHistogramData(:,i) = FinalHistogramData(:,i)/SumForThatColumn;
    end
end

%%% Saves this info in a variable, FigureSettings, which
%%% will be stored later with the figure.
FigureSettings{3} = FinalHistogramData;
PlotBinLocations = FigureSettings{1};

AxesHandles = findobj(h_Parent,'Tag','BarTag');
for i = 1:length(AxesHandles)
    h2 = axes('Position',get(AxesHandles(i),'Position'));
    plot(PlotBinLocations,FinalHistogramData(:,1),'LineWidth',2,'tag',LineColor);
    set(h2,'YAxisLocation','right','Color','none','ActivePositionProperty','Position','XTickLabel',[],'XTick',[],'YTickLabel',[],'YTick',[]);
    set(h2,'XLim',get(AxesHandles(i),'XLim'),'Layer','top');
    set(findobj(findobj('tag',LineColor),'type','line'),'color',LineColor);
    set(h2,'XLim',get(AxesHandles(i),'XLim'))
    set(h2,'YLim',get(AxesHandles(i),'YLim'))
end