function CPcumhistogram(h_Parent,tempData,Flip)

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
% $Revision: 2802 $

Logical = tempData.Logical;
ThresholdVal = tempData.ThresholdVal;
BinVar = tempData.BinVar;
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
try
    [ObjectTypename,FeatureType,FeatureNo] = CPgetfeature(handles);
catch
    ErrorMessage = lasterr;
    error(ErrorMessage(30:end))
end
if isempty(ObjectTypename),return,end
MeasurementToExtract = [handles.Measurements.(ObjectTypename).([FeatureType,'Features']){FeatureNo},' of ', ObjectTypename];

%%% Put the measurements for this feature in a cell array, one
%%% cell for each image set.
tmp = handles.Measurements.(ObjectTypename).(FeatureType);
%%% Calculates some values for the next dialog box.
TotalNumberImageSets = length(tmp);
TextTotalNumberImageSets = num2str(TotalNumberImageSets);


%%% Opens a window that lets the user choose histogram settings
%%% This function returns a UserAnswers structure with the
%%% information required to carry out the calculations.
try UserAnswers = UserAnswersWindow(handles);
catch CPerrordlg(lasterr)
    return
end

% If Cancel button pressed, return
if ~isfield(UserAnswers, 'FirstSample')
    return
end

FirstImage=UserAnswers.FirstSample;
LastImage=UserAnswers.LastSample;

switch UserAnswers.Color
    case 'Red'
        LineTag='Red';
        LineColor='r';
    case 'Blue'
        LineTag='Blue';
        LineColor='b';
    case 'Green'
        LineTag='Green';
        LineColor='g';
    case 'Yellow'
        LineTag='Yellow';
        LineColor='y';
    case 'Magenta'
        LineTag='Magenta';
        LineColor='m';
    case 'Cyan'
        LineTag='Cyan';
        LineColor='c';
    case 'Black'
        LineTag='Black';
        LineColor='k';
    case 'White'
        LineTag='White';
        LineColor='w';
    otherwise
        LineTag='CellProfiler background';
        LineColor=[.7 .7 .9];
end


Measurements = cell((FirstImage-LastImage),1);
i = 1;
for k = FirstImage:LastImage
    Measurements{i} = tmp{k}(:,FeatureNo);
    i = i+1;
end

if ~strcmp(Logical,'None')
    try % try/catch not really necessary, but just in case.
        [ObjectTypename,FeatureType,FeatureNo] = CPgetfeature(handles);
    catch
        ErrorMessage = lasterr;
        error(ErrorMessage(30:end))
    end
    if isempty(ObjectTypename),return,end
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
if ~strcmp(Logical,'None')
    SelectMeasurementsCellArray = MeasurementToThresholdValueOn(1:end);
    OutputMeasurements{1,2} = cell2mat(SelectMeasurementsCellArray(:));
    AdditionalInfoForTitle = [' for objects where ', MeasurementToThresholdValueOnName, Logical, num2str(ThresholdVal)];
else AdditionalInfoForTitle = [];
end
%%% Applies the specified ThresholdValue and gives a cell
%%% array as output.
if strcmp(Logical,'None')
    %%% If the user selected None, the measurements are not
    %%% altered.
elseif strcmpi(Logical,'>')
    OutputMeasurements{1,1} = OutputMeasurements{1,1}(OutputMeasurements{1,2} > ThresholdVal);
elseif strcmpi(Logical,'>=')
    OutputMeasurements{1,1} = OutputMeasurements{1,1}(OutputMeasurements{1,2} >= ThresholdVal);
elseif strcmpi(Logical,'<')
    OutputMeasurements{1,1} = OutputMeasurements{1,1}(OutputMeasurements{1,2} < ThresholdVal);
elseif strcmpi(Logical,'<=')
    OutputMeasurements{1,1} = OutputMeasurements{1,1}(OutputMeasurements{1,2} <= ThresholdVal);
else 
    OutputMeasurements{1,1} = OutputMeasurements{1,1}(OutputMeasurements{1,2} == ThresholdVal);
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

if strcmp(BinVar,'Percentages')
    for i = 1: size(FinalHistogramData,2)
        SumForThatColumn = sum(FinalHistogramData(:,i));
        FinalHistogramData(:,i) = 100*FinalHistogramData(:,i)/SumForThatColumn;
    end
end

%%% Saves this info in a variable, FigureSettings, which
%%% will be stored later with the figure.
FigureSettings{3} = FinalHistogramData;
PlotBinLocations = FigureSettings{1};

AxesHandles = findobj(h_Parent,'Tag','BarTag');
for i = 1:length(AxesHandles)
    h2 = axes('Position',get(AxesHandles(i),'Position'));
    if strcmp(Flip,'AxesFlipped')
        plot(FinalHistogramData(:,1),PlotBinLocations,'LineWidth',2, 'Tag', LineTag);
    else
        plot(PlotBinLocations,FinalHistogramData(:,1),'LineWidth',2, 'Tag', LineTag);
    end
    set(h2,'YAxisLocation','right','Color','none','ActivePositionProperty','Position','XTickLabel',[],'XTick',[],'YTickLabel',[],'YTick',[]);
    set(h2,'XLim',get(AxesHandles(i),'XLim'),'Layer','top','YLim',get(AxesHandles(i),'YLim'));
    set(findobj(findobj('tag',LineTag),'type','line'),'color',LineColor);
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function UserAnswers = UserAnswersWindow(handles)
% This function displays a window for user input. If the return variable 'UserAnswers' is empty
% it means that either no measurements were found or the user pressed
% the Cancel button (or the window was closed). 


% Store font size
FontSize = handles.Preferences.FontSize;

% Create UserWindow window
UserWindow = figure;
set(UserWindow,'units','inches','resize','on','menubar','none','toolbar','none','numbertitle','off','Name','Choose control histogram settings','Color',[.7 .7 .9]);
% Some variables controling the sizes of uicontrols
uiheight = 0.3;
% Set window size in inches, depends on the number of prompts
pos = get(UserWindow,'position');
Height = uiheight*8;
Width  = 5.2;
set(UserWindow,'position',[pos(1)+1 pos(2) Width Height]);

ypos = Height - uiheight*2.5;

NumMat=[];
for x=1:handles.Current.NumberOfImageSets
    NumMat=[NumMat;x];
end

ReverseNumMat=NumMat(end:-1:1);

% UserWindow user input
uicontrol(UserWindow,'style','text','String','First sample number to use:','FontSize',FontSize,'FontWeight', 'bold',...
    'HorizontalAlignment','left','units','inches','position',[0.2 ypos 2.8 uiheight],'BackgroundColor',get(UserWindow,'color'));
FirstSample = uicontrol(UserWindow,'style','popupmenu','String',{NumMat},'FontSize',FontSize,...
    'HorizontalAlignment','left','units','inches','position',[3 ypos+.05 1.8 uiheight],'BackgroundColor',get(UserWindow, 'color'));

ypos = ypos - uiheight;

uicontrol(UserWindow,'style','text','String','Last sample number to use:','FontSize',FontSize,'FontWeight', 'bold',...
    'HorizontalAlignment','left','units','inches','position',[0.2 ypos 2.8 uiheight],'BackgroundColor',get(UserWindow,'color'));
LastSample = uicontrol(UserWindow,'style','popupmenu','String',{ReverseNumMat},'FontSize',FontSize,...
    'HorizontalAlignment','left','units','inches','position',[3 ypos+.05 1.8 uiheight],'BackgroundColor',get(UserWindow, 'color'));

%Help button
LastSample_Help_Callback = 'CPhelpdlg(''To display data from only one image, choose the image number as both the first and last sample number.'')';
uicontrol(UserWindow,'style','pushbutton','String','?','FontSize',FontSize,...
    'HorizontalAlignment','center','units','inches','position',[4.85 ypos+.1 0.14 uiheight-0.05],'BackgroundColor',get(UserWindow,'color'),'FontWeight', 'bold',...
    'Callback', LastSample_Help_Callback);


ypos = ypos - uiheight;

uicontrol(UserWindow,'style','text','String','Color of the control histogram plot:','FontSize',FontSize,'FontWeight', 'bold',...
    'HorizontalAlignment','left','units','inches','position',[0.2 ypos 2.8 uiheight],'BackgroundColor',get(UserWindow,'color'));
Color = uicontrol(UserWindow,'style','popupmenu','String',{'Red','Blue','Green','Yellow','Magenta','Cyan','Black','White','CellProfiler background'},'FontSize',FontSize,...
    'HorizontalAlignment','left','units','inches','position',[3 ypos+.05 1.8 uiheight],'BackgroundColor',get(UserWindow, 'color'));


%%% OK AND CANCEL BUTTONS
posx = (Width - 1.7)/2;               % Centers buttons horizontally
okbutton = uicontrol(UserWindow,'style','pushbutton','String','OK','Fontweight','bold','FontSize',FontSize,'units','inches',...
    'position',[posx 0.1 0.75 0.3],'BackgroundColor',[.7 .7 .9],'Callback','[cobj,cfig] = gcbo;set(cobj,''UserData'',1);uiresume(cfig);clear cobj cfig;','BackgroundColor',[.7 .7 .9]);
cancelbutton = uicontrol(UserWindow,'style','pushbutton','String','Cancel','Fontweight','bold','FontSize',FontSize,'units','inches',...
    'position',[posx+0.95 0.1 0.75 0.3],'Callback','close(gcf)','BackgroundColor',[.7 .7 .9]);


% Repeat until valid input has been entered or the window is destroyed
while 1

    % Wait until window is destroyed or uiresume() is called
    uiwait(UserWindow)

    % Action depending on user input
    if ishandle(okbutton)               % The OK button pressed
        %UserAnswers = get(UserWindow,'UserData');

        % Populate structure array
        UserAnswers.FirstSample = get(FirstSample,'value');
        UserAnswers.LastSample = ReverseNumMat(get(LastSample,'value'));
        UserAnswers.Color = get(Color,'value');

        if UserAnswers.FirstSample > UserAnswers.LastSample         % Error check for sample numbers
            warnfig=CPwarndlg('Please make the first sample number less than or equal to the last sample number! Please try again.');
            uiwait(warnfig);
            set(okbutton,'UserData',[]);
        else
            switch UserAnswers.Color
                case 1
                    UserAnswers.Color='Red';
                case 2
                    UserAnswers.Color='Blue';
                case 3
                    UserAnswers.Color='Green';
                case 4
                    UserAnswers.Color='Yellow';
                case 5
                    UserAnswers.Color='Magenta';
                case 6
                    UserAnswers.Color='Cyan';
                case 7
                    UserAnswers.Color='Black';
                case 8
                    UserAnswers.Color='White';
                otherwise
                    UserAnswers.Color='CellProfiler background';
            end
            delete(UserWindow);
            return
        end
    else
        UserAnswers = [];
        if ishandle(UserWindow),delete(UserWindow);end
        return
    end
end
        
