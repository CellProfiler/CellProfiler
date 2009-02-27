function varargout = CPShowPlateMapDataFig(varargin)
% CPSHOWPLATEMAPDATAFIG M-file for CPShowPlateMapDataFig.fig
%      CPSHOWPLATEMAPDATAFIG, by itself, creates a new CPSHOWPLATEMAPDATAFIG or raises the existing
%      singleton*.
%
%      H = CPSHOWPLATEMAPDATAFIG returns the handle to a new CPSHOWPLATEMAPDATAFIG or the handle to
%      the existing singleton*.
%
%      CPSHOWPLATEMAPDATAFIG('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in CPSHOWPLATEMAPDATAFIG.M with the given input arguments.
%
%      CPSHOWPLATEMAPDATAFIG('Property','Value',...) creates a new CPSHOWPLATEMAPDATAFIG or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before CPShowPlateMapDataFig_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to CPShowPlateMapDataFig_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help CPShowPlateMapDataFig

% Last Modified by GUIDE v2.5 07-Nov-2008 13:35:11

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @CPShowPlateMapDataFig_OpeningFcn, ...
                   'gui_OutputFcn',  @CPShowPlateMapDataFig_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

% --- Executes just before CPShowPlateMapDataFig is made visible.
function CPShowPlateMapDataFig_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to CPShowPlateMapDataFig (see VARARGIN)

% Choose default command line output for CPShowPlateMapDataFig
handles.output = hObject;

handles.Metadata = varargin{1};
handles.handles  = varargin{2};

% Update handles structure
guidata(hObject, handles);
UpdateObjectSourcePopup(handles);
UpdatePlateNamePopup(handles);

% UIWAIT makes CPShowPlateMapDataFig wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = CPShowPlateMapDataFig_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

varargout{1} = handles.output;

% --- Executes on button press in UpdateButton.
function UpdateButton_Callback(hObject, eventdata, handles)
% hObject    handle to UpdateButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
UpdateAxes(handles);

% --------------------------------------------------------------------
function FileMenu_Callback(hObject, eventdata, handles)
% hObject    handle to FileMenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function OpenMenuItem_Callback(hObject, eventdata, handles)
% hObject    handle to OpenMenuItem (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
file = uigetfile('*.fig');
if ~isequal(file, 0)
    open(file);
end

% --------------------------------------------------------------------
function PrintMenuItem_Callback(hObject, eventdata, handles)
% hObject    handle to PrintMenuItem (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
printdlg(handles.figure1)

% --------------------------------------------------------------------
function CloseMenuItem_Callback(hObject, eventdata, handles)
% hObject    handle to CloseMenuItem (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
selection = questdlg(['Close ' get(handles.figure1,'Name') '?'],...
                     ['Close ' get(handles.figure1,'Name') '...'],...
                     'Yes','No','Yes');
if strcmp(selection,'No')
    return;
end

delete(handles.figure1)


% --- Executes on selection change in popupmenu1.
function popupmenu1_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns popupmenu1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu1


% --- Executes during object creation, after setting all properties.
function popupmenu1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
     set(hObject,'BackgroundColor','white');
end

set(hObject, 'String', {'plot(rand(5))', 'plot(sin(1:0.01:25))', 'bar(1:.5:10)', 'plot(membrane)', 'surf(peaks)'});


% --- Executes on selection change in ObjectSource.
function ObjectSource_Callback(hObject, eventdata, handles)
% hObject    handle to ObjectSource (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns ObjectSource contents as cell array
%        contents{get(hObject,'Value')} returns selected item from ObjectSource
UpdateCategoryPopup(handles);

% --- Executes during object creation, after setting all properties.
function ObjectSource_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ObjectSource (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in Category.
function Category_Callback(hObject, eventdata, handles)
% hObject    handle to Category (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns Category contents as cell array
%        contents{get(hObject,'Value')} returns selected item from Category
UpdateMeasurementPopup(handles);

% --- Executes during object creation, after setting all properties.
function Category_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Category (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in Measurement.
function Measurement_Callback(hObject, eventdata, handles)
% hObject    handle to Measurement (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns Measurement contents as cell array
%        contents{get(hObject,'Value')} returns selected item from Measurement
UpdateImageSourcePopup(handles);

% --- Executes during object creation, after setting all properties.
function Measurement_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Measurement (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in ImageSource.
function ImageSource_Callback(hObject, eventdata, handles)
% hObject    handle to ImageSource (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns ImageSource contents as cell array
%        contents{get(hObject,'Value')} returns selected item from ImageSource
UpdateScalePopup(handles);

% --- Executes during object creation, after setting all properties.
function ImageSource_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ImageSource (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in Scale.
function Scale_Callback(hObject, eventdata, handles)
% hObject    handle to Scale (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns Scale contents as cell array
%        contents{get(hObject,'Value')} returns selected item from Scale


% --- Executes during object creation, after setting all properties.
function Scale_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Scale (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in PlateName.
function PlateName_Callback(hObject, eventdata, handles)
% hObject    handle to PlateName (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns PlateName contents as cell array
%        contents{get(hObject,'Value')} returns selected item from PlateName
UpdateAxes(handles);

% --- Executes during object creation, after setting all properties.
function PlateName_CreateFcn(hObject, eventdata, handles)
% hObject    handle to PlateName (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Update the object source popup from the measurements in the handles
function UpdateObjectSourcePopup(handles)
% handles   handles to the objects + the loaded .MAT file
hObject = handles.ObjectSource;
OldSelection = GetSelectedPopupValue(hObject);
Measurements = handles.handles.Measurements;
set(hObject,'String',fieldnames(Measurements));
SetSelectedPopupValue(hObject,OldSelection);
UpdateCategoryPopup(handles);

% --- Update the measurement category popup from the measurements
%     and use the object source to constrain the allowable values
function UpdateCategoryPopup(handles)
% handles   handles to the objects + the loaded .MAT file
hObject = handles.Category;
OldSelection = GetSelectedPopupValue(hObject);
ObjectSource = GetSelectedPopupValue(handles.ObjectSource);
Measurements = fieldnames(handles.handles.Measurements.(ObjectSource));
Categories = unique(strtok(Measurements,'_'));
set(hObject,'String',Categories);
SetSelectedPopupValue(hObject,OldSelection);
UpdateMeasurementPopup(handles);

% --- Update the measurement popup from the measurements
%     and use the object source and category to constrain 
%     the allowable values
function UpdateMeasurementPopup(handles)
% handles   handles to the objects + the loaded .MAT file
hObject = handles.Measurement;
OldSelection = GetSelectedPopupValue(hObject);
ObjectSource = GetSelectedPopupValue(handles.ObjectSource);
Category = GetSelectedPopupValue(handles.Category);
Measurements = fieldnames(handles.handles.Measurements.(ObjectSource));
[Categories,Measurements] = strtok(Measurements,'_');
Measurements = Measurements(strcmp(Categories,Category));
Measurements = cellfun(@(x) x(2:end),Measurements,'UniformOutput',0);
Measurements = unique(strtok(Measurements,'_'));
set(hObject,'String',Measurements);
SetSelectedPopupValue(hObject,OldSelection);
UpdateImageSourcePopup(handles);

% --- Update the image source popup from the measurements
%     and use the object source, measurement and category to constrain 
%     the allowable values. Hide the popup and its label if not relevant.
function UpdateImageSourcePopup(handles)
% handles   handles to the objects + the loaded .MAT file
hObject = handles.ImageSource;
Visible = 'off';
OldSelection = GetSelectedPopupValue(hObject);
ObjectSource = GetSelectedPopupValue(handles.ObjectSource);
Category = GetSelectedPopupValue(handles.Category);
Measurement = GetSelectedPopupValue(handles.Measurement);
Measurements = fieldnames(handles.handles.Measurements.(ObjectSource));
[Categories,Measurements] = strtok(Measurements,'_');
Measurements = Measurements(strcmp(Categories,Category));
Measurements = cellfun(@(x) x(2:end),Measurements,'UniformOutput',0);
[Measurements,ImageSources] = strtok(Measurements,'_');
ImageSources = ImageSources(strcmp(Measurements,Measurement));
if ~ (isempty(ImageSources) | isempty(ImageSources{1}))
    Visible='on';
    ImageSources = cellfun(@(x) x(2:end),ImageSources,'UniformOutput',0);
    ImageSources = unique(strtok(ImageSources,'_'));
    set(hObject,'String',ImageSources);
    SetSelectedPopupValue(hObject,OldSelection);
end
set(hObject,'Visible',Visible);
set(handles.ImageSourceLabel,'Visible',Visible);
UpdateScalePopup(handles);

% --- Update the scale popup from the measurements
%     and use the object source, measurement, category, and image source
%     to constrain the allowable values. 
%     Hide the popup and its label if not relevant.
function UpdateScalePopup(handles)
% handles   handles to the objects + the loaded .MAT file
hObject = handles.Scale;
Visible = 'off';
if ~ strcmp(get(handles.ImageSource,'Visible'),'off')
    OldSelection = GetSelectedPopupValue(hObject);
    ObjectSource = GetSelectedPopupValue(handles.ObjectSource);
    Category = GetSelectedPopupValue(handles.Category);
    Measurement = GetSelectedPopupValue(handles.Measurement);
    ImageSource = GetSelectedPopupValue(handles.ImageSource);
    Measurements = fieldnames(handles.handles.Measurements.(ObjectSource));
    [Categories,Measurements] = strtok(Measurements,'_');
    Measurements = Measurements(strcmp(Categories,Category));
    Measurements = cellfun(@(x) x(2:end),Measurements,'UniformOutput',0);
    [Measurements,ImageSources] = strtok(Measurements,'_');
    ImageSources = ImageSources(strcmp(Measurements,Measurement));
    if ~ (isempty(ImageSources) | isempty(ImageSources{1}))
        ImageSources = cellfun(@(x) x(2:end),ImageSources,'UniformOutput',0);
        [ImageSources, Scales] = strtok(ImageSources,'_');
        Scales = Scales(strcmp(ImageSources,ImageSource));
        if ~ (isempty(Scales) | isempty(Scales{1}))
            Visible = 'on';
            Scales = cellfun(@(x) x(2:end),Scales,'UniformOutput',0);
            set(hObject,'String',Scales);
            SetSelectedPopupValue(hObject, OldSelection);
        end
    end
end
set(hObject,'Visible',Visible);
set(handles.ScaleLabel,'Visible',Visible);

% --- Update the plate name popup. Fill it with all of the plate names
%     in the .MAT file
function UpdatePlateNamePopup(handles)
% handles   handles to the objects + the loaded .MAT file + metadata tags
PlateNameField = handles.Metadata.PlateNameMeasurement;
PlateNames = unique(handles.handles.Measurements.Image.(PlateNameField));
set(handles.PlateName,'String',PlateNames);

% --- Update the axes with a platemap that has the selected measurement
function UpdateAxes(handles)
data=zeros(GetPlateSize(handles));
map=cell(GetPlateSize(handles));
data(:) = NaN;
PlateNameFieldname = handles.Metadata.PlateNameMeasurement;
PlateName = GetSelectedPopupValue(handles.PlateName);
ImageIndices = find(strcmp(handles.handles.Measurements.Image.(PlateNameFieldname),PlateName));
ObjectSourceFieldname = GetSelectedPopupValue(handles.ObjectSource);
MeasurementFieldname  = GetMeasurementFieldname(handles);
WellRowFieldname      = handles.Metadata.WellRowMeasurement;
WellColumnFieldname   = handles.Metadata.WellColumnMeasurement;
WellNames = unique(arrayfun(@(x) WellName(handles,x),ImageIndices,'UniformOutput',0));
AggregateOperation  = GetSelectedPopupValue(handles.AggregateOperation);
if all(cellfun(@iscell,handles.handles.Measurements.Image.(PlateNameFieldname))),
    handles.handles.Measurements.Image.(PlateNameFieldname) = cat(2,handles.handles.Measurements.Image.(PlateNameFieldname){:});
end
if all(cellfun(@iscell,handles.handles.Measurements.Image.(WellRowFieldname))),
    handles.handles.Measurements.Image.(WellRowFieldname) = cat(2,handles.handles.Measurements.Image.(WellRowFieldname){:});
end
if all(cellfun(@iscell,handles.handles.Measurements.Image.(WellColumnFieldname))),
    handles.handles.Measurements.Image.(WellColumnFieldname) = cat(2,handles.handles.Measurements.Image.(WellColumnFieldname){:});
end
for WellNameIndex=1:length(WellNames)
    IndivWellName = WellNames{WellNameIndex};
    [WellRow,WellColumn] = strtok(IndivWellName,'_');
    WellColumn=WellColumn(2:end);
    ImageIndices = find(...
        strcmp(handles.handles.Measurements.Image.(PlateNameFieldname),PlateName) &...
        strcmp(handles.handles.Measurements.Image.(WellRowFieldname),WellRow) &...
        strcmp(handles.handles.Measurements.Image.(WellColumnFieldname),WellColumn));
    measurements = vertcat(handles.handles.Measurements.(ObjectSourceFieldname).(MeasurementFieldname){ImageIndices});
    switch(AggregateOperation)
        case 'Mean'
            value=mean(measurements);
        case 'Median'
            value=median(measurements);
        case 'Minimum'
            value=min(measurements);
        case 'Maximum'
            value=max(measurements);
        case 'Standard deviation'
            value=std(measurements);
    end
    [x,y] = GetWellIndices(handles,ImageIndices(1));
    data(x,y) = value;
    map(x,y) = {ImageIndices};
end
Minimum = min(data(~isnan(data)));
Maximum = max(data(~isnan(data)));
set(handles.InfoText,'String',sprintf('Min: %.4f, Max: %.4f',Minimum,Maximum));
if size(data,1)==16
    inner = 9;
    outer = 11;
else
    inner = 15;
    outer = 18;
end;
[image, hImage] = CPplatemap(handles.PlateAxes,data,inner,outer,3,Minimum,Maximum);
cbar=colorbar('peer',handles.PlateAxes);
if Maximum ~= Minimum
    ytick = (0:8)*16;
    set(cbar,'YTick',ytick);
    set(cbar,'YTickLabel',ytick*(Maximum-Minimum)/128+Minimum);
end
set(hImage,'ButtonDownFcn',@(hObject,event_data) ButtonDownFcn(hObject,event_data,handles,map));

% --- Get a canonical well name (e.g. A_01) for the indexed image
function name=WellName(handles,ImageIndex)
WellRowFieldname    = handles.Metadata.WellRowMeasurement;
WellColumnFieldname = handles.Metadata.WellColumnMeasurement;
WellRow             = char(handles.handles.Measurements.Image.(WellRowFieldname){ImageIndex});
WellColumn          = char(handles.handles.Measurements.Image.(WellColumnFieldname){ImageIndex});
name = [WellRow,'_',WellColumn];

% --- Return the size (columns x rows) of a plate.
function PlateSize=GetPlateSize(handles)
% handles   handles to the graphics objext + the loaded .MAT file
% PlateSize a vector with the # of columns and rows
if handles.Metadata.PlateSize == 96
    PlateSize = [8,12];
else
    PlateSize = [16,24];
end

% --- Return the fieldname of the measurement selected by the user
function MeasurementFieldname = GetMeasurementFieldname(handles)
% handles    handles to the graphics objects
Category    = GetSelectedPopupValue(handles.Category);
Measurement = GetSelectedPopupValue(handles.Measurement);
if strcmpi(get(handles.ImageSource,'Visible'),'on')
    ImageSource = GetSelectedPopupValue(handles.ImageSource);
    if strcmpi(get(handles.Scale,'Visible'),'on')
        Scale = GetSelectedPopupValue(handles.Scale);
        MeasurementFieldname = CPjoinstrings(Category, Measurement, ImageSource, Scale);
    else
        MeasurementFieldname = CPjoinstrings(Category, Measurement, ImageSource);
    end
else
    MeasurementFieldname = CPjoinstrings(Category, Measurement);
end

% --- Set a well value, given an image index for that well
function [WellRow,WellColumn]=GetWellIndices(handles, ImageIndex)
% data       the array that holds the plate map data
% handles    handles to the graphics object + the loaded .MAT file
% value      the value to display in the well
% ImageIndex the index of the image of that well
WellRowFieldname    = handles.Metadata.WellRowMeasurement;
WellColumnFieldname = handles.Metadata.WellColumnMeasurement;
WellRow             = handles.handles.Measurements.Image.(WellRowFieldname){ImageIndex};
WellColumn          = handles.handles.Measurements.Image.(WellColumnFieldname){ImageIndex};
WellRow = find(strcmp([{'A'},{'B'},{'C'},{'D'},{'E'},{'F'},{'G'},{'H'},...
                       {'I'},{'J'},{'K'},{'L'},{'M'},{'N'},{'O'},{'P'}],WellRow));
WellColumn = str2double(WellColumn);

function selection=GetSelectedPopupValue(hObject)
choices = get(hObject,'String');
index   = get(hObject,'Value');
if iscell(choices)
    selection = choices{index};
else
    selection = choices;
end

function SetSelectedPopupValue(hObject, value)
choices = get(hObject,'String');
index   = find(strcmp(choices,value));
if ~ isempty(index)
    set(hObject,'Value',index);
else
    set(hObject,'Value',1);
end


% --- Executes on selection change in AggregateOperation.
function AggregateOperation_Callback(hObject, eventdata, handles)
% hObject    handle to AggregateOperation (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns AggregateOperation contents as cell array
%        contents{get(hObject,'Value')} returns selected item from AggregateOperation


% --- Executes during object creation, after setting all properties.
function AggregateOperation_CreateFcn(hObject, eventdata, handles)
% hObject    handle to AggregateOperation (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on mouse press over figure background.
function ButtonDownFcn(hObject, eventdata, handles, map)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if strcmpi(get(handles.figure1,'SelectionType'),'Open')
    CurrentPoint = get(handles.PlateAxes,'CurrentPoint');
    xtick = get(handles.PlateAxes,'XTick');
    ytick = get(handles.PlateAxes,'YTick');
    xtick_distance = xtick(2)-xtick(1);
    ytick_distance = ytick(2)-ytick(1);
    xtick_idx = find(xtick > CurrentPoint(1,1)-xtick_distance/2);
    ytick_idx = find(ytick > CurrentPoint(1,2)-ytick_distance/2);
    if (~ isempty(xtick_idx)) && (~ isempty(ytick_idx))
        fig           = CPfigure(handles.handles,'Image');
        ImageIndices  = map{ytick_idx(1),xtick_idx(1)};
        ImageFields   = fieldnames(handles.handles.Measurements.Image);
        FileFields    = ImageFields(strncmp(ImageFields,'FileName_',9));
        PathFields    = ImageFields(strncmp(ImageFields,'PathName_',9));
        FilesPerImage = size(FileFields,1);
        ImagesPerWell = length(ImageIndices);
        WellRowFieldname    = handles.Metadata.WellRowMeasurement;
        WellColumnFieldname = handles.Metadata.WellColumnMeasurement;
        PlateFieldname      = handles.Metadata.PlateNameMeasurement;
        WellRow       = handles.handles.Measurements.Image.(WellRowFieldname){ImageIndices(1)};
        WellColumn    = handles.handles.Measurements.Image.(WellColumnFieldname){ImageIndices(1)};
        PlateName     = handles.handles.Measurements.Image.(PlateFieldname){ImageIndices(1)};
        set(fig,'Name',sprintf('Plate: %s, Well: %s%s',PlateName,WellRow,WellColumn));
        for i=1:FilesPerImage
            for j=1:ImagesPerWell
                axis=subplot(ImagesPerWell,FilesPerImage,i+(j-1)*FilesPerImage,'Parent',fig);
                FileField = FileFields{i};
                PathField = PathFields{i};
                File      = handles.handles.Measurements.Image.(FileField){ImageIndices(j)};
                Path      = handles.handles.Measurements.Image.(PathField){ImageIndices(j)};
                FileName  = fullfile(Path,File);
                image     = CPimread(FileName);
                if i==1 && j==1
                    ImageWidth = size(image,1);
                    ImageHeight = size(image,2);
                    ScaledWidth = 200;
                    AspectRatio = ImageWidth / ImageHeight;
                    figpos      = get(fig,'Position');
                    Width       = (ScaledWidth+40) * FilesPerImage;
                    Height      = (ScaledWidth / AspectRatio + 40)*ImagesPerWell;
                    set(fig,'Position',[figpos(1),figpos(2),Width,Height]);
                end
                CPimagesc(image, handles.handles, axis);
                set(axis,'DataAspectRatioMode','manual');
                title(axis,File);
            end
        end
    end
end


