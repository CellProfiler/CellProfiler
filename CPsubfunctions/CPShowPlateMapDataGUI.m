function varargout = CPShowPlateMapDataGUI(varargin)
% CPSHOWPLATEMAPDATAGUI M-file for CPShowPlateMapDataGUI.fig
%      CPSHOWPLATEMAPDATAGUI, by itself, creates a new CPSHOWPLATEMAPDATAGUI or raises the existing
%      singleton*.
%
%      H = CPSHOWPLATEMAPDATAGUI returns the handle to a new CPSHOWPLATEMAPDATAGUI or the handle to
%      the existing singleton*.
%
%      CPSHOWPLATEMAPDATAGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in CPSHOWPLATEMAPDATAGUI.M with the given input arguments.
%
%      CPSHOWPLATEMAPDATAGUI('Property','Value',...) creates a new CPSHOWPLATEMAPDATAGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before CPShowPlateMapDataGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to CPShowPlateMapDataGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help CPShowPlateMapDataGUI

% Last Modified by GUIDE v2.5 06-Nov-2008 13:41:08

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @CPShowPlateMapDataGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @CPShowPlateMapDataGUI_OutputFcn, ...
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


% --- Executes just before CPShowPlateMapDataGUI is made visible.
function CPShowPlateMapDataGUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to CPShowPlateMapDataGUI (see VARARGIN)

% Choose default command line output for CPShowPlateMapDataGUI
handles.output = hObject;

% Update handles structure
handles.handles = varargin{1};
guidata(hObject, handles);
setappdata(handles.OKButton,'Accepted',0);
ImageCategory_OpenFcn(handles.PlateNameMeasurementCategory, handles, varargin{1});
ImageCategory_OpenFcn(handles.WellColumnCategory, handles, varargin{1});
ImageCategory_OpenFcn(handles.WellRowCategory, handles, varargin{1});

ImageMeasurement_OpenFcn(handles.PlateNameMeasurement, handles.PlateNameMeasurementCategory, handles, varargin{1});
ImageMeasurement_OpenFcn(handles.WellColumnMeasurement, handles.WellColumnCategory, handles, varargin{1});
ImageMeasurement_OpenFcn(handles.WellRowMeasurement, handles.WellRowCategory, handles, varargin{1});

% UIWAIT makes CPShowPlateMapDataGUI wait for user response (see UIRESUME)
uiwait(handles.figure1);

% --- Outputs from this function are returned to the command line.
function varargout = CPShowPlateMapDataGUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if ~ isempty(handles)
    PlateNameMeasurement = [GetSelectedPopupValue(handles.PlateNameMeasurementCategory),'_',...
                            GetSelectedPopupValue(handles.PlateNameMeasurement)];
    WellRowMeasurement = [GetSelectedPopupValue(handles.WellRowCategory),'_',...
                            GetSelectedPopupValue(handles.WellRowMeasurement)];
    WellColumnMeasurement = [GetSelectedPopupValue(handles.WellColumnCategory),'_',...
                            GetSelectedPopupValue(handles.WellColumnMeasurement)];
    Accepted = getappdata(handles.OKButton,'Accepted');
    varargout{1} = struct(...
        'PlateNameMeasurement',PlateNameMeasurement,...
        'WellRowMeasurement', WellRowMeasurement,...
        'WellColumnMeasurement', WellColumnMeasurement,...
        'PlateSize',str2num(GetSelectedPopupValue(handles.PlateSize)),...
        'Accepted',Accepted);
    close(handles.figure1);
end

% --- Executes on selection change in PlateNameMeasurementCategory.
function PlateNameMeasurementCategory_Callback(hObject, eventdata, handles)
% hObject    handle to PlateNameMeasurementCategory (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns PlateNameMeasurementCategory contents as cell array
%        contents{get(hObject,'Value')} returns selected item from PlateNameMeasurementCategory
ImageMeasurement_OpenFcn(handles.PlateNameMeasurement, hObject, handles, handles.handles);

% --- Executes during object creation, after setting all properties.
function PlateNameMeasurementCategory_CreateFcn(hObject, eventdata, handles)
% hObject    handle to PlateNameMeasurementCategory (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function ImageCategory_OpenFcn(hObject,handles,cphandles)
ImageMeasurements = fieldnames(cphandles.Measurements.Image);
ImageCategories = unique(strtok(ImageMeasurements,'_'));
set(hObject,'String',ImageCategories);
SetSelectedPopupValue(hObject,'Metadata');

% --- Executes on selection change in PlateNameMeasurement.
function PlateNameMeasurement_Callback(hObject, eventdata, handles)
% hObject    handle to PlateNameMeasurement (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns PlateNameMeasurement contents as cell array
%        contents{get(hObject,'Value')} returns selected item from PlateNameMeasurement

% --- Executes during object creation, after setting all properties.
function PlateNameMeasurement_CreateFcn(hObject, eventdata, handles)
% hObject    handle to PlateNameMeasurement (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function ImageMeasurement_OpenFcn(hObject, hParent, handles, cphandles)
CurrentValue = GetSelectedPopupValue(hObject);
CurrentCategory = GetSelectedPopupValue(hParent);
ImageMeasurements = fieldnames(cphandles.Measurements.Image);
[Categories ,Measurements] = strtok(ImageMeasurements,'_');
Measurements = Measurements(strcmp(Categories,CurrentCategory));
Measurements = unique(cellfun(@(x) x(2:end),Measurements,'UniformOutput',0));
set(hObject,'String',Measurements);
SetSelectedPopupValue(hObject, CurrentValue);

% --- Executes on selection change in PlateSize.
function PlateSize_Callback(hObject, eventdata, handles)
% hObject    handle to PlateSize (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns PlateSize contents as cell array
%        contents{get(hObject,'Value')} returns selected item from PlateSize


% --- Executes during object creation, after setting all properties.
function PlateSize_CreateFcn(hObject, eventdata, handles)
% hObject    handle to PlateSize (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in OKButton.
function OKButton_Callback(hObject, eventdata, handles)
% hObject    handle to OKButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
setappdata(hObject,'Accepted',1);
uiresume(gcbf);

% --- Executes on button press in CancelButton.
function CancelButton_Callback(hObject, eventdata, handles)
% hObject    handle to CancelButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
uiresume(gcbf);

% --- Executes on selection change in MeasurementSource.
function MeasurementSource_Callback(hObject, eventdata, handles)
% hObject    handle to MeasurementSource (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns MeasurementSource contents as cell array
%        contents{get(hObject,'Value')} returns selected item from MeasurementSource


% --- Executes during object creation, after setting all properties.
function MeasurementSource_CreateFcn(hObject, eventdata, handles)
% hObject    handle to MeasurementSource (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in MeasurementCategory.
function MeasurementCategory_Callback(hObject, eventdata, handles)
% hObject    handle to MeasurementCategory (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns MeasurementCategory contents as cell array
%        contents{get(hObject,'Value')} returns selected item from MeasurementCategory


% --- Executes during object creation, after setting all properties.
function MeasurementCategory_CreateFcn(hObject, eventdata, handles)
% hObject    handle to MeasurementCategory (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in MeasurementMeasurement.
function MeasurementMeasurement_Callback(hObject, eventdata, handles)
% hObject    handle to MeasurementMeasurement (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns MeasurementMeasurement contents as cell array
%        contents{get(hObject,'Value')} returns selected item from MeasurementMeasurement


% --- Executes during object creation, after setting all properties.
function MeasurementMeasurement_CreateFcn(hObject, eventdata, handles)
% hObject    handle to MeasurementMeasurement (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in MeasurementImage.
function MeasurementImage_Callback(hObject, eventdata, handles)
% hObject    handle to MeasurementImage (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns MeasurementImage contents as cell array
%        contents{get(hObject,'Value')} returns selected item from MeasurementImage


% --- Executes during object creation, after setting all properties.
function MeasurementImage_CreateFcn(hObject, eventdata, handles)
% hObject    handle to MeasurementImage (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in MeasurementScale.
function MeasurementScale_Callback(hObject, eventdata, handles)
% hObject    handle to MeasurementScale (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns MeasurementScale contents as cell array
%        contents{get(hObject,'Value')} returns selected item from MeasurementScale


% --- Executes during object creation, after setting all properties.
function MeasurementScale_CreateFcn(hObject, eventdata, handles)
% hObject    handle to MeasurementScale (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in WellColumnCategory.
function WellColumnCategory_Callback(hObject, eventdata, handles)
% hObject    handle to WellColumnCategory (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

ImageMeasurement_OpenFcn(handles.WellColumnMeasurement,hObject, eventdata, handles);

% --- Executes during object creation, after setting all properties.
function WellColumnCategory_CreateFcn(hObject, eventdata, handles)
% hObject    handle to WellColumnCategory (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in WellColumnMeasurement.
function WellColumnMeasurement_Callback(hObject, eventdata, handles)
% hObject    handle to WellColumnMeasurement (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns WellColumnMeasurement contents as cell array
%        contents{get(hObject,'Value')} returns selected item from WellColumnMeasurement


% --- Executes during object creation, after setting all properties.
function WellColumnMeasurement_CreateFcn(hObject, eventdata, handles)
% hObject    handle to WellColumnMeasurement (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in WellRowCategory.
function WellRowCategory_Callback(hObject, eventdata, handles)
% hObject    handle to WellRowCategory (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns WellRowCategory contents as cell array
%        contents{get(hObject,'Value')} returns selected item from WellRowCategory
ImageMeasurement_OpenFcn(handles.WellRowMeasurement,hObject, eventdata, handles);


% --- Executes during object creation, after setting all properties.
function WellRowCategory_CreateFcn(hObject, eventdata, handles)
% hObject    handle to WellRowCategory (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in WellRowMeasurement.
function WellRowMeasurement_Callback(hObject, eventdata, handles)
% hObject    handle to WellRowMeasurement (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns WellRowMeasurement contents as cell array
%        contents{get(hObject,'Value')} returns selected item from WellRowMeasurement


% --- Executes during object creation, after setting all properties.
function WellRowMeasurement_CreateFcn(hObject, eventdata, handles)
% hObject    handle to WellRowMeasurement (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

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
end