function varargout = HelpLoadSavedVariables(varargin);
% HelpLoadSavedVariables M-file for HelpLoadSavedVariables.fig
%      HelpLoadSavedVariables, by itself, creates a new HelpLoadSavedVariables or raises the existing
%      singleton*.
%
%      H = HelpLoadSavedVariables returns the handle to a new HelpLoadSavedVariables or the handle to
%      the existing singleton*.
%
%      HelpLoadSavedVariables('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in HelpLoadSavedVariables.M with the given input arguments.
%
%      HelpLoadSavedVariables('Property','Value',...) creates a new HelpLoadSavedVariables or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before HelpLoadSavedVariables_OpeningFunction gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to HelpLoadSavedVariables_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Copyright 2002-2003 The MathWorks, Inc.

% Edit the above text to modify the response to HelpLoadSavedVariables HelpLoadSavedVariables

% Last Modified by GUIDE v2.5 06-Dec-2004 21:32:18

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @HelpLoadSavedVariables_OpeningFcn, ...
                   'gui_OutputFcn',  @HelpLoadSavedVariables_OutputFcn, ...
                   'gui_LayoutFcn',  @HelpLoadSavedVariables_LayoutFcn, ...
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


% --- Executes just before HelpLoadSavedVariables is made visible.
function HelpLoadSavedVariables_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to HelpLoadSavedVariables (see VARARGIN)

handles.savedVariables = varargin{1};
handles.defaultVariables = varargin{2};
handles.errorString = varargin{3};
handles.algorithmName = varargin{4};

% Choose default command line output for HelpLoadSavedVariables
handles.output = 0;

% Update handles structure
guidata(hObject, handles);

defaultString = get(handles.text3, 'String');
errorString{1} = handles.errorString;
errorString{2} = defaultString;
set(handles.text3, 'String', errorString);

textString1 = get(handles.text1, 'String');
newString1{1} = textString1;
newString1{2} = ['for ' handles.algorithmName];
set(handles.text1, 'String', newString1);

textString2 = get(handles.text2, 'String');
newString2{1} = textString2;
newString2{2} = ['for ' handles.algorithmName];
set(handles.text2, 'String', newString2);

savedbox_Callback(handles.savedbox, eventdata, handles);
defaultbox_Callback(handles.defaultbox, eventdata, handles);

% UIWAIT makes HelpLoadSavedVariables wait for user response (see UIRESUME)
uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = HelpLoadSavedVariables_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;
delete(handles.figure1);

% --- Executes on selection change in savedbox.
function savedbox_Callback(hObject, eventdata, handles)
% hObject    handle to savedbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns savedbox contents as cell array
%        contents{get(hObject,'Value')} returns selected item from savedbox
set(hObject, 'String', handles.savedVariables);


% --- Executes during object creation, after setting all properties.
function savedbox_CreateFcn(hObject, eventdata, handles)
% hObject    handle to savedbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end



% --- Executes on selection change in defaultbox.
function defaultbox_Callback(hObject, eventdata, handles)
% hObject    handle to defaultbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns defaultbox contents as cell array
%        contents{get(hObject,'Value')} returns selected item from defaultbox
set(hObject,'String',handles.defaultVariables);


% --- Executes during object creation, after setting all properties.
function defaultbox_CreateFcn(hObject, eventdata, handles)
% hObject    handle to defaultbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end


% --- Executes on button press in savedbutton.
function savedbutton_Callback(hObject, eventdata, handles)
% hObject    handle to savedbutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.output = 1;
guidata(hObject, handles);
uiresume(handles.figure1);


% --- Executes on button press in defaultbutton.
function defaultbutton_Callback(hObject, eventdata, handles)
% hObject    handle to defaultbutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.output = 2;
guidata(hObject, handles);
uiresume(handles.figure1);

% --- Creates and returns a handle to the GUI figure. 
function h1 = HelpLoadSavedVariables_LayoutFcn(policy)
% policy - create a new figure or use a singleton. 'new' or 'reuse'.

persistent hsingleton;
if strcmpi(policy, 'reuse') & ishandle(hsingleton)
    h1 = hsingleton;
    return;
end

appdata = [];
appdata.GUIDEOptions = struct(...
    'active_h', [], ...
    'taginfo', struct(...
    'figure', 2, ...
    'listbox', 3, ...
    'text', 4, ...
    'pushbutton', 3), ...
    'override', 0, ...
    'release', 13, ...
    'resize', 'none', ...
    'accessibility', 'callback', ...
    'mfile', 1, ...
    'callbacks', 1, ...
    'singleton', 1, ...
    'syscolorfig', 1, ...
    'blocking', 0, ...
    'lastSavedFile', 'C:\mit\urop\CellProfiler\HelpLoadSavedVariables.m');
appdata.lastValidTag = 'figure1';
appdata.GUIDELayoutEditor = [];

h1 = figure(...
'Units','characters',...
'Color',[0.925490196078431 0.913725490196078 0.847058823529412],...
'Colormap',[0 0 0.5625;0 0 0.625;0 0 0.6875;0 0 0.75;0 0 0.8125;0 0 0.875;0 0 0.9375;0 0 1;0 0.0625 1;0 0.125 1;0 0.1875 1;0 0.25 1;0 0.3125 1;0 0.375 1;0 0.4375 1;0 0.5 1;0 0.5625 1;0 0.625 1;0 0.6875 1;0 0.75 1;0 0.8125 1;0 0.875 1;0 0.9375 1;0 1 1;0.0625 1 1;0.125 1 0.9375;0.1875 1 0.875;0.25 1 0.8125;0.3125 1 0.75;0.375 1 0.6875;0.4375 1 0.625;0.5 1 0.5625;0.5625 1 0.5;0.625 1 0.4375;0.6875 1 0.375;0.75 1 0.3125;0.8125 1 0.25;0.875 1 0.1875;0.9375 1 0.125;1 1 0.0625;1 1 0;1 0.9375 0;1 0.875 0;1 0.8125 0;1 0.75 0;1 0.6875 0;1 0.625 0;1 0.5625 0;1 0.5 0;1 0.4375 0;1 0.375 0;1 0.3125 0;1 0.25 0;1 0.1875 0;1 0.125 0;1 0.0625 0;1 0 0;0.9375 0 0;0.875 0 0;0.8125 0 0;0.75 0 0;0.6875 0 0;0.625 0 0;0.5625 0 0],...
'IntegerHandle','off',...
'InvertHardcopy',get(0,'defaultfigureInvertHardcopy'),...
'MenuBar','none',...
'Name','Check the settings',...
'NumberTitle','off',...
'PaperPosition',get(0,'defaultfigurePaperPosition'),...
'Position',[103.8 29.1538461538462 112 32.3076923076923],...
'Renderer',get(0,'defaultfigureRenderer'),...
'RendererMode','manual',...
'Resize','off',...
'HandleVisibility','callback',...
'Tag','figure1',...
'UserData',[],...
'Behavior',get(0,'defaultfigureBehavior'),...
'Visible','on',...
'CreateFcn', {@local_CreateFcn, '', appdata} );

appdata = [];
appdata.lastValidTag = 'savedbox';

h2 = uicontrol(...
'Parent',h1,...
'Units','characters',...
'Callback','HelpLoadSavedVariables(''savedbox_Callback'',gcbo,[],guidata(gcbo))',...
'Position',[9.8 12 30.2 15.4615384615385],...
'String',{  'Listbox' },...
'Style','listbox',...
'Value',1,...
'CreateFcn', {@local_CreateFcn, 'HelpLoadSavedVariables(''savedbox_CreateFcn'',gcbo,[],guidata(gcbo))', appdata} ,...
'Tag','savedbox',...
'Behavior',get(0,'defaultuicontrolBehavior'));

appdata = [];
appdata.lastValidTag = 'defaultbox';

h3 = uicontrol(...
'Parent',h1,...
'Units','characters',...
'Callback','HelpLoadSavedVariables(''defaultbox_Callback'',gcbo,[],guidata(gcbo))',...
'Position',[49.8 12 30.2 15.4615384615385],...
'String',{  'Listbox' },...
'Style','listbox',...
'Value',1,...
'CreateFcn', {@local_CreateFcn, 'HelpLoadSavedVariables(''defaultbox_CreateFcn'',gcbo,[],guidata(gcbo))', appdata} ,...
'Tag','defaultbox',...
'Behavior',get(0,'defaultuicontrolBehavior'));

appdata = [];
appdata.lastValidTag = 'text1';

h4 = uicontrol(...
'Parent',h1,...
'Units','characters',...
'Position',[9 29.1538461538462 30.2 2],...
'String','Saved Settings',...
'Style','text',...
'Tag','text1',...
'Behavior',get(0,'defaultuicontrolBehavior'),...
'CreateFcn', {@local_CreateFcn, '', appdata} );

appdata = [];
appdata.lastValidTag = 'text2';

h5 = uicontrol(...
'Parent',h1,...
'Units','characters',...
'Position',[49 29.1538461538462 30.2 2],...
'String','Default Settings',...
'Style','text',...
'Tag','text2',...
'Behavior',get(0,'defaultuicontrolBehavior'),...
'CreateFcn', {@local_CreateFcn, '', appdata} );

appdata = [];
appdata.lastValidTag = 'savedbutton';

h6 = uicontrol(...
'Parent',h1,...
'Units','characters',...
'Callback','HelpLoadSavedVariables(''savedbutton_Callback'',gcbo,[],guidata(gcbo))',...
'Position',[13.2 3.38461538461539 22.6 1.92307692307692],...
'String','Saved Settings',...
'Tag','savedbutton',...
'Behavior',get(0,'defaultuicontrolBehavior'),...
'CreateFcn', {@local_CreateFcn, '', appdata} );

appdata = [];
appdata.lastValidTag = 'defaultbutton';

h7 = uicontrol(...
'Parent',h1,...
'Units','characters',...
'Callback','HelpLoadSavedVariables(''defaultbutton_Callback'',gcbo,[],guidata(gcbo))',...
'Position',[53.8 3.38461538461539 22.6 1.92307692307692],...
'String','Default Settings',...
'Tag','defaultbutton',...
'Behavior',get(0,'defaultuicontrolBehavior'),...
'CreateFcn', {@local_CreateFcn, '', appdata} );

appdata = [];
appdata.lastValidTag = 'text3';

h8 = uicontrol(...
'Parent',h1,...
'Units','characters',...
'Position',[9.8 6.92307692307692 70.2 3.69230769230769],...
'String','Do you want to try to load the saved or default settings?',...
'Style','text',...
'Tag','text3',...
'Behavior',get(0,'defaultuicontrolBehavior'),...
'CreateFcn', {@local_CreateFcn, '', appdata} );


hsingleton = h1;


% --- Set application data first then calling the CreateFcn. 
function local_CreateFcn(hObject, eventdata, createfcn, appdata)

if ~isempty(appdata)
   names = fieldnames(appdata);
   for i=1:length(names)
       name = char(names(i));
       setappdata(hObject, name, getfield(appdata,name));
   end
end

if ~isempty(createfcn)
   eval(createfcn);
end


% --- Handles default GUIDE GUI creation and callback dispatch
function varargout = gui_mainfcn(gui_State, varargin);


%   GUI_MAINFCN provides these command line APIs for dealing with GUIs
%
%      HelpLoadSavedVariables, by itself, creates a new HelpLoadSavedVariables or raises the existing
%      singleton*.
%
%      H = HelpLoadSavedVariables returns the handle to a new HelpLoadSavedVariables or the handle to
%      the existing singleton*.
%
%      HelpLoadSavedVariables('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in HelpLoadSavedVariables.M with the given input arguments.
%
%      HelpLoadSavedVariables('Property','Value',...) creates a new HelpLoadSavedVariables or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before untitled_OpeningFunction gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to untitled_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".

%   Copyright 1984-2004 The MathWorks, Inc.
%   $Revision$ $Date$

gui_StateFields =  {'gui_Name'
                    'gui_Singleton'
                    'gui_OpeningFcn'
                    'gui_OutputFcn'
                    'gui_LayoutFcn'
                    'gui_Callback'};
gui_Mfile = '';
for i=1:length(gui_StateFields)
    if ~isfield(gui_State, gui_StateFields{i})
        error('Could not find field %s in the gui_State struct in GUI M-file %s', gui_StateFields{i}, gui_Mfile);        
    elseif isequal(gui_StateFields{i}, 'gui_Name')
        gui_Mfile = [gui_State.(gui_StateFields{i}), '.m'];
    end
end

numargin = length(varargin);

if numargin == 0
    % HelpLoadSavedVariables
    % create the GUI
    gui_Create = 1;
elseif isequal(ishandle(varargin{1}), 1) && ispc && iscom(varargin{1}) && isequal(varargin{1},gcbo)
    % HelpLoadSavedVariables(ACTIVEX,...)    
    vin{1} = gui_State.gui_Name;
    vin{2} = [get(varargin{1}.Peer, 'Tag'), '_', varargin{end}];
    vin{3} = varargin{1};
    vin{4} = varargin{end-1};
    vin{5} = guidata(varargin{1}.Peer);
    feval(vin{:});
    return;
elseif ischar(varargin{1}) && numargin>1 && isequal(ishandle(varargin{2}), 1)
    % HelpLoadSavedVariables('CALLBACK',hObject,eventData,handles,...)
    gui_Create = 0;
else
    % HelpLoadSavedVariables(...)
    % create the GUI and hand varargin to the openingfcn
    gui_Create = 1;
end

if gui_Create == 0
    varargin{1} = gui_State.gui_Callback;
    if nargout
        [varargout{1:nargout}] = feval(varargin{:});
    else
        feval(varargin{:});
    end
else
    if gui_State.gui_Singleton
        gui_SingletonOpt = 'reuse';
    else
        gui_SingletonOpt = 'new';
    end
    
    % Open fig file with stored settings.  Note: This executes all component
    % specific CreateFunctions with an empty HANDLES structure.
    
    % Do feval on layout code in m-file if it exists
    if ~isempty(gui_State.gui_LayoutFcn)
        gui_hFigure = feval(gui_State.gui_LayoutFcn, gui_SingletonOpt);
        % openfig (called by local_openfig below) does this for guis without
        % the LayoutFcn. Be sure to do it here so guis show up on screen.
        movegui(gui_hFigure,'onscreen')
    else
        gui_hFigure = local_openfig(gui_State.gui_Name, gui_SingletonOpt);            
        % If the figure has InGUIInitialization it was not completely created
        % on the last pass.  Delete this handle and try again.
        if isappdata(gui_hFigure, 'InGUIInitialization')
            delete(gui_hFigure);
            gui_hFigure = local_openfig(gui_State.gui_Name, gui_SingletonOpt);            
        end
    end
    
    % Set flag to indicate starting GUI initialization
    setappdata(gui_hFigure,'InGUIInitialization',1);

    % Fetch GUIDE Application options
    gui_Options = getappdata(gui_hFigure,'GUIDEOptions');
    
    if ~isappdata(gui_hFigure,'GUIOnScreen')
        % Adjust background color
        if gui_Options.syscolorfig 
            set(gui_hFigure,'Color', get(0,'DefaultUicontrolBackgroundColor'));
        end

        % Generate HANDLES structure and store with GUIDATA
        guidata(gui_hFigure, guihandles(gui_hFigure));
    end
    
    % If user specified 'Visible','off' in p/v pairs, don't make the figure
    % visible.
    gui_MakeVisible = 1;
    for ind=1:2:length(varargin)
        if length(varargin) == ind
            break;
        end
        len1 = min(length('visible'),length(varargin{ind}));
        len2 = min(length('off'),length(varargin{ind+1}));
        if ischar(varargin{ind}) && ischar(varargin{ind+1}) && ...
                strncmpi(varargin{ind},'visible',len1) && len2 > 1
            if strncmpi(varargin{ind+1},'off',len2)
                gui_MakeVisible = 0;
            elseif strncmpi(varargin{ind+1},'on',len2)
                gui_MakeVisible = 1;
            end
        end
    end
    
    % Check for figure param value pairs
    for index=1:2:length(varargin)
        if length(varargin) == index
            break;
        end
        try set(gui_hFigure, varargin{index}, varargin{index+1}), catch break, end
    end

    % If handle visibility is set to 'callback', turn it on until finished
    % with OpeningFcn
    gui_HandleVisibility = get(gui_hFigure,'HandleVisibility');
    if strcmp(gui_HandleVisibility, 'callback')
        set(gui_hFigure,'HandleVisibility', 'on');
    end
    
    feval(gui_State.gui_OpeningFcn, gui_hFigure, [], guidata(gui_hFigure), varargin{:});
    
    if ishandle(gui_hFigure)
        % Update handle visibility
        set(gui_hFigure,'HandleVisibility', gui_HandleVisibility);
        
        % Make figure visible
        if gui_MakeVisible
            set(gui_hFigure, 'Visible', 'on')
            if gui_Options.singleton 
                setappdata(gui_hFigure,'GUIOnScreen', 1);
            end
        end

        % Done with GUI initialization
        rmappdata(gui_hFigure,'InGUIInitialization');
    end
    
    % If handle visibility is set to 'callback', turn it on until finished with
    % OutputFcn
    if ishandle(gui_hFigure)
        gui_HandleVisibility = get(gui_hFigure,'HandleVisibility');
        if strcmp(gui_HandleVisibility, 'callback')
            set(gui_hFigure,'HandleVisibility', 'on');
        end
        gui_Handles = guidata(gui_hFigure);
    else
        gui_Handles = [];
    end
    
    if nargout
        [varargout{1:nargout}] = feval(gui_State.gui_OutputFcn, gui_hFigure, [], gui_Handles);
    else
        feval(gui_State.gui_OutputFcn, gui_hFigure, [], gui_Handles);
    end
    
    if ishandle(gui_hFigure)
        set(gui_hFigure,'HandleVisibility', gui_HandleVisibility);
    end
end    

function gui_hFigure = local_openfig(name, singleton)

% openfig with three arguments was new from R13. Try to call that first, if
% failed, try the old openfig.
try 
    gui_hFigure = openfig(name, singleton, 'auto');
catch
    % OPENFIG did not accept 3rd input argument until R13,
    % toggle default figure visible to prevent the figure
    % from showing up too soon.
    gui_OldDefaultVisible = get(0,'defaultFigureVisible');
    set(0,'defaultFigureVisible','off');
    gui_hFigure = openfig(name, singleton);
    set(0,'defaultFigureVisible',gui_OldDefaultVisible);
end
