function varargout = CellProfiler(varargin)
% CellProfiler M-file for CellProfiler.fig
%      CellProfiler, by itself, creates a new CellProfiler or raises the existing
%      singleton*.
%
%      H = CellProfiler returns the handle to a new CellProfiler or the handle to
%      the existing singleton*.
%
%      CellProfiler('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in CellProfiler.M with the given input arguments.
%
%      CellProfiler('Property','Value',...) creates a new CellProfiler or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before CellProfiler_OpeningFunction gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to CellProfiler_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% HINTS:
% hObject    handle to the object of the function (see GCBO)
% handles    structure with handles and user data (see GUIDATA)

% Edit the above text to modify the response to help CellProfiler
% Last Modified by GUIDE v2.5 30-Sep-2004 16:09:26
% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @CellProfiler_OpeningFcn, ...
                   'gui_OutputFcn',  @CellProfiler_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin & isstr(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end

% End initialization code - DO NOT EDIT

% --- Executes just before CellProfiler is made visible.
function CellProfiler_OpeningFcn(hObject, eventdata, handles, varargin)

% Choose default command line output for CellProfiler
handles.output = hObject;

% The Number of Algorithms/Variables hardcoded in
handles.numAlgorithms = 8;
handles.numVariables = zeros(1,99);
handles.MaxAlgorithms = 8;
handles.MaxVariables = 11;

% Turn on debugging
handles.Debug = 1;

% Update handles structure
guidata(hObject, handles);

% --- Outputs from this function are returned to the command line.
function varargout = CellProfiler_OutputFcn(hObject, eventdata, handles)

% Get default command line output from handles structure
varargout{1} = handles.output;

%%%%%%%%%%%%%%%%%%%%%%%
%%% INITIAL SETTINGS %%%
%%%%%%%%%%%%%%%%%%%%%%%

%%% Checks whether the user has the Image Processing Toolbox.
Answer = license('test','image_toolbox');
if Answer ~= 1
    warndlg('It appears that you do not have a license for the Image Processing Toolbox of Matlab.  Many of the image analysis modules of CellProfiler may not function properly.') 
end
% Determines the current directory in order to switch back to it later.
CurrentDirectory = pwd;

%%% Retrieves preferences from CellProfilerPreferences.mat, if possible.
%%% Try loading CellProfilerPreferences.mat first from the matlabroot
%%% directory and then the current directory.  This is not necessary for
%%% CellProfiler to function; it just allows defaults to be pre-loaded. If
%%% successful, this produces three variables in the workspace:
%%% PixelSize, DefaultAlgorithmDirectory, WorkingDirectory.
try cd(matlabroot)
    load CellProfilerPreferences
    PreferencesExist = 1;
catch try cd(CurrentDirectory);
     load CellProfilerPreferences
        PreferencesExist = 1;
    catch PreferencesExist = 0;
    end
end

%%% Stores some initial values in the handles structure based either on the
%%% Preferences, if they were successfully loaded, or on the current
%%% directory.
if PreferencesExist == 1
    handles.Vpixelsize = PixelSize;
    handles.Vdefaultalgorithmdirectory = DefaultAlgorithmDirectory;
    handles.Vworkingdirectory = WorkingDirectory;
    handles.Vpathname = WorkingDirectory;
    handles.Vtestpathname = handles.Vworkingdirectory;
    set(handles.PixelSizeEditBox,'string',PixelSize);
    set(handles.PathToLoadEditBox,'String',handles.Vworkingdirectory);
else
    handles.Vpixelsize = get(handles.PixelSizeEditBox,'string');
    handles.Vdefaultalgorithmdirectory = pwd;
    handles.Vworkingdirectory = pwd;
    handles.Vpathname = pwd;
    handles.Vtestpathname = pwd;
    set(handles.PathToLoadEditBox,'String',pwd);
end

%%% Sets up the main program window (Main GUI window) so that it asks for
%%% confirmation prior to closing.
%%% First, obtains the handle for the main GUI window (aka figure1).
global MainGUIhandle
MainGUIhandle = handles.figure1;
ClosingFunction = ...
    'global MainGUIhandle ; answer = questdlg(''Do you really want to quit?'', ''Confirm quit'',''Yes'',''No'',''Yes''); switch answer; case ''Yes''; delete(MainGUIhandle); case ''No''; return; end; clear answer MainGUIhandle; clear HandleFigureDisplay';
%%% Sets the closing function of the Main GUI window to be the line above.
set(MainGUIhandle,'CloseRequestFcn',ClosingFunction);

%%% Obtains the screen size.
ScreenSize = get(0,'ScreenSize');
ScreenWidth = ScreenSize(3);
ScreenHeight = ScreenSize(4);
%%% Sets the position of the Main GUI window so it is in the center of the
%%% screen. I designed the
%%% GUI window itself to be 800 pixels wide and 600 high, but the following
%%% code looks up the size anyway to be sure.
GUIsize = get(MainGUIhandle,'position');
GUIwidth = GUIsize(3);
GUIheight = GUIsize(4);
Left = 0.5*(ScreenWidth - GUIwidth);
Bottom = 0.5*(ScreenHeight - GUIheight);
set(MainGUIhandle,'Position',[Left Bottom GUIwidth GUIheight]);

%%% Retrieves the list of image file names from the chosen directory and
%%% stores them in the handles structure, using the function
%%% RetrieveImageFileNames.
PathName = handles.Vpathname;
handles = RetrieveImageFileNames(handles, PathName);
guidata(hObject, handles);
if isempty(handles.Vfilenames)
    set(handles.ListBox,'String','No image files recognized',...
    'Value',1)
else
    %%% Loads these image names into the ListBox.
set(handles.ListBox,'String',handles.Vfilenames,...
    'Value',1)
end
cd(CurrentDirectory)
% Update handles structure
guidata(hObject, handles);

%%%%%%%%%%%%%%%%%

function handles = RetrieveImageFileNames(handles, PathName)
%%% Lists all the contents of that path into a structure which includes the
%%% name of each object as well as whether the object is a file or
%%% directory.
FilesAndDirsStructure = dir(PathName);
%%% Puts the names of each object into a list.
FileAndDirNames = sortrows({FilesAndDirsStructure.name}');
%%% Puts the logical value of whether each object is a directory into a list.
LogicalIsDirectory = [FilesAndDirsStructure.isdir];
%%% Eliminates directories from the list of file names.
FileNamesNoDir = FileAndDirNames(~LogicalIsDirectory);

if isempty(FileNamesNoDir)
    handles.Vfilenames = [];
    %%% Test whether this is during CellProfiler launching, in which case
    %%% the following error is unnecessary.
    if strcmp(get(handles.ListBox,'String'),'Listbox') ~= 1
    errordlg('There are no files in the chosen directory')
    end
else

DiscardsHidden = strncmp(FileNamesNoDir,'.',1);
DiscardsByExtension = regexpi(FileNamesNoDir, '\.(m|mat|m~|frk~|xls|doc|txt)$', 'once');
if strcmp(class(DiscardsByExtension), 'cell')
  DiscardsByExtension = cellfun('prodofsize',DiscardsByExtension);
else 
  DiscardsByExtension = []
end

%%% Combines all of the DiscardLogical arrays into one.
Discards = DiscardsHidden | DiscardsByExtension;
%%% Eliminates filenames to be discarded.
if isempty(Discards)
  FileNames = FileNamesNoDir;
else 
  FileNames = FileNamesNoDir(~Discards);
end
%%% Checks whether any files are left.
if isempty(FileNames)
    handles.Vfilenames = [];
    %%% Test whether this is during CellProfiler launching, in which case
    %%% the following error is unnecessary.
    if strcmp(get(handles.ListBox,'String'),'Listbox') ~= 1
    errordlg('There are no files in the chosen directory')
    end
else
%%% Stores the final list of file names in the handles structure
handles.Vfilenames = FileNames;
guidata(handles.figure1,handles);
end
end

%%%%%%%%%%%%%%%%%

% --- Executes on button press in BrowseToLoad.
function BrowseToLoad_Callback(hObject, eventdata, handles)
CurrentDirectory = pwd;
cd(handles.Vworkingdirectory)
%%% Opens a dialog box to allow the user to choose a directory and loads
%%% that directory name into the edit box.  Also, changes the current 
%%% directory to the chosen directory.
pathname = uigetdir('','Choose the directory of images to be analyzed');
%%% If the user presses "Cancel", the pathname will = 0 and nothing will
%%% happen.
if pathname == 0
else
    %%% Saves the pathname in the handles structure.
    handles.Vpathname = pathname;
    guidata(hObject,handles)
    %%% Retrieves the list of image file names from the chosen directory and
    %%% stores them in the handles structure, using the function
    %%% RetrieveImageFileNames.
    handles = RetrieveImageFileNames(handles,pathname);
    guidata(hObject, handles);
    if isempty(handles.Vfilenames)
        set(handles.ListBox,'String','No image files recognized',...
            'Value',1)
    else
        %%% Loads these image names into the ListBox.
        set(handles.ListBox,'String',handles.Vfilenames,...
            'Value',1)
        %%% Retrieves the SelectedTestImageName from the ListBox.
        Contents = get(handles.ListBox,'String');
        SelectedTestImageName = Contents{get(handles.ListBox,'Value')};
        set(handles.TestImageName,'String',[pathname,'/',SelectedTestImageName])
    end
    %%% Displays the chosen directory in the PathToLoadEditBox.
    set(handles.PathToLoadEditBox,'String',pathname);
end
cd(CurrentDirectory)
%%%%%%%%%%%%%%%%%

% --- Executes during object creation, after setting all properties.
function PathToLoadEditBox_CreateFcn(hObject, eventdata, handles)
    set(hObject,'BackgroundColor',[1 1 1]);

function PathToLoadEditBox_Callback(hObject, eventdata, handles)
%%% Retrieves the text that was typed in.
pathname = get(hObject,'string');
%%% Checks whether a directory with that name exists.
if exist(pathname) ~= 0
%%% Saves the pathname in the handles structure.
handles.Vpathname = pathname;
guidata(hObject,handles)
%%% Retrieves the list of image file names from the chosen directory and
%%% stores them in the handles structure, using the function
%%% RetrieveImageFileNames.
handles = RetrieveImageFileNames(handles,pathname);
guidata(hObject, handles);
%%% Display the path in the edit box.
set(handles.PathToLoadEditBox,'String',handles.Vpathname);
if isempty(handles.Vfilenames)
    set(handles.ListBox,'String','No image files recognized',...
    'Value',1)
else
    %%% Loads these image names into the ListBox.
set(handles.ListBox,'String',handles.Vfilenames,...
    'Value',1)
end
%%% If the directory entered in the box does not exist, give an error
%%% message and change the contents of the edit box back to the current
%%% directory.
else errordlg('A directory with that name does not exist')
    set(handles.PathToLoadEditBox,'String',pwd)
end

%%%%%%%%%%%%%%%%%

% --- Executes on button press in LoadSampleInfo.
function output = LoadSampleInfo_Callback(hObject, eventdata, handles)

CurrentDirectory = pwd;
cd(handles.Vworkingdirectory)
%%% Opens a dialog box to retrieve a file name that contains a list of 
%%% sample descriptions, like gene names or sample numbers.
[fname,pname] = uigetfile('*.*','Choose sample info text file');
%%% If the user presses "Cancel", the fname will = 0 and nothing will
%%% happen.
if fname == 0
else
    extension = fname(end-2:end);
    %%% Checks whether the chosen file is a text file.
    if strcmp(extension,'txt') == 0;
        errordlg('Sorry, the list of sample descriptions must be in a text file (.txt).');
    else 
        %%% Saves the text from the file into a new variable, "SampleNames".  The
        %%% '%s' command tells it to select groups of strings not separated by
        %%% things like carriage returns, and maybe spaces and commas, too. (Not
        %%% sure about that).
        SampleNames = textread([pname fname],'%s');
        NumberSamples = length(SampleNames);
        %%% Displays a warning.  The buttons don't do anything except proceed.
        Warning = {'The next window will show you a preview'; ...
                'of the sample info you have loaded'; ...
                ''; ...
                ['You have ', num2str(NumberSamples), ' lines of sample information.'];...
                ''; ...
                'Press either ''OK'' button to continue.';...
                '-------'; ...
                'Warning:'; 'Please note that any spaces or weird'; ...
                '(e.g. punctuation) characters on any line of your'; ...
                'text file will split the entry into two entries!';...
                '-------'; ...
                'Also check that the order of the image files within Matlab is';...
                'as expected.  For example, If you are running Matlab within';...
                'X Windows on a Macintosh, the Mac will show the files as: ';...
                '(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11) whereas the X windows ';...
                'system that Matlab uses will show them as ';...
                '(1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9) and so on.  So be sure that ';...
                'the order of your sample info matches the order that Matlab ';...
                'is using.  Look in the Current Directory window of Matlab to ';...
                'see the order.  Go to View > Current Directory to open the ';...
                'window if it is not already visible.'};
        listdlg('ListString',Warning,'ListSize', [300 600],'Name','Warning',...
            'PromptString','Press any button to continue.', 'CancelString','Ok',...
            'SelectionMode','single');
        %%% Displays a listbox so the user can preview the data.  The buttons in
        %%% this listbox window don't do anything except proceed.
        listdlg('ListString',SampleNames, 'ListSize', [300 600],...
            'Name','Preview your sample data',...
            'PromptString','Press any button to continue.','CancelString','Ok',...
            'SelectionMode','single');
        %%% Retrieves a name for the heading of the sample data just entered.
        A = inputdlg('Enter the heading for these sample descriptions (e.g. GeneNames                 or SampleNumber). Your entry must be one word with letters and                   numbers only, and must begin with a letter.','Name the Sample Info',1);
        %%% If the user presses cancel A will be an empty array.  If the user
        %%% doesn't enter anything in the dialog box but presses "OK" anyway,
        %%% A is equal to '' (two quote marks).  In either case, skip to the end; 
        %%% don't save anything. 
        if (isempty(A)) 
        elseif strcmp(A,'') == 1, 
            errordlg('Sample info was not saved, because no heading was entered.');
        elseif isfield(handles, A) == 1
            errordlg('Sample info was not saved, because sample info with that heading has already been stored.');
        else
            %%% Uses the heading the user entered to name the field in the handles
            %%% structure array and save the SampleNames list there.
            try    handles.(char(A)) = SampleNames;
                %%% Also need to add this heading name (field name) to the headings
                %%% field of the handles structure (if it already exists), in the last position. 
                if isfield(handles, 'headings') == 1
                    N = length(handles.headings) + 1;
                    handles.headings(N)  = A;
                    %%% If the headings field doesn't yet exist, create it and put the heading 
                    %%% name in position 1.
                else handles.headings(1)  = A;
                end
                guidata(hObject, handles);
            catch errordlg('Sample info was not saved, because the heading contained illegal characters.');
            end % Goes with catch
        end
    end
    %%% One of these "end"s goes with the if A is empty, when user presses
    %%% cancel. 
end 
cd(CurrentDirectory)
% Some random advice from Ganesh:
% SampleNames is a n x m (n - number of rows, m - 1 column) cell array
% If you want to make it into a 1x1 cell array where the cell element
% contains the text, you can do the following
%
% cell_data = {strvcat(SampleNames)};
%
% This will assign the string matrix being created into a single cell
% element.

%%%%%%%%%%%%%%%%%

% --- Executes on button press in ClearSampleInfo.
function ClearSampleInfo_Callback(hObject, eventdata, handles)
%%% The Clear Sample Info button allows deleting any list of 
%%% sample info, specified by its heading, from the handles structure.

%%% Checks whether any headings are loaded yet.
if isfield(handles,'headings') == 0
    errordlg('No sample info has been loaded.')
else
%%% Opens a listbox which displays the list of headings so that one can be
%%% selected.  The OK button has been assigned to mean "Delete".
Headings = handles.headings;
[Selected,Action] = listdlg('ListString',Headings, 'ListSize', [300 600],...
    'Name','Current sample info loaded',...
    'PromptString','Select the sample descriptions you would like to delete',...
    'OKString','Delete','CancelString','Cancel','SelectionMode','single');

%%% Extracts the actual heading name from the Headings variable.
SelectedFieldName = Headings(Selected);

% Action = 1 if the user pressed the OK (DELETE) button.  If they pressed
% the cancel button or closed the window Action == 0 and nothing happens.
if Action == 1
    %%% Delete the selected heading (with its contents, the sample data) 
    %%% from the structure.
    handles = rmfield(handles,SelectedFieldName);
    %%% Delete the selected heading from the headings list in the
    %%% structure, by assigning it the empty structure.
    handles.headings(Selected) = [];
    %%% If no sample info remains, the field "headings" is removed
    %%% so that when the user clicks Clear or View, the proper error
    %%% message is generated, telling the user that no sample info has been
    %%% loaded.
    if isempty(handles.headings) ==1
          handles = rmfield(handles, 'headings');
    end
    %%% Handles structure is updated
    guidata(gcbo,handles)
end
%%% This end goes with the error-detecting - "Do you have any sample info
%%% loaded?"
end

%%%%%%%%%%%%%%%%%

% --- Executes on button press in ViewSampleInfo.
function ViewSampleInfo_Callback(hObject, eventdata, handles)
%%% The View Sample Info button allows viewing any list of 
%%% sample info, specified by its heading, taken from the handles structure.

%%% Checks whether any sample info has been loaded by determining 
%%% whether handles.headings is empty or not.
if isfield(handles,'headings') == 0
    errordlg('No sample info has been loaded.')
%%% Opens a listbox which displays the list of headings so that one can be
%%% selected.  The OK button has been assigned to mean "View".
else Headings = handles.headings;
[Selected,Action] = listdlg('ListString',Headings, 'ListSize', [300 600],...
    'Name','Current sample info loaded',...
    'PromptString','Select the sample descriptions you would like to view.',...
    'OKString','View','CancelString','Cancel','SelectionMode','single');

%%% Extracts the actual heading name from the Headings variable.
SelectedFieldName = Headings(Selected);

% Action = 1 if the user pressed the OK (VIEW) button.  If they pressed
% the cancel button or closed the window Action == 0.
if Action == 1
    ListToShow = handles.(char(SelectedFieldName));
    listdlg('ListString',ListToShow, 'ListSize', [300 600],...
        'Name','Preview your sample data','PromptString',...
        'Press any button to continue','CancelString','Ok','SelectionMode','single');
    %%% The OK buttons within this window don't do anything.
else
    %%% If the user pressed "cancel" or closes the window, Action = 0, so
    %%% nothing happens. 
end
%%% This "end" goes with the "isempty" if no sample info is loaded.
end
%%%%%%%%%%%%%%%%%

% --- Executes during object creation, after setting all properties.
function OutputFileName_CreateFcn(hObject, eventdata, handles)
    set(hObject,'BackgroundColor',[1 1 1]);

function OutputFileName_Callback(hObject, eventdata, handles)
CurrentDirectory = cd;
PathName = get(handles.PathToLoadEditBox,'string');
%%% Gets the user entry and stores it in the handles structure using the
%%% store1variable function.
InitialUserEntry = get(handles.OutputFileName,'string');
fieldname = {'Voutputfilename'};
if isempty(InitialUserEntry)
    handles.Voutputfilename =[]
    guidata(gcbo, handles);
else
    if length(InitialUserEntry) >=4
        if strncmp(lower(InitialUserEntry(end-3:end)),'.mat',4) == 0
            UserEntry = [InitialUserEntry,'OUT.mat'];
        else UserEntry = [InitialUserEntry(1:end-4) 'OUT.mat'];
        end
    else  UserEntry = [InitialUserEntry,'OUT.mat'];
    end
    guidata(gcbo, handles);
    %%% Checks whether a file with that name already exists, to warn the user
    %%% that the file will be overwritten.
    CurrentDirectory = cd;
    if exist([PathName,'/',UserEntry]) ~= 0
        errordlg(['A file already exists at ', [PathName,'/',UserEntry],...
            '. Enter a different name. Click the help button for an explanation of why you cannot just overwrite an existing file.'], 'Warning!');
        set(handles.OutputFileName,'string',[])
    else guidata(gcbo, handles);
        handles = store1variable('Voutputfilename',UserEntry, handles);
        set(handles.OutputFileName,'string',UserEntry)
    end
end
guidata(gcbo, handles);
cd(CurrentDirectory)

function handles = store1variable(VariableName,UserEntry, handles);
%%% This function stores a variable's value in the handles structure, 
%%% when given the Algorithm Number, the Variable Number, 
%%% the UserEntry (from the Edit box), and the initial handles
%%% structure.
handles.(VariableName) = UserEntry;
guidata(gcbo, handles);

%%%%%%%%%%%%%%%%%

% --- Executes on button press in LoadSettingsFromFileButton.
function LoadSettingsFromFileButton_Callback(hObject, eventdata, handles)

CurrentDirectory = pwd;
cd(handles.Vworkingdirectory)
[SettingsFileName, SettingsPathName] = uigetfile('*.mat','Choose the settings file');
%%% If the user presses "Cancel", the SettingsFileName.m will = 0 and
%%% nothing will happen.
if SettingsFileName == 0
else
  %%% Loads the Settings file.
  LoadedSettings = load([SettingsPathName SettingsFileName]);

  if ~ (isfield(LoadedSettings, 'Settings') | isfield(LoadedSettings, 'handles')),
    errordlg(['The file ' SettingsPathName SettingsFilename ' does not appear to be a valid settings or output file (does not contain a variable named ''Settings'' or ''handles'').']);
    cd(CurrentDirectory);
    return;
  end

  %%% Figure out whether we loaded a Settings or Output file, and put the correct values into Settings
  if (isfield(LoadedSettings, 'Settings')),
    Settings = LoadedSettings.Settings;
  else 
    Settings = LoadedSettings.handles;
  end;
  
  if handles.numAlgorithms > 0,
    %%% Clears the current settings, using the clearalgorithm function.
    for i=1:handles.numAlgorithms,
      handles = ClearAlgorithm_Helper(TwoDigitString(i), handles, 'NoConfirm');
    end
    guidata(gcbo, handles);
    
    %%% The last clearalgorithm function leaves the indicator bar set at
    %%% the last algorithm, so the following makes it invisible.
    set(handles.(['Indicator',TwoDigitString(handles.numAlgorithms)]),'Visible','off');
  end

  %%% Splice the subset of variables from the "settings" structure into the
  %%% handles structure.  For each one, it checks whether the value is empty
  %%% before creating a field for it in the handles structure.  For the
  %%% algorithm names and the pixel size, this code also displays the values
  %%% in the GUI.

  handles.numAlgorithms = 0;
  for AlgorithmNumber=1:handles.MaxAlgorithms,
    AlgorithmFieldName = ['Valgorithmname', TwoDigitString(AlgorithmNumber)];
    if isfield(Settings, AlgorithmFieldName),
      handles.(AlgorithmFieldName) = Settings.(AlgorithmFieldName);
      set(handles.(['AlgorithmName' TwoDigitString(AlgorithmNumber)]) ,'string', handles.(AlgorithmFieldName));
      handles.numAlgorithms = AlgorithmNumber;

      handles.numVariables(AlgorithmNumber) = 0;
      for VariableNumber=1:handles.MaxVariables,
        VariableFieldName = ['Vvariable' TwoDigitString(AlgorithmNumber) '_' TwoDigitString(VariableNumber)];
        if isfield(Settings, VariableFieldName),
          handles.([VariableFieldName]) = Settings.([VariableFieldName]);
          handles.numVariables(AlgorithmNumber) = VariableNumber;
        end
      end
    end
  end

  if isfield(Settings, 'VpixelSize'),
    handles.VpixelSize = Settings.VpixelSize;
  end
  
  %%% Update handles structure.
  guidata(hObject,handles);

  %%% If the user loaded settings from an output file, prompt them to
  %%% save it as a separate Settings file for future use.
  if isfield(LoadedSettings, 'handles'),
    if questdlg('Save settings from output file in a separate, settings-only file?','','Yes','No','Yes') == 'Yes',
      SaveCurrentSettingsButton_Callback(hObject, eventdata, handles)
    end
  end  
      

end
cd(CurrentDirectory)

%%%%%%%%%%%%%%%%%

% --- Executes on button press in SaveCurrentSettingsButton.
function SaveCurrentSettingsButton_Callback(hObject, eventdata, handles)
CurrentDirectory = pwd;
cd(handles.Vworkingdirectory)
%%% The "Settings" variable is saved to the file name the user chooses.
[FileName,PathName] = uiputfile('*.mat', 'Save Settings As...');
%%% Allows canceling.
if FileName ~= 0

  %%% Checks if a field is present, and if it is, the value is stored in the 
  %%% structure 'Settings' with the same name
  
  for AlgorithmNumber=1:handles.numAlgorithms,
    AlgorithmFieldName = ['Valgorithmname', TwoDigitString(AlgorithmNumber)];
    if isfield(handles, AlgorithmFieldName),
      Settings.(AlgorithmFieldName) = handles.(AlgorithmFieldName);
      for VariableNumber=1:handles.MaxVariables,
        VariableFieldName = ['Vvariable' TwoDigitString(AlgorithmNumber) '_' TwoDigitString(VariableNumber)];
        if isfield(handles, VariableFieldName),
          Settings.(VariableFieldName) = handles.(VariableFieldName);
        end
      end
    end
  end
  
  if isfield(handles,'Vpixelsize'),
    Settings.Vpixelsize = handles.Vpixelsize;
  end

  save([PathName FileName],'Settings')
  helpdlg('The settings file has been written.')
end
cd(CurrentDirectory)

%%%%%%%%%%%%%%%%%

% --- Executes during object creation, after setting all properties.
function PixelSizeEditBox_CreateFcn(hObject, eventdata, handles)
    set(hObject,'BackgroundColor',[1 1 1]);

function PixelSizeEditBox_Callback(hObject, eventdata, handles)
%%% Checks to see whether the user input is a number, and generates an
%%% error message if it is not a number.
user_entry = str2double(get(hObject,'string'));
if isnan(user_entry)
    errordlg('You must enter a numeric value','Bad Input','modal')
    set(hObject,'string','0.25')
%%% Checks to see whether the user input is positive, and generates an
%%% error message if it is not.
elseif user_entry<=0
   errordlg('You entered a value less than or equal to zero','Bad Input','modal')
   set(hObject,'string', '0.25')
else
%%% Gets the user entry and stores it in the handles structure using the
%%% store1variable function.
UserEntry = get(handles.PixelSizeEditBox,'string');
store1variable('Vpixelsize',UserEntry, handles);
end

%%%%%%%%%%%%%%%%%

% --- Executes on button press in SetPreferencesButton.
function SetPreferencesButton_Callback(hObject, eventdata, handles)
%%% Determine what the current directory is, so we can change back 
%%% when this process is done.
CurrentDirectory = cd;
%%% Change to the Matlab root directory.
cd(matlabroot)
%%% If the CellProfilerPreferences.mat file does not exist in the matlabroot
%%% directory, change to the current directory.
if exist('CellProfilerPreferences.mat') == 0
  cd(CurrentDirectory);
else
    %%% If the CellProfilerPreferences.mat file exists, load it and change to the
%%% default algorithm directory.
  load CellProfilerPreferences
end

%%% (1) GET DEFAULT PIXEL SIZE
%%% Tries to load the pixel size from the existing file, to use it in the
%%% dialog box below.
try PixelSizeForDialogBox = PixelSize;
catch PixelSizeForDialogBox = {'1'};
end
%%% Asks for the default pixel size.
PixelSize = inputdlg('How many micrometers per pixel?','Set default pixel size',1,PixelSizeForDialogBox);
%%% Allows canceling.
if isempty(PixelSize) == 1
    cd(CurrentDirectory);
    return
end

%%% (2) GET DEFAULT ALGORITHM DIRECTORY
try   
  %%% Tries to change to the default algorithm directory, whose name is a variable
  %%% that is stored in the CellProfilerPreferences.mat file.
  cd(DefaultAlgorithmDirectory)
end
%%% Open a dialog box to get the directory from the user.
DefaultAlgorithmDirectory = uigetdir(pwd, 'Where are the analysis modules?'); 
%%% Allows canceling.
if DefaultAlgorithmDirectory == 0
    %%% Change back to the original directory and do nothing.
    cd(CurrentDirectory);
    return
end

%%% (3) GET WORKING DIRECTORY
try   
  %%% Tries to change to the working directory, whose name is a variable
  %%% that is stored in the CellProfilerPreferences.mat file.
  cd(WorkingDirectory)
end
%%% Open a dialog box to get the directory from the user.
WorkingDirectory = uigetdir(pwd, 'Which folder should be the default for your output and settings files?'); 
%%% Allows canceling.
if WorkingDirectory == 0
    %%% Change back to the original directory and do nothing.
    cd(CurrentDirectory);
    return
end

%%% (4) SAVE PREFERENCES
%%% The pathname is saved as a variable in a .mat file in the Matlab root
%%% directory. In this way, the file can always be found by the Load
%%% algorithm function. The first argument is the name of the .mat file;
%%% the remaining arguments are the names of the variables which are saved.
try
    %%% Change the directory to the Matlab root directory
    cd(matlabroot)
    save CellProfilerPreferences DefaultAlgorithmDirectory PixelSize WorkingDirectory
    helpdlg('Your CellProfiler Preferences were successfully set.  They are contained within a folder in the Matlab root directory in a file called CellProfilerPreferences.mat.')
    handles.Vpixelsize = PixelSize;
    handles.Vdefaultalgorithmdirectory = DefaultAlgorithmDirectory;
    handles.Vworkingdirectory = WorkingDirectory;
    set(handles.PixelSizeEditBox,'string',PixelSize);
    %%% Update handles structure.
    guidata(hObject,handles);

catch
    cd(CurrentDir)
    try
        save CellProfilerPreferences DefaultAlgorithmDirectory PixelSize WorkingDirectory
        helpdlg('You do not have permission to write anything to the Matlab root directory, which is required to save your preferences permanently.  Instead, your preferences will only function properly while you are in the current directory.')
        handles.Vpixelsize = PixelSize;
        handles.Vdefaultalgorithmdirectory = DefaultAlgorithmDirectory;
        handles.Vworkingdirectory = WorkingDirectory;
        set(handles.PixelSizeEditBox,'string',PixelSize);
        %%% Update handles structure.
        guidata(hObject,handles);

    catch
        helpdlg('CellProfiler was unable to save your desired preferences, probably because you lack write permission for both the Matlab root directory as well as the current directory.  Your preferences were not saved.');
    end
end % Goes with try/catch.
cd(CurrentDirectory);

%%%%%%%%%%%%%%%%%%%
%%% LOAD BUTTONS %%%
%%%%%%%%%%%%%%%%%%%

% --- Executes on button press in LoadAlgorithm.
function LoadAlgorithm_Callback(hObject,eventdata,handles)
% Find which algorithm slot number this callback was called for.
LoadAlgorithmButtonTag = get(hObject,'tag');
AlgorithmNumber = trimstr(LoadAlgorithmButtonTag,'LoadAlgorithm','left');

%%% 1. Opens a user interface to retrieve the .m file you want to use.  The
%%% name of that .m file is stored as the variablebox2_1
%%% "FirstImageAlgorithmName".

%%% First, the current directory is stored so we can switch back to it at
%%% the end of this step:
CurrentDir = cd;
%%% Change to the default algorithm directory, whose name is a variable
%%% that is stored in that .mat file. It is within a try-end pair because
%%% the user may have changed the folder names leading up to this directory
%%% sometime after saving the Preferences.
try
    cd(handles.Vdefaultalgorithmdirectory)
end 
%%% Now, when the dialog box is opened to retrieve an algorithm, the
%%% directory will be the default algorithm directory.
[AlgorithmNamedotm,PathName] = uigetfile('*.m',...
    'Choose an image analysis module');
%%% Change back to the original directory.
cd(CurrentDir)

%%% 2. If the user presses "Cancel", the AlgorithmNamedotm = 0, and
%%% everything should be left as it was.  If the algorithm is not on
%%% Matlab's search path, the user is warned.
if AlgorithmNamedotm == 0,
  %%% If the algorithm's .m file is not found on the search path, the result
  %%% of exist is zero.
elseif exist(AlgorithmNamedotm) == 0
  msgbox(['The .m file ', AlgorithmNamedotm, ...
        ' was not initially found by Matlab, so the folder containing it was added to the Matlab search path.  Please reload the analysis module; It should work fine from now on. If for some reason you did not want to add that folder to the path, go to Matlab > File > Set Path and remove the folder from the path.  If you have no idea what this means, don''t worry about it.'])
  %%% The folder containing the desired .m file is added to Matlab's search path.
  addpath(PathName)
  %%% Doublecheck that the algorithm exists on Matlab's search path.
  if exist(AlgorithmNamedotm) == 0
    errordlg('Something is wrong; Matlab still cannot find the .m file for the analysis module you selected.')
  end
else

  %%% 3. Set all the indicator bars (which tell you which algorithm
  %%% you are editing settings for) to be invisible and then set
  %%% the one you are working on to be visible.
  for i=1:handles.numAlgorithms;
    set(handles.(['Indicator' TwoDigitString(i)]),'Visible','off');
  end;
  set(handles.(['Indicator' AlgorithmNumber]),'Visible','on');

  %%% 4. Sets all 11 VariableBox edit boxes and all 11 VariableDescriptions
  %%% to be invisible.
  for VariableNumber = 1:handles.MaxVariables;
    set(handles.(['VariableBox' TwoDigitString(VariableNumber)]),'visible','off');
    set(handles.(['VariableDescription' TwoDigitString(VariableNumber)]),'visible','off');
  end;

  %%% 5. Clears the variable values in the handles structure in case some are
  %%% not used in the new algorithm (they would remain intact and not be
  %%% overwritten). Before removing a variable, you have to check that the
  %%% variable exists or else the 'rmfield' function gives an error.
  for VariableNumber=1:handles.MaxVariables
    VarNumber = TwoDigitString(VariableNumber);
    ConstructedName = ['Vvariable' AlgorithmNumber '_' VarNumber];
    if isfield(handles,ConstructedName) == 1;
      handles = rmfield(handles, ConstructedName);
    end;
  end;

  %%% 6. The last two characters (=.m) are removed from the
  %%% AlgorithmName.m and called AlgorithmName.
  AlgorithmName = AlgorithmNamedotm(4:end-2);
  %%% The name of the algorithm is shown in a text box in the GUI (the text
  %%% box is called AlgorithmName1.) and in a text box in the GUI which
  %%% displays the current algorithm (whose settings are shown).
  set(handles.(['AlgorithmName' AlgorithmNumber]),'String',AlgorithmName);

  %%% 7. Saves the AlgorithmName to the handles structure.
  handles.(['Valgorithmname' AlgorithmNumber]) = AlgorithmName;

  %%% 8. The text description for each variable for the chosen algorithm is 
  %%% extracted from the algorithm's .m file and displayed.  
  fid=fopen([PathName AlgorithmNamedotm]);

  while 1;
    output = fgetl(fid); if ~ischar(output); break; end;
    
    % FIXME: this doesn't need to loop over MaxVariables
    
    for i=1:handles.MaxVariables,
      if (strncmp(output,['%textVAR',TwoDigitString(i)],10) == 1);
        set(handles.(['VariableDescription',TwoDigitString(i)]), 'string', output(13:end),'visible', 'on');
        break;
      end
    end

    for i=1:handles.MaxVariables,
      if (strncmp(output,['%defaultVAR' TwoDigitString(i)],13) == 1),
        displayval = output(17:end);
        set(handles.(['VariableBox' TwoDigitString(i)]), 'string', displayval,'visible', 'on');
        set(handles.(['VariableDescription' TwoDigitString(i)]), 'visible', 'on');
        handles.(['Vvariable' AlgorithmNumber '_' TwoDigitString(i)]) = displayval;
        handles.numVariables(str2num(AlgorithmNumber)) = i;
        break;
      end
    end
   end
  fclose(fid);

  %%% Updates the handles structure to incorporate all the changes.
  guidata(gcbo, handles);
end

%%%%%%%%%%%%%%%%%%%%
%%% CLEAR BUTTONS %%%
%%%%%%%%%%%%%%%%%%%%


% --- Executes on button press for all ClearAlgorithm buttons.
function ClearAlgorithm_Callback(hObject, eventdata, handles)
AlgorithmName = get(hObject,'tag');
AlgorithmNumber = trimstr(AlgorithmName,'ClearAlgorithm','left');
handles = ClearAlgorithm_Helper(AlgorithmNumber, handles, 'Confirm');

% separated because it's called elsewhere
function handles = ClearAlgorithm_Helper(AlgorithmNumber, handles, ConfirmOrNot)

if strcmp(ConfirmOrNot, 'Confirm') == 1
    %%% Confirms the choice to clear the algorithm.
    Answer = questdlg('Are you sure you want to clear this analysis module and its settings?','Confirm','Yes','No','Yes');
    if strcmp(Answer,'No') == 1
        return
    end
end
%%% 1. Sets the proper algorithm name to "No analysis module loaded" 
set(handles.(['AlgorithmName' AlgorithmNumber]),'String','No analysis module loaded');

%%% 2. Removes the AlgorithmName from the handles structure.
             ConstructedName = strcat('Valgorithmname',AlgorithmNumber);
             if isfield(handles,ConstructedName) == 1
             handles = rmfield(handles,ConstructedName); end

%%% 3. Set all the indicator bars (which tell you which algorithm 
%%% you are editing settings for) to be invisible and then set 
%%% the one you are working on to be visible.
for i=1:handles.numAlgorithms
   set(handles.(['Indicator' TwoDigitString(i)]),'Visible','off');
end
set(handles.(['Indicator' AlgorithmNumber]),'Visible','on');

%%% 4. Sets all 11 VariableBox edit boxes and all 11 VariableDescriptions
%%% to be invisible.
for i = 1:handles.numVariables(str2num(AlgorithmNumber));
   set(handles.(['VariableBox' TwoDigitString(i)]),'visible','off')
   set(handles.(['VariableDescription' TwoDigitString(i)]),'visible','off')
end

%%% 5. Clears the variable values in the handles structure.

for i=1:handles.numVariables(str2num(AlgorithmNumber));
    ConstructedName = ['Vvariable' AlgorithmNumber '_' TwoDigitString(i)];
    if isfield(handles,ConstructedName) == 1;
        handles = rmfield(handles, ConstructedName);
    end;
end;

guidata(gcbo, handles);

%%%%%%%%%%%%%%%%%%%
%%% VIEW BUTTONS %%%
%%%%%%%%%%%%%%%%%%%

% --- Executes on button press in ViewAlgorithm.
function ViewAlgorithm_Callback(hObject,eventdata,handles)
ButtonTag = get(hObject,'tag');
AlgorithmNumber = trimstr(ButtonTag,'ViewAlgorithm','left');

%%% 1. Set all the indicator bars (which tell you which algorithm 
%%% you are editing settings for) to be invisible and then set 
%%% the one you are working on to be visible.
for i=1:handles.numAlgorithms
    set(handles.(['Indicator' TwoDigitString(i)]),'Visible','off');
end
set(handles.(['Indicator' AlgorithmNumber]),'Visible','on');


%%% 2. Sets all 11 VariableBox edit boxes and all 11
%%% VariableDescriptions to be invisible.
for i = 1:handles.MaxVariables
    set(handles.(['VariableBox' TwoDigitString(i)]),'visible','off','String','n/a')
    set(handles.(['VariableDescription' TwoDigitString(i)]),'visible','off')
end

%%% 2.5 Checks whether an algorithm is loaded in this slot.
AlgorithmName = get(handles.(['AlgorithmName' AlgorithmNumber]),'String');
IsItNotChosen = strncmp(AlgorithmName,'No a',4);
if IsItNotChosen == 1
    helpdlg('You do not have an analysis module selected.  Click "?" next to "Image analysis settings" to get help in choosing an analysis module, or click "View" next to an analysis module that has been loaded already.','Help for choosing an analysis module')
else

    %%% 3. Extracts and displays the variable descriptors from the .m file.
    AlgorithmName = get(handles.(['AlgorithmName' AlgorithmNumber]), 'string');
    AlgorithmNamedotm = strcat('Alg',AlgorithmName,'.m');
    if exist(AlgorithmNamedotm) ~= 2
        errordlg(['The image analysis module named ', AlgorithmNamedotm, ' was not found. Is it stored in the folder with the other modules?  Has its name changed?  The settings stored for this algorithm will be displayed, but this module will not run properly.']);
    else
        fid=fopen(AlgorithmNamedotm);
        
        while 1;
            output = fgetl(fid); if ~ischar(output); break; end;
            for i=1:handles.MaxVariables,
              if (strncmp(output,['%textVAR' TwoDigitString(i)],10) == 1);
                set(handles.(['VariableDescription' TwoDigitString(i)]), 'string', output(13:end),'visible', 'on');
                break;
              end
            end
        end
        fclose(fid);
    end
    %%% 4. The stored values for the variables are extracted from the handles
    %%% structure and displayed in the edit boxes.
    for i=1:handles.numVariables(str2num(AlgorithmNumber)),
        VariableNumber = TwoDigitString(i);
        ConstructedName = strcat('Vvariable',AlgorithmNumber,'_',VariableNumber);
        if isfield(handles,ConstructedName) == 1;
            set(handles.(['VariableBox' VariableNumber]),...
                'string',handles.(['Vvariable' AlgorithmNumber '_' VariableNumber]),...
                'visible','on');
        else set(handles.(['VariableBox' VariableNumber]),'string','n/a','visible','off');
        end;
    end;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% FIGURE DISPLAY BUTTONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% NOTE: These buttons appear after analysis has begun, and disappear 
%%% when it is over.
function FigureDisplay_Callback(hObject, eventdata, handles)
AlgorithmName = get(hObject,'tag');
AlgorithmNumber = trimstr(AlgorithmName,'FigureDisplay','left');
CurrentHandle = handles.(['FigureDisplay' AlgorithmNumber]);
ButtonStatus = get(CurrentHandle, 'string');
%%% First case: closing or opening is already in progress; Don't do
%%% anything.
if strcmp(ButtonStatus, 'Closing...') == 1 
elseif strcmp(ButtonStatus, 'Opening...') == 1 
elseif strcmp(ButtonStatus, 'Close Figure') == 1 
    %%% Setting the text to "Closing" will allow this window to close at the
    %%% end of the current image set.
    set(CurrentHandle,'string', 'Closing...')
    %%% Refreshes the Main GUI window.
    drawnow
elseif strcmp(ButtonStatus, 'Open Figure') == 1 
    %%% Setting the text to "Opening" will allow this window to open at the
    %%% end of the current image set.
    set(CurrentHandle,'string', 'Opening...')
    %%% Refreshes the Main GUI window.
    drawnow
end

%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% VARIABLE EDIT BOXES %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%

function storevariable(AlgorithmNumberString, VariableNumber, UserEntry, handles);
%%% This function stores a variable's value in the handles structure, 
%%% when given the Algorithm Number, the Variable Number, 
%%% the UserEntry (from the Edit box), and the initial handles
%%% structure.
handles.(['Vvariable' AlgorithmNumberString '_' VariableNumber]) = UserEntry;
guidata(gcbo, handles);

function [AlgorithmNumber] = whichactive(handles);
tempJ = 0;
for i=1:handles.numAlgorithms;
    if strncmp(get(handles.(['Indicator' TwoDigitString(i)]),'visible'),'on',2) == 1
        AlgorithmNumber = i;
        tempJ = 1;
    end
end
if tempJ == 0;
    AlgorithmNumber = 0;
end
    
% --- Executes during object creation, after setting all properties.
function VariableBox_CreateFcn(hObject, eventdata, handles);
    set(hObject,'BackgroundColor',[1 1 1]);


function VariableBox_Callback(hObject, eventdata, handles);
%%% The following lines fetch the contents of the edit box,
%%% determine which algorithm we are dealing with at the moment (by
%%% running the "whichactive" subfunction), and call the storevariable
%%% function.
VariableName = get(hObject,'tag');
VariableNumber = trimstr(VariableName,'VariableBox','left');

UserEntry = get(handles.(['VariableBox' VariableNumber]),'string');
AlgorithmNumber = whichactive(handles);
AlgorithmNumberString = TwoDigitString(AlgorithmNumber);
if isempty(UserEntry)
    errordlg('Variable boxes must not be left blank')
    set(handles.(['VariableBox' VariableNumber]),'string', 'Fill in');
    storevariable(AlgorithmNumberString,VariableNumber, 'Fill in', handles);
else
    AlgorithmNumber = whichactive(handles);
    if AlgorithmNumber == 0,     
        errordlg('Something strange is going on: none of the analysis modules are active right now but somehow you were able to edit a setting.','weirdness has occurred')
    else
        AlgorithmNumberString = TwoDigitString(AlgorithmNumber);
        storevariable(AlgorithmNumberString,VariableNumber,UserEntry, handles);
    end
end



%%%%%%%%%%%%%%%%%

% --- Executes on button press in SelectTestImageBrowseButton.
function SelectTestImageBrowseButton_Callback(hObject, eventdata, handles)
CurrentDirectory = pwd;
cd(handles.Vworkingdirectory)
%%% Opens a user interface window which retrieves a file name and path 
%%% name for the image to be used as a test image.
[FileName,PathName] = uigetfile('*.*','Select a test image (nuclei)');
%%% If the user presses "Cancel", the FileName will = 0 and nothing will
%%% happen.
if FileName == 0
else
    set(handles.TestImageName,'String',[PathName,FileName])
    handles.Vtestpathname = PathName;
    %%% Retrieves the list of image file names from the chosen directory and
    %%% stores them in the handles structure, using the function
    %%% RetrieveImageFileNames.
    handles = RetrieveImageFileNames(handles,PathName);
    guidata(gcbo, handles);
    if isempty(handles.Vfilenames)
        set(handles.ListBox,'String','No image files recognized',...
            'Value',1)
    else
        %%% Loads these image names into the ListBox.
        set(handles.ListBox,'String',handles.Vfilenames,...
            'Value',1)
    end
end
cd(CurrentDirectory)
%%%%%%%%%%%%%%%%%

% --- Executes during object creation, after setting all properties.
function TestImageName_CreateFcn(hObject, eventdata, handles)
set(hObject,'BackgroundColor',[1 1 1]);

function TestImageName_Callback(hObject, eventdata, handles)

%%% Retrieves the contents of the edit box.
UserEntry = get(hObject, 'string');
%%% Checks whether a file with that name exists.
if exist(UserEntry) ~= 0
    %%% If the file exists, leave the contents of the box as is.
    %%% Retrieves the list of image file names from the chosen directory and
    %%% stores them in the handles structure, using the function
    %%% RetrieveImageFileNames.
    PathName = handles.Vtestpathname;
    handles = RetrieveImageFileNames(handles,PathName);
    guidata(hObject, handles);
    if isempty(handles.Vfilenames)
        set(handles.ListBox,'String','No image files recognized',...
            'Value',1)
    else
        %%% Loads these image names into the ListBox.
        set(handles.ListBox,'String',handles.Vfilenames,...
            'Value',1)
    end
else 
    %%% If the file entered in the box does not exist, give an error
    %%% message and change the contents of the edit box to blank.
    errordlg('A file with that name does not exist')
    set(handles.TestImageName,'String','')
end

%%%%%%%%%%%%%%%%%

% --- Executes during object creation, after setting all properties.
function ListBox_CreateFcn(hObject, eventdata, handles)
    set(hObject,'BackgroundColor',[1 1 1]);

% --- Executes on selection change in ListBox.
function ListBox_Callback(hObject, eventdata, handles)
%%% Retrieves the SelectedTestImageName from the ListBox.
Contents = get(hObject,'String');
SelectedTestImageName = Contents{get(hObject,'Value')};
PathName = handles.Vtestpathname;
set(handles.TestImageName,'String',[PathName,SelectedTestImageName])

%%%%%%%%%%%%%%%%%

% --- Executes on button press in TechnicalDiagnosisButton.
function TechnicalDiagnosisButton_Callback(hObject, eventdata, handles)
%%% I am using this button to show the handles structure in the
%%% main Matlab window.
handles
msgbox('The handles structure has been printed out at the command line of Matlab.')

%%%%%%%%%%%%%%%%%

% --- Executes on button press in AnalyzeTestImageButton.
function AnalyzeTestImageButton_Callback(hObject, eventdata, handles)
errordlg('This button is not yet functional')

%%%%%%%%%%%%%%%%%

% --- Executes on button press in SaveImageAsButton.
function SaveImageAsButton_Callback(hObject, eventdata, handles)
MsgboxHandle = msgbox('Click twice on the image you wish to save. This window will be closed automatically - do not close it or click OK.');
waitforbuttonpress
ClickedImage = getimage(gca);
Answers = inputdlg({'Enter file name (no extension)','Enter image file format (e.g. tif,jpg)','If compatible with that file format, save as 16-bit image?'},'Save Image As',1,{'A','tif','no'});
if isempty(Answers) ~= 1
    FileName = char(Answers{1});
    Extension = char(Answers{2});
    SixteenBit = char(Answers{3});
    if strcmp(SixteenBit,'yes') == 1
        ClickedImage = uint16(65535*ClickedImage);
    end
    CompleteFileName = [FileName,'.',Extension];
    %%% Checks whether the specified file name will overwrite an
    %%% existing file. 
    OutputFileOverwrite = exist([cd,'/',CompleteFileName]);
    if OutputFileOverwrite ~= 0
        Answer = questdlg(['A file with the name ', CompleteFileName, ' already exists. Do you want to overwrite it?'],'Confirm file overwrite','Yes','No','No');
        if strcmp(Answer,'Yes') == 1;
            imwrite(ClickedImage, CompleteFileName, Extension)
            MsgboxHandle2 = msgbox(['The file ', CompleteFileName, ' has been saved to the current directory']);
            
        end
    else
        imwrite(ClickedImage, CompleteFileName, Extension)
        MsgboxHandle2 = msgbox(['The file ', CompleteFileName, ' has been saved to the current directory']);
    end
end
delete(MsgboxHandle)

%%%%%%%%%%%%%%%%%

% --- Executes on button press in ShowImageButton.
function ShowImageButton_Callback(hObject, eventdata, handles)
CurrentDirectory = cd;
Directory = handles.Vpathname;
cd(Directory)
%%% Opens a user interface window which retrieves a file name and path 
%%% name for the image to be used as a test image.
[FileName,PathName] = uigetfile('*.*','Select the image to view');
%%% If the user presses "Cancel", the FileName will = 0 and nothing will
%%% happen.
if FileName == 0
else
    %%% Acquires basic screen info for making buttons in the
    %%% display window.
    StdUnit = 'point';
    StdColor = get(0,'DefaultUIcontrolBackgroundColor');
    PointsPerPixel = 72/get(0,'ScreenPixelsPerInch');
    
    %%% Reads the image.
    Image = im2double(imread([PathName,'/',FileName]));
    FigureHandle = figure; imagesc(Image), colormap(gray)
    pixval
    %%% The following adds the Interactive Zoom button, which relies
    %%% on the InteractiveZoomSubfunction.m being in the CellProfiler
    %%% folder.
    set(FigureHandle, 'Unit',StdUnit)
    FigurePosition = get(FigureHandle, 'Position');
    %%% Specifies the function that will be run when the zoom button is
    %%% pressed.
    ZoomButtonCallback = 'try, InteractiveZoomSubfunction, catch msgbox(''Could not find the file called InteractiveZoomSubfunction.m which should be located in the CellProfiler folder.''), end';
    ZoomButtonHandle = uicontrol('Parent',FigureHandle, ...
        'CallBack',ZoomButtonCallback, ...
        'BackgroundColor',StdColor, ...
        'Position',PointsPerPixel*[FigurePosition(3)-108 5 105 22], ...
        'String','Interactive Zoom', ...
        'Style','pushbutton');
end
cd(CurrentDirectory)

%%%%%%%%%%%%%%%%%

% --- Executes on button press in ShowPixelDataButton.
function ShowPixelDataButton_Callback(hObject, eventdata, handles)
FigureNumber = inputdlg('In which figure number would like to see pixel data?','',1);
if isempty(FigureNumber) ~= 1
    FigureNumber = str2num(FigureNumber{1});
    pixval(FigureNumber,'on')
end

%%%%%%%%%%%%%%%%%

% --- Executes on button press in CloseAllFigureWindowsButton.
function CloseAllFigureWindowsButton_Callback(hObject, eventdata, handles)

%%% Requests confirmation to really delete all the figure windows.
Answer = questdlg('Are you sure you want to close all open figure windows and timers?','Confirm','Yes','No','Yes');
if strcmp(Answer, 'Yes') == 1
    %%% Lists all of the figure/graphics handles.
    AllHandles = findobj;
    %%% Checks which handles are integers (remainder after dividing by 1 =
    %%% zero). The regular figure windows and the Matlab root all have integer
    %%% handles, whereas the main CellProfiler window and the Timer window have
    %%% noninteger handles.
    WhichIntegers = rem(AllHandles,1);
    RootAndFigureHandles = AllHandles(WhichIntegers ==0);
    %%% Removes the handle "0" which is the root handle so that the main Matlab
    %%% window is not attempted to be deleted.
    FigureHandlesToBeDeleted = RootAndFigureHandles(RootAndFigureHandles ~= 0);
    %%% Closes the figure windows.
    delete(FigureHandlesToBeDeleted)
    %%% Finds and closes timer windows.
    TimerHandles = findall(findobj, 'Name', 'Timer');
    delete(TimerHandles)
end

%%%%%%%%%%%%%%%%%

% --- Executes on button press in ExportDataButton.
function ExportDataButton_Callback(hObject, eventdata, handles)

%%% Determines the current directory so it can switch back when done.
CurrentDirectory = pwd;
cd(handles.Vworkingdirectory)
%%% Ask the user to choose the file from which to extract measurements.
[RawFileName, RawPathName] = uigetfile('*.mat','Select the raw measurements file');
if RawFileName == 0
else
    cd(RawPathName);
    load(RawFileName);
    
    %%% Extract the fieldnames of measurements from the handles structure. Also
    %%% adds the fieldnames of file names that have been analyzed.
    Fieldnames = fieldnames(handles);
    MeasFieldnames = Fieldnames(strncmp(Fieldnames,'dMT',3)==1 | strncmp(Fieldnames,'dOTFilename', 11)==1 | strncmp(Fieldnames,'dOTTimeElapsed', 14)==1);
    %%% Determines whether any sample info has been loaded.  If sample info has
    %%% been loaded, the heading for that sample info would be listed in
    %%% handles.headings.  If sample info is present, the fieldnames for those
    %%% are also added to the list of data to extract.
    if isfield(handles, 'headings') == 1
        HeadingNames = handles.headings';
        MeasFieldnames = cat(1,HeadingNames, MeasFieldnames);
    end
    
    %%% Error detection.
    if isempty(MeasFieldnames)
        errordlg('No measurements were found in the file you selected. In the handles structure contained within the output file, the fieldnames of data to be extracted must be preceded by the prefix dMT, dOTFilename, or a heading loaded via the "Load sample info" button.')
    else
        
        %%% Determine the number of image sets for which there are data.
        fieldname = MeasFieldnames{1};
        TotalNumberImageSets = num2str(length(handles.(fieldname)));
        %%% Ask the user to specify the number of image sets to extract.
        NumberOfImages = inputdlg({['How many image sets do you want to extract? As a shortcut,                     type the numeral 0 to extract data from all ', TotalNumberImageSets, ' image sets.']},'Specify number of image sets',1,{'0';' '});
        %%% If the user presses the Cancel button, the program goes to the end.
        if isempty(NumberOfImages)
        else
            %%% Calculate the appropriate number of image sets.
            NumberOfImages = str2num(NumberOfImages{1});
            if NumberOfImages == 0
                NumberOfImages = length(handles.(char(MeasFieldnames(1))));
            elseif NumberOfImages > length(handles.(char(MeasFieldnames(1))));
                errordlg(['There are only ', length(handles.(char(MeasFieldnames(1)))), ' image sets total.'])
            end
            %%% Determines the suggested file name.
            try
                %%% Find and remove the file format extension within the original file
                %%% name, but only if it is at the end. Strip the original file format extension 
                %%% off of the file name, if it is present, otherwise, leave the original
                %%% name intact.
                CharFileName = char(RawFileName);
                PotentialDot = CharFileName(end-3:end-3);
                if strcmp(PotentialDot,'.') == 1
                    BareFileName = [CharFileName(1:end-4),'.xls'];
                else BareFileName = [CharFileName,'.xls'];
                end
            catch BareFileName = 'TempData';
            end
            
            %%% Ask the user to name the file.
            FileName = inputdlg('What do you want to call the resulting measurements file?  To open the file easily in Excel, add ".xls" to the name.','Name the file',1,{BareFileName});
            %%% If the user presses the Cancel button, the program goes to the end.
            if isempty(FileName)
            else
                FileName = FileName{1};
                OutputFileOverwrite = exist([cd,'/',FileName]);
                if OutputFileOverwrite ~= 0
                    errordlg('A file with that name already exists in the directory containing the raw measurements file.  Repeat and choose a different file name.')
                else
                    %%% Extract the measurements.  Waitbar shows the percentage of image sets
                    %%% remaining.
                    WaitbarHandle = waitbar(0,'Extracting measurements...');
                    TimeStart = clock;
                    for imagenumber = 1:NumberOfImages
                        for FieldNumber = 1:length(MeasFieldnames)
                            Fieldname = cell2mat(MeasFieldnames(FieldNumber));
                            Measurements(imagenumber,FieldNumber) = {handles.(Fieldname){imagenumber}};
                        end
                        CurrentTime = clock;
                        TimeSoFar = etime(CurrentTime,TimeStart);
                        TimePerSet = TimeSoFar/imagenumber;
                        ImagesRemaining = NumberOfImages - imagenumber;
                        TimeRemaining = round(TimePerSet*ImagesRemaining);
                        WaitbarText = ['Extracting measurements... ', num2str(TimeRemaining), ' seconds remaining.'];
                        waitbar(imagenumber/NumberOfImages, WaitbarHandle, WaitbarText)
                    end
                    close(WaitbarHandle) 
                    %%% Open the file and name it appropriately.
                    fid = fopen(FileName, 'wt');
                    %%% Write the MeasFieldnames as headings for columns.
                    for i = 1:size(MeasFieldnames,1),
                        fwrite(fid, char(MeasFieldnames(i)), 'char');
                        fwrite(fid, sprintf('\t'), 'char');
                    end
                    fwrite(fid, sprintf('\n'), 'char');
                    %%% Write the Measurements.
                    WaitbarHandle = waitbar(0,'Writing the measurements file...');
                    NumberMeasurements = size(Measurements,1);
                    TimeStart = clock;
                    for i = 1:NumberMeasurements
                        for measure = 1:size(MeasFieldnames,1),
                            val = Measurements(i,measure);
                            val = val{1};
                            if ischar(val),
                                fwrite(fid, sprintf('%s\t', val), 'char');
                            else
                                fwrite(fid, sprintf('%g\t', val), 'char');
                            end
                        end
                        fwrite(fid, sprintf('\n'), 'char');
                          CurrentTime = clock;
                        TimeSoFar = etime(CurrentTime,TimeStart);
                        TimePerSet = TimeSoFar/i;
                        ImagesRemaining = NumberOfImages - i;
                        TimeRemaining = round(TimePerSet*ImagesRemaining);
                        WaitbarText = ['Writing the measurements file... ', num2str(TimeRemaining), ' seconds remaining.'];
                        waitbar(i/NumberMeasurements, WaitbarHandle, WaitbarText)
                    end
                    close(WaitbarHandle) 
                    %%% Close the file
                    fclose(fid);
                    helpdlg(['The file ', FileName, ' has been written to the directory where the raw measurements file is located.'])
                end % This goes with the error catching at the beginning of the file.
            end % This goes with the error catching "No measurements found" at the beginning of the file.
        end % This goes with the "Cancel" button on the Number of Image Sets dialog.
    end % This goes with the "Cancel" button on the FileName dialog.
end % This goes with the "Cancel" button on the RawFileName dialog.

cd(CurrentDirectory);
% In case I want to save data that is 
% all numbers, with different numbers of rows for each column, the
% following code might be helpful:
% fid = fopen(filename,'w');
% for i = 1:length(Measurements)
%     fprintf(fid,'%d   ',Measurements{i});
%     fprintf(fid,'\n');
% end
% fclose(fid);
% type eval(filename)

%%%%%%%%%%%%%%%%%%%%%

% --- Executes on button press in ExportCellByCellButton.
function ExportCellByCellButton_Callback(hObject, eventdata, handles)
%%% Determines the current directory so it can switch back when done.
CurrentDirectory = pwd;
cd(handles.Vworkingdirectory)
%%% Ask the user to choose the file from which to extract measurements.
[RawFileName, RawPathName] = uigetfile('*.mat','Select the raw measurements file');
if RawFileName == 0
    cd(CurrentDirectory);
    return
end
cd(RawPathName);
load(RawFileName);

Answer = questdlg('Do you want to export cell by cell data for all measurements from one image, or data from all images for one measurement?','','All measurements','All images','All measurements');

if strcmp(Answer, 'All images') == 1
    %%% Extract the fieldnames of cell by cell measurements from the handles structure. 
    Fieldnames = fieldnames(handles);
    MeasFieldnames = Fieldnames(strncmp(Fieldnames,'dMC',3)==1);
    %%% Error detection.
    if isempty(MeasFieldnames)
        errordlg('No measurements were found in the file you selected.  They would be found within the output file''s handles structure preceded by ''dMC''.')
        cd(CurrentDirectory);
        return
    end
    %%% Removes the 'dMC' prefix from each name for display purposes.
    for Number = 1:length(MeasFieldnames)
        EditedMeasFieldnames{Number} = MeasFieldnames{Number}(4:end);
    end
    
    %%% Allows the user to select a measurement from the list.
    [Selection, ok] = listdlg('ListString',EditedMeasFieldnames, 'ListSize', [300 600],...
        'Name','Select measurement',...
        'PromptString','Choose a measurement to export','CancelString','Cancel',...
        'SelectionMode','single');
    if ok == 0
        cd(CurrentDirectory);
        return
    end
    EditedMeasurementToExtract = char(EditedMeasFieldnames(Selection));
    MeasurementToExtract = ['dMC', EditedMeasurementToExtract];
    TotalNumberImageSets = length(handles.(MeasurementToExtract));
    Measurements = handles.(MeasurementToExtract);
    
    %%% Extract the fieldnames of non-cell by cell measurements from the
    %%% handles structure. This will be used as headings for each column of
    %%% measurements.
    HeadingFieldnames = Fieldnames(strncmp(Fieldnames,'dOTFilename',11)==1);
    %%% Error detection.
    if isempty(HeadingFieldnames)
        errordlg('No headings were found in the file you selected.  They would be found within the output file''s handles structure preceded by ''dOTFilename''.')
        cd(CurrentDirectory);
        return
    end
    %%% Removes the 'dOT' prefix from each name for display purposes.
    for Number = 1:length(HeadingFieldnames)
        EditedHeadingFieldnames{Number} = HeadingFieldnames{Number}(4:end);
    end
    
    %%% Allows the user to select a heading name from the list.
    [Selection, ok] = listdlg('ListString',EditedHeadingFieldnames, 'ListSize', [300 600],...
        'Name','Select measurement',...
        'PromptString','Choose a field to label each column of data with','CancelString','Cancel',...
        'SelectionMode','single');
    if ok == 0
        cd(CurrentDirectory);
        return
    end
    EditedHeadingToDisplay = char(EditedHeadingFieldnames(Selection));
    HeadingToDisplay = ['dOT', EditedHeadingToDisplay];
    %%% Extracts the headings.
    ListOfHeadings = handles.(HeadingToDisplay);
    
    %%% Determines the suggested file name.
    try
        %%% Find and remove the file format extension within the original file
        %%% name, but only if it is at the end. Strip the original file format extension 
        %%% off of the file name, if it is present, otherwise, leave the original
        %%% name intact.
        CharFileName = char(RawFileName);
        PotentialDot = CharFileName(end-3:end-3);
        if strcmp(PotentialDot,'.') == 1
            BareFileName = [CharFileName(1:end-4),'.xls'];
        else BareFileName = [CharFileName,'.xls'];
        end
    catch BareFileName = 'TempData';
    end
    %%% Ask the user to name the file.
    FileName = inputdlg('What do you want to call the resulting measurements file?  To open the file easily in Excel, add ".xls" to the name.','Name the file',1,{BareFileName});
    %%% If the user presses the Cancel button, the program goes to the end.
    if isempty(FileName)
        cd(CurrentDirectory);
        return
    end
    FileName = FileName{1};
    OutputFileOverwrite = exist([cd,'/',FileName]);
    if OutputFileOverwrite ~= 0
        Answer = questdlg('A file with that name already exists in the directory containing the raw measurements file. Do you wish to overwrite?','Confirm overwrite','Yes','No','No');
        if strcmp(Answer, 'No') == 1
            cd(CurrentDirectory);
            return    
        end
    end
    
    %%% Opens the file and names it appropriately.
    fid = fopen(FileName, 'wt');
    %%% Writes MeasurementToExtract as the heading for the first column/row.
    fwrite(fid, char(MeasurementToExtract), 'char');
    fwrite(fid, sprintf('\n'), 'char');
    
    %%% Writes the data, row by row: one row for each image.
    for ImageNumber = 1:TotalNumberImageSets
        %%% Writes the heading in the first column.
        fwrite(fid, char(ListOfHeadings(ImageNumber)), 'char');
        %%% Tabs to the next column.
        fwrite(fid, sprintf('\t'), 'char');
        %%% Writes the measurements for that image in successive columns.
        fprintf(fid,'%d\t',Measurements{ImageNumber});
        %%% Returns to the next row.
        fwrite(fid, sprintf('\n'), 'char');
    end
    %%% Closes the file
    fclose(fid);
    helpdlg(['The file ', FileName, ' has been written to the directory where the raw measurements file is located.'])
    
elseif strcmp(Answer, 'All measurements') == 1
    TotalNumberImageSets = handles.setbeinganalyzed;
    %%% Asks the user to specify which image set to export.
    Answers = inputdlg({['Enter the sample number to export. There are ', num2str(TotalNumberImageSets), ' total.']},'Choose samples to export',1,{'1'});
    if isempty(Answers{1})
        cd(CurrentDirectory);
        return
    end
    try ImageNumber = str2num(Answers{1});
    catch errordlg('The text entered was not a number.')
        cd(CurrentDirectory);
        return
    end
    if ImageNumber > TotalNumberImageSets
        errordlg(['There are only ', num2str(TotalNumberImageSets), ' image sets total.'])
        cd(CurrentDirectory);
        return
    end
    
    %%% Extract the fieldnames of cell by cell measurements from the handles structure. 
    Fieldnames = fieldnames(handles);
    MeasFieldnames = Fieldnames(strncmp(Fieldnames,'dMC',3)==1);
    %%% Error detection.
    if isempty(MeasFieldnames)
        errordlg('No measurements were found in the file you selected.  They would be found within the output file''s handles structure preceded by ''dMC''.')
        cd(CurrentDirectory);
        return
    end
    
    %%% Extract the fieldnames of non-cell by cell measurements from the
    %%% handles structure. This will be used as headings for each column of
    %%% measurements.
    HeadingFieldnames = Fieldnames(strncmp(Fieldnames,'dOTFilename',11)==1);
    %%% Error detection.
    if isempty(HeadingFieldnames)
        errordlg('No headings were found in the file you selected.  They would be found within the output file''s handles structure preceded by ''dOTFilename''.')
        cd(CurrentDirectory);
        return
    end
    %%% Removes the 'dOT' prefix from each name for display purposes.
    for Number = 1:length(HeadingFieldnames)
        EditedHeadingFieldnames{Number} = HeadingFieldnames{Number}(4:end);
    end
    
    %%% Allows the user to select a heading name from the list.
    [Selection, ok] = listdlg('ListString',EditedHeadingFieldnames, 'ListSize', [300 600],...
        'Name','Select measurement',...
        'PromptString','Choose a field to label this data.','CancelString','Cancel',...
        'SelectionMode','single');
    if ok == 0
        cd(CurrentDirectory);
        return
    end
    EditedHeadingToDisplay = char(EditedHeadingFieldnames(Selection));
    HeadingToDisplay = ['dOT', EditedHeadingToDisplay];
    %%% Extracts the headings.
    ImageNamesToDisplay = handles.(HeadingToDisplay);
    ImageNameToDisplay = ImageNamesToDisplay(ImageNumber);
    
    %%% Determines the suggested file name.
    try
        %%% Find and remove the file format extension within the original file
        %%% name, but only if it is at the end. Strip the original file format extension 
        %%% off of the file name, if it is present, otherwise, leave the original
        %%% name intact.
        CharFileName = char(RawFileName);
        PotentialDot = CharFileName(end-3:end-3);
        if strcmp(PotentialDot,'.') == 1
            BareFileName = [CharFileName(1:end-4),'.xls'];
        else BareFileName = [CharFileName,'.xls'];
        end
    catch BareFileName = 'TempData';
    end
    %%% Asks the user to name the file.
    FileName = inputdlg('What do you want to call the resulting measurements file?  To open the file easily in Excel, add ".xls" to the name.','Name the file',1,{BareFileName});
    %%% If the user presses the Cancel button, the program goes to the end.
    if isempty(FileName)
        cd(CurrentDirectory);
        return
    end
    FileName = FileName{1};
    OutputFileOverwrite = exist([cd,'/',FileName]);
    if OutputFileOverwrite ~= 0
        Answer = questdlg('A file with that name already exists in the directory containing the raw measurements file. Do you wish to overwrite?','Confirm overwrite','Yes','No','No');
        if strcmp(Answer, 'No') == 1
            cd(CurrentDirectory);
            return    
        end
    end
    
    %%% Opens the file and names it appropriately.
    fid = fopen(FileName, 'wt');
    %%% Writes ImageNameToDisplay as the heading for the first column/row.
    fwrite(fid, char(ImageNameToDisplay), 'char');
    fwrite(fid, sprintf('\n'), 'char');
    %%% Writes the data, row by row: one row for each measurement type.
    for MeasNumber = 1:length(MeasFieldnames)
        FieldName = char(MeasFieldnames(MeasNumber));
        %%% Writes the measurement heading in the first column.
        fwrite(fid, FieldName, 'char');
        %%% Tabs to the next column.
        fwrite(fid, sprintf('\t'), 'char');
        Measurements = handles.(FieldName);
        Measurements = Measurements';
        %%% Writes the measurements for that measurement type in successive columns.
        fprintf(fid,'%d\t',Measurements{ImageNumber});
        %%% Returns to the next row.
        fwrite(fid, sprintf('\n'), 'char');
    end
    %%% Closes the file
    fclose(fid);
    helpdlg(['The file ', FileName, ' has been written to the directory where the raw measurements file is located.'])
end
cd(CurrentDirectory);

%%%%%%%%%%%%%%%%%

% --- Executes on button press in HistogramButton.
function HistogramButton_Callback(hObject, eventdata, handles)
%%% Determines the current directory so it can switch back when done.
CurrentDirectory = pwd;
cd(handles.Vworkingdirectory)
%%% Ask the user to choose the file from which to extract measurements.
[RawFileName, RawPathName] = uigetfile('*.mat','Select the raw measurements file');
if RawFileName == 0
    cd(CurrentDirectory);
    return
end
cd(RawPathName);
load(RawFileName);
%%% Extract the fieldnames of measurements from the handles structure. 
Fieldnames = fieldnames(handles);
MeasFieldnames = Fieldnames(strncmp(Fieldnames,'dMC',3)==1);
%%% Error detection.
if isempty(MeasFieldnames)
    errordlg('No measurements were found in the file you selected.  They would be found within the output file''s handles structure preceded by ''dMC''.')
    cd(CurrentDirectory);
    return
end
%%% Removes the 'dMC' prefix from each name for display purposes.
for Number = 1:length(MeasFieldnames)
    EditedMeasFieldnames{Number} = MeasFieldnames{Number}(4:end);
end
%%% Allows the user to select a measurement from the list.
[Selection, ok] = listdlg('ListString',EditedMeasFieldnames, 'ListSize', [300 600],...
    'Name','Select measurement',...
    'PromptString','Choose a measurement to display as histograms','CancelString','Cancel',...
    'SelectionMode','single');
if ok ~= 0
    EditedMeasurementToExtract = char(EditedMeasFieldnames(Selection));
    MeasurementToExtract = ['dMC', EditedMeasurementToExtract];
    %%% Allows the user to load sample info.
    Answer = questdlg('Do you want to load names for each image set, other than names that are already embedded in the output file?','','Yes','No','No');
    if strcmp(Answer,'Yes') ==1
        %%% START OF LOAD SAMPLE INFO SECTION. THIS IS JUST LIKE
        %%% THE CODE FOR THE load sample info button, but I
        %%% couldn't figure out how to make the handles structure
        %%% be passed properly.
        
        %%% Opens a dialog box to retrieve a file name that contains a list of 
        %%% sample descriptions, like gene names or sample numbers.
        [fname,pname] = uigetfile('*.*','Choose sample info text file');
        %%% If the user presses "Cancel", the fname will = 0 and nothing will
        %%% happen.
        if fname == 0
        else
            extension = fname(end-2:end);
            %%% Checks whether the chosen file is a text file.
            if strcmp(extension,'txt') == 0;
                errordlg('Sorry, the list of sample descriptions must be in a text file (.txt).');
                cd(CurrentDirectory);
                return
            else 
                %%% Saves the text from the file into a new variable, "SampleNames".  The
                %%% '%s' command tells it to select groups of strings not separated by
                %%% things like carriage returns, and maybe spaces and commas, too. (Not
                %%% sure about that).
                SampleNames = textread([pname fname],'%s');
                NumberSamples = length(SampleNames);
                %%% Displays a warning.  The buttons don't do anything except proceed.
                Warning = {'The next window will show you a preview'; ...
                        'of the sample info you have loaded'; ...
                        ''; ...
                        ['You have ', num2str(NumberSamples), ' lines of sample information.'];...
                        ''; ...
                        'Press either ''OK'' button to continue.';...
                        '-------'; ...
                        'Warning:'; 'Please note that any spaces or weird'; ...
                        '(e.g. punctuation) characters on any line of your'; ...
                        'text file will split the entry into two entries!';...
                        '-------'; ...
                        'Also check that the order of the image files within Matlab is';...
                        'as expected.  For example, If you are running Matlab within';...
                        'X Windows on a Macintosh, the Mac will show the files as: ';...
                        '(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11) whereas the X windows ';...
                        'system that Matlab uses will show them as ';...
                        '(1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9) and so on.  So be sure that ';...
                        'the order of your sample info matches the order that Matlab ';...
                        'is using.  Look in the Current Directory window of Matlab to ';...
                        'see the order.  Go to View > Current Directory to open the ';...
                        'window if it is not already visible.'};
                listdlg('ListString',Warning,'ListSize', [300 600],'Name','Warning',...
                    'PromptString','Press any button to continue.', 'CancelString','Ok',...
                    'SelectionMode','single');
                %%% Displays a listbox so the user can preview the data.  The buttons in
                %%% this listbox window don't do anything except proceed.
                listdlg('ListString',SampleNames, 'ListSize', [300 600],...
                    'Name','Preview your sample data',...
                    'PromptString','Press any button to continue.','CancelString','Ok',...
                    'SelectionMode','single');
                %%% Retrieves a name for the heading of the sample data just entered.
                A = inputdlg('Enter the heading for these sample descriptions (e.g. GeneNames                 or SampleNumber). Your entry must be one word with letters and                   numbers only, and must begin with a letter.','Name the Sample Info',1);
                %%% If the user presses cancel A will be an empty array.  If the user
                %%% doesn't enter anything in the dialog box but presses "OK" anyway,
                %%% A is equal to '' (two quote marks).  In either case, skip to the end; 
                %%% don't save anything. 
                if (isempty(A)) 
                elseif strcmp(A,'') == 1, 
                    errordlg('Sample info was not saved, because no heading was entered.');
                    cd(CurrentDirectory);
                    return
                elseif isfield(handles, A) == 1
                    errordlg('Sample info was not saved, because sample info with that heading has already been stored.');
                    cd(CurrentDirectory);
                    return    
                else
                    %%% Uses the heading the user entered to name the field in the handles
                    %%% structure array and save the SampleNames list there.
                    try    handles.(char(A)) = SampleNames;
                        %%% Also need to add this heading name (field name) to the headings
                        %%% field of the handles structure (if it already exists), in the last position. 
                        if isfield(handles, 'headings') == 1
                            N = length(handles.headings) + 1;
                            handles.headings(N)  = A;
                            %%% If the headings field doesn't yet exist, create it and put the heading 
                            %%% name in position 1.
                        else handles.headings(1)  = A;
                        end
                        guidata(hObject, handles);
                    catch errordlg('Sample info was not saved, because the heading contained illegal characters.');
                        cd(CurrentDirectory);
                        return
                    end % Goes with catch
                end
            end
            %%% One of these "end"s goes with the if A is empty, when user presses
            %%% cancel. 
        end 
    end
    
    %%% Determines whether any sample info has been loaded.  If sample info has
    %%% been loaded, the heading for that sample info would be listed in
    %%% handles.headings.  If sample info is present, the fieldnames for those
    %%% are extracted.
    if isfield(handles, 'headings') == 1
        HeadingNames = handles.headings';
        %%% Allows the user to select a heading from the list.
        [Selection, ok] = listdlg('ListString',HeadingNames, 'ListSize', [300 600],...
            'Name','Select sample info',...
            'PromptString','Choose the sample info with which to label each histogram.','CancelString','Cancel',...
            'SelectionMode','single');
        if ok ~= 0
            HeadingName = char(HeadingNames(Selection));
            SampleNames = handles.(HeadingName);
        else cd(CurrentDirectory);
            return
        end                    
    end
    %%% Asks the user whether a histogram should be shown for all image
    %%% sets or just a few.
    TotalNumberImageSets = length(handles.(MeasurementToExtract));
    TextTotalNumberImageSets = num2str(TotalNumberImageSets);
    %%% Ask the user to specify which image sets to display.
    Prompts = {'Enter the first sample number to display','Enter the last sample number to display'};
    Defaults = {'1',TextTotalNumberImageSets};
    Answers = inputdlg(Prompts,'Choose samples to display',1,Defaults);
    if isempty(Answers) ~= 1
        FirstImage = str2num(Answers{1});
        LastImage = str2num(Answers{2});
        if isempty(FirstImage)
            errordlg('No number was entered for the first sample number to display.')
            cd(CurrentDirectory);
            return
        end
        if isempty(LastImage)
            errordlg('No number was entered for the last sample number to display.')
            cd(CurrentDirectory);
            return
        end
        NumberOfImages = LastImage - FirstImage + 1;
        if NumberOfImages == 0
            NumberOfImages = TotalNumberImageSets;
        elseif NumberOfImages > TotalNumberImageSets
            errordlg(['There are only ', TextTotalNumberImageSets, ' image sets total.'])
            cd(CurrentDirectory);
            return
        end
        
        %%% Ask the user to specify histogram settings.
        Prompts = {'Enter the number of bins you want to be displayed in the histogram','Enter the minimum value to display', 'Enter the maximum value to display', 'Do you want to calculate one histogram for all of the specified data?', 'Do you want the Y-axis (number of cells) to be absolute or relative?','Display as a compressed histogram?','To save the histogram data, enter a filename (with ".xls" to open easily in Excel).'};
        Defaults = {'20','automatic','automatic','no','relative','no','no'};
        Answers = inputdlg(Prompts,'Choose histogram settings',1,Defaults);
        %%% Error checking/canceling.
        if isempty(Answers)
            cd(CurrentDirectory);
            return
        end
        try NumberOfBins = str2num(Answers{1});
        catch errordlg('The text entered for the question "Enter the number of bins you want to be displayed in the histogram" was not a number.')
            cd(CurrentDirectory);
            return
        end
        if isempty(NumberOfBins) ==1
            errordlg('No text was entered for "Enter the number of bins you want to be displayed in the histogram".')
            cd(CurrentDirectory);
            return
        end
        MinHistogramValue = Answers{2};
        if isempty(MinHistogramValue) ==1
            errordlg('No text was entered for "Enter the minimum value to display".')
            cd(CurrentDirectory);
            return
        end
        MaxHistogramValue = Answers{3};
        if isempty(MaxHistogramValue) ==1
            errordlg('No text was entered for "Enter the maximum value to display".')
            cd(CurrentDirectory);
            return
        end
        CumulativeHistogram = Answers{4};
        %%% Error checking for the Y Axis Scale question.
        try YAxisScale = lower(Answers{5});
        catch errordlg('The text you entered for ''Do you want the Y-axis (number of cells) to be absolute or relative?'' was not recognized.');
            cd(CurrentDirectory);
            return    
        end
        if strcmp(YAxisScale, 'relative') ~= 1 & strcmp(YAxisScale, 'absolute') ~= 1
            errordlg('The text you entered for ''Do you want the Y-axis (number of cells) to be absolute or relative?'' was not recognized.');
            cd(CurrentDirectory);
            return
        end
        CompressedHistogram = Answers{6};
        if strcmp(CompressedHistogram,'yes') ~= 1 & strcmp(CompressedHistogram,'no') ~= 1 
            errordlg('You must enter "yes" or "no" for displaying the histograms in compressed format.');
            cd(CurrentDirectory);
            return
        end
        SaveData = Answers{7};
        if isempty(SaveData)
            errordlg('You must enter "no", or a filename, in answer to the question about saving the data.');
            cd(CurrentDirectory);
            return
        end
        OutputFileOverwrite = exist([cd,'/',SaveData]);
        if OutputFileOverwrite ~= 0
            Answer = questdlg('A file with that name already exists in the directory containing the raw measurements file. Do you wish to overwrite?','Confirm overwrite','Yes','No','No');
            if strcmp(Answer, 'No') == 1
                cd(CurrentDirectory);
                return    
            end
        end
        
        %%% Calculates the default bin size and range based on all
        %%% the data.
        AllMeasurementsCellArray = handles.(MeasurementToExtract);
        SelectedMeasurementsCellArray = AllMeasurementsCellArray(:,FirstImage:LastImage);
        SelectedMeasurementsMatrix = cell2mat(SelectedMeasurementsCellArray(:));
        PotentialMaxHistogramValue = max(SelectedMeasurementsMatrix);
        PotentialMinHistogramValue = min(SelectedMeasurementsMatrix);
        %%% See whether the min and max histogram values were user-entered numbers or should be automatically calculated.
        if isempty(str2num(MinHistogramValue))
            if strcmp(MinHistogramValue,'automatic') == 1
                MinHistogramValue = PotentialMinHistogramValue;
            else
                errordlg('The value entered for the minimum histogram value must be either a number or the word ''automatic''.')
                cd(CurrentDirectory);
                return
            end
        else MinHistogramValue = str2num(MinHistogramValue);
        end
        if isempty(str2num(MaxHistogramValue))
            if strcmp(MaxHistogramValue,'automatic') == 1
                MaxHistogramValue = PotentialMaxHistogramValue;
            else
                errordlg('The value entered for the maximum histogram value must be either a number or the word ''automatic''.')
                cd(CurrentDirectory);
                return
            end
        else MaxHistogramValue = str2num(MaxHistogramValue);
        end
        %%% Determine plot bin locations.
        HistogramRange = MaxHistogramValue - MinHistogramValue;
        if HistogramRange <= 0
            errordlg('The numbers you entered for the minimum or maximum, or the number which was calculated automatically for one of these values, results in the range being zero or less.  For example, this would occur if you entered a minimum that is greater than the maximum which you asked to be automatically calculated.')
            cd(CurrentDirectory);
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
        if strcmp(lower(CumulativeHistogram), 'no') ~= 1
            [HistogramData,Ignore] = histc(SelectedMeasurementsMatrix,BinLocations);
            %%% Deletes the last value of HistogramData, which is
            %%% always a zero (because it's the number of values
            %%% that match + inf).
            HistogramData(n+1) = [];
            FinalHistogramData(:,1) = HistogramData;
            HistogramTitles{1} = ['Histogram of data from Image #', num2str(FirstImage), ' to #', num2str(LastImage)];
            FirstImage = 1;
            LastImage = 1;
            ImageNumber = 1;
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
                h = helpdlg(['The file ', SaveData, ' has been written to the directory where the raw measurements file is located.'])
                waitfor(h)
            end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%% Calculates histogram data for non-cumulative histogram %%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        else
            for ImageNumber = FirstImage:LastImage
                ListOfMeasurements{ImageNumber} = handles.(MeasurementToExtract){ImageNumber};
                [HistogramData,Ignore] = histc(ListOfMeasurements{ImageNumber},BinLocations);
                %%% Deletes the last value of HistogramData, which
                %%% is always a zero (because it's the number of values that match
                %%% + inf).
                HistogramData(n+1) = [];
                FinalHistogramData(:,ImageNumber) = HistogramData;
                if exist('SampleNames') == 1
                    SampleName = SampleNames{ImageNumber};
                    HistogramTitles{ImageNumber} = ['#', num2str(ImageNumber), ': ' , SampleName];
                else HistogramTitles{ImageNumber} = ['Image #', num2str(ImageNumber)];
                end                 
            end 
            
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
            LeftFrameHandle = uicontrol('Parent',FigureHandle, ...
                'Unit',StdUnit, ...
                'BackgroundColor',[0.8 0.8 0.8], ...
                'Position',PointsPerPixel*[0 NewHeight-52 0.5*NewFigurePosition(3) 60], ...
                'Units','Normalized',...
                'Style','frame');
            RightFrameHandle = uicontrol('Parent',FigureHandle, ...
                'Unit',StdUnit, ...
                'BackgroundColor',[0.8 0.8 0.8], ...
                'Position',PointsPerPixel*[0.5*NewFigurePosition(3) NewHeight-52 0.5*NewFigurePosition(3) 60], ...
                'Units','Normalized',...
                'Style','frame');
            MiddleFrameHandle = uicontrol('Parent',FigureHandle, ...
                'Unit',StdUnit, ...
                'BackgroundColor',[0.8 0.8 0.8], ...
                'Position',PointsPerPixel*[100 NewHeight-26 180 30], ...
                'Units','Normalized',...
                'Style','frame');
            TextHandle1 = uicontrol('Parent',FigureHandle, ...
                'BackgroundColor',get(FigureHandle,'Color'), ...
                'Unit',StdUnit, ...
                'Position',PointsPerPixel*[5 NewHeight-30 85 22], ...
                'Units','Normalized',...
                'String','Change plots:', ...
                'Style','text');
            TextHandle2 = uicontrol('Parent',FigureHandle, ...
                'BackgroundColor',get(FigureHandle,'Color'), ...
                'Unit',StdUnit, ...
                'Position',PointsPerPixel*[0.5*NewFigurePosition(3)+15 NewHeight-30 85 22], ...
                'Units','Normalized',...
                'String','Change bars:', ...
                'Style','text');
            TextHandle3 = uicontrol('Parent',FigureHandle, ...
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
            ButtonHandle1 = uicontrol('Parent',FigureHandle, ...
                'Unit',StdUnit, ...
                'BackgroundColor',StdColor, ...
                'CallBack',Button1Callback, ...
                'Position',PointsPerPixel*[25 NewHeight-48 85 22], ...
                'Units','Normalized',...
                'String','This window', ...
                'Style','pushbutton');
            Button2Callback = 'propedit(gca,''v6''); drawnow';
            ButtonHandle2 = uicontrol('Parent',FigureHandle, ...
                'Unit',StdUnit, ...
                'BackgroundColor',StdColor, ...
                'CallBack',Button2Callback, ...
                'Position',PointsPerPixel*[115 NewHeight-48 70 22], ...
                'Units','Normalized',...
                'String','Current', ...
                'Style','pushbutton');
            Button3Callback = 'FigureHandles = findobj(''Parent'', 0); AxesHandles = findobj(FigureHandles, ''Type'', ''axes''); axis(AxesHandles, ''manual''); try, propedit(AxesHandles,''v6''), catch, msgbox(''A bug in Matlab is preventing this function from working. Service Request #1-RR6M1''), end, drawnow';
            ButtonHandle3 = uicontrol('Parent',FigureHandle, ...
                'Unit',StdUnit, ...
                'BackgroundColor',StdColor, ...
                'CallBack', Button3Callback, ...
                'Position',PointsPerPixel*[190 NewHeight-48 85 22], ...
                'Units','Normalized',...
                'String','All windows', ...
                'Style','pushbutton');
            Button4Callback = 'FigureHandle = gcf; PatchHandles = findobj(FigureHandle, ''Type'', ''patch''); try, propedit(PatchHandles,''v6''), catch, msgbox(''A bug in Matlab is preventing this function from working. Service Request #1-RR6M1''), end; drawnow';
            ButtonHandle4 = uicontrol('Parent',FigureHandle, ...
                'Unit',StdUnit, ...
                'BackgroundColor',StdColor, ...
                'CallBack', Button4Callback, ...
                'Position',PointsPerPixel*[0.5*NewFigurePosition(3)+5 NewHeight-48 85 22], ...
                'Units','Normalized',...
                'String','This window', ...
                'Style','pushbutton');
            Button5Callback = 'AxisHandle = gca; PatchHandles = findobj(''Parent'', AxisHandle, ''Type'', ''patch''); try, propedit(PatchHandles,''v6''), catch, msgbox(''A bug in Matlab is preventing this function from working. Service Request #1-RR6M1''), end; drawnow';
            ButtonHandle5 = uicontrol('Parent',FigureHandle, ...
                'Unit',StdUnit, ...
                'BackgroundColor',StdColor, ...
                'CallBack', Button5Callback, ...
                'Position',PointsPerPixel*[0.5*NewFigurePosition(3)+95 NewHeight-48 70 22], ...
                'Units','Normalized',...
                'String','Current', ...
                'Style','pushbutton');
            Button6Callback = 'FigureHandles = findobj(''Parent'', 0); PatchHandles = findobj(FigureHandles, ''Type'', ''patch''); try, propedit(PatchHandles,''v6''), catch, msgbox(''A bug in Matlab is preventing this function from working. Service Request #1-RR6M1''), end; drawnow';
            ButtonHandle6 = uicontrol('Parent',FigureHandle, ...
                'Unit',StdUnit, ...
                'BackgroundColor',StdColor, ...
                'CallBack', Button6Callback, ...
                'Position',PointsPerPixel*[0.5*NewFigurePosition(3)+170 NewHeight-48 85 22], ...
                'Units','Normalized',...
                'String','All windows', ...
                'Style','pushbutton');
            Button7Callback = 'msgbox(''Histogram display info: (1) Data outside the range you specified to calculate histogram bins are added together and displayed in the first and last bars of the histogram.  (2) Only the display can be changed in this window, including axis limits.  The histogram bins themselves cannot be changed here because the data must be recalculated. (3) If a change you make using the "Change display" buttons does not seem to take effect in all of the desired windows, try pressing enter several times within that box, or look in the bottom of the Property Editor window that opens when you first press one of those buttons.  There may be a message describing why.  For example, you may need to deselect "Auto" before changing the limits of the axes. (4) The labels for each bar specify the low bound for that bin.  In other words, each bar includes data equal to or greater than the label, but less than the label on the bar to its right. (5) If the tick mark labels are overlapping each other on the X axis, click a "Change display" button and either change the font size on the "Style" tab, or check the boxes marked "Auto" for "Ticks" and "Labels" on the "X axis" tab. Be sure to check both boxes, or the labels will not be accurate.  Changing the labels to "Auto" cannot be undone, and you will lose the detailed info about what values were actually used for the histogram bins.'')';
            ButtonHandle7 = uicontrol('Parent',FigureHandle, ...
                'Unit',StdUnit, ...
                'BackgroundColor',StdColor, ...
                'CallBack', Button7Callback, ...
                'Position',PointsPerPixel*[5 NewHeight-48 15 22], ...
                'Units','Normalized',...
                'String','?', ...
                'Style','pushbutton');
            %%% Hide every other label button.
            Button8Callback = 'FigureSettings = get(gca,''UserData'');  PlotBinLocations = FigureSettings{1}; XTickLabels = FigureSettings{2}; AxesHandles = findobj(gcf, ''Type'', ''axes''); if ceil(length(PlotBinLocations)/2) ~= length(PlotBinLocations)/2, PlotBinLocations(length(PlotBinLocations)) = []; XTickLabels(length(XTickLabels)) = []; end; PlotBinLocations2 = reshape(PlotBinLocations,2,[]); XTickLabels2 = reshape(XTickLabels,2,[]); set(AxesHandles,''XTick'',PlotBinLocations2(1,:)); set(AxesHandles,''XTickLabel'',XTickLabels2(1,:)); clear';
            ButtonHandle8 = uicontrol('Parent',FigureHandle, ...
                'Unit',StdUnit, ...
                'BackgroundColor',StdColor, ...
                'CallBack',Button8Callback, ...
                'Position',PointsPerPixel*[177 NewHeight-22 45 22], ...
                'Units','Normalized',...
                'String','Fewer', ...
                'Style','pushbutton');
            %%% Restore original X axis labels.
            Button9Callback = 'FigureSettings = get(gca,''UserData'');  PlotBinLocations = FigureSettings{1}; XTickLabels = FigureSettings{2}; AxesHandles = findobj(gcf, ''Type'', ''axes''); set(AxesHandles,''XTick'',PlotBinLocations); set(AxesHandles,''XTickLabel'',XTickLabels); clear';
            ButtonHandle9 = uicontrol('Parent',FigureHandle, ...
                'Unit',StdUnit, ...
                'BackgroundColor',StdColor, ...
                'CallBack',Button9Callback, ...
                'Position',PointsPerPixel*[227 NewHeight-22 50 22], ...
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
            cd(CurrentDirectory);
            return
        end
    end % Goes with cancel button when selecting the measurement to display.
end
cd(CurrentDirectory);

%%%%%%%%%%%%%%%%%

% --- Executes on button press in NormalizationButton.
function NormalizationButton_Callback(hObject, eventdata, handles)
h = msgbox('Copy your data to the clipboard then press OK');
waitfor(h)

uiimport('-pastespecial');
h = msgbox('After importing your data and pressing "Finish", click OK');
waitfor(h)
if exist('clipboarddata') == 0
    return
end
IncomingData = clipboarddata;

Prompts = {'Enter the number of rows','Enter the number of columns','Enter the percentile below which values will be excluded from fitting the normalization function.','Enter the percentile above which values will be excluded from fitting the normalization function.'};
Defaults = {'24','16','.05','.95'};
Answers = inputdlg(Prompts,'Describe Array/Slide Format',1,Defaults);
if isempty(Answers)
    return
end
NumberRows = str2num(Answers{1});
NumberColumns = str2num(Answers{2});
LowPercentile = str2num(Answers{3});
HighPercentile = str2num(Answers{4});
TotalSamplesToBeGridded = NumberRows*NumberColumns;
NumberSamplesImported = length(IncomingData);
if TotalSamplesToBeGridded > NumberSamplesImported
    h = warndlg(['You have specified a layout of ', num2str(TotalSamplesToBeGridded), ' samples in the layout, but only ', num2str(NumberSamplesImported), ' measurements were imported. The remaining spaces in the layout will be filled in with the value of the last sample.']); 
    waitfor(h)
    IncomingData(NumberSamplesImported+1:TotalSamplesToBeGridded) = IncomingData(NumberSamplesImported);
elseif TotalSamplesToBeGridded < NumberSamplesImported
    h = warndlg(['You have specified a layout of ', num2str(TotalSamplesToBeGridded), ' samples in the layout, but ', num2str(NumberSamplesImported), ' measurements were imported. The imported measurements at the end will be ignored.']); 
    waitfor(h)
    IncomingData(TotalSamplesToBeGridded+1:NumberSamplesImported) = [];
end

%%% The data is shaped into the appropriate grid.
MeanImage = reshape(IncomingData,NumberRows,NumberColumns);

%%% The data are listed in ascending order.
AscendingData = sort(IncomingData);

%%% The percentiles are calculated. (Statistics Toolbox has a percentile
%%% function, but many users may not have that function.)
%%% The values to be ignored are set to zero in the mask.
mask = MeanImage;
if LowPercentile ~= 0
RankOrderOfLowThreshold = floor(LowPercentile*NumberSamplesImported);
LowThreshold = AscendingData(RankOrderOfLowThreshold);
mask(mask <= LowThreshold) = 0;
end
if HighPercentile ~= 1
RankOrderOfHighThreshold = ceil(HighPercentile*NumberSamplesImported);
HighThreshold = AscendingData(RankOrderOfHighThreshold);
mask(mask >= HighThreshold) = 0;
end
ThrownOutDataForDisplay = mask;
ThrownOutDataForDisplay(mask > 0) = 1;

%%% Fits the data to a third-dimensional polynomial 
% [x,y] = meshgrid(1:size(MeanImage,2), 1:size(MeanImage,1));
% x2 = x.*x;
% y2 = y.*y;
% xy = x.*y;
% x3 = x2.*x;
% x2y = x2.*y;
% xy2 = y2.*x;
% y3 = y2.*y;
% o = ones(size(MeanImage));
% ind = find((MeanImage > 0) & (mask > 0));
% coeffs = [x3(ind) x2y(ind) xy2(ind) y3(ind) x2(ind) y2(ind) xy(ind) x(ind) y(ind) o(ind)] \ double(MeanImage(ind));
% IlluminationImage = reshape([x3(:) x2y(:) xy2(:) y3(:) x2(:) y2(:) xy(:) x(:) y(:) o(:)] * coeffs, size(MeanImage));
% 

%%% Fits the data to a fourth-dimensional polynomial 
[x,y] = meshgrid(1:size(MeanImage,2), 1:size(MeanImage,1));
x2 = x.*x;
y2 = y.*y;
xy = x.*y;
x3 = x2.*x;
x2y = x2.*y;
xy2 = y2.*x;
y3 = y2.*y;
x4 = x2.*x2;
y4 = y2.*y2;
x3y = x3.*y;
x2y2 = x2.*y2;
xy3 = x.*y3;
o = ones(size(MeanImage));
ind = find((MeanImage > 0) & (mask > 0));
coeffs = [x4(ind) x3y(ind) x2y2(ind) xy3(ind) y4(ind) ...
          x3(ind) x2y(ind) xy2(ind) y3(ind) ...
          x2(ind) xy(ind) y2(ind)  ...
          x(ind) y(ind) ...
          o(ind)] \ double(MeanImage(ind));
IlluminationImage = reshape([x4(:) x3y(:) x2y2(:) xy3(:) y4(:) ...
          x3(:) x2y(:) xy2(:) y3(:) ...
          x2(:) xy(:) y2(:)  ...
          x(:) y(:) ...
          o(:)] * coeffs, size(MeanImage));
CorrFactorsRaw = reshape(IlluminationImage,TotalSamplesToBeGridded,1);
IlluminationImage2 = IlluminationImage ./ mean(CorrFactorsRaw);
  
%%% Shows the results.
figure, subplot(1,3,1), imagesc(MeanImage), title('Imported Data'), colorbar
subplot(1,3,2), imagesc(ThrownOutDataForDisplay), title('Ignored Samples'),
subplot(1,3,3), imagesc(IlluminationImage2), title('Correction Factors'), colorbar

%%% Puts the results in a column and displays in the main Matlab window.
OrigData = reshape(MeanImage,TotalSamplesToBeGridded,1)
CorrFactors = reshape(IlluminationImage2,TotalSamplesToBeGridded,1);
CorrectedData = OrigData./CorrFactors

msgbox('The original data and the corrected data are now displayed in the Matlab window. You can cut and paste from there.')

% %%% Exports the results to the clipboard.
% clipboard('copy',CorrFactors);
% h = msgbox('The correction factors are now on the clipboard. Paste them where desired and press OK.  The data is also displayed in column format in the main Matlab window, so you can copy and paste from there as well.');
% waitfor(h)
% clipboard('copy',OrigData);
% h = msgbox('The original data used to generate those normalization factors is now on the clipboard. Paste them where desired (if desired) and press OK.  The data is also displayed in column format in the main Matlab window, so you can copy and paste from there as well.');
% waitfor(h)

%%%%%%%%%%%%%%%%%

% --- Executes on button press in DisplayDataOnImageButton.
function DisplayDataOnImageButton_Callback(hObject, eventdata, handles)
%%% Determines the current directory so it can switch back when done.
CurrentDirectory = pwd;
cd(handles.Vworkingdirectory)
%%% Asks the user to choose the file from which to extract measurements.
[RawFileName, RawPathName] = uigetfile('*.mat','Select the raw measurements file');
if RawFileName ~= 0
    cd(RawPathName);
    load(RawFileName);
    %%% Extracts the fieldnames of measurements from the handles structure. 
    Fieldnames = fieldnames(handles);
    MeasFieldnames = Fieldnames(strncmp(Fieldnames,'dMC',3)==1);
    %%% Error detection.
    if isempty(MeasFieldnames)
        errordlg('No measurements were found in the file you selected.  They would be found within the output file''s handles structure preceded by ''dMC''.')
        cd(CurrentDirectory);
        return
    else
        %%% Removes the 'dMC' prefix from each name for display purposes.
        for Number = 1:length(MeasFieldnames)
            EditedMeasFieldnames{Number} = MeasFieldnames{Number}(4:end);
        end
        %%% Allows the user to select a measurement from the list.
        [Selection, ok] = listdlg('ListString',EditedMeasFieldnames, 'ListSize', [300 600],...
            'Name','Select measurement',...
            'PromptString','Choose a measurement to display on the image','CancelString','Cancel',...
            'SelectionMode','single');
        if ok ~= 0
            EditedMeasurementToExtract = char(EditedMeasFieldnames(Selection));
            MeasurementToExtract = ['dMC', EditedMeasurementToExtract];
            %%% Allows the user to select the X Locations from the list.
            [Selection, ok] = listdlg('ListString',EditedMeasFieldnames, 'ListSize', [300 600],...
                'Name','Select the X locations to be used',...
                'PromptString','Select the X locations to be used','CancelString','Cancel',...
                'SelectionMode','single');
            if ok ~= 0
                EditedXLocationMeasurementName = char(EditedMeasFieldnames(Selection));
                XLocationMeasurementName = ['dMC', EditedXLocationMeasurementName];
                %%% Allows the user to select the Y Locations from the list.
                [Selection, ok] = listdlg('ListString',EditedMeasFieldnames, 'ListSize', [300 600],...
                    'Name','Select the Y locations to be used',...
                    'PromptString','Select the Y locations to be used','CancelString','Cancel',...
                    'SelectionMode','single');
                if ok ~= 0
                    EditedYLocationMeasurementName = char(EditedMeasFieldnames(Selection));
                    YLocationMeasurementName = ['dMC', EditedYLocationMeasurementName];
                    %%% Prompts the user to choose a sample number to be displayed.
                    Answer = inputdlg({'Which sample number do you want to display?'},'Choose sample number',1,{'1'});
                    if isempty(Answer)
                        cd(CurrentDirectory);
                        return
                    end
                    SampleNumber = str2num(Answer{1});
                    TotalNumberImageSets = length(handles.(MeasurementToExtract));
                    if SampleNumber > TotalNumberImageSets
                        error('The number you entered exceeds the number of samples in the file.  You entered ', num2str(SampleNumber), ' but there are only ', num2str(TotalNumberImageSets), ' in the file.')
                        cd(CurrentDirectory);
                        return
                    end
                    %%% Looks up the corresponding image file name.
                    PotentialImageNames = Fieldnames(strncmp(Fieldnames,'dOTFilename',11)==1);
                    %%% Error detection.
                    if isempty(PotentialImageNames)
                        errordlg('CellProfiler was not able to look up the image file names used to create these measurements to help you choose the correct image on which to display the results. You may continue, but you are on your own to choose the correct image file.')
                    end
                    %%% Removes the 'dOT' prefix from each name for display purposes.
                    for Number = 1:length(PotentialImageNames)
                        EditedPotentialImageNames{Number} = PotentialImageNames{Number}(4:end);
                    end
                    %%% Allows the user to select a filename from the list.
                    [Selection, ok] = listdlg('ListString',EditedPotentialImageNames, 'ListSize', [300 600],...
                        'Name','Choose the image whose filename you want to display',...
                        'PromptString','Choose the image whose filename you want to display','CancelString','Cancel',...
                        'SelectionMode','single');
                    if ok ~= 0
                        EditedSelectedImageName = char(EditedPotentialImageNames(Selection));
                        SelectedImageName = ['dOT', EditedSelectedImageName];
                        ImageFileName = handles.(SelectedImageName){SampleNumber};
                        %%% Prompts the user with the image file name.
                        h = msgbox(['Browse to find the image called ', ImageFileName,'.']);
                        %%% Opens a user interface window which retrieves a file name and path 
                        %%% name for the image to be displayed.
                        [FileName,PathName] = uigetfile('*.*','Select the image to view');
                        delete(h)
                        %%% If the user presses "Cancel", the FileName will = 0 and nothing will
                        %%% happen.
                        if FileName == 0
                            cd(CurrentDirectory);
                            return
                        else
                            %%% Opens and displays the image, with pixval shown.
                            ImageToDisplay = im2double(imread([PathName,'/',FileName]));
                            %%% Allows underscores to be displayed properly.
                            ImageFileName = strrep(ImageFileName,'_','\_');
                            FigureHandle = figure; imagesc(ImageToDisplay), colormap(gray), title([EditedMeasurementToExtract, ' on ', ImageFileName])
                            %%% Extracts the XY locations and the measurement values.
                            global StringListOfMeasurements
                            ListOfMeasurements = handles.(MeasurementToExtract){SampleNumber};
                            StringListOfMeasurements = cellstr(num2str(ListOfMeasurements));
                            Xlocations(:,FigureHandle) = handles.(XLocationMeasurementName){SampleNumber};
                            Ylocations(:,FigureHandle) = handles.(YLocationMeasurementName){SampleNumber};
                            %%% A button is created in the display window which
                            %%% allows altering the properties of the text.
                            StdUnit = 'point';
                            StdColor = get(0,'DefaultUIcontrolBackgroundColor');
                            PointsPerPixel = 72/get(0,'ScreenPixelsPerInch');                            
                            DisplayButtonCallback1 = 'global TextHandles, FigureHandle = gcf; CurrentTextHandles = TextHandles{FigureHandle}; propedit(CurrentTextHandles,''v6''); drawnow, clear TextHandles';
                            DisplayButtonHandle1 = uicontrol('Parent',FigureHandle, ...
                                'Unit',StdUnit, ...
                                'BackgroundColor',StdColor, ...
                                'CallBack',DisplayButtonCallback1, ...
                                'Position',PointsPerPixel*[2 2 90 22], ...
                                'Units','Normalized',...
                                'String','Text Properties', ...
                                'Style','pushbutton');
                            DisplayButtonCallback2 = 'global TextHandles StringListOfMeasurements, FigureHandle = gcf; NumberOfDecimals = inputdlg(''Enter the number of decimal places to display'',''Enter the number of decimal places'',1,{''0''}); CurrentTextHandles = TextHandles{FigureHandle}; NumberValues = str2num(cell2mat(StringListOfMeasurements)); Command = [''%.'',num2str(NumberOfDecimals{1}),''f'']; NewNumberValues = num2str(NumberValues,Command); CellNumberValues = cellstr(NewNumberValues); PropName(1) = {''string''}; set(CurrentTextHandles,PropName, CellNumberValues); drawnow, clear TextHandles StringListOfMeasurements';
                            DisplayButtonHandle2 = uicontrol('Parent',FigureHandle, ...
                                'Unit',StdUnit, ...
                                'BackgroundColor',StdColor, ...
                                'CallBack',DisplayButtonCallback2, ...
                                'Position',PointsPerPixel*[100 2 135 22], ...
                                'Units','Normalized',...
                                'String','Fewer significant digits', ...
                                'Style','pushbutton');
                            DisplayButtonCallback3 = 'global TextHandles StringListOfMeasurements, FigureHandle = gcf; CurrentTextHandles = TextHandles{FigureHandle}; PropName(1) = {''string''}; set(CurrentTextHandles,PropName, StringListOfMeasurements); drawnow, clear TextHandles StringListOfMeasurements';
                            DisplayButtonHandle3 = uicontrol('Parent',FigureHandle, ...
                                'Unit',StdUnit, ...
                                'BackgroundColor',StdColor, ...
                                'CallBack',DisplayButtonCallback3, ...
                                'Position',PointsPerPixel*[240 2 135 22], ...
                                'Units','Normalized',...
                                'String','Restore labels', ...
                                'Style','pushbutton');
                            %%% Overlays the values in the proper location in the
                            %%% image.
                            global TextHandles
                            TextHandles{FigureHandle} = text(Xlocations(:,FigureHandle) , Ylocations(:,FigureHandle) , StringListOfMeasurements,...
                                'HorizontalAlignment','center', 'color', 'white');
                            %%% Puts the menu and tool bar in the figure window.
                            set(FigureHandle,'toolbar', 'figure')
                        end
                    end
                end    
            end
        end
    end
end
cd(CurrentDirectory);

%%%%%%%%%%%%%%%%%

% --- Executes on button press in AnalyzeAllImagesButton.
function AnalyzeAllImagesButton_Callback(hObject, eventdata, handles)
CurrentDirectory = cd;
%%% Checks whether any algorithms are loaded.
sum = 0;
for i = 1:handles.numAlgorithms;
    sum = sum + isfield(handles,['Valgorithmname' TwoDigitString(i)]);
end
if sum == 0, errordlg('You do not have any analysis modules loaded')
else 
    %%% Checks whether an output file name has been specified. 
    if isfield(handles, 'Voutputfilename') == 0
        errordlg('You have not entered an output file name in Step 2.')
    elseif isempty(handles.Voutputfilename)
        errordlg('You have not entered an output file name in Step 2.')
    else
        %%% Retrieves the chosen directory.
        pathname = handles.Vpathname;
        %%% If the directory exists, change to that directory.
        DirDoesNotExist = 0;
        try cd(pathname);
        catch DirDoesNotExist = 1;
        end
        
        if DirDoesNotExist == 1
            errordlg('The chosen directory does not exist')
        else
            
            %%% Checks whether the specified output file name will overwrite an
            %%% existing file. 
            OutputFileOverwrite = exist([cd,'/',handles.Voutputfilename]);
            if OutputFileOverwrite ~= 0
                errordlg('An output file with the name you entered in Step 2 already exists. Overwriting is not allowed, so please enter a new filename.')
            else
                
                %%% Retrieves the list of image file names in the chosen directory and
                %%% stores them in the handles structure, using the function
                %%% RetrieveImageFileNames.  This should already have been done when the
                %%% directory was chosen, but in case some files were moved or changed in
                %%% the meantime, this will refresh the list.
                handles = RetrieveImageFileNames(handles,pathname);
                guidata(hObject, handles);
                if isempty(handles.Vfilenames)
                    set(handles.ListBox,'String','No image files recognized',...
                        'Value',1)
                else
                    %%% Loads these image names into the ListBox.
                    set(handles.ListBox,'String',handles.Vfilenames,...
                        'Value',1)
                end
                %%% Update the handles structure. Not sure if it's necessary here.
                guidata(gcbo, handles);
                %%% Disables a lot of the buttons on the GUI so that the program doesn't
                %%% get messed up.  The View buttons and Help buttons are left enabled.
                set(handles.BrowseToLoad,'enable','off')
                set(handles.PathToLoadEditBox,'enable','inactive','foregroundcolor',[0.7,0.7,0.7])
                set(handles.LoadSampleInfo,'enable','off')
                set(handles.ClearSampleInfo,'enable','off')
                set(handles.ViewSampleInfo,'enable','off')
                set(handles.OutputFileName,'enable','inactive','foregroundcolor',[0.7,0.7,0.7])
                set(handles.SetPreferencesButton,'enable','off')
                set(handles.PixelSizeEditBox,'enable','inactive','foregroundcolor',[0.7,0.7,0.7])
                set(handles.LoadSettingsFromFileButton,'enable','off')
                set(handles.SaveCurrentSettingsButton,'enable','off')
                for i=1:handles.numAlgorithms;
                    set(handles.(['LoadAlgorithm' TwoDigitString(i)]),'visible','off');
                    set(handles.(['ClearAlgorithm' TwoDigitString(i)]),'visible','off');
                    set(handles.(['ViewAlgorithm' TwoDigitString(i)]),'visible','off');
                end
                % FIXME: This should loop just over the number of actual variables in the display.
                for VariableNumber=1:handles.MaxVariables;
                    set(handles.(['VariableBox' TwoDigitString(VariableNumber)]),'enable','inactive','foregroundcolor',[0.7,0.7,0.7]);
                end
                set(handles.SelectTestImageBrowseButton,'enable','off')
                set(handles.ListBox,'enable','off')
                set(handles.TestImageName,'enable','inactive','foregroundcolor',[0.7,0.7,0.7])
                set(handles.TechnicalDiagnosisButton,'enable','off')
                set(handles.AnalyzeTestImageButton,'enable','off')
                set(handles.SaveImageAsButton,'enable','off')
                set(handles.ShowImageButton,'enable','off')
                set(handles.ShowPixelDataButton,'enable','off')
                set(handles.CloseAllFigureWindowsButton,'enable','off')
                set(handles.AnalyzeAllImagesButton,'enable','off')
                set(handles.AnalyzeAllImagesClusterButton,'enable','off')
                set(handles.ExportDataButton,'enable','off')
                set(handles.ExportCellByCellButton,'enable','off')                
                set(handles.HistogramButton,'enable','off')
                set(handles.NormalizationButton,'enable','off')
                set(handles.DisplayDataOnImageButton,'enable','off')
                
                %%% The following code prevents the warning message in the Matlab
                %%% main window: "Warning: Image is too big to fit on screen":
                %%% This warning appears due to the truesize command which 
                %%% rescales an image so that it fits on the screen.  Truesize is often
                %%% called by imshow.
                try
                    iptsetpref('TruesizeWarning','off')
                catch error('Apparently, you do not have the Image Processing Toolbox installed or licensed on this computer. This is likely to be necessary for running most of the modules. If you know you do not need the image processing toolbox for the modules you want to run, you might want to try removing the "iptsetpref" lines in the main CellProfiler program and trying again.')
                end
                %%% In the following code, the Timer window and 
                %%% timer_text is created.  Each time around the loop, 
                %%% the text will be updated using the string property.
                
                %%% Obtains the screen size.
                ScreenSize = get(0,'ScreenSize');
                ScreenWidth = ScreenSize(3);
                ScreenHeight = ScreenSize(4);
                
                %%% Makes the timer_handle variable global for future use.
                global timer_handle
                %%% Determines where to place the timer window: We want it below the image
                %%% windows, which means at about 720 pixels from the top of the screen,
                %%% but in case the screen doesn't have that many pixels, we don't want it
                %%% to be below zero.
                PotentialBottom = [0, (ScreenHeight-720)];
                BottomOfTimer = max(PotentialBottom);
                %%% Creates the Timer window.
                timer_handle = figure('name','Timer','position',[0 BottomOfTimer 495 120],...
                    'menubar','none','NumberTitle','off','IntegerHandle','off', 'HandleVisibility', 'off', ...
                    'color',[0.7,0.7,0.9]);
                %%% Sets initial text to be displayed in the text box within the timer window.
                timertext = 'First image set is being processed';
                %%% The text_handle, CancelAfterImageSetButton_handle, and PauseButton_handle 
                %%% variables are made global so that the Cancel and Pause
                %%% button functions are able to find them when the time comes to set the
                %%% string to "Canceling in progress" or to disable the Cancel and Pause
                %%% buttons when they have already been pressed.
                global text_handle CancelAfterImageSetButton_handle CancelAfterModuleButton_handle PauseButton_handle
                %%% Creates the text box within the timer window which will display the
                %%% timer text.  
                text_handle = uicontrol(timer_handle,'string',timertext,'style','text',...
                    'parent',timer_handle,'position', [0 40 494 64],'FontName','Times',...
                    'FontSize',14,'FontWeight','bold','BackgroundColor',[0.7,0.7,0.9]);
                %%% Saves text handle to the handles structure.
                handles.timertexthandle = text_handle;
                %%% Sets the functions to be called when the Cancel and Pause buttons
                %%% within the Timer window are pressed.
                PauseButtonFunction = 'h = msgbox(''Image processing is paused without causing any damage. Processing will restart when you close the Pause window or click OK.''); waitfor(h); clear h;';
                CancelAfterImageSetButtonFunction = 'deleteme = questdlg(''Paused. Are you sure you want to cancel after this image set? Processing will continue on the current image set, the data up to and including the current image set will be saved in the output file, and then the analysis will be canceled.'', ''Confirm close'',''Yes'',''No'',''Yes''); switch deleteme; case ''Yes''; global text_handle CancelAfterModuleButton_handle CancelAfterImageSetButton_handle; set(CancelAfterImageSetButton_handle,''enable'',''off''); set(text_handle,''string'',''Canceling in progress; Waiting for the processing of current image set to be complete. You can press the Cancel after module button to cancel more quickly, but data relating to the current image set will not be saved in the output file.''); clear Cancel*; clear text_handle deleteme; case ''No''; return; end;';
                CancelAfterModuleButtonFunction = 'deleteme = questdlg(''Paused. Are you sure you want to cancel after this module? Processing will continue until the current image analysis module is completed, to avoid corrupting the current settings of CellProfiler. Data up to the *previous* image set are saved in the output file and processing is canceled.'', ''Confirm close'',''Yes'',''No'',''Yes''); switch deleteme; case ''Yes''; global text_handle CancelAfterModuleButton_handle CancelAfterImageSetButton_handle; set(CancelAfterImageSetButton_handle,''enable'',''off''); set(CancelAfterModuleButton_handle,''enable'',''off''); set(text_handle,''string'',''Immediate canceling in progress; Waiting for the processing of current module to be complete in order to avoid corrupting the current CellProfiler settings.''); clear Cancel*; clear text_handle deleteme; case ''No''; return; end;';
                CancelNowCloseButtonFunction = 'global MainGUIhandle; deleteme = questdlg(''Paused. Are you sure you want to cancel immediately and close CellProfiler? The CellProfiler program will close, losing your current settings. The data up to the *previous* image set will be saved in the output file, but the current image set data will be stored incomplete in the output file, which might be confusing when using the output file.'', ''Confirm close'',''Yes'',''No'',''Yes''); helpdlg(''The CellProfiler program should have closed itself. Important: Go to the command line of Matlab and press Control-C to stop processes in progress. Then type clear and press the enter key at the command line.  Figure windows will not close properly: to close them, type delete(N) at the command line of Matlab, where N is the figure number. The data up to the *previous* image set will be saved in the output file, but the current image set data will be stored incomplete in the output file, which might be confusing when using the output file.''), switch deleteme; case ''Yes''; delete(MainGUIhandle); case ''No''; return; end; clear MainGUIhandle, clear deleteme';
                HelpButtonFunction = 'msgbox(''Pause button: The current processing is immediately suspended without causing any damage. Processing restarts when you close the Pause window or click OK. Cancel after image set: Processing will continue on the current image set, the data up to and including the current image set will be saved in the output file, and then the analysis will be canceled.  Cancel after module: Processing will continue until the current image analysis module is completed, to avoid corrupting the current settings of CellProfiler. Data up to the *previous* image set are saved in the output file and processing is canceled. Cancel now & close CellProfiler: CellProfiler will immediately close itself. The data up to the *previous* image set will be saved in the output file, but the current image set data will be stored incomplete in the output file, which might be confusing when using the output file.'')';
                
                %%% Creates the Cancel and Pause buttons.
                PauseButton_handle = uicontrol('Style', 'pushbutton', ...
                    'String', 'Pause', 'Position', [5 10 40 30], ...
                    'Callback', PauseButtonFunction, 'parent',timer_handle, 'BackgroundColor',[0.7,0.7,0.9]);
                CancelAfterImageSetButton_handle = uicontrol('Style', 'pushbutton', ...
                    'String', 'Cancel after image set', 'Position', [50 10 120 30], ...
                    'Callback', CancelAfterImageSetButtonFunction, 'parent',timer_handle, 'BackgroundColor',[0.7,0.7,0.9]);
                CancelAfterModuleButton_handle = uicontrol('Style', 'pushbutton', ...
                    'String', 'Cancel after module', 'Position', [175 10 115 30], ...
                    'Callback', CancelAfterModuleButtonFunction, 'parent',timer_handle, 'BackgroundColor',[0.7,0.7,0.9]);
                CancelNowCloseButton_handle = uicontrol('Style', 'pushbutton', ...
                    'String', 'Cancel now & close CellProfiler', 'Position', [295 10 160 30], ...
                    'Callback', CancelNowCloseButtonFunction, 'parent',timer_handle, 'BackgroundColor',[0.7,0.7,0.9]);
                HelpButton_handle = uicontrol('Style', 'pushbutton', ...
                    'String', '?', 'Position', [460 10 15 30], ...
                    'Callback', HelpButtonFunction, 'parent',timer_handle, 'BackgroundColor',[0.7,0.7,0.9]);
                %%% The timertext string is read by the analyze all images button's callback
                %%% at the end of each time around the loop (i.e. at the end of each image
                %%% set).  If it notices that the string says "Cancel...", it breaks out of
                %%% the loop and finishes up.
                
                %%% Update the handles structure. Not sure if it's necessary here.
                guidata(gcbo, handles);
                %%% Sets the timer window to show a warning box before allowing it to be
                %%% closed.
                CloseFunction = 'deleteme = questdlg(''DO NOT CLOSE the Timer window while image processing is in progress!! Are you sure you want to close the timer?'', ''Confirm close'',''Yes'',''No'',''Yes''); switch deleteme; case ''Yes''; global timer_handle; delete(timer_handle); case ''No''; return; end; clear deleteme; clear timer_handle;';
                set(timer_handle,'CloseRequestFcn',CloseFunction);
                %%% Note: The size of the text object that fits inside the timer window is
                %%% officially 1 pixel smaller than the size of the timer window itself. 
                %%%  There is, however, a ~20 pixel gap at the top of the timer window: I
                %%%  think this is because there is space allotted for a menu bar which is
                %%%  not utilized in this window.  However, when I increased the size of
                %%%  the text object relative to the timer window, I ended up with the
                %%%  program crashing and creating segmentation faults, so I gave up and
                %%%  decided to live with that gap being present.  I am not absolutely sure
                %%%  that this was causing the problems, though.
                
                %%% If an algorithm is chosen in this slot, assign it an output figure
                %%% window and write the figure window number to the handles structure so
                %%% that the algorithms know where to write to.  Each algorithm should
                %%% resize the figure window appropriately.  The closing function of the
                %%% figure window is set to wait until an image set is done processing
                %%% before closing the window, to avoid unexpected results.  The handles
                %%% for each algorithm's figure must be made global so the closing function
                %%% can find the handles.

                
                for i=1:handles.numAlgorithms;
                    if isfield(handles,strcat('Valgorithmname',TwoDigitString(i))) == 1
                        set(handles.(['FigureDisplay' TwoDigitString(i)]),'visible','on')
                        set(handles.(['ViewAlgorithm' TwoDigitString(i)]),'visible','on')
                        handles.(['figurealgorithm' TwoDigitString(i)]) = ...
                            figure('name',[handles.(['Valgorithmname' TwoDigitString(i)]), ' Display'], 'Position',[(ScreenWidth*((i-1)/12)) (ScreenHeight-522) 560 442],'color',[0.7,0.7,0.7]);
                        global HandleFigureDisplay
                        HandleFigureDisplay(i) = handles.(['FigureDisplay' TwoDigitString(i)]);
                        ClosingFunction = ['global HandleFigureDisplay; set(HandleFigureDisplay(' int2str(i) '), ''string'', ''Closing...''); drawnow; clear HandleFigureDisplay'];
                        %%% Sets the closing function of the figure window to be the line above.
                        set(handles.(['figurealgorithm' TwoDigitString(i)]),'CloseRequestFcn',ClosingFunction);
                    end
                end
                
                %%% For the first time through, the number of image sets
                %%% will not yet have been determined.  So, the Number of
                %%% image sets is set temporarily.
                handles.Vnumberimagesets = 1;
                handles.setbeinganalyzed = 1;
                %%% Marks the time that analysis was begun.
                handles.Vtimestarted = datestr(now);
                %%% Start the timer.
                tic
                %%% Update the handles structure.
                guidata(gcbo, handles);
                
                %%%%%%
                %%% Begin loop (going through all the image sets).
                %%%%%%
                
                %%% variable to allow breaking out of nested loops.
                break_outer_loop = 0;
                
                while handles.setbeinganalyzed <= handles.Vnumberimagesets
                    setbeinganalyzed = handles.setbeinganalyzed;
                    
                    %%% Loop through normally if this is the first
                    %%% image set or this evaluation is being run
                    %%% sequentially (i.e., not in parallel)
                    if ((setbeinganalyzed == 1) | (~ isfield(handles, 'parallel_machines')))
                      % clear the Pending flag if we're using parallel machines
                      if isfield(handles, 'parallel_machines')
                        Pending(handles.parallel_machines + 1) = 0;
                      end

                      for SlotNumber = 1:handles.numAlgorithms,
                          %%% If an algorithm is not chosen in this slot, continue on to the next.
                          AlgNumberAsString = TwoDigitString(SlotNumber);
                          AlgName = ['Valgorithmname' AlgNumberAsString];
                          if isfield(handles,AlgName) == 0
                          else 
                              %%% Saves the current algorithm number in the handles structure.
                              handles.currentalgorithm = AlgNumberAsString;
                              %%% The try/catch/end set catches any errors that occur during the
                              %%% running of algorithm 1, notifies the user, breaks out of the image
                              %%% analysis loop, and completes the refreshing process.
                              try
                                  %%% Runs the appropriate algorithm, with the handles structure as an
                                  %%% input argument and as the output argument.
                                  eval(['handles = Alg',handles.(AlgName),'(handles);'])
                              catch
                                  if exist(['Alg',handles.(AlgName),'.m']) ~= 2,
                                      errordlg(['Image processing was canceled because the image analysis module named ', (['Alg',handles.(AlgName),'.m']), ' was not found. Is it stored in the folder with the other modules?  Has its name changed?'])
                                  else
                                      %%% Runs the errorfunction function that catches errors and
                                      %%% describes to the user what to do.
                                      errorfunction(AlgNumberAsString)
                                  end
                                  %%% Causes break out of the image analysis loop (see below)
                                  break_outer_loop = 1;
                                  break;
                              end % Goes with try/catch.
                          end
                          %%% If an immediate "cancel" signal is waiting, break and go to the "end" that goes
                          %%% with the "while" loop.  The output file is not saved since it would only
                          %%% be partially complete.
                          CancelWaiting = get(handles.timertexthandle,'string');
                          if strncmp(CancelWaiting,'Immedi',6) == 1
                                  break_outer_loop = 1;
                              break
                          end
                          
                        %%% In a similar manner to determining whether a
                        %%% cancel request is pending (see below),
                        %%% determine whether a figure closing/opening
                        %%% request is pending.
                        ThisFigDisplay = handles.(['FigureDisplay' AlgNumberAsString]);
                        if strcmp(get(ThisFigDisplay, 'string'), 'Closing...') == 1
                          Thisfigurealgorithm = handles.(['figurealgorithm' AlgNumberAsString]);
                          delete(Thisfigurealgorithm)
                          %%% Set the button's text to "Open Figure".
                          set(ThisFigDisplay, 'string', 'Open Figure')
                          %%% Refreshes the Main GUI window, or else "Open Figure" is not
                          %%% displayed.
                          drawnow
                        elseif strcmp(get(ThisFigDisplay, 'string'), 'Opening...') == 1
                          Thisfigurealgorithm = handles.(['figurealgorithm' AlgNumberAsString]);
                          figure(Thisfigurealgorithm)
                          set(Thisfigurealgorithm, 'name',[handles.(AlgName), ' Display'])
                          %%% Sets the closing function of the window appropriately. (See way
                          %%% above where 'ClosingFunction's are defined).
                          set(Thisfigurealgorithm,'CloseRequestFcn',eval(['ClosingFunction' AlgNumberAsString]));
                          %%% Set the button's text to "Close Figure".
                          set(ThisFigDisplay, 'string', 'Close Figure')
                          %%% Refreshes the Main GUI window, or else "Close Figure" is not
                          %%% displayed.
                          drawnow
                        end
                      end %%% ends loop over slot number
                      if (break_outer_loop),
                        break;  %%% this break is out of the outer loop of image analysis
                      end

                      %%% Get a list of the measurement fields (after the first pass has run through
                      %%% all the modules)
                      Fields = fieldnames(handles);
                      mFields = (strncmp(Fields,'dM',2) | strncmp(Fields,'dOTFilename',11));
                      MeasurementFields = Fields(mFields);
                      
                      % If we are using parallel machines, copy the handles structure to them.
                      if (isfield(handles, 'parallel_machines')),
                        handles_culled = handles;
                        deleteFields = strncmp(Fields,'dOT',2);
                        keepFields = strncmp(Fields,'dOTFileList',11) | ...
                            strncmp(Fields,'dOTPathName',11) | strncmp(Fields,'dOTFilename',11) | ...
                            strncmp(Fields,'dOTIllumImage',13) | strncmp(Fields,'dOTIntensityToShift',19) | ...
                            strncmp(Fields,'dOTTimeElapsed',14);
                        handles_culled = rmfield(handles_culled, Fields(deleteFields & (~keepFields)));
                        pnet_remote(handles.parallel_machines, 'PUT', 'handles', handles_culled);
                      end
                      
                    else %%% goes with the check for first-time or parallel machines
                      NumParallelMachines = length(handles.parallel_machines);
                      
                      CurrMachine = handles.parallel_machines(1 + mod(setbeinganalyzed, NumParallelMachines));
                      % Check if we need to get something back from this machine, and if so, fetch
                      % it.  It would be nice if we could just fetch back the results once at the
                      % end (and allow the remote machine to fill up the handles structure with
                      % all of its results), but since the pnet_remote('EVAL') function is
                      % non-blocking, I don't think that will work.  I guess we could ship off the
                      % entire set to the remote machine, but this way allows for more
                      % fine-grained error-catching.
                      if Pending(CurrMachine+1),
                        Pending(CurrMachine+1) = 0;
                        % Check the status of the remote evaluation
                        
                        RemoteResults = pnet_remote(CurrMachine, 'GET', 'handles_results');
                        RemoteError = RemoteResults.CellProfilerError;
                        RemoteSet = RemoteResults.setbeinganalyzed;

                        if isstr(RemoteError),
                          %%% Remote machine returned a string.  It must be an error.
                          errordlg(['Error in parallel evaluation on set ' int2str(RemoteSet) ' : ' RemoteError]);
                          break;  %%% breaks out of image analysis loop
                        end
                        


                        %%% Loop over measurement fields, merging them in
                        for FieldIndex = 1:length(MeasurementFields),
                          handles.(cell2mat(MeasurementFields(FieldIndex))){RemoteSet} = ...
                              RemoteResults.(cell2mat(MeasurementFields(FieldIndex))){RemoteSet};
                        end
                      end
                      pnet_remote(CurrMachine, 'WAITNOBUSY');
                      pnet_remote(CurrMachine, 'PUT', 'setbeinganalyzed', setbeinganalyzed);
                      pnet_remote(CurrMachine, 'WAITNOBUSY');
                      pnet_remote(CurrMachine, 'EVAL', 'cellprofiler_one_loop');
                      Pending(CurrMachine+1) = 1;


                      %%% If an immediate "cancel" signal is waiting, break and go to the "end" that goes
                      %%% with the "while" loop.  The output file is not saved since it would only
                      %%% be partially complete.
                      CancelWaiting = get(handles.timertexthandle,'string');
                      if strncmp(CancelWaiting,'Immedi',6) == 1
                        break
                      end

                      %%% TODO: need to check for figure closing requests here.

                    end
                    
                    %%% Reads the text in the timer window to check whether it is a cancel
                    %%% signal, since the text will be overwritten in the calculations for the
                    %%% timer.  The timer calculations have to be done before canceling because
                    %%% the time elapsed must be stored in the handles structure and therefore
                    %%% in the output file.
                    CancelWaiting = get(handles.timertexthandle,'string');
                    
                    %%% Make calculations for the Timer window.
                    time_elapsed = num2str(toc);
                    timer_elapsed_text =  ['Time elapsed (seconds) = ',time_elapsed];
                    number_analyzed = ['Number of image sets analyzed = ',...
                            num2str(setbeinganalyzed), ' of ', num2str(handles.Vnumberimagesets)];
                    if setbeinganalyzed ~=0
                        time_per_set = ['Time per image set (seconds) = ', ...
                                num2str(toc/setbeinganalyzed)];
                    else time_per_set = 'Time per image set (seconds) = none completed'; 
                    end
                    timertext = [{timer_elapsed_text; number_analyzed; time_per_set}];
                    %%% Display calculations in 
                    %%% the "Timer" window by changing the string property.
                    set(text_handle,'string',timertext)
                    drawnow    
                    %%% Save the time elapsed so far in the handles structure.
                    %%% Check first to see that the set being analyzed is not zero, or else an
                    %%% error will be produced when trying to do this.
                    if setbeinganalyzed ~= 0
                        handles.dOTTimeElapsed{setbeinganalyzed} = toc;
                        guidata(gcbo, handles)
                    end
                    %%% Save all data that is in the handles structure to the output file 
                    %%% name specified by the user.
                    eval(['save ',handles.Voutputfilename, ' handles;'])
                    %%% The setbeinganalyzed is increased by one and stored in the handles structure.
                    setbeinganalyzed = setbeinganalyzed + 1;
                    handles.setbeinganalyzed = setbeinganalyzed;
                    guidata(gcbo, handles)
                    %%% If a "cancel" signal is waiting, break and go to the "end" that goes
                    %%% with the "while" loop.
                    if strncmp(CancelWaiting,'Cancel',6) == 1
                        break
                    end
                end %%% This "end" goes with the "while" loop (going through the image sets).
                
                %%% If we were using parallel machines, and there are still results pending on them,
                %%% we need to fetch it back and merge it in.
                if isfield(handles, 'parallel_machines'),
                  NumParallelMachines = length(handles.parallel_machines);
                  
                  for CurrMachine = handles.parallel_machines,
                    if Pending(CurrMachine+1),
                      Pending(CurrMachine+1) = 0;

                      RemoteResults = pnet_remote(CurrMachine, 'GET', 'handles_results');
                      RemoteError = RemoteResults.CellProfilerError;
                      RemoteSet = RemoteResults.setbeinganalyzed;

                      if isstr(RemoteError),
                        %%% Remote machine returned a string.  It must be an error.
                        errordlg(['Error in parallel evaluation on set ' int2str(RemoteSet) ' : ' RemoteError]);
                        break;  %%% breaks out of image analysis loop
                      end

                      %%% Loop over measurement fields, merging them in
                      for FieldIndex = 1:length(MeasurementFields),
                        handles.(cell2mat(MeasurementFields(FieldIndex))){RemoteSet} = ...
                            RemoteResults.(cell2mat(MeasurementFields(FieldIndex))){RemoteSet};
                      end
                    end

                    % save results
                    eval(['save ',handles.Voutputfilename, ' handles;'])
                  end
                end

                %%% After all the image sets have been processed, the following checks to
                %%% be sure that the data loaded as "Sample Info" has the proper number of
                %%% entries.  If not, the data is removed from the handles structure so
                %%% that the extract data button will work later on.
                
                %%% Create a vector that contains the length of each headings field.  In other
                %%% words, determine the number of entries for each column of Sample Info.
                
                if isfield(handles,'headings') == 1
                    HeadingsNames = [handles.headings];
                    for i = 1:length(HeadingsNames);
                        fieldname = char(HeadingsNames{i});
                        Lengths(i) = length(handles.(fieldname));
                    end   
                    %%% Create a logical array that indicates which headings do not have the
                    %%% same number of entries as the number of image sets analyzed.
                    IsWrongNumber = Lengths ~= setbeinganalyzed - 1;
                    %%% Determine which heading names to remove.
                    HeadingsToBeRemoved = HeadingsNames(IsWrongNumber);
                    %%% Remove headings names from handles.headings and remove the sample
                    %%% info from the field named after the heading.
                    if isempty(HeadingsToBeRemoved) == 0
                        handles = rmfield(handles, HeadingsToBeRemoved);
                        handles.headings(IsWrongNumber == 1) = [];
                        %%% Tell the user that fields have been removed.
                        HeadingsErrorMessage(1) = {'Some of the sample info you'};
                        HeadingsErrorMessage(2) = {'loaded does not have the'};
                        HeadingsErrorMessage(3) = {'same number of entries as'};
                        HeadingsErrorMessage(4) = {'the number of image sets'};
                        HeadingsErrorMessage(5) = {['analyzed (which is ', num2str(setbeinganalyzed), ').']};
                        HeadingsErrorMessage(6) = {'This mismatch will prevent'};
                        HeadingsErrorMessage(7) = {'the extract data button from'};
                        HeadingsErrorMessage(8) = { 'working, so the following'};
                        HeadingsErrorMessage(9) = {'columns of sample info have'};
                        HeadingsErrorMessage(10) = {'been deleted from the output'};
                        HeadingsErrorMessage(11) = {'file. Click OK to continue.'};
                        HeadingsErrorMessage = cellstr(HeadingsErrorMessage);
                        listdlg('ListString', HeadingsToBeRemoved, 'PromptString', HeadingsErrorMessage, 'CancelString', 'OK');
                        %%% If no sample info remains, the field "headings" is removed
                        %%% so that when the user clicks Clear or View, the proper error
                        %%% message is generated, telling the user that no sample info has been
                        %%% loaded.
                        if isempty(handles.headings) ==1
                            handles = rmfield(handles, 'headings');
                        end
                        %%% Save all data that is in the handles structure to the output file 
                        %%% name specified by the user.
                        eval(['save ',handles.Voutputfilename, ' handles;'])
                    end % This end goes with the "isempty" line.
                end % This end goes with the 'isfield' line.    
                %%% Update the handles structure.
                guidata(gcbo, handles)
                
                %%% Calculate total time elapsed and display Complete in the Timer window.
                total_time_elapsed = ['Total time elapsed (seconds) = ',num2str(toc)];
                number_analyzed = ['Number of image sets analyzed = ',...
                        num2str(setbeinganalyzed - 1)];
                if setbeinganalyzed ~=1
                    time_per_set = ['Time per image set (seconds) = ', ...
                            num2str(toc/(setbeinganalyzed - 1))];
                else time_per_set = 'Time per image set (seconds) = none completed'; 
                end
                text_handle = uicontrol(timer_handle,'string',timertext,'style','text',...
                    'parent',timer_handle,'position', [0 40 494 64],'FontName','Times',... 
                    'FontSize',14,'FontWeight','bold','backgroundcolor',[0.7,0.7,0.9]);
                timertext = [{'IMAGE PROCESSING IS COMPLETE!';total_time_elapsed; ...
                            number_analyzed; time_per_set}];
                set(text_handle,'string',timertext)
                set(timer_handle,'CloseRequestFcn','closereq');
                
                %%% Re-enable/disable appropriate buttons.
                set(handles.BrowseToLoad,'enable','on')
                set(handles.PathToLoadEditBox,'enable','on','foregroundcolor','black')
                set(handles.LoadSampleInfo,'enable','on')
                set(handles.ClearSampleInfo,'enable','on')
                set(handles.ViewSampleInfo,'enable','on')
                set(handles.OutputFileName,'enable','on','foregroundcolor','black')
                set(handles.SetPreferencesButton,'enable','on')
                set(handles.PixelSizeEditBox,'enable','on','foregroundcolor','black')
                set(handles.LoadSettingsFromFileButton,'enable','on')
                set(handles.SaveCurrentSettingsButton,'enable','on')
                for AlgorithmNumber=1:handles.numAlgorithms;
                    set(handles.(['LoadAlgorithm' TwoDigitString(AlgorithmNumber)]),'visible','on');
                    set(handles.(['ClearAlgorithm' TwoDigitString(AlgorithmNumber)]),'visible','on');
                    set(handles.(['FigureDisplay' TwoDigitString(AlgorithmNumber)]),'visible','off');
                    set(handles.(['FigureDisplay' TwoDigitString(AlgorithmNumber)]),'string', 'Close Figure');
                    set(handles.(['ViewAlgorithm' TwoDigitString(AlgorithmNumber)]),'visible','on');
                    for VariableNumber = 1:handles.numVariables(AlgorithmNumber);
                        set(handles.(['VariableBox' TwoDigitString(VariableNumber)]),'enable','on','foregroundcolor','black');
                    end
                end
                set(handles.SelectTestImageBrowseButton,'enable','on')
                set(handles.ListBox,'enable','on')
                set(handles.TestImageName,'enable','on','foregroundcolor','black')
                set(handles.TechnicalDiagnosisButton,'enable','on')
                set(handles.AnalyzeTestImageButton,'enable','on')
                set(handles.SaveImageAsButton,'enable','on')
                set(handles.ShowImageButton,'enable','on')
                set(handles.ShowPixelDataButton,'enable','on')
                set(handles.CloseAllFigureWindowsButton,'enable','on')
                set(handles.AnalyzeAllImagesButton,'enable','on')
                set(handles.AnalyzeAllImagesClusterButton,'enable','on')
                set(handles.ExportDataButton,'enable','on')
                set(handles.ExportCellByCellButton,'enable','on')                
                set(handles.HistogramButton,'enable','on')
                set(handles.NormalizationButton,'enable','on')
                set(handles.DisplayDataOnImageButton,'enable','on')
                set(CancelAfterModuleButton_handle,'enable','off')
                set(CancelAfterImageSetButton_handle,'enable','off')
                set(PauseButton_handle,'enable','off')
                set(CancelNowCloseButton_handle,'enable','off')

                %%% The following code turns the warning message back on that 
                %%% I turned off when the GUI was launched. 
                %%% "Warning: Image is too big to fit on screen":
                %%% This warning appears due to the truesize command which 
                %%% rescales an image so that it fits on the screen.  Truesize is often
                %%% called by imshow.
                iptsetpref('TruesizeWarning','on')
                
                %%% Sets the figure windows' Closing Functions back to normal, if 
                %%% the figure windows are still open.  If this is not done,
                %%% after the analysis is complete, these windows cannot be closed with the
                %%% ctrl-W or "X" buttons in the window, because the closing function would
                %%% wait for the image analysis to complete a loop (which would never
                %%% happen).  Has to check to see whether the figure exists first before
                %%% setting the close request function.  That requires looking up
                %%% handles.figurealgorithm1.  Before looking that up, you have to check to
                %%% see if it exists or else an error occurs.
                
                for i=1:handles.numAlgorithms
                    AlgorithmNum = TwoDigitString(i);
                    if isfield(handles,['figurealgorithm' AlgorithmNum]) ==1
                        if any(findobj == handles.(['figurealgorithm' AlgorithmNum])) == 1;
                            properhandle = handles.(['figurealgorithm' AlgorithmNum]);
                            set(properhandle,'CloseRequestFcn','delete(gcf)');
                        end
                    end
                end
                
                %%% Clears the global variables, if they exist.
                clear text_handle HandleFigureDisplay timer_handle
                
                %%% Removes the temporary measurements and image files from the "buffer",
                %%% i.e. the handles structure.
                %%% Lists the fields that are present in the handles structure.
                Fields = fieldnames(handles);
                %%% Produces a logical array called dFields which contains a 0 in every
                %%% spot that does not begin 'dMT' and a 1 in every spot that does.
                dFields = strncmp(Fields,'dMT',3);
                %%% Produces a list of fields to remove by selecting those fields from
                %%% "Fields" that correspond to a "1" in dFields.
                FieldsToRemove = Fields(dFields);
                %%% Sets the new handles structure as the old one with those fields
                %%% removed.
                handles = rmfield(handles,FieldsToRemove);
                
                Fields = fieldnames(handles);
                dFields = strncmp(Fields,'dMC',3);
                FieldsToRemove = Fields(dFields);
                handles = rmfield(handles,FieldsToRemove);
                
                Fields = fieldnames(handles);
                dFields = strncmp(Fields,'dOT',3);
                FieldsToRemove = Fields(dFields);
                handles = rmfield(handles,FieldsToRemove);
                
                %%% Clears the output file name to prevent it from being reused.
                set(handles.OutputFileName,'string',[])
                handles = rmfield(handles,'Voutputfilename');
                guidata(gcbo, handles)
                
                %%% This "end" goes with the error-detecting "You have no analysis modules
                %%% loaded". 
            end
            %%% This "end" goes with the error-detecting "You have not specified an
            %%% output file name".
        end
        %%% This "end" goes with the error-detecting "An output file with that name
        %%% already exists."
    end
    %%% This "end" goes with the error-detecting "The chosen directory does not
    %%% exist."
end
cd(CurrentDirectory);

%%% Note: an improvement I would like to make:
%%% Currently, it is possible to use the Zoom tool in the figure windows to
%%% zoom in on any of the subplots.  However, when new image data comes
%%% into the window, the Zoom factor is reset. If the processing is fairly
%%% rapid, there isn't really time to zoom in on an image before it
%%% refreshes. It would be nice if the
%%% Zoom factor was applied to the new incoming image.  I think that this
%%% would require redefining the Zoom tool's action, which is not likely to
%%% be a simple task.

function errorfunction(CurrentAlgorithmNumber)
Error = lasterr;
%%% If an error occurred in an image analysis module, the error message
%%% should begin with "Error using ==> Alg", which will be recognized here.
if strncmp(Error,'Error using ==> Alg', 19) == 1
    ErrorExplanation = ['There was a problem running the analysis module number ',CurrentAlgorithmNumber, '.', Error];
    %%% The following are errors that may have occured within the analyze all
    %%% images callback itself.
elseif isempty(strfind(Error,'bad magic')) == 0
    ErrorExplanation = 'There was a problem running the image analysis. It seems likely that there are files in your image directory that are not images or are not the image format that you indicated. Probably the data for the image sets up to the one which generated this error are OK in the output file.';
else
    ErrorExplanation = ['There was a problem running the image analysis. Sorry, it is unclear what the problem is. It would be wise to close the entire CellProfiler program in case something strange has happened to the settings. The output file may be unreliable as well. Matlab says the error is: ', Error, ' in Algorithm', CurrentAlgorithmNumber];
end
errordlg(ErrorExplanation)

%%%%%%%%%%%%%%%%%

% --- Executes on button press in AnalyzeAllImagesClusterButton.
function AnalyzeAllImagesClusterButton_Callback(hObject, eventdata, handles)
Prompts = {'Path to CellProfiler on the remote machine(s)','Path to the images on the remote machine(s)','File containing the list of remote machine(s)'};

% set up default values for the answers
if (~ isfield(handles, 'RemoteCellProfilerPathName'))
  LocationOfGUI = which('CellProfiler');
  Slashes = findstr(LocationOfGUI, '/');
  handles.RemoteCellProfilerPathName = LocationOfGUI(1:Slashes(end));
  handles.RemoteImagePathName = handles.Vpathname;
  handles.RemoteMachineListFile = LocationOfGUI(1:Slashes(end));
end

% pop up the dialog
Defaults = {handles.RemoteCellProfilerPathName,handles.RemoteImagePathName,handles.RemoteMachineListFile};
Answers = inputdlg(Prompts,'Provide cluster information',1,Defaults,'on');

if isempty(Answers)
    return
end

% Store the answers as new defaults
handles.RemoteCellProfilerPathName = Answers{1};
handles.RemoteImagePathName = Answers{2};
handles.RemoteMachineListFile = Answers{3};
guidata(hObject, handles);

% Load the list of machines, and connect to each one
[fid, reason] = fopen(handles.RemoteMachineListFile, 'rt');
if fid == -1,
  errordlg(['CellProfiler could not open the list of remote machines (' handles.RemoteMachineListFile ').  The error message was "' reason '"']);
  return;
end

pnet_remote('closeall');
handles.parallel_machines = [];

while 1,
  RemoteMachine = fgetl(fid)
  % We should put up a dialog here with a CANCEL button.  Also need to
  % modify pnet_remote to return after a few retries, rather than just
  % giving up.
  if (~ ischar(RemoteMachine)),
    break;
  end
  if (~ isempty(RemoteMachine)),
    handles.parallel_machines(length(handles.parallel_machines)+1) = pnet_remote('connect', RemoteMachine);
  end
end

if length(handles.parallel_machines) == 0,
  errordlg(['CellProfiler could not connetct to any remote machines.  Is the list of machines an empty file (' handles.RemoteMachineListFile ')?']);
  handles = rmfield(handles, 'parallel_machines');
  guidata(hObject, handles);
  return;
end

% set up the path on the remote machines
pnet_remote(handles.parallel_machines, 'eval', ['addpath ' handles.RemoteCellProfilerPathName]);

% fake a click on the analyze images button
AnalyzeAllImagesButton_Callback(hObject, eventdata, handles);

% clear the list of parallel machines
rmfield(handles, 'parallel_machines');
guidata(hObject, handles);



%%%%%%%%%%%%%%%%%%%%%
%%% Aux Functions %%%
%%%%%%%%%%%%%%%%%%%%%

function twodigit = TwoDigitString(val)
%TwoDigitString is a function like num2str(int) but it returns a two digit
%representation of a string for our purposes.
if ((val > 99) | (val < 0)),
  error(['TwoDigitString: Can''t convert ' num2str(val) ' to a 2 digit number']);
end
twodigit = sprintf('%02d', val);


function y = trimstr( s, stripchars, leftorright )
%TRIMSTR  Strip the whitespace and the defined characters from string.
%   TRIMSTR( S, CHARS, LEFTORRIGHT ) strips the whitespace and the
%   characters defined in CHARS from the string S. By default both sides 
%   will be trimmed. With 'left' or 'right' you may choose to trim the  
%   leading or trailing part only.
%
%   Examples:    o trimstr( ', a, b, c,', ',' )         --> 'a, b, c'
%                o trimstr( ', a, b, c,', 'left' )      --> ', a, b, c,'
%                o trimstr( ', a, b, c,', ',', 'left' ) --> 'a, b, c,'

if isempty( s )
  y = s([]);
else
 if ~ischar( s )
   error( 'Input must be a string (char array).' );
 end
 
   % arguments
 sc = '';        % stripchars
 lor = 0;        % leftorright (0: both, 1: left, 2: right)
 if (nargin == 2)
   if strcmpi( stripchars, 'left' )
     lor = 1; 
   elseif strcmpi( stripchars, 'right' )
     lor = 2; 
   else
     sc = stripchars;
   end;
  elseif nargin == 3
    sc = stripchars;
    if strcmpi( leftorright, 'left' )
      lor = 1; 
    elseif strcmpi( leftorright, 'right' )
      lor = 2; 
    else
      error( 'Third argument must be ''left'' or ''right''' );
    end;
  end
     
    % start, end index
  ind1 = 1;               
  ind2 = size( s, 2 );
  
    % get indexes (avoiding ismember if possible is faster)
  if isempty( sc )
    ind = find( ~isspace( s ) );
  else
    ind = find( ~isspace( s ) & ~ismember( s, sc ) );
  end;
  if (lor == 0) | (lor == 1)
    ind1 = min( ind );
  end; %if both or left side
  if (lor == 0) | (lor == 2)
    ind2 = max( ind );
  end; %if both or right side
   
    % output the trimmed string
  y = s(ind1:ind2);
  if isempty( y )
    y = s([]);
  end;  
     
end; %if s isempty

%%%%%%%%%%%%%%%%%%%
%%% HELP BUTTONS %%%
%%%%%%%%%%%%%%%%%%%

%%% --- Executes on button press in the permanent Help buttons.
%%% (The permanent Help buttons are the ones that don't change 
%%% depending on the algorithm loaded.) 
function HelpStep1_Callback(hObject, eventdata, handles)
helpdlg('Select the main folder containing the images you want to analyze. You will have the option within load images modules to retrieve images from more than one folder, but the folder selected here will be the default folder.  Use the Browse button to select the folder, or carefully type the full pathname in the box to the right.','Step 1 Help')
function HelpStep2_Callback(hObject, eventdata, handles)
helpdlg('OUTPUT FILE NAME: Type in the text you want to use to name the output file, which is where all of the information about the analysis as well as any measurements are stored. It is strongly recommended that all output files begin with ?OUT? to avoid confusion.  You do not need to type ?.mat? at the end of the file name, it will be added automatically. The program prevents you from entering a name which, when ''.mat'' is appended, exists already. This prevents overwriting an output data file by accident.  It also prevents intentionally overwriting an output file for the following reason: when a file is ''overwritten'', instead of completely overwriting the output file, Matlab just replaces some of the old data with the new data.  So, if you have an output file with 12 measurements and the new set of data has only 4 measurements, saving the output file to the same name would produce a file with 12 measurements: the new 4 followed by 8 old measurements.       PIXELS PER MICROMETER: Enter the pixel size of the images.  This is based on the resolution and binning of the camera and the magnification of the objective lens. This number is used to convert measurements to micrometers instead of pixels. If you do not know the pixel size or you want the measurements to be reported in pixels, enter "1".          SAMPLE INFO: If you would like text information about each image to be recorded in the output file along with measurements (e.g. Gene names, accession numbers, or sample numbers), click the Load button.  You will then be guided through the process of choosing a text file that contains the text data for each image. More than one set of text information can be entered for each image; each set of text will be a separate column in the output file.        SET DEFAULT FOLDER: Click this button and choose a folder to permanently set the folder to go to when you load analysis modules. This only needs to be done once, because a file called CellProfilerPreferences.mat is created in the root directory of Matlab that stores this information.','Step 2 Help')
function HelpStep3_Callback(hObject, eventdata, handles)
helpdlg('FOR HELP ON INDIVIDUAL MODULES: Click the "Help for this analysis module" button towards the right of the CellProfiler window.       LOAD/CLEAR/VIEW BUTTONS:  Choose image analysis modules in the desired order by clicking "Load" and selecting the corresponding Matlab ".m" file.      SHORTCUTS: Once you have loaded the desired image analysis modules and modified all of the settings as desired, you may save these settings for future use by clicking "Save Settings" and naming the file.  Later, you can click "Load Settings", select this file that you made, and all of the modules and settings will be restored.  ALTERNATELY, if you previously ran an image analysis and you want to repeat the exact analysis, you may click "Extract Settings from an output file".  Select the output file, and the modules and settings used to create it will be extracted.  You then name the settings file and load it using the "Load Settings" button.  Troubleshooting: If you loaded an analysis module by loading a settings file, and then obtained error messages in the Matlab main window, the most likely cause is that the analysis modules loaded are not on the Matlab search path. Be sure that the folder immediately containing the analysis module is on the search path. The search path can be edited by choosing File > Set Path.  Another possibility is that the Settings file was created with old versions of CellProfiler or with old versions of modules.  The Settings file can be opened with any word processor as plain text and you should be able to figure out what the settings were.        TECHNICAL DIAGNOSIS: Clicking here causes text to appear in the main Matlab window.  This text shows the "handles structure" which is sometimes useful for diagnosing problems with the software.','Step 3 Help')
function HelpStep4_Callback(hObject, eventdata, handles)
helpdlg('THIS DOES NOT YET WORK!!   CHOOSE TEST IMAGE: Using either the Browse button, the pull-down menu, or the text box (type carefully!), choose the image file that is the first in the set you would like to analyze.  Then click "Analyze test image".','Step 4 Help')
function HelpStep5_Callback(hObject, eventdata, handles)
helpdlg('SHOW PIXEL DATA: If you have an image displayed in a figure window and would like to determine the X, Y position or the intensity at a particular pixel, click this button.        CLOSE ALL FIGURES AND TIMERS: Click this button to close all open figure/image windows and timers. The main CellProfiler window and any error/message windows will remain open. You will be asked for confirmation first before the windows are all closed.           EXTRACT DATA: Once image analysis is complete, click this button and select the output file to extract the measurements and other information about the analysis.  The data will be converted to a delimited text file which can be read by most programs.  By naming the file with the extension for Microsoft Excel (.xls), the file is usually easily openable by that program.       ANALYZE ALL IMAGES: All of the images in the selected directory/directories will be analyzed using the modules and settings you have specified.  You will have the option to cancel at any time.  At the end of each data set, the data are stored in the output file.','Step 5 Help')
function HelpStep6_Callback(hObject, eventdata, handles)
helpdlg('Help will be displayed here.','Step 6 Help')

% --- Executes on button press in HelpForThisAnalysisModule.  
function HelpForThisAnalysisModule_Callback(hObject, eventdata, handles)
%%% First, check to see whether there is a specific algorithm loaded.
%%% If not, it opens a help dialog which explains how to pick one.
AlgorithmNumber = whichactive(handles);
if AlgorithmNumber == 0
    helpdlg('You do not have an analysis module selected.  Click "?" next to "Image analysis settings" to get help in choosing an analysis module, or click "View" next to an analysis module that has been loaded already.','Help for choosing an analysis module')
else
    AlgorithmName = get(handles.(['AlgorithmName' TwoDigitString(AlgorithmNumber)]),'String');
    IsItNotChosen = strncmp(AlgorithmName,'No a',4);
    if IsItNotChosen == 1
        helpdlg('You do not have an analysis module selected.  Click "?" next to "Image analysis settings" to get help in choosing an analysis module, or click "View" next to an analysis module that has been loaded already.','Help for choosing an analysis module')
    else
        
        %%% This is the function that actually reads the algorithm's help
        %%% data.
        Algorithm = strcat('Alg',AlgorithmName,'.m');
        fid=fopen(Algorithm);
        while 1;
            output = fgetl(fid);
            testifpercent = strncmp(output,'%%%%% ',6);
            if ~ischar(output); break; end;
            if testifpercent == 1;
                doesHelpTextexist = exist('HelpText','var');
                if doesHelpTextexist == 0 
                    HelpText = output(6:end);
                else HelpText = strvcat(HelpText,output(6:end));
                end;
            end;
        end;
        fclose(fid);
        DoesHelpExist = exist('HelpText','var');
        if DoesHelpExist == 1
            helpdlg(HelpText, 'Algorithm Help'); 
        else helpdlg('Sorry, there is no help information for this analysis module.','This is not helpful')
        end;
    end;
end;

%%% ^ END OF HELP HELP HELP HELP HELP HELP BUTTONS ^ %%%


% --- Executes on button press in RevealDataAnalysisButtons.
function RevealDataAnalysisButtons_Callback(hObject, eventdata, handles)
CurrentButtonLabel = get(hObject,'string');
if strcmp(CurrentButtonLabel,'Hide')
    set(handles.CoverDataAnalysisFrame,'visible','on')
    set(hObject,'String','Data')
else
        set(handles.CoverDataAnalysisFrame,'visible','off')
            set(hObject,'String','Hide')
end