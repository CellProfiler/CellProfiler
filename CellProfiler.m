function varargout = CellProfiler(varargin)

% CellProfilerTM cell image analysis software
%
% CellProfiler cell image analysis software is designed for
% biologists without training in computer vision or programming to
% quantitatively measure phenotypes from thousands of images
% automatically. CellProfiler.m and CellProfiler.fig work together to
% create a user interface which allows the analysis of large numbers
% of images.  New modules can be written for the software using
% Matlab.
%
%  Typing CellProfiler at the command line launches the program.
%
%  To run an example image analysis, browse to choose the
%  ExampleFlyImages folder within CellProfiler/ExampleImages/, type in
%  the name of an output file (e.g. 'Temp1'), click 'Load settings'
%  and choose 'ExampleFlySettings' and click 'Analyze all images'. An
%  analysis run should begin.
%
%      H = CellProfiler returns the handle to a new CellProfiler or
%      the handle to the existing singleton*.

% The contents of this file are subject to the Mozilla Public License Version 
% 1.1 (the "License"); you may not use this file except in compliance with 
% the License. You may obtain a copy of the License at 
% http://www.mozilla.org/MPL/
% 
% Software distributed under the License is distributed on an "AS IS" basis,
% WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
% for the specific language governing rights and limitations under the
% License.
% 
% 
% The Original Code is the CellProfiler.m and CellProfiler.fig files.
% 
% The Initial Developer of the Original Code is
% Whitehead Institute for Biomedical Research
% Portions created by the Initial Developer are Copyright (C) 2003,2004
% the Initial Developer. All Rights Reserved.
% 
% Contributor(s):
%   Anne Carpenter <carpenter@wi.mit.edu>
%   Thouis Jones   <thouis@csail.mit.edu>
%   In Han Kang    <inthek@mit.edu>
%
% $Revision$

% Last Modified by GUIDE v2.5 01-Nov-2004 14:17:07
% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @CellProfiler_OpeningFcn, ...
                   'gui_OutputFcn',  @CellProfiler_OutputFcn, ...
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

% --- Executes just before CellProfiler is made visible.
function CellProfiler_OpeningFcn(hObject, eventdata, handles, varargin) %#ok We want to ignore MLint error checking for this line.

%create additional gui elements
handles = createVariablePanel(handles);

% Choose default command line output for CellProfiler
handles.output = hObject;

% The Number of Algorithms/Variables hardcoded in
handles.numAlgorithms = 0;
handles.MaxAlgorithms = 99;
handles.MaxVariables = 99;
global closeFigures openFigures;
closeFigures = [];
openFigures = [];

% Update handles structure
guidata(hObject, handles);

% --- Outputs from this function are returned to the command line.
function varargout = CellProfiler_OutputFcn(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.

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
    handles.Settings.Vpixelsize = PixelSize;
    handles.Vdefaultalgorithmdirectory = DefaultAlgorithmDirectory;
    handles.Vworkingdirectory = WorkingDirectory;
    handles.Vpathname = WorkingDirectory;
    handles.Vtestpathname = handles.Vworkingdirectory;
    set(handles.PixelSizeEditBox,'string',PixelSize);
    set(handles.PathToLoadEditBox,'String',handles.Vworkingdirectory);
else
    handles.Settings.Vpixelsize = get(handles.PixelSizeEditBox,'string');
    handles.Vdefaultalgorithmdirectory = pwd;
    handles.Vworkingdirectory = pwd;
    handles.Vpathname = pwd;
    handles.Vtestpathname = pwd;
    set(handles.PathToLoadEditBox,'String',pwd);
end

%%% Sets up the main program window (Main GUI window) so that it asks for
%%% confirmation prior to closing.
%%% First, obtains the handle for the main GUI window (aka figure1).
ClosingFunction = ...
    ['deleteme = questdlg(''Do you really want to quit?'', ''Confirm quit'',''Yes'',''No'',''Yes''); switch deleteme; case ''Yes''; delete(', num2str((handles.figure1)*8192), '/8192); case ''No''; return; end; clear deleteme'];
%%% Sets the closing function of the Main GUI window to be the line above.
set(handles.figure1,'CloseRequestFcn',ClosingFunction);

%%% Obtains the screen size.
ScreenSize = get(0,'ScreenSize');
ScreenWidth = ScreenSize(3);
ScreenHeight = ScreenSize(4);
%%% Sets the position of the Main GUI window so it is in the center of
%%% the screen. At one point, I designed the GUI window itself to be
%%% 800 pixels wide and 600 high, but it has changed since then.
GUIsize = get(handles.figure1,'position');
GUIwidth = GUIsize(3);
GUIheight = GUIsize(4);
Left = 0.5*(ScreenWidth - GUIwidth);
Bottom = 0.5*(ScreenHeight - GUIheight);
set(handles.figure1,'Position',[Left Bottom GUIwidth GUIheight]);


%%% Retrieves the list of image file names from the chosen directory and
%%% stores them in the handles structure, using the function
%%% RetrieveImageFileNames.
Pathname = handles.Vpathname;
handles = RetrieveImageFileNames(handles, Pathname);
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

function handles = RetrieveImageFileNames(handles, Pathname)
%%% Lists all the contents of that path into a structure which includes the
%%% name of each object as well as whether the object is a file or
%%% directory.
FilesAndDirsStructure = dir(Pathname);
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
  DiscardsByExtension = [];
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
function BrowseToLoad_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
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
    end
    %%% Displays the chosen directory in the PathToLoadEditBox.
    set(handles.PathToLoadEditBox,'String',pathname);
end
cd(CurrentDirectory)
%%%%%%%%%%%%%%%%%

% --- Executes during object creation, after setting all properties.
function PathToLoadEditBox_CreateFcn(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
    set(hObject,'BackgroundColor',[1 1 1]);

function PathToLoadEditBox_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
%%% Retrieves the text that was typed in.
pathname = get(hObject,'string');
%%% Checks whether a directory with that name exists.
if exist(pathname,'dir') ~= 0
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
function LoadSampleInfo_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.

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
          ;
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
function ClearSampleInfo_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
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
function ViewSampleInfo_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
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
function OutputFileName_CreateFcn(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
set(hObject,'BackgroundColor',[1 1 1]);

function OutputFileName_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
CurrentDirectory = cd;
Pathname = get(handles.PathToLoadEditBox,'string');
%%% Gets the user entry and stores it in the handles structure.
InitialUserEntry = get(handles.OutputFileName,'string');
if isempty(InitialUserEntry)
    handles.Voutputfilename =[];
    guidata(gcbo, handles);
else
    if length(InitialUserEntry) >=7
        if strncmpi(InitialUserEntry(end-6:end),'out.mat',7) == 1
            UserEntry = InitialUserEntry;
        elseif strncmpi(InitialUserEntry(end-3:end),'.mat',4) == 1
            UserEntry = [InitialUserEntry(1:end-4) 'OUT.mat'];
        else UserEntry = [InitialUserEntry,'OUT.mat'];
        end
    elseif length(InitialUserEntry) >=4
        if strncmp(InitialUserEntry(end-3:end),'.mat',4) == 1
        UserEntry = [InitialUserEntry(1:end-4) 'OUT.mat'];
        else UserEntry = [InitialUserEntry,'OUT.mat'];
        end
    else UserEntry = [InitialUserEntry,'OUT.mat'];
    end
    guidata(gcbo, handles);
    %%% Checks whether a file with that name already exists, to warn the user
    %%% that the file will be overwritten.
    CurrentDirectory = cd;
    if exist([Pathname,'/',UserEntry],'file') ~= 0
        errordlg(['A file already exists at ', [Pathname,'/',UserEntry],...
            '. Enter a different name. Click the help button for an explanation of why you cannot just overwrite an existing file.'], 'Warning!');
        set(handles.OutputFileName,'string',[])
    else guidata(gcbo, handles);
        handles.Voutputfilename = UserEntry;
        set(handles.OutputFileName,'string',UserEntry)
    end
end
guidata(gcbo, handles);
cd(CurrentDirectory)

%%%%%%%%%%%%%%%%%

% --- Executes on button press in LoadSettingsFromFileButton.
function LoadSettingsFromFileButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.

CurrentDirectory = pwd;
cd(handles.Vworkingdirectory)
[SettingsFileName, SettingsPathname] = uigetfile('*.mat','Choose a settings or output file');
%%% If the user presses "Cancel", the SettingsFileName.m will = 0 and
%%% nothing will happen.
if SettingsFileName == 0
    return
end
%%% Loads the Settings file.
LoadedSettings = load([SettingsPathname SettingsFileName]);

if ~ (isfield(LoadedSettings, 'Settings') || isfield(LoadedSettings, 'handles')),
    errordlg(['The file ' SettingsPathname SettingsFilename ' does not appear to be a valid settings or output file. Settings can be extracted from an output file created when analyzing images with CellProfiler or from a small settings file saved using the "Save Settings" button.  Either way, this file must have the extension ".mat" and contain a variable named "Settings" or "handles".']);
    cd(CurrentDirectory);
    return;
end

%%% Figure out whether we loaded a Settings or Output file, and put the correct values into Settings
%%% Splice the subset of variables from the "settings" structure into the
%%% handles structure.

if (isfield(LoadedSettings, 'Settings')),
    Settings = LoadedSettings.Settings;
    handles.Settings.Valgorithmname = Settings.Valgorithmname;
    handles.Settings.Vvariable = Settings.Vvariable;
else
    Settings = LoadedSettings.handles;
    if isfield(Settings,'Settings'),
        handles.Settings.Valgorithmname = Settings.Settings.Valgorithmname;
        handles.Settings.Vvariable = Settings.Settings.Vvariable;
    end
end

if isfield(Settings,'numVariables'),
    handles.numVariables = Settings.numVariables;
end
handles.numAlgorithms = 0;
handles.numAlgorithms = length(handles.Settings.Valgorithmname);
contents = handles.Settings.Valgorithmname;
set(handles.AlgorithmBox,'String',contents);
set(handles.AlgorithmBox,'Value',1);
handles.Settings.Vpixelsize = Settings.Vpixelsize;
set(handles.PixelSizeEditBox,'string',Settings.Vpixelsize);

%%% Update handles structure.
guidata(hObject,handles);
ViewAlgorithm(handles);

%%% If the user loaded settings from an output file, prompt them to
%%% save it as a separate Settings file for future use.
if isfield(LoadedSettings, 'handles'),
    Answer = questdlg('The settings have been extracted from the output file you selected.  Would you also like to save these settings in a separate, smaller, settings-only file?','','Yes','No','Yes');
    if strcmp(Answer, 'Yes') == 1
        SaveCurrentSettingsButton_Callback(hObject, eventdata, handles);
    end
end
cd(CurrentDirectory)

%%%%%%%%%%%%%%%%%

% --- Executes on button press in SaveSettingsButton.
function SaveSettingsButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
CurrentDirectory = pwd;
cd(handles.Vworkingdirectory)
%%% The "Settings" variable is saved to the file name the user chooses.
[FileName,Pathname] = uiputfile('*.mat', 'Save Settings As...');
%%% Allows canceling.
if FileName ~= 0
  %%% Checks if a field is present, and if it is, the value is stored in the 
  %%% structure 'Settings' with the same name
  
  if isfield(handles.Settings,'Vvariable'),
      Settings.Vvariable = handles.Settings.Vvariable;
  end
  if isfield(handles.Settings,'Valgorithmname'),
      Settings.Valgorithmname = handles.Settings.Valgorithmname;
  end
  if isfield(handles,'numVariables'),
      Settings.numVariables = handles.numVariables;
  end
  if isfield(handles.Settings,'Vpixelsize'),
    Settings.Vpixelsize = handles.Settings.Vpixelsize;
  end
  save([Pathname FileName],'Settings')
  helpdlg('The settings file has been written.')
end
cd(CurrentDirectory)

%%%%%%%%%%%%%%%%%

% --- Executes during object creation, after setting all properties.
function PixelSizeEditBox_CreateFcn(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
    set(hObject,'BackgroundColor',[1 1 1]);

function PixelSizeEditBox_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
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
%%% Gets the user entry and stores it in the handles structure.
UserEntry = get(handles.PixelSizeEditBox,'string');
handles.Settings.Vpixelsize = UserEntry;
guidata(gcbo, handles);
end

%%%%%%%%%%%%%%%%%

% --- Executes on button press in SetPreferencesButton.
function SetPreferencesButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
%%% Determine what the current directory is, so we can change back 
%%% when this process is done.
CurrentDirectory = cd;
%%% Change to the Matlab root directory.
cd(matlabroot)
%%% If the CellProfilerPreferences.mat file does not exist in the matlabroot
%%% directory, change to the current directory.
if exist('CellProfilerPreferences.mat','file') == 0
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
try   %#ok We want to ignore MLint error checking for this line.
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
%%% Tries to change to the working directory, whose name is a variable
%%% that is stored in the CellProfilerPreferences.mat file.
try cd(WorkingDirectory) %#ok We want to ignore MLint error checking for this line.
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
try cd(matlabroot)
    save CellProfilerPreferences DefaultAlgorithmDirectory PixelSize WorkingDirectory
    helpdlg('Your CellProfiler Preferences were successfully set.  They are contained within a folder in the Matlab root directory in a file called CellProfilerPreferences.mat.')
    handles.Settings.Vpixelsize = PixelSize;
    handles.Vdefaultalgorithmdirectory = DefaultAlgorithmDirectory;
    handles.Vworkingdirectory = WorkingDirectory;
    set(handles.PixelSizeEditBox,'string',PixelSize);
    %%% Update handles structure.
    guidata(hObject,handles);

catch
    cd(CurrentDirectory)
    try save CellProfilerPreferences DefaultAlgorithmDirectory PixelSize WorkingDirectory
        helpdlg('You do not have permission to write anything to the Matlab root directory, which is required to save your preferences permanently.  Instead, your preferences will only function properly while you are in the current directory.')
        handles.Settings.Vpixelsize = PixelSize;
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

%%%%%%%%%%%%%%%%%
%%% ADD BUTTON %%%
%%%%%%%%%%%%%%%%%

% --- Executes on button press in AddAlgorithm.
function AddAlgorithm_Callback(hObject,eventdata,handles) %#ok We want to ignore MLint error checking for this line.
% Find which algorithm slot number this callback was called for.
AlgorithmNumber = TwoDigitString(handles.numAlgorithms+1);
AlgorithmNums = handles.numAlgorithms+1;

%%% 1. Opens a user interface to retrieve the .m file you want to use.  The
%%% name of that .m file is stored as the variablebox2_1
%%% "FirstImageAlgorithmName".

%%% First, the current directory is stored so we can switch back to it at
%%% the end of this step:
CurrentDirectory = cd;
%%% Change to the default algorithm directory, whose name is a variable
%%% that is stored in that .mat file. It is within a try-end pair because
%%% the user may have changed the folder names leading up to this directory
%%% sometime after saving the Preferences.
try cd(handles.Vdefaultalgorithmdirectory) %#ok We want to ignore MLint error checking for this line.
end 
%%% Now, when the dialog box is opened to retrieve an algorithm, the
%%% directory will be the default algorithm directory.
[AlgorithmNamedotm,Pathname] = uigetfile('Alg*.m',...
    'Choose an image analysis module');
%%% Change back to the original directory.
cd(CurrentDirectory)

%%% 2. If the user presses "Cancel", the AlgorithmNamedotm = 0, and
%%% everything should be left as it was.  If the algorithm is not on
%%% Matlab's search path, the user is warned.
if AlgorithmNamedotm == 0,
  %%% If the algorithm's .m file is not found on the search path, the result
  %%% of exist is zero.
elseif exist(AlgorithmNamedotm,'file') == 0
  msgbox(['The .m file ', AlgorithmNamedotm, ...
        ' was not initially found by Matlab, so the folder containing it was added to the Matlab search path.  Please reload the analysis module; It should work fine from now on. If for some reason you did not want to add that folder to the path, go to Matlab > File > Set Path and remove the folder from the path.  If you have no idea what this means, don''t worry about it.'])
  %%% The folder containing the desired .m file is added to Matlab's search path.
  addpath(Pathname)
  %%% Doublecheck that the algorithm exists on Matlab's search path.
  if exist(AlgorithmNamedotm,'file') == 0
    errordlg('Something is wrong; Matlab still cannot find the .m file for the analysis module you selected.')
  end
else

  %%% 3. The last two characters (=.m) are removed from the
  %%% AlgorithmName.m and called AlgorithmName.
  AlgorithmName = AlgorithmNamedotm(4:end-2);
  %%% The name of the algorithm is shown in a text box in the GUI (the text
  %%% box is called AlgorithmName1.) and in a text box in the GUI which
  %%% displays the current algorithm (whose settings are shown).
  %set(handles.(['AlgorithmName' AlgorithmNumber]),'String',AlgorithmName);

  %%% 4. Saves the AlgorithmName to the handles structure.
  handles.Settings.Valgorithmname{AlgorithmNums} = AlgorithmName;
  contents = get(handles.AlgorithmBox,'String');
  contents{AlgorithmNums} = AlgorithmName;
  set(handles.AlgorithmBox,'String',contents);

  %%% 5. The text description for each variable for the chosen algorithm is 
  %%% extracted from the algorithm's .m file and displayed.  
  fid=fopen([Pathname AlgorithmNamedotm]);

  while 1;
      output = fgetl(fid); if ~ischar(output); break; end;

      if (strncmp(output,'%defaultVAR',11) == 1),
          displayval = output(17:end);
          istr = output(12:13);
          i = str2num(istr);
          set(handles.(['VariableBox' istr]), 'string', displayval,'visible', 'on');
          set(handles.(['VariableDescription' istr]), 'visible', 'on');
          handles.Settings.Vvariable(AlgorithmNums, i) = {displayval};
          handles.numVariables(str2double(AlgorithmNumber)) = i;
      end
  end
  fclose(fid);

  %%% 6. Update handles.numAlgorithms
  if str2double(AlgorithmNumber) > handles.numAlgorithms,
    handles.numAlgorithms = str2double(AlgorithmNumber);
  end
  
  %%% 7. Choose Loaded Algorithm in Listbox
  set(handles.AlgorithmBox,'Value',handles.numAlgorithms);
    
  %%% Updates the handles structure to incorporate all the changes.
  guidata(gcbo, handles);
  ViewAlgorithm(handles);
end

%%%%%%%%%%%%%%%%%

function ViewAlgorithm(handles)
AlgorithmHighlighted = get(handles.AlgorithmBox,'Value');
if (length(AlgorithmHighlighted) > 0)
    AlgorithmNumber = AlgorithmHighlighted(1);
    if( handles.numAlgorithms > 0 )

        %%% 2. Sets all VariableBox edit boxes and all
        %%% VariableDescriptions to be invisible.
        for i = 1:handles.MaxVariables,
            set(handles.(['VariableBox' TwoDigitString(i)]),'visible','off','String','n/a')
            set(handles.(['VariableDescription' TwoDigitString(i)]),'visible','off')
        end

        %%% 2.25 Remove slider and move panel back to original position
        set(handles.variablepanel, 'position', [46 5.3846 108.4 23.154]);
        set(handles.slider1,'visible','off');

        %%% 2.5 Checks whether an algorithm is loaded in this slot.
        contents = get(handles.AlgorithmBox,'String');
        AlgorithmName = contents{AlgorithmNumber};

        %%% 3. Extracts and displays the variable descriptors from the .m file.
        AlgorithmNamedotm = strcat('Alg',AlgorithmName,'.m');
        if exist(AlgorithmNamedotm,'file') ~= 2
            errordlg(['The image analysis module named ', AlgorithmNamedotm, ' was not found. Is it stored in the folder with the other modules?  Has its name changed?  The settings stored for this module will be displayed, but this module will not run properly.']);
        else
            fid=fopen(AlgorithmNamedotm);

            while 1;
                output = fgetl(fid); if ~ischar(output); break; end;
                if (strncmp(output,'%textVAR',8) == 1);
                    set(handles.(['VariableDescription',output(9:10)]), 'string', output(13:end),'visible', 'on');
                end
            end
            fclose(fid);
        end
        %%% 4. The stored values for the variables are extracted from the handles
        %%% structure and displayed in the edit boxes.
        for i=1:handles.numVariables(AlgorithmNumber),
            if iscellstr(handles.Settings.Vvariable(AlgorithmNumber, i));
                set(handles.(['VariableBox' TwoDigitString(i)]),...
                    'string',char(handles.Settings.Vvariable(AlgorithmNumber,i)),...
                    'visible','on');
            else set(handles.(['VariableBox' TwoDigitString(i)]),'string','n/a','visible','off');
            end
        end

        %%% 5.  Set the slider

        if(handles.numVariables(AlgorithmNumber) > 12)
            set(handles.slider1,'visible','on');
            set(handles.slider1,'max',(handles.numVariables(AlgorithmNumber)-12)*1.77);
            set(handles.slider1,'value',(handles.numVariables(AlgorithmNumber)-12)*1.77);
        end

    else
        helpdlg('Module not loaded.');
    end
else
    helpdlg('No module highlighted.');
end

%%%%%%%%%%%%%%%%%%%%
%%% REMOVE BUTTON %%%
%%%%%%%%%%%%%%%%%%%%

% --- Executes on button press for RemoveAlgorithm button.
function RemoveAlgorithm_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
AlgorithmHighlighted = get(handles.AlgorithmBox,'Value');
RemoveAlgorithm_Helper(AlgorithmHighlighted, hObject, eventdata, handles, 'Confirm');

% separated because it's called elsewhere
function RemoveAlgorithm_Helper(AlgorithmHighlighted, hObject, eventdata, handles, ConfirmOrNot) %#ok We want to ignore MLint error checking for this line.

if strcmp(ConfirmOrNot, 'Confirm') == 1
    %%% Confirms the choice to clear the algorithm.
    Answer = questdlg('Are you sure you want to clear this analysis module and its settings?','Confirm','Yes','No','Yes');
    if strcmp(Answer,'No') == 1
        return
    end
end

%%% 1. Sets all 11 VariableBox edit boxes and all 11
%%% VariableDescriptions to be invisible.
for i = 1:handles.MaxVariables
    set(handles.(['VariableBox' TwoDigitString(i)]),'visible','off','String','n/a')
    set(handles.(['VariableDescription' TwoDigitString(i)]),'visible','off')
end

for AlgDelete = 1:length(AlgorithmHighlighted);
    %%% 2. Removes the AlgorithmName from the handles structure.
    handles.Settings.Valgorithmname(AlgorithmHighlighted(AlgDelete)-AlgDelete+1) = [];
    %%% 3. Clears the variable values in the handles structure.
    handles.Settings.Vvariable(AlgorithmHighlighted(AlgDelete)-AlgDelete+1,:) = [];
    %%% 4. Clears the number of variables in each algorithm slot from handles structure.
    handles.numVariables(AlgorithmHighlighted(AlgDelete)-AlgDelete+1) = [];
end

%%% 5. Update the number of algorithms loaded
handles.numAlgorithms = 0;
handles.numAlgorithms = length(handles.Settings.Valgorithmname);

%%% 6. Sets the proper algorithm name to "No analysis module loaded"
if(isempty(handles.Settings.Valgorithmname))
    contents = {'No Algorithms Loaded'};
else
    contents = handles.Settings.Valgorithmname;
end

set(handles.AlgorithmBox,'String',contents);

while((isempty(AlgorithmHighlighted)==0) && (AlgorithmHighlighted(length(AlgorithmHighlighted)) > handles.numAlgorithms) )
    AlgorithmHighlighted(length(AlgorithmHighlighted)) = [];
end

if(handles.numAlgorithms == 0)
    AlgorithmHighlighted = 1;
elseif (isempty(AlgorithmHighlighted))
    AlgorithmHighlighted = handles.numAlgorithms;
end

set(handles.AlgorithmBox,'Value',AlgorithmHighlighted);

guidata(gcbo, handles);
ViewAlgorithm(handles);

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% MOVE UP/DOWN BUTTONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

function MoveUpButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
AlgorithmHighlighted = get(handles.AlgorithmBox,'Value');
if(handles.numAlgorithms < 1 || AlgorithmHighlighted(1) == 1)
else
    for AlgUp = 1:length(AlgorithmHighlighted);
        AlgorithmUp = AlgorithmHighlighted(AlgUp)-1;
        AlgorithmNow = AlgorithmHighlighted(AlgUp);
        %%% 1. Switches AlgorithmNames
        AlgorithmUpName = char(handles.Settings.Valgorithmname(AlgorithmUp));
        AlgorithmName = char(handles.Settings.Valgorithmname(AlgorithmNow));
        handles.Settings.Valgorithmname{AlgorithmUp} = AlgorithmName;
        handles.Settings.Valgorithmname{AlgorithmNow} = AlgorithmUpName;
        %%% 2. Copy then clear the variable values in the handles structure.
        copyVariables = handles.Settings.Vvariable(AlgorithmNow,:);
        handles.Settings.Vvariable(AlgorithmNow,:) = handles.Settings.Vvariable(AlgorithmUp,:);
        handles.Settings.Vvariable(AlgorithmUp,:) = copyVariables;
        %%% 3. Copy then clear the num of variables in the handles
        %%% structure.
        copyNumVariables = handles.numVariables(AlgorithmNow);
        handles.numVariables(AlgorithmNow) = handles.numVariables(AlgorithmUp);
        handles.numVariables(AlgorithmUp) = copyNumVariables;
    end
    %%% 4. Changes the Listbox to show the changes
    contents = handles.Settings.Valgorithmname;
    AlgorithmHighlighted = AlgorithmHighlighted-1;
    set(handles.AlgorithmBox,'String',contents);
    set(handles.AlgorithmBox,'Value',AlgorithmHighlighted);
    %%% Updates the handles structure to incorporate all the changes.
    guidata(gcbo, handles);
    ViewAlgorithm(handles)
end

%%%%%%%

function MoveDownButton_Callback(hObject,eventdata,handles) %#ok We want to ignore MLint error checking for this line.
AlgorithmHighlighted = get(handles.AlgorithmBox,'Value');
if(handles.numAlgorithms<1 || AlgorithmHighlighted(length(AlgorithmHighlighted)) >= handles.numAlgorithms)
else
    for AlgDown = 1:length(AlgorithmHighlighted);
        AlgorithmDown = AlgorithmHighlighted(AlgDown) + 1;
        AlgorithmNow = AlgorithmHighlighted(AlgDown);
        %%% 1. Saves the AlgorithmName
        AlgorithmDownName = char(handles.Settings.Valgorithmname(AlgorithmDown));
        AlgorithmName = char(handles.Settings.Valgorithmname(AlgorithmNow));
        handles.Settings.Valgorithmname{AlgorithmDown} = AlgorithmName;
        handles.Settings.Valgorithmname{AlgorithmNow} = AlgorithmDownName;
        %%% 2. Copy then clear the variable values in the handles structure.
        copyVariables = handles.Settings.Vvariable(AlgorithmNow,:);
        handles.Settings.Vvariable(AlgorithmNow,:) = handles.Settings.Vvariable(AlgorithmDown,:);
        handles.Settings.Vvariable(AlgorithmDown,:) = copyVariables;
        %%% 3. Copy then clear the num of variables in the handles
        %%% structure.
        copyNumVariables = handles.numVariables(AlgorithmNow);
        handles.numVariables(AlgorithmNow) = handles.numVariables(AlgorithmDown);
        handles.numVariables(AlgorithmDown) = copyNumVariables;
    end
    %%% 4. Changes the Listbox to show the changes
    contents = handles.Settings.Valgorithmname;
    set(handles.AlgorithmBox,'String',contents);
    set(handles.AlgorithmBox,'Value',AlgorithmHighlighted+1);
    AlgorithmHighlighted = AlgorithmHighlighted+1;
    %%% Updates the handles structure to incorporate all the changes.
    guidata(gcbo, handles);
    ViewAlgorithm(handles)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% FIGURE DISPLAY BUTTONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% NOTE: These buttons appear after analysis has begun, and disappear 
%%% when it is over.

function CloseFigureButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
global closeFigures;
AlgorithmHighlighted = get(handles.AlgorithmBox,'Value');
for i=1:length(AlgorithmHighlighted),
        closeFigures(length(closeFigures)+1) = AlgorithmHighlighted(i);
end
guidata(hObject, handles);


% --- Executes on button press in OpenFigureButton.
function OpenFigureButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
global openFigures;
AlgorithmHighlighted = get(handles.AlgorithmBox,'Value');
for i=1:length(AlgorithmHighlighted),
        openFigures(length(openFigures)+1) = AlgorithmHighlighted(i);
end
guidata(hObject, handles);


%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% VARIABLE EDIT BOXES %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%

function storevariable(AlgorithmNumber, VariableNumber, UserEntry, handles)
%%% This function stores a variable's value in the handles structure, 
%%% when given the Algorithm Number, the Variable Number, 
%%% the UserEntry (from the Edit box), and the initial handles
%%% structure.
handles.Settings.Vvariable(AlgorithmNumber, str2double(VariableNumber)) = {UserEntry};
guidata(gcbo, handles);

function [AlgorithmNumber] = whichactive(handles)
AlgorithmHighlighted = get(handles.AlgorithmBox,'Value');
AlgorithmNumber = AlgorithmHighlighted(1);
    
% --- Executes during object creation, after setting all properties.
function VariableBox_CreateFcn(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
    set(hObject,'BackgroundColor',[1 1 1])


function VariableBox_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
%%% The following lines fetch the contents of the edit box,
%%% determine which algorithm we are dealing with at the moment (by
%%% running the "whichactive" subfunction), and call the storevariable
%%% function.
VariableName = get(hObject,'tag');
VariableNumberStr = VariableName(12:13);

UserEntry = get(handles.(['VariableBox' VariableNumberStr]),'string');
AlgorithmNumber = whichactive(handles);
if isempty(UserEntry)
  errordlg('Variable boxes must not be left blank')
  set(handles.(['VariableBox' VariableNumberStr]),'string', 'Fill in');
  storevariable(AlgorithmNumber,VariableNumberStr, 'Fill in', handles);
else
  if AlgorithmNumber == 0,     
    errordlg('Something strange is going on: none of the analysis modules are active right now but somehow you were able to edit a setting.','weirdness has occurred')
  else
    storevariable(AlgorithmNumber,VariableNumberStr,UserEntry, handles);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% AlGORITHM BOXES %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes on selection change in AlgorithmBox.
function AlgorithmBox_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.

% Update handles structure
guidata(hObject, handles);
ViewAlgorithm(handles)

% --- Executes during object creation, after setting all properties.
function AlgorithmBox_CreateFcn(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
set(hObject,'BackgroundColor',[1 1 1]);
initialString = 'No Algorithms Loaded';
initialContents{1} = initialString;
set(hObject, 'String', initialContents);

% Update handles structure
guidata(hObject, handles);

%%%%%%%%%%%%%%%%%

% --- Executes during object creation, after setting all properties.
function ListBox_CreateFcn(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
    set(hObject,'BackgroundColor',[1 1 1]);

% --- Executes on selection change in ListBox.
function ListBox_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.

%%%%%%%%%%%%%%%%%

% --- Executes on button press in TechnicalDiagnosisButton.
function TechnicalDiagnosisButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
%%% I am using this button to show the handles structure in the
%%% main Matlab window.
handles %#ok We want to ignore MLint error checking for this line.
handles.Settings %#ok We want to ignore MLint error checking for this line.
handles.Measurements %#ok We want to ignore MLint error checking for this line.
msgbox('The handles structure has been printed out at the command line of Matlab.')

%%%%%%%%%%%%%%%%%

% --- Executes on button press in SaveImageAsButton.
function SaveImageAsButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
CurrentDirectory = cd;
cd(handles.Vworkingdirectory)
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
    OutputFileOverwrite = exist([cd,'/',CompleteFileName],'file');
    if OutputFileOverwrite ~= 0
        Answer = questdlg(['A file with the name ', CompleteFileName, ' already exists. Do you want to overwrite it?'],'Confirm file overwrite','Yes','No','No');
        if strcmp(Answer,'Yes') == 1;
            imwrite(ClickedImage, CompleteFileName, Extension)
            msgbox(['The file ', CompleteFileName, ' has been saved to the current directory']);
        end
    else
        imwrite(ClickedImage, CompleteFileName, Extension)
        msgbox(['The file ', CompleteFileName, ' has been saved to the current directory']);
    end
end
delete(MsgboxHandle)
cd(CurrentDirectory)

%%%%%%%%%%%%%%%%%

% --- Executes on button press in ShowImageButton.
function ShowImageButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
CurrentDirectory = cd;
cd(handles.Vworkingdirectory)
%%% Opens a user interface window which retrieves a file name and path 
%%% name for the image to be used as a test image.
[FileName,Pathname] = uigetfile('*.*','Select the image to view');
%%% If the user presses "Cancel", the FileName will = 0 and nothing will
%%% happen.
if FileName == 0
else
%%% REMOVED DUE TO CONFLICTS WITH THE NORMAL ZOOM FUNCTION
%%% SHOULD CONSIDER ADDING IT BACK.
%     %%% Acquires basic screen info for making buttons in the
%     %%% display window.
%     StdUnit = 'point';
%     StdColor = get(0,'DefaultUIcontrolBackgroundColor');
%     PointsPerPixel = 72/get(0,'ScreenPixelsPerInch');
%%% REMOVED DUE TO CONFLICTS WITH THE NORMAL ZOOM FUNCTION
    
    %%% Reads the image.
    Image = im2double(imread([Pathname,'/',FileName]));
    figure; imagesc(Image), colormap(gray)
    pixval
%%% REMOVED DUE TO CONFLICTS WITH THE NORMAL ZOOM FUNCTION
%%% SHOULD CONSIDER ADDING IT BACK.
%     %%% The following adds the Interactive Zoom button, which relies
%     %%% on the InteractiveZoomSubfunction.m being in the CellProfiler
%     %%% folder.
%     set(FigureHandle, 'Unit',StdUnit)
%     FigurePosition = get(FigureHandle, 'Position');
%     %%% Specifies the function that will be run when the zoom button is
%     %%% pressed.
%     ZoomButtonCallback = 'try, InteractiveZoomSubfunction, catch msgbox(''Could not find the file called InteractiveZoomSubfunction.m which should be located in the CellProfiler folder.''), end';
%     uicontrol('Parent',FigureHandle, ...
%         'CallBack',ZoomButtonCallback, ...
%         'BackgroundColor',StdColor, ...
%         'Position',PointsPerPixel*[FigurePosition(3)-108 5 105 22], ...
%         'String','Interactive Zoom', ...
%         'Style','pushbutton');
%%% REMOVED DUE TO CONFLICTS WITH THE NORMAL ZOOM FUNCTION
end
cd(CurrentDirectory)

%%%%%%%%%%%%%%%%%

% --- Executes on button press in ShowPixelDataButton.
function ShowPixelDataButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
FigureNumber = inputdlg('In which figure number would like to see pixel data?','',1);
if ~isempty(FigureNumber)
    FigureNumber = str2double(FigureNumber{1});
    pixval(FigureNumber,'on')
end

%%%%%%%%%%%%%%%%%

% --- Executes on button press in CloseAllFigureWindowsButton.
function CloseAllFigureWindowsButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.

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

% --- Executes on button press in RevealDataAnalysisButtons.
function RevealDataAnalysisButtons_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
CurrentButtonLabel = get(hObject,'string');
if strcmp(CurrentButtonLabel,'Hide')
    set(handles.CoverDataAnalysisFrame,'visible','on')
    set(hObject,'String','Data')
else
        set(handles.CoverDataAnalysisFrame,'visible','off')
            set(hObject,'String','Hide')
end

%%%%%%%%%%%%%%%%%
% --- Executes on button press in ExportDataButton.
function ExportDataButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.

%%% Determines the current directory so it can switch back when done.
CurrentDirectory = pwd;
cd(handles.Vworkingdirectory)
%%% Ask the user to choose the file from which to extract measurements.
[RawFileName, RawPathname] = uigetfile('*.mat','Select the raw measurements file');
if RawFileName == 0
else
    cd(RawPathname);
    load(RawFileName);
    
    %%% Extract the fieldnames of measurements from the handles structure.
    Fieldnames = fieldnames(handles.Measurements);
    MeasFieldnames = Fieldnames(strncmp(Fieldnames,'Image',5)==1);
    Fieldnames = fieldnames(handles.Pipeline);
    FileFieldNames = Fieldnames(strncmp(Fieldnames, 'Filename', 8)==1);
    %%% Determines whether any sample info has been loaded.  If sample info has
    %%% been loaded, the heading for that sample info would be listed in
    %%% handles.headings.  If sample info is present, the fieldnames for those
    %%% are also added to the list of data to extract.
    if isfield(handles, 'headings') == 1
      HeadingNames = handles.headings';
    else
      HeadingNames = {};
    end
    
    %%% Error detection.
    if isempty(MeasFieldnames)
        errordlg('No measurements were found in the file you selected. In the handles structure contained within the output file, the Measurements substructure must have fieldnames prefixed by ''Image''.')
    else
        
        %%% Determine the number of image sets for which there are data.
        fieldname = MeasFieldnames{1};
        TotalNumberImageSets = num2str(length(handles.Measurements.(fieldname)));
        %%% Ask the user to specify the number of image sets to extract.
        NumberOfImages = inputdlg({['How many image sets do you want to extract? As a shortcut,                     type the numeral 0 to extract data from all ', TotalNumberImageSets, ' image sets.']},'Specify number of image sets',1,{'0';' '});
        %%% If the user presses the Cancel button, the program goes to the end.
        if isempty(NumberOfImages)
        else
            %%% Calculate the appropriate number of image sets.
            NumberOfImages = str2double(NumberOfImages{1});
            if NumberOfImages == 0
                NumberOfImages = length(handles.Measurements.(char(MeasFieldnames(1))));
            elseif NumberOfImages > length(handles.Measurements.(char(MeasFieldnames(1))));
                errordlg(['There are only ', length(handles.Measurements.(char(MeasFieldnames(1)))), ' image sets total.'])
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
                OutputFileOverwrite = exist([cd,'/',FileName],'file');
                if OutputFileOverwrite ~= 0
                    errordlg('A file with that name already exists in the directory containing the raw measurements file.  Repeat and choose a different file name.')
                else
                    %%% Extract the measurements.  Waitbar shows the percentage of image sets
                    %%% remaining.
                    WaitbarHandle = waitbar(0,'Extracting measurements...');
                    %%% Preallocate the variable Measurements.
                    Fieldname = cell2mat(MeasFieldnames(length(MeasFieldnames)));
                    Measurements(NumberOfImages,length(MeasFieldnames)) = {handles.Measurements.(Fieldname){NumberOfImages}};
                    %%% Finished preallocating the variable Measurements.
                    TimeStart = clock;
                    for imagenumber = 1:NumberOfImages
                        for FieldNumber = 1:length(MeasFieldnames)
                            Fieldname = cell2mat(MeasFieldnames(FieldNumber));
                            Measurements(imagenumber,FieldNumber) = {handles.Measurements.(Fieldname){imagenumber}};
                        end
                        for FileNameNumber = 1:length(FileFieldNames)
                            Fieldname = cell2mat(FileFieldNames(FileNameNumber));
                            Measurements(imagenumber,FieldNumber) = {handles.Pipeline.(Fieldname){imagenumber}};
                            FieldNumber = FieldNumber + 1;
                        end
                        for HeadingNumber = 1:length(HeadingNames)
                            Fieldname = cell2mat(HeadingNames(HeadingNumber));
                            Measurements(imagenumber, FieldNumber) = {handles.(Fieldname){imagenumber}};
                            FieldNumber = FieldNumber + 1;
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
function ExportCellByCellButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
%%% Determines the current directory so it can switch back when done.
CurrentDirectory = pwd;
cd(handles.Vworkingdirectory)
%%% Ask the user to choose the file from which to extract measurements.
[RawFileName, RawPathname] = uigetfile('*.mat','Select the raw measurements file');
if RawFileName == 0
    cd(CurrentDirectory);
    return
end
cd(RawPathname);
load(RawFileName);

Answer = questdlg('Do you want to export cell by cell data for all measurements from one image, or data from all images for one measurement?','','All measurements','All images','All measurements');

if strcmp(Answer, 'All images') == 1
    %%% Extract the fieldnames of cell by cell measurements from the handles structure. 
    Fieldnames = fieldnames(handles.Measurements);
    MeasFieldnames = Fieldnames(strncmp(Fieldnames,'Object',6)==1);
    %%% Error detection.
    if isempty(MeasFieldnames)
        errordlg('No measurements were found in the file you selected.  They would be found within the output file''s handles.Measurements structure preceded by ''Object''.')
        cd(CurrentDirectory);
        return
    end
    %%% Removes the 'Object' prefix from each name for display purposes.
    for Number = 1:length(MeasFieldnames)
        EditedMeasFieldnames{Number} = MeasFieldnames{Number}(7:end);
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
    MeasurementToExtract = ['Object', EditedMeasurementToExtract];
    TotalNumberImageSets = length(handles.Measurements.(MeasurementToExtract));
    Measurements = handles.Measurements.(MeasurementToExtract);
    
    %%% Extract the fieldnames of non-cell by cell measurements from the
    %%% handles structure. This will be used as headings for each column of
    %%% measurements.
    Fieldnames = fieldnames(handles.Pipeline);
    HeadingFieldnames = Fieldnames(strncmp(Fieldnames,'Filename',8)==1);
    %%% Error detection.
    if isempty(HeadingFieldnames)
        errordlg('No headings were found in the file you selected.  They would be found within the output file''s handles.Pipeline structure preceded by ''Filename''.')
        cd(CurrentDirectory);
        return
    end

    %%% Allows the user to select a heading name from the list.
    [Selection, ok] = listdlg('ListString',HeadingFieldnames, 'ListSize', [300 600],...
        'Name','Select measurement',...
        'PromptString','Choose a field to label each column of data with','CancelString','Cancel',...
        'SelectionMode','single');
    if ok == 0
        cd(CurrentDirectory);
        return
    end
    HeadingToDisplay = char(HeadingFieldnames(Selection));
    %%% Extracts the headings.
    ListOfHeadings = handles.Pipeline.(HeadingToDisplay);

    %%% Have the user choose which of image/cells should be rows/columns
    RowColAnswer = questdlg('Which layout do you want images and cells to follow in the exported data?  WARNING: Excel spreadsheets can only have 256 columns.','','Rows = Cells, Columns = Images','Rows = Images, Columns = Cells','Rows = Cells, Columns = Images');
    
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
    OutputFileOverwrite = exist([cd,'/',FileName],'file');
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
    
    TooWideForXLS = 0;
    if (strcmp(RowColAnswer, 'Rows = Images, Columns = Cells')),
      %%% Writes the data, row by row: one row for each image.
      for ImageNumber = 1:TotalNumberImageSets
        %%% Writes the heading in the first column.
        fwrite(fid, char(ListOfHeadings(ImageNumber)), 'char');
        %%% Tabs to the next column.
        fwrite(fid, sprintf('\t'), 'char');
        %%% Writes the measurements for that image in successive columns.
        if (length(Measurements{ImageNumber}) > 256),
          TooWideForXLS = 1;
        end
        fprintf(fid,'%d\t',Measurements{ImageNumber});
        %%% Returns to the next row.
        fwrite(fid, sprintf('\n'), 'char');
      end
    else
      %%% Writes the data, row by row: one column for each image.

      % Check for truncation
      if (TotalNumberImageSets > 255),
        TooWideForXLS = 1;
      end
      
      %%% Writes the heading in the first row.
      for ImageNumber = 1:TotalNumberImageSets
        fwrite(fid, char(ListOfHeadings(ImageNumber)), 'char');
        %%% Tabs to the next column.
        fwrite(fid, sprintf('\t'), 'char');
      end
      %%% Returns to the next row.
      fwrite(fid, sprintf('\n'), 'char');

      %%% find the number of cells in the largest set
      maxlength = 0;
      for ImageNumber = 1:TotalNumberImageSets
        maxlength = max(maxlength, length(Measurements{ImageNumber}));
      end
      
      for CellNumber = 1:maxlength,
        for ImageNumber = 1:TotalNumberImageSets
          if (length(Measurements{ImageNumber}) >= CellNumber),
            fprintf(fid, '%d\t', Measurements{ImageNumber}(CellNumber));
          else
            fprintf(fid, '\t');
          end
        end
        %%% Returns to the next row.
        fwrite(fid, sprintf('\n'), 'char');
      end
      %%% Closes the file
    end
    fclose(fid);
    
    if TooWideForXLS,
      helpdlg(['The file ', FileName, ' has been written to the directory where the raw measurements file is located.  WARNING: This file contains more than 256 columns, and will not be readable in Excel.'])
    else 
      helpdlg(['The file ', FileName, ' has been written to the directory where the raw measurements file is located.'])
    end

    
elseif strcmp(Answer, 'All measurements') == 1
    TotalNumberImageSets = handles.setbeinganalyzed;
    %%% Asks the user to specify which image set to export.
    Answers = inputdlg({['Enter the sample number to export. There are ', num2str(TotalNumberImageSets), ' total.']},'Choose samples to export',1,{'1'});
    if isempty(Answers{1})
        cd(CurrentDirectory);
        return
    end
    try ImageNumber = str2double(Answers{1});
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
    Fieldnames = fieldnames(handles.Measurements);
    MeasFieldnames = Fieldnames(strncmp(Fieldnames,'Object',6)==1);
    %%% Error detection.
    if isempty(MeasFieldnames)
        errordlg('No measurements were found in the file you selected.  They would be found within the output file''s handles.Measurements structure preceded by ''Object''.')
        cd(CurrentDirectory);
        return
    end
    
    %%% Extract the fieldnames of non-cell by cell measurements from the
    %%% handles structure. This will be used as headings for each column of
    %%% measurements.
    Fieldnames = fieldnames(handles.Pipeline);
    HeadingFieldnames = Fieldnames(strncmp(Fieldnames,'Filename',8)==1);
    %%% Error detection.
    if isempty(HeadingFieldnames)
        errordlg('No headings were found in the file you selected.  They would be found within the output file''s handles.Pipeline structure preceded by ''Filename''.')
        cd(CurrentDirectory);
        return
    end

    %%% Allows the user to select a heading name from the list.
    [Selection, ok] = listdlg('ListString',HeadingFieldnames, 'ListSize', [300 600],...
        'Name','Select measurement',...
        'PromptString','Choose a field to label this data.','CancelString','Cancel',...
        'SelectionMode','single');
    if ok == 0
        cd(CurrentDirectory);
        return
    end
    HeadingToDisplay = char(HeadingFieldnames(Selection));
    %%% Extracts the headings.
    ImageNamesToDisplay = handles.Pipeline.(HeadingToDisplay);
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
    OutputFileOverwrite = exist([cd,'/',FileName],'file');
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
        Measurements = handles.Measurements.(FieldName);
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
function HistogramButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
%%% Determines the current directory so it can switch back when done.
CurrentDirectory = pwd;
cd(handles.Vworkingdirectory)
%%% Ask the user to choose the file from which to extract measurements.
[RawFileName, RawPathname] = uigetfile('*.mat','Select the raw measurements file');
if RawFileName == 0
    cd(CurrentDirectory);
    return
end
cd(RawPathname);
load(RawFileName);
%%% Extract the fieldnames of measurements from the handles structure. 
Fieldnames = fieldnames(handles.Measurements);
MeasFieldnames = Fieldnames(strncmp(Fieldnames,'Object',6)==1);
%%% Error detection.
if isempty(MeasFieldnames)
    errordlg('No measurements were found in the file you selected.  They would be found within the output file''s handles.Measurements structure preceded by ''Object''.')
    cd(CurrentDirectory);
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
        try NumberOfBins = str2double(Answers{1});
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
        if strcmp(YAxisScale, 'relative') ~= 1 && strcmp(YAxisScale, 'absolute') ~= 1
            errordlg('The text you entered for ''Do you want the Y-axis (number of cells) to be absolute or relative?'' was not recognized.');
            cd(CurrentDirectory);
            return
        end
        CompressedHistogram = Answers{6};
        if strcmp(CompressedHistogram,'yes') ~= 1 && strcmp(CompressedHistogram,'no') ~= 1 
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
        OutputFileOverwrite = exist([cd,'/',SaveData],'file');
        if OutputFileOverwrite ~= 0
            Answer = questdlg('A file with that name already exists in the directory containing the raw measurements file. Do you wish to overwrite?','Confirm overwrite','Yes','No','No');
            if strcmp(Answer, 'No') == 1
                cd(CurrentDirectory);
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
                cd(CurrentDirectory);
                return
            end
        else MinHistogramValue = str2num(MinHistogramValue); %#ok
        end
        if isempty(str2num(MaxHistogramValue)) %#ok
            if strcmp(MaxHistogramValue,'automatic') == 1
                MaxHistogramValue = PotentialMaxHistogramValue;
            else
                errordlg('The value entered for the maximum histogram value must be either a number or the word ''automatic''.')
                cd(CurrentDirectory);
                return
            end
        else MaxHistogramValue = str2num(MaxHistogramValue); %#ok
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
            FigureSettings{3} = FinalHistogramData

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
            Button9Callback = 'FigureSettings = get(gca,''UserData''); PlotBinLocations = FigureSettings{1}; PreXTickLabels = FigureSettings{2}, XTickLabels = PreXTickLabels(2:end-1), AxesHandles = findobj(gcf, ''Type'', ''axes''); set(AxesHandles,''XTick'',PlotBinLocations); NumberOfDecimals = inputdlg(''Enter the number of decimal places to display'',''Enter the number of decimal places'',1,{''0''}); NumberValues = cell2mat(XTickLabels), Command = [''%.'',num2str(NumberOfDecimals{1}),''f'']; NewNumberValues = num2str(NumberValues'',Command), NewNumberValuesPlusFirstLast = [char(PreXTickLabels(1)); NewNumberValues; char(PreXTickLabels(end-1))], CellNumberValues = cellstr(NewNumberValuesPlusFirstLast); set(AxesHandles,''XTickLabel'',CellNumberValues), drawnow'
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
            cd(CurrentDirectory);
            return
        end
    end % Goes with cancel button when selecting the measurement to display.
end
cd(CurrentDirectory);

%%%%%%%%%%%%%%%%%

% --- Executes on button press in NormalizationButton.
function NormalizationButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
h = msgbox('Copy your data to the clipboard then press OK');
waitfor(h)

uiimport('-pastespecial');
h = msgbox('After importing your data and pressing "Finish", click OK');
waitfor(h)
if exist('clipboarddata','var') == 0
    return
end
IncomingData = clipboarddata;

Prompts = {'Enter the number of rows','Enter the number of columns','Enter the percentile below which values will be excluded from fitting the normalization function.','Enter the percentile above which values will be excluded from fitting the normalization function.'};
Defaults = {'24','16','.05','.95'};
Answers = inputdlg(Prompts,'Describe Array/Slide Format',1,Defaults);
if isempty(Answers)
    return
end
NumberRows = str2double(Answers{1});
NumberColumns = str2double(Answers{2});
LowPercentile = str2double(Answers{3});
HighPercentile = str2double(Answers{4});
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
OrigData = reshape(MeanImage,TotalSamplesToBeGridded,1) %#ok We want to ignore MLint error checking for this line.
CorrFactors = reshape(IlluminationImage2,TotalSamplesToBeGridded,1);
CorrectedData = OrigData./CorrFactors %#ok We want to ignore MLint error checking for this line.

msgbox('The original data and the corrected data are now displayed in the Matlab window. You can cut and paste from there.')

% %%% Exports the results to the clipboard.
% clipboard('copy',CorrFactors);
% h = msgbox('The correction factors are now on the clipboard. Paste them where desired and press OK.  The data is also displayed in column format in the main Matlab window, so you can copy and paste from there as well.');
% waitfor(h)
% clipboard('copy',OrigData);
% h = msgbox('The original data used to generate those normalization factors is now on the clipboard. Paste them where desired (if desired) and press OK.  The data is also displayed in column format in the main Matlab window, so you can copy and paste from there as well.');
% waitfor(h)

%%%%%%%%%%%%%%%%%

% --- Executes on button press in ShowDataOnImageButton.
function ShowDataOnImageButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
%%% Determines the current directory so it can switch back when done.
CurrentDirectory = pwd;
cd(handles.Vworkingdirectory)
%%% Asks the user to choose the file from which to extract measurements.
[RawFileName, RawPathname] = uigetfile('*.mat','Select the raw measurements file');
if RawFileName ~= 0
    cd(RawPathname);
    load(RawFileName);
    %%% Extracts the fieldnames of measurements from the handles structure. 
    Fieldnames = fieldnames(handles.Measurements);
    MeasFieldnames = Fieldnames(strncmp(Fieldnames,'Object',6)==1);
    %%% Error detection.
    if isempty(MeasFieldnames)
        errordlg('No measurements were found in the file you selected.  They would be found within the output file''s handles.Measurements structure preceded by ''Object''.')
        cd(CurrentDirectory);
        return
    else
        %%% Removes the 'Object' prefix from each name for display purposes.
        for Number = 1:length(MeasFieldnames)
            EditedMeasFieldnames{Number} = MeasFieldnames{Number}(7:end);
        end
        %%% Allows the user to select a measurement from the list.
        [Selection, ok] = listdlg('ListString',EditedMeasFieldnames, 'ListSize', [300 600],...
            'Name','Select measurement',...
            'PromptString','Choose a measurement to display on the image','CancelString','Cancel',...
            'SelectionMode','single');
        if ok ~= 0
            EditedMeasurementToExtract = char(EditedMeasFieldnames(Selection));
            MeasurementToExtract = ['Object', EditedMeasurementToExtract];
            %%% Allows the user to select the X Locations from the list.
            [Selection, ok] = listdlg('ListString',EditedMeasFieldnames, 'ListSize', [300 600],...
                'Name','Select the X locations to be used',...
                'PromptString','Select the X locations to be used','CancelString','Cancel',...
                'SelectionMode','single');
            if ok ~= 0
                EditedXLocationMeasurementName = char(EditedMeasFieldnames(Selection));
                XLocationMeasurementName = ['Object', EditedXLocationMeasurementName];
                %%% Allows the user to select the Y Locations from the list.
                [Selection, ok] = listdlg('ListString',EditedMeasFieldnames, 'ListSize', [300 600],...
                    'Name','Select the Y locations to be used',...
                    'PromptString','Select the Y locations to be used','CancelString','Cancel',...
                    'SelectionMode','single');
                if ok ~= 0
                    EditedYLocationMeasurementName = char(EditedMeasFieldnames(Selection));
                    YLocationMeasurementName = ['Object', EditedYLocationMeasurementName];
                    %%% Prompts the user to choose a sample number to be displayed.
                    Answer = inputdlg({'Which sample number do you want to display?'},'Choose sample number',1,{'1'});
                    if isempty(Answer)
                        cd(CurrentDirectory);
                        return
                    end
                    SampleNumber = str2double(Answer{1});
                    TotalNumberImageSets = length(handles.Measurements.(MeasurementToExtract));
                    if SampleNumber > TotalNumberImageSets
                        cd(CurrentDirectory);
                        error(['The number you entered exceeds the number of samples in the file.  You entered ', num2str(SampleNumber), ' but there are only ', num2str(TotalNumberImageSets), ' in the file.'])
                    end
                    %%% Looks up the corresponding image file name.
                    Fieldnames = fieldnames(handles.Measurements);
                    PotentialImageNames = Fieldnames(strncmp(Fieldnames,'Filename',8)==1);
                    %%% Error detection.
                    if isempty(PotentialImageNames)
                        errordlg('CellProfiler was not able to look up the image file names used to create these measurements to help you choose the correct image on which to display the results. You may continue, but you are on your own to choose the correct image file.')
                    end
                    %%% Allows the user to select a filename from the list.
                    [Selection, ok] = listdlg('ListString',PotentialImageNames, 'ListSize', [300 600],...
                        'Name','Choose the image whose filename you want to display',...
                        'PromptString','Choose the image whose filename you want to display','CancelString','Cancel',...
                        'SelectionMode','single');
                    if ok ~= 0
                        SelectedImageName = char(PotentialImageNames(Selection));
                        ImageFileName = handles.Measurements.(SelectedImageName){SampleNumber};
                        %%% Prompts the user with the image file name.
                        h = msgbox(['Browse to find the image called ', ImageFileName,'.']);
                        %%% Opens a user interface window which retrieves a file name and path 
                        %%% name for the image to be displayed.
                        [FileName,Pathname] = uigetfile('*.*','Select the image to view');
                        delete(h)
                        %%% If the user presses "Cancel", the FileName will = 0 and nothing will
                        %%% happen.
                        if FileName == 0
                            cd(CurrentDirectory);
                            return
                        else
                            %%% Opens and displays the image, with pixval shown.
                            ImageToDisplay = im2double(imread([Pathname,'/',FileName]));
                            %%% Allows underscores to be displayed properly.
                            ImageFileName = strrep(ImageFileName,'_','\_');
                            FigureHandle = figure; imagesc(ImageToDisplay), colormap(gray), title([EditedMeasurementToExtract, ' on ', ImageFileName])
                            %%% Extracts the XY locations and the measurement values.
                            global StringListOfMeasurements
                            ListOfMeasurements = handles.Measurements.(MeasurementToExtract){SampleNumber};
                            StringListOfMeasurements = cellstr(num2str(ListOfMeasurements));
                            Xlocations(:,FigureHandle) = handles.Measurements.(XLocationMeasurementName){SampleNumber};
                            Ylocations(:,FigureHandle) = handles.Measurements.(YLocationMeasurementName){SampleNumber};
                            %%% A button is created in the display window which
                            %%% allows altering the properties of the text.
                            StdUnit = 'point';
                            StdColor = get(0,'DefaultUIcontrolBackgroundColor');
                            PointsPerPixel = 72/get(0,'ScreenPixelsPerInch');                            
                            DisplayButtonCallback1 = 'global TextHandles, FigureHandle = gcf; CurrentTextHandles = TextHandles{FigureHandle}; try, propedit(CurrentTextHandles,''v6''); catch, msgbox(''A bug in Matlab is preventing this function from working. Service Request #1-RR6M1''), end; drawnow, clear TextHandles';
                            uicontrol('Parent',FigureHandle, ...
                                'Unit',StdUnit, ...
                                'BackgroundColor',StdColor, ...
                                'CallBack',DisplayButtonCallback1, ...
                                'Position',PointsPerPixel*[2 2 90 22], ...
                                'Units','Normalized',...
                                'String','Text Properties', ...
                                'Style','pushbutton');
                            DisplayButtonCallback2 = 'global TextHandles StringListOfMeasurements, FigureHandle = gcf; NumberOfDecimals = inputdlg(''Enter the number of decimal places to display'',''Enter the number of decimal places'',1,{''0''}); CurrentTextHandles = TextHandles{FigureHandle}; NumberValues = str2num(cell2mat(StringListOfMeasurements)); Command = [''%.'',num2str(NumberOfDecimals{1}),''f'']; NewNumberValues = num2str(NumberValues,Command); CellNumberValues = cellstr(NewNumberValues); PropName(1) = {''string''}; set(CurrentTextHandles,PropName, CellNumberValues); drawnow, clear TextHandles StringListOfMeasurements';
                            uicontrol('Parent',FigureHandle, ...
                                'Unit',StdUnit, ...
                                'BackgroundColor',StdColor, ...
                                'CallBack',DisplayButtonCallback2, ...
                                'Position',PointsPerPixel*[100 2 135 22], ...
                                'Units','Normalized',...
                                'String','Fewer significant digits', ...
                                'Style','pushbutton');
                            DisplayButtonCallback3 = 'global TextHandles StringListOfMeasurements, FigureHandle = gcf; CurrentTextHandles = TextHandles{FigureHandle}; PropName(1) = {''string''}; set(CurrentTextHandles,PropName, StringListOfMeasurements); drawnow, clear TextHandles StringListOfMeasurements';
                            uicontrol('Parent',FigureHandle, ...
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

% --- Executes on button press in AnalyzeImagesButton.
function AnalyzeImagesButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
global closeFigures openFigures;
CurrentDirectory = cd;
%%% Checks whether any algorithms are loaded.
sum = 0;
for i = 1:handles.numAlgorithms;
    sum = sum + iscellstr(handles.Settings.Valgorithmname(i));
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
            OutputFileOverwrite = exist([cd,'/',handles.Voutputfilename],'file');
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
                %%% get messed up.  The Help buttons are left enabled.
                set(handles.BrowseToLoad,'enable','off')
                set(handles.PathToLoadEditBox,'enable','inactive','foregroundcolor',[0.7,0.7,0.7])
                set(handles.LoadSampleInfo,'enable','off')
                set(handles.ClearSampleInfo,'enable','off')
                set(handles.ViewSampleInfo,'enable','off')
                set(handles.OutputFileName,'enable','inactive','foregroundcolor',[0.7,0.7,0.7])
                set(handles.SetPreferencesButton,'enable','off')
                set(handles.PixelSizeEditBox,'enable','inactive','foregroundcolor',[0.7,0.7,0.7])
                set(handles.LoadSettingsFromFileButton,'enable','off')
                set(handles.SaveSettingsButton,'enable','off')
                set(handles.AddAlgorithm,'visible','off');
                set(handles.RemoveAlgorithm,'visible','off');
                set(handles.MoveUpButton,'visible','off');
                set(handles.MoveDownButton,'visible','off');

                % FIXME: This should loop just over the number of actual variables in the display.
                for VariableNumber=1:handles.MaxVariables;
                    set(handles.(['VariableBox' TwoDigitString(VariableNumber)]),'enable','inactive','foregroundcolor',[0.7,0.7,0.7]);
                end
                set(handles.ListBox,'enable','off')
                set(handles.TechnicalDiagnosisButton,'enable','off')
                set(handles.SaveImageAsButton,'enable','off')
                set(handles.ShowImageButton,'enable','off')
                set(handles.ShowPixelDataButton,'enable','off')
                set(handles.CloseAllFigureWindowsButton,'enable','off')
                set(handles.AnalyzeImagesButton,'enable','off')
                set(handles.ExportDataButton,'enable','off')
                set(handles.ExportCellByCellButton,'enable','off')
                set(handles.HistogramButton,'enable','off')
                set(handles.NormalizationButton,'enable','off')
                set(handles.ShowDataOnImageButton,'enable','off')

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
                %%% Creates the text box within the timer window which will display the
                %%% timer text.
                text_handle = uicontrol(timer_handle,'string',timertext,'style','text',...
                    'parent',timer_handle,'position', [0 40 494 64],'FontName','Times',...
                    'FontSize',14,'FontWeight','bold','BackgroundColor',[0.7,0.7,0.9]);
                %%% Saves text handle to the handles structure.
                handles.timertexthandle = text_handle;
                %%% Creates the Cancel and Pause buttons.
                PauseButton_handle = uicontrol('Style', 'pushbutton', ...
                    'String', 'Pause', 'Position', [5 10 40 30], ...
                    'parent',timer_handle, 'BackgroundColor',[0.7,0.7,0.9]);
                CancelAfterImageSetButton_handle = uicontrol('Style', 'pushbutton', ...
                    'String', 'Cancel after image set', 'Position', [50 10 120 30], ...
                    'parent',timer_handle, 'BackgroundColor',[0.7,0.7,0.9]);
                CancelAfterModuleButton_handle = uicontrol('Style', 'pushbutton', ...
                    'String', 'Cancel after module', 'Position', [175 10 115 30], ...
                    'parent',timer_handle, 'BackgroundColor',[0.7,0.7,0.9]);
                CancelNowCloseButton_handle = uicontrol('Style', 'pushbutton', ...
                    'String', 'Cancel now & close CellProfiler', 'Position', [295 10 160 30], ...
                    'parent',timer_handle, 'BackgroundColor',[0.7,0.7,0.9]);
                %%% Sets the functions to be called when the Cancel and Pause buttons
                %%% within the Timer window are pressed.
                PauseButtonFunction = 'h = msgbox(''Image processing is paused without causing any damage. Processing will restart when you close the Pause window or click OK.''); waitfor(h); clear h;';
                set(PauseButton_handle,'Callback', PauseButtonFunction)
                CancelAfterImageSetButtonFunction = ['deleteme = questdlg(''Paused. Are you sure you want to cancel after this image set? Processing will continue on the current image set, the data up to and including the current image set will be saved in the output file, and then the analysis will be canceled.'', ''Confirm close'',''Yes'',''No'',''Yes''); switch deleteme; case ''Yes''; set(',num2str(CancelAfterImageSetButton_handle*8192), '/8192,''enable'',''off''); set(', num2str(text_handle*8192), '/8192,''string'',''Canceling in progress; Waiting for the processing of current image set to be complete. You can press the Cancel after module button to cancel more quickly, but data relating to the current image set will not be saved in the output file.''); case ''No''; return; end; clear deleteme'];
                set(CancelAfterImageSetButton_handle, 'Callback', CancelAfterImageSetButtonFunction)
                CancelAfterModuleButtonFunction = ['deleteme = questdlg(''Paused. Are you sure you want to cancel after this module? Processing will continue until the current image analysis module is completed, to avoid corrupting the current settings of CellProfiler. Data up to the *previous* image set are saved in the output file and processing is canceled.'', ''Confirm close'',''Yes'',''No'',''Yes''); switch deleteme; case ''Yes''; set(', num2str(CancelAfterImageSetButton_handle*8192), '/8192,''enable'',''off''); set(', num2str(CancelAfterModuleButton_handle*8192), '/8192,''enable'',''off''); set(', num2str(text_handle*8192), '/8192,''string'',''Immediate canceling in progress; Waiting for the processing of current module to be complete in order to avoid corrupting the current CellProfiler settings.''); case ''No''; return; end; clear deleteme'];
                set(CancelAfterModuleButton_handle,'Callback', CancelAfterModuleButtonFunction)
                CancelNowCloseButtonFunction = ['deleteme = questdlg(''Paused. Are you sure you want to cancel immediately and close CellProfiler? The CellProfiler program will close, losing your current settings. The data up to the *previous* image set will be saved in the output file, but the current image set data will be stored incomplete in the output file, which might be confusing when using the output file.'', ''Confirm close'',''Yes'',''No'',''Yes''); helpdlg(''The CellProfiler program should have closed itself. Important: Go to the command line of Matlab and press Control-C to stop processes in progress. Then type clear and press the enter key at the command line.  Figure windows will not close properly: to close them, type delete(N) at the command line of Matlab, where N is the figure number. The data up to the *previous* image set will be saved in the output file, but the current image set data will be stored incomplete in the output file, which might be confusing when using the output file.''), switch deleteme; case ''Yes''; delete(', num2str((handles.figure1)*8192), '/8192); case ''No''; return; end; clear deleteme'];
                set(CancelNowCloseButton_handle,'Callback', CancelNowCloseButtonFunction)
                HelpButtonFunction = 'msgbox(''Pause button: The current processing is immediately suspended without causing any damage. Processing restarts when you close the Pause window or click OK. Cancel after image set: Processing will continue on the current image set, the data up to and including the current image set will be saved in the output file, and then the analysis will be canceled.  Cancel after module: Processing will continue until the current image analysis module is completed, to avoid corrupting the current settings of CellProfiler. Data up to the *previous* image set are saved in the output file and processing is canceled. Cancel now & close CellProfiler: CellProfiler will immediately close itself. The data up to the *previous* image set will be saved in the output file, but the current image set data will be stored incomplete in the output file, which might be confusing when using the output file.'')';
                %%% HelpButton
                uicontrol('Style', 'pushbutton', ...
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
                CloseFunction = ['deleteme = questdlg(''DO NOT CLOSE the Timer window while image processing is in progress!! Are you sure you want to close the timer?'', ''Confirm close'',''Yes'',''No'',''Yes''); switch deleteme; case ''Yes''; delete(',num2str(timer_handle*8192), '/8192); case ''No''; return; end; clear deleteme'];
                set(timer_handle,'CloseRequestFcn',CloseFunction)
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
                %%% before closing the window, to avoid unexpected results.              
                set(handles.CloseFigureButton,'visible','on')
                set(handles.OpenFigureButton,'visible','on')
                %listbox changes

                for i=1:handles.numAlgorithms;
                    if iscellstr(handles.Settings.Valgorithmname(i)) == 1
                        handles.(['figurealgorithm' TwoDigitString(i)]) = ...
                            figure('name',[char(handles.Settings.Valgorithmname(i)), ' Display'], 'Position',[(ScreenWidth*((i-1)/12)) (ScreenHeight-522) 560 442],'color',[0.7,0.7,0.7]);
                    end
                end

                %%% For the first time through, the number of image sets
                %%% will not yet have been determined.  So, the Number of
                %%% image sets is set temporarily.
                handles.Vnumberimagesets = 1;
                handles.setbeinganalyzed = 1;
                %%% Marks the time that analysis was begun.
                handles.Vtimestarted = datestr(now);
                %%% Clear the buffers (Pipeline and Measurements)
                handles.Pipeline = struct;
                handles.Measurements = struct;
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

                    for SlotNumber = 1:handles.numAlgorithms,
                        %%% If an algorithm is not chosen in this slot, continue on to the next.
                        AlgNumberAsString = TwoDigitString(SlotNumber);
                        AlgName = char(handles.Settings.Valgorithmname(SlotNumber));
                        if iscellstr(handles.Settings.Valgorithmname(SlotNumber)) == 0
                        else
                            %%% Saves the current algorithm number in the handles structure.
                            handles.currentalgorithm = AlgNumberAsString;
                            %%% The try/catch/end set catches any errors that occur during the
                            %%% running of algorithm 1, notifies the user, breaks out of the image
                            %%% analysis loop, and completes the refreshing process.
                            try
                                %%% Runs the appropriate algorithm, with the handles structure as an
                                %%% input argument and as the output argument.
                                eval(['handles = Alg',AlgName,'(handles);'])
                            catch
                                if exist(['Alg',AlgName,'.m'],'file') ~= 2,
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
                                            
                        openFig = openFigures;
                        openFigures = [];
                        for i=1:length(openFig),
                            algNumber = openFig(i);
                            try
                                Thisfigurealgorithm = handles.(['figurealgorithm' TwoDigitString(algNumber)]);
                                figure(Thisfigurealgorithm);
                                set(Thisfigurealgorithm, 'name',[(char(handles.Settings.Valgorithmname(algNumber))), ' Display']);
                                set(Thisfigurealgorithm, 'Position',[(ScreenWidth*((algNumber-1)/12)) (ScreenHeight-522) 560 442]);
                                set(Thisfigurealgorithm,'color',[0.7,0.7,0.7]);
                                %%% Sets the closing function of the window appropriately. (See way
                                %%% above where 'ClosingFunction's are defined).
                                %set(Thisfigurealgorithm,'CloseRequestFcn',eval(['ClosingFunction' TwoDigitString(algNumber)]));
                            catch
                            end
                        end
                    end %%% ends loop over slot number

                    closeFig = closeFigures;
                    closeFigures = [];
                    for i=1:length(closeFig),
                        algNumber = closeFig(i);
                        try
                            Thisfigurealgorithm = handles.(['figurealgorithm' TwoDigitString(algNumber)]);
                            delete(Thisfigurealgorithm);
                        catch
                        end
                    end
                    
                    openFig = openFigures;
                    openFigures = [];
                    for i=1:length(openFig),
                        algNumber = openFig(i);
                        try
                            Thisfigurealgorithm = handles.(['figurealgorithm' TwoDigitString(algNumber)]);
                            figure(Thisfigurealgorithm);
                            set(Thisfigurealgorithm, 'name',[(char(handles.Settings.Valgorithmname(algNumber))), ' Display']);
                            set(Thisfigurealgorithm, 'Position',[(ScreenWidth*((algNumber-1)/12)) (ScreenHeight-522) 560 442]);
                            set(Thisfigurealgorithm,'color',[0.7,0.7,0.7]);
                            %%% Sets the closing function of the window appropriately. (See way
                            %%% above where 'ClosingFunction's are defined).
                            %set(Thisfigurealgorithm,'CloseRequestFcn',eval(['ClosingFunction' TwoDigitString(algNumber)]));
                        catch
                        end
                    end
                    


                    if (break_outer_loop),
                        break;  %%% this break is out of the outer loop of image analysis
                    end


% PARALLEL DEAD CODE
%                     %%% Get a list of the measurement fields (after the first pass has run through
%                     %%% all the modules)
%                     Fields = fieldnames(handles);
%                     mFields = (strncmp(Fields,'dM',2) | strncmp(Fields,'dOTFilename',11));
%                     MeasurementFields = Fields(mFields);
%                     
%                    % If we are using parallel machines, copy the handles structure to them.
%                    if (isfield(handles, 'parallel_machines')),
%                        handles_culled = handles;
%                        deleteFields = strncmp(Fields,'dOT',2);
%                        keepFields = strncmp(Fields,'dOTFileList',11) | ...
%                            strncmp(Fields,'dOTPathname',11) | strncmp(Fields,'dOTFilename',11) | ...
%                            strncmp(Fields,'dOTIllumImage',13) | strncmp(Fields,'dOTIntensityToShift',19) | ...
%                            strncmp(Fields,'dOTTimeElapsed',14);
%                        handles_culled = rmfield(handles_culled, Fields(deleteFields & (~keepFields)));
%                        % Make sure all the functions are cleared, so
%                        % that any changes to the modules are noticed.
%                        pnet_remote(handles.parallel_machines, 'EVAL', 'clear functions')
%                        pnet_remote(handles.parallel_machines, 'PUT', 'handles', handles_culled);
%                    end
                    
                    
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
                    timertext = {timer_elapsed_text; number_analyzed; time_per_set};
                    %%% Display calculations in 
                    %%% the "Timer" window by changing the string property.
                    set(text_handle,'string',timertext)
                    drawnow    
                    %%% Save the time elapsed so far in the handles structure.
                    %%% Check first to see that the set being analyzed is not zero, or else an
                    %%% error will be produced when trying to do this.
                    if setbeinganalyzed ~= 0
                        handles.Measurements.TimeElapsed{setbeinganalyzed} = toc;
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
                    IsWrongNumber = (Lengths ~= setbeinganalyzed - 1);
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
                timertext = {'IMAGE PROCESSING IS COMPLETE!';total_time_elapsed; number_analyzed; time_per_set};
                set(text_handle,'string',timertext)
                set(timer_handle,'CloseRequestFcn','closereq')
                
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
                set(handles.SaveSettingsButton,'enable','on')
                %listbox changes
                for AlgorithmNumber=1:handles.numAlgorithms;
                    for VariableNumber = 1:handles.numVariables(AlgorithmNumber);
                        set(handles.(['VariableBox' TwoDigitString(VariableNumber)]),'enable','on','foregroundcolor','black');
                    end
                end
                set(handles.AddAlgorithm,'visible','on');
                set(handles.RemoveAlgorithm,'visible','on');
                set(handles.MoveUpButton,'visible','on');
                set(handles.MoveDownButton,'visible','on');
                set(handles.CloseFigureButton,'visible','off');
                set(handles.OpenFigureButton,'visible','off');
                set(handles.ListBox,'enable','on')
                set(handles.TechnicalDiagnosisButton,'enable','on')
                set(handles.SaveImageAsButton,'enable','on')
                set(handles.ShowImageButton,'enable','on')
                set(handles.ShowPixelDataButton,'enable','on')
                set(handles.CloseAllFigureWindowsButton,'enable','on')
                set(handles.AnalyzeImagesButton,'enable','on')
                set(handles.ExportDataButton,'enable','on')
                set(handles.ExportCellByCellButton,'enable','on')                
                set(handles.HistogramButton,'enable','on')
                set(handles.NormalizationButton,'enable','on')
                set(handles.ShowDataOnImageButton,'enable','on')
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
    ErrorExplanation = ['There was a problem running the image analysis. Sorry, it is unclear what the problem is. It would be wise to close the entire CellProfiler program in case something strange has happened to the settings. The output file may be unreliable as well. Matlab says the error is: ', Error, ' in module ', CurrentAlgorithmNumber];
end
errordlg(ErrorExplanation)

%%%%%%%%%%%%%%%%%

% --- Executes on button press in AnalyzeAllImagesClusterButton.
function AnalyzeAllImagesClusterButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
Prompts = {'Path to CellProfiler on the remote machine(s)','Path to the images on the remote machine(s)','File containing the list of remote machine(s)'};

% set up default values for the answers
if (~ isfield(handles, 'RemoteCellProfilerPathname'))
  LocationOfGUI = which('CellProfiler');
  Slashes = findstr(LocationOfGUI, '/');
  handles.RemoteCellProfilerPathname = LocationOfGUI(1:Slashes(end));
  handles.RemoteImagePathname = handles.Vpathname;
  handles.RemoteMachineListFile = LocationOfGUI(1:Slashes(end));
end

% pop up the dialog
Defaults = {handles.RemoteCellProfilerPathname,handles.RemoteImagePathname,handles.RemoteMachineListFile};
Answers = inputdlg(Prompts,'Provide cluster information',1,Defaults,'on');

if isempty(Answers)
    return
end

% Store the answers as new defaults
handles.RemoteCellProfilerPathname = Answers{1};
handles.RemoteImagePathname = Answers{2};
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
  RemoteMachine = fgetl(fid) %#ok We want to ignore MLint error checking for this line.
  % We should put up a dialog here with a CANCEL button.  Also need to
  % modify pnet_remote to return after a few retries, rather than just
  % giving up.
  if (~ ischar(RemoteMachine)),
    break
  end
  if (~ isempty(RemoteMachine)),
    handles.parallel_machines(length(handles.parallel_machines)+1) = pnet_remote('connect', RemoteMachine);
  end
end

if isempty(handles.parallel_machines)
  errordlg(['CellProfiler could not connetct to any remote machines.  Is the list of machines an empty file (' handles.RemoteMachineListFile ')?']);
  handles = rmfield(handles, 'parallel_machines');
  guidata(hObject, handles);
  return;
end

% set up the path on the remote machines
pnet_remote(handles.parallel_machines, 'eval', ['addpath ' handles.RemoteCellProfilerPathname]);

% fake a click on the analyze images button
AnalyzeImagesButton_Callback(hObject, eventdata, handles);

% clear the list of parallel machines
handles = rmfield(handles, 'parallel_machines');
guidata(hObject, handles);



%%%%%%%%%%%%%%%%%%%%%
%%% Aux Functions %%%
%%%%%%%%%%%%%%%%%%%%%

function twodigit = TwoDigitString(val)
%TwoDigitString is a function like num2str(int) but it returns a two digit
%representation of a string for our purposes.
if ((val > 99) || (val < 0)),
  error(['TwoDigitString: Can''t convert ' num2str(val) ' to a 2 digit number']);
end
twodigit = sprintf('%02d', val);

%%%%%%%%%%%%%%%%%%%
%%% HELP BUTTONS %%%
%%%%%%%%%%%%%%%%%%%

%%% --- Executes on button press in the permanent Help buttons.
%%% (The permanent Help buttons are the ones that don't change 
%%% depending on the algorithm loaded.) 
function HelpButton1_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
HelpText = help('Help1.m');
helpdlg(HelpText,'CellProfiler Help #1')

function HelpButton2_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
HelpText = help('Help2.m');
msgbox(HelpText,'CellProfiler Help #2')

function HelpButton3_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
HelpText = help('Help3.m');
helpdlg(HelpText,'CellProfiler Help #3')

function HelpButton4_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
HelpText = help('Help4.m');
helpdlg(HelpText,'CellProfiler Help #4')

% --- Executes on button press in HelpExportMeanDataButton.
function HelpExportMeanDataButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
HelpText = help('HelpExportMeanData.m');
helpdlg(HelpText,'CellProfiler Help: Export mean data')

% --- Executes on button press in HelpExportCellByCellDataButton.
function HelpExportCellByCellDataButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
HelpText = help('HelpExportCellByCellData.m');
helpdlg(HelpText,'CellProfiler Help: Export cell by cell data')

% --- Executes on button press in HelpShowDataOnImageButton.
function HelpShowDataOnImageButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
HelpText = help('HelpShowDataOnImage.m');
helpdlg(HelpText,'CellProfiler Help: Show data on image')

% --- Executes on button press in HelpHistogramsButton.
function HelpHistogramsButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
HelpText = help('HelpHistograms.m');
helpdlg(HelpText,'CellProfiler Help: Histograms')

% --- Executes on button press in HelpNormalizationButton.
function HelpNormalizationButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
HelpText = help('HelpNormalization.m');
helpdlg(HelpText,'CellProfiler Help: Normalization')

function HelpAnalyzeImagesButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
HelpText = help('HelpAnalyzeImages.m');
helpdlg(HelpText,'CellProfiler Help: Analyze images')

% --- Executes on button press in HelpForThisAnalysisModule.  
function HelpForThisAnalysisModule_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
%%% First, check to see whether there is a specific algorithm loaded.
%%% If not, it opens a help dialog which explains how to pick one.
AlgorithmNumber = whichactive(handles);
if AlgorithmNumber == 0
    helpdlg('You do not have an analysis module selected.  Click "?" next to "Image analysis settings" to get help in choosing an analysis module, or click "View" next to an analysis module that has been loaded already.','Help for choosing an analysis module')
else
    AlgorithmName = handles.Settings.Valgorithmname(AlgorithmNumber);
    IsItNotChosen = strncmp(AlgorithmName,'No a',4);
    if IsItNotChosen == 1
        helpdlg('You do not have an analysis module selected.  Click "?" next to "Image analysis settings" to get help in choosing an analysis module, or click "View" next to an analysis module that has been loaded already.','Help for choosing an analysis module')
    else
        %%% This is the function that actually reads the algorithm's help
        %%% data.
        AlgorithmNoDotM = strcat('Alg',AlgorithmName);
        HelpText = help(char(AlgorithmNoDotM));
        DoesHelpExist = exist('HelpText','var');
        if DoesHelpExist == 1
            helpFig = figure;
            set(helpFig,'NumberTitle','off');
            set(helpFig,'name', 'Image analysis module help');
            set(helpFig,'units','characters','color',[0.7 0.7 0.9]);
            helpFigPos = get(helpFig,'position');
            set(helpFig,'position',[helpFigPos(1),helpFigPos(2),87,helpFigPos(4)]);
            
            helpUI = uicontrol(...
                'Parent',helpFig,...
                'Enable','inactive',...
                'Units','characters',...
                'HorizontalAlignment','left',...
                'Max',2,...
                'Min',0,...
                'Position',[1 1 helpFigPos(3) helpFigPos(4)],...
                'String',HelpText,...
                'BackgroundColor',[0.7 0.7 0.9],...
                'Style','text');
            outstring = textwrap(helpUI,{HelpText});
            set(helpUI,'position',[1 1.5+(27-length(outstring))*1.09 80 length(outstring)*1.09]);
            if(length(outstring) > 27),
            
                helpUIPosition = get(helpUI,'position');
                helpScrollCallback = ['set(',num2str(helpUI,'%.13f'),',''position'',[', ...
                    num2str(helpUIPosition(1)),' ',num2str(helpUIPosition(2)),'+get(gcbo,''max'')-get(gcbo,''value'') ', num2str(helpUIPosition(3)), ...
                    ' ', num2str(helpUIPosition(4)),'])'];
                    
                helpScrollUI = uicontrol(...
                    'Parent',helpFig,...
                    'Callback',helpScrollCallback,...
                    'Units','characters',...
                    'Visible', 'on',...
                                    'BackgroundColor',[0.7 0.7 0.9],...
                    'Style', 'slider',...
                    'Position',[81 1 4 30]);
                set(helpScrollUI,'max',(length(outstring)-27)*1.09);
                set(helpScrollUI,'value',(length(outstring)-27)*1.09);
            end
        else helpdlg('Sorry, there is no help information for this image analysis module.','Image analysis module help')
        end
    end
end

%%% ^ END OF HELP HELP HELP HELP HELP HELP BUTTONS ^ %%%

% --- Executes on slider movement.
function slider1_Callback(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider

scrollPos = get(hObject,'max') - get(hObject, 'Value');
set(handles.variablepanel, 'position', [46 5.3846+scrollPos 108.4 23.154]);


% --- Executes during object creation, after setting all properties.
function slider1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background, change
%       'usewhitebg' to 0 to use default.  See ISPC and COMPUTER.
usewhitebg = 1;
if usewhitebg
    set(hObject,'BackgroundColor',[.9 .9 .9]);
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end


function handles = createVariablePanel(handles)

for i=1:99,
    handles.(['VariableBox' TwoDigitString(i)]) = uicontrol(...
        'Parent',handles.variablepanel,...
        'Units','characters',...
        'BackgroundColor',[1 1 1],...
        'Callback','CellProfiler(''VariableBox_Callback'',gcbo,[],guidata(gcbo))',...
        'FontName','Times',...
        'FontSize',12,...
        'Position',[92 22.7-1.77*i 15.6 1.61538461538462],...
        'String','n/a',...
        'Style','edit',...
        'CreateFcn', 'CellProfiler(''VariableBox_CreateFcn'',gcbo,[],guidata(gcbo))',...
        'Tag',['VariableBox' TwoDigitString(i)],...
        'Behavior',get(0,'defaultuicontrolBehavior'),...
        'Visible','off');

    handles.(['VariableDescription' TwoDigitString(i)]) = uicontrol(...
        'Parent',handles.variablepanel,...
        'Units','characters',...
        'BackgroundColor',[0.699999988079071 0.699999988079071 0.899999976158142],...
        'CData',[],...
        'FontName','Times',...
        'FontSize',12,...
        'FontWeight','bold',...
        'HorizontalAlignment','right',...
        'Position',[0.1 22.7-1.77*i 90 1.30769230769231],...
        'String','No analysis module has been loaded',...
        'Style','text',...
        'Tag',['VariableDescription' TwoDigitString(i)],...
        'UserData',[],...
        'Behavior',get(0,'defaultuicontrolBehavior'),...
        'Visible','off',...
        'CreateFcn', '');
end


% --- Executes during object creation, after setting all properties.
function topPanelFrame_CreateFcn(hObject, eventdata, handles)
% hObject    handle to topPanelFrame (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


