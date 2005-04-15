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
%

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
%confi
% $Revision$


% Last Modified by GUIDE v2.5 19-Dec-2004 02:52:31
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

%%%%%%%%%%%%%%%%%%%%%%%
%%% INITIAL SETTINGS %%%
%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes just before CellProfiler is made visible.
function CellProfiler_OpeningFcn(hObject, eventdata, handles, varargin) %#ok We want to ignore MLint error checking for this line.

% Creates the variable panel.
handles = createVariablePanel(handles);

% Chooses default command line output for CellProfiler
handles.output = hObject;

%%% Creates variables for later use.
handles.Settings = struct;
handles.Pipeline = struct;
handles.Measurements = struct;
handles.Preferences = struct;
handles.Current.NumberOfModules = 0;

global closeFigures openFigures;
closeFigures = [];
openFigures = [];

%%% Determines the startup directory.
handles.Current.StartupDirectory = pwd;
addpath(pwd);

%%% Retrieves preferences from CellProfilerPreferences.mat, if possible.
%%% Try loading CellProfilerPreferences.mat first from the matlabroot
%%% directory and then the current directory.  This is not necessary for
%%% CellProfiler to function; it just allows defaults to be
%%% pre-loaded.
LoadedPreferencesExist = 0;
try
    load(fullfile(matlabroot,'CellProfilerPreferences.mat'))
    LoadedPreferences = SavedPreferences;
    LoadedPreferencesExist = 1;
    clear SavedPreferences
catch
    try 
        load(fullfile(handles.Current.StartupDirectory, CellProfilerPreferences))
        LoadedPreferences = SavedPreferences;
        LoadedPreferencesExist = 1;
        clear SavedPreferences
    end
end

%%% Stores some initial values in the handles structure based on the
%%% SavedPreferences, if they were successfully loaded.  Otherwise,
%%% defaults are used.
try handles.Preferences.PixelSize = LoadedPreferences.PixelSize;
catch
    %%% If not present in the loaded preferences, the pixel size shown
    %%% in the display is used (this is set in the CellProfiler.fig
    %%% file).
    handles.Preferences.PixelSize = get(handles.PixelSizeEditBox,'string');
end

try
    if exist(LoadedPreferences.DefaultModuleDirectory, 'dir')
        handles.Preferences.DefaultModuleDirectory = LoadedPreferences.DefaultModuleDirectory;
    end
end
%%% If the Default Module Directory has not yet been successfully
%%% identified (i.e., it is not present in the loaded preferences or
%%% the directory does not exist), look at where the CellProfiler.m
%%% file is located and see whether there is a subdirectory within
%%% that directory, called "Modules".  If so, use that subdirectory as
%%% the default module directory. If not, use the current directory.
if isfield(handles.Preferences,'DefaultModuleDirectory') == 0
    [CellProfilerPathname,FileName,ext,versn] = fileparts(which('CellProfiler'));
    %%% Checks whether the Modules subdirectory exists.
    if exist(fullfile(CellProfilerPathname,'Modules'), 'dir')
        CellProfilerModulePathname = fullfile(CellProfilerPathname,'Modules');
        handles.Preferences.DefaultModuleDirectory = CellProfilerModulePathname;
    else
        handles.Preferences.DefaultModuleDirectory = handles.Current.StartupDirectory;
    end
end
%%% Similar approach for the DefaultOutputDirectory.
try
    if exist(LoadedPreferences.DefaultOutputDirectory, 'dir')
        handles.Preferences.DefaultOutputDirectory = LoadedPreferences.DefaultOutputDirectory;
    end
end
if isfield(handles.Preferences,'DefaultOutputDirectory') == 0
    handles.Preferences.DefaultOutputDirectory = handles.Current.StartupDirectory;
end
%%% Similar approach for the DefaultImageDirectory.
try
    if exist(LoadedPreferences.DefaultImageDirectory, 'dir')
        handles.Preferences.DefaultImageDirectory = LoadedPreferences.DefaultImageDirectory;
    end
end
if isfield(handles.Preferences,'DefaultImageDirectory') == 0
    handles.Preferences.DefaultImageDirectory = handles.Current.StartupDirectory;
end

%%% Now that handles.Preferences.(4 different variables) has been filled
%%% in, the handles.Current values and edit box displays are set.
handles.Current.DefaultOutputDirectory = handles.Preferences.DefaultOutputDirectory;
handles.Current.DefaultImageDirectory = handles.Preferences.DefaultImageDirectory;
handles.Settings.PixelSize = handles.Preferences.PixelSize;
%%% (No need to set a current module directory or display it in an
%%% edit box; the one stored in preferences is the only one ever
%%% used).
set(handles.PixelSizeEditBox,'String',handles.Preferences.PixelSize)
set(handles.DefaultOutputDirectoryEditBox,'String',handles.Preferences.DefaultOutputDirectory)
set(handles.DefaultImageDirectoryEditBox,'String',handles.Preferences.DefaultImageDirectory)
%%% Retrieves the list of image file names from the chosen directory,
%%% stores them in the handles structure, and displays them in the
%%% filenameslistbox, by faking a click on the DefaultImageDirectoryEditBox.
handles = DefaultImageDirectoryEditBox_Callback(hObject, eventdata, handles);

%%% Adds the default module directory to Matlab's search path. Also
%%% adds the Modules subfolder of the folder that contains CellProfiler.m to
%%% Matlab's search path, if possible.
addpath(handles.Preferences.DefaultModuleDirectory)
[CellProfilerPathname,FileName,ext,versn] = fileparts(which('CellProfiler'));
CellProfilerModulePathname = fullfile(CellProfilerPathname,'Modules');
handles.Current.CellProfilerPathname = CellProfilerPathname;
try
    addpath(CellProfilerModulePathname)
end

%%% Sets a suitable fontsize. An automatic font size is calculated,
%%% but it is overridden if the user has set a default font size.
%%% The fontsize is also saved in the main window's (i.e. "0") UserData property so that
%%% it can be used for setting the fontsize in dialog boxes.
if exist('LoadedPreferences') && isfield(LoadedPreferences,'FontSize') && ~isempty(str2num(LoadedPreferences.FontSize))
    handles.Current.FontSize = str2num(LoadedPreferences.FontSize);
else
    ScreenResolution = get(0,'ScreenPixelsPerInch');
    handles.Current.FontSize = (220 - ScreenResolution)/13;       % 90 pix/inch => 10pts, 116 pix/inch => 8pts
end
names = fieldnames(handles);
for k = 1:length(names)
    if ishandle(handles.(names{k}))
        set(findobj(handles.(names{k}),'-property','FontSize'),'FontSize',handles.Current.FontSize,'FontName','Times')
    end
end
set(0,'UserData',handles.Current.FontSize)

%%% Checks whether the user has the Image Processing Toolbox.
Answer = license('test','image_toolbox');
if Answer ~= 1
    warndlg('It appears that you do not have a license for the Image Processing Toolbox of Matlab.  Many of the image analysis modules of CellProfiler may not function properly. Typing ''ver'' or ''license'' at the Matlab command line may provide more information about your current license situation.');
end

%%%% Sets up the data and image tools popup menus using the
%%%% LoadToolsPopUpMenu subfunction.
handles.Current.ImageToolHelp = LoadToolsPopUpMenu(handles, 'Image');
handles.Current.DataToolHelp = LoadToolsPopUpMenu(handles, 'Data');

%%% Adds the Help folder to Matlab's search path.
try Pathname = fullfile(handles.Current.CellProfilerPathname,'Help');
addpath(Pathname)
catch 
    errordlg('CellProfiler could not find its help files, which should be located in a folder called Help within the folder containing CellProfiler.m. The help buttons will not be functional.');
end

%%% Checks figure handles for current open windows.
handles.Current.CurrentHandles = findobj;

%%% Sets up the main program window (Main GUI window) so that it asks for
%%% confirmation prior to closing.
%%% First, obtains the handle for the main GUI window (aka figure1).
ClosingFunction = ...
    ['deleteme = CPquestdlg(''Do you really want to quit?'', ''Confirm quit'',''Yes'',''No'',''Yes''); switch deleteme; case ''Yes''; delete(', num2str((handles.figure1)*8192), '/8192); case ''No''; return; end; clear deleteme'];
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


% Update handles structure
guidata(hObject, handles);

% Set default output filename
set(handles.OutputFileNameEditBox,'string','DefaultOUT.mat')

% --- Outputs from this function are returned to the command line.
function varargout = CellProfiler_OutputFcn(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
% Get default command line output from handles structure
varargout{1} = handles.output;
%%% SUBFUNCTION %%%
function ToolHelp = LoadToolsPopUpMenu(handles, ImageOrData)
if strcmp(ImageOrData, 'Image') == 1
    FolderName = 'ImageTools';
    NoneLoadedText = 'Image tools: none loaded';
    PopUpMenuLabel = 'Image tools';
    PopUpMenuHandle = 'ImageToolsPopUpMenu';
    ToolHelpInfo = ['Help information from individual image tool files, which are Matlab m-files located within the ImageTools directory:' 10];
elseif strcmp(ImageOrData, 'Data') == 1
    FolderName = 'DataTools';
    NoneLoadedText = 'Data tools: none loaded';
    PopUpMenuLabel = 'Data tools';
    PopUpMenuHandle = 'DataToolsPopUpMenu';
    ToolHelpInfo = ['Help information from individual data tool files, which are Matlab m-files located within the DataTools directory:' 10];
end
%%% Finds all available tools, which are .m files residing in the
%%% DataTools or ImageTools folder. CellProfilerPathname was defined
%%% upon launching CellProfiler.
Pathname = fullfile(handles.Current.CellProfilerPathname,FolderName);
ListOfTools{1} = NoneLoadedText;
try addpath(Pathname)
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
    if isempty(FileNamesNoDir) ~= 1
        %%% Looks for .m files.
        for i = 1:length(FileNamesNoDir),
            if strncmp(FileNamesNoDir{i}(end-1:end),'.m',2) == 1,
                ListOfTools(length(ListOfTools)+1) = {FileNamesNoDir{i}(1:end-2)};
                ToolHelp{i} = [ToolHelpInfo, '-----------' 10 help(char(FileNamesNoDir{i}(1:end-2)))];
            end
        end
        if length(ListOfTools) > 1
            ListOfTools(1) = {PopUpMenuLabel};
        else ToolHelp = 'No image tools were loaded upon starting up CellProfiler. Image tools are Matlab m-files ending in ''.m'', and should be located in a folder called ImageTools within the folder containing CellProfiler.m';
        end
    end
end
set(handles.(PopUpMenuHandle), 'string', ListOfTools)

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% LOAD PIPELINE BUTTON %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes on button press in LoadPipelineButton.
function [SettingsPathname, SettingsFileName, errFlg, handles] = ...
    LoadPipelineButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.

errFlg = 0;
if exist(handles.Current.DefaultOutputDirectory, 'dir')
    [SettingsFileName, SettingsPathname] = uigetfile(fullfile(handles.Current.DefaultOutputDirectory,'.', '*.mat'),'Choose a settings or output file');
else
    [SettingsFileName, SettingsPathname] = uigetfile('*.mat','Choose a settings or output file');
end

%%% If the user presses "Cancel", the SettingsFileName.m will = 0 and
%%% nothing will happen.
if SettingsFileName == 0
    return
end
%%% Loads the Settings file.
LoadedSettings = load(fullfile(SettingsPathname,SettingsFileName));
%%% Error Checking for valid settings file.
if ~ (isfield(LoadedSettings, 'Settings') || isfield(LoadedSettings, 'handles'))
    errordlg(['The file ' SettingsPathname SettingsFileName ' does not appear to be a valid settings or output file. Settings can be extracted from an output file created when analyzing images with CellProfiler or from a small settings file saved using the "Save Settings" button.  Either way, this file must have the extension ".mat" and contain a variable named "Settings" or "handles".']);
    errFlg = 1;
    return
end
%%% Figures out whether we loaded a Settings or Output file, and puts
%%% the correct values into Settings. Splices the subset of variables
%%% from the "settings" structure into the handles structure.
if (isfield(LoadedSettings, 'Settings')),
    Settings = LoadedSettings.Settings;
else
    Settings = LoadedSettings.handles.Settings;
    Settings.NumbersOfVariables = LoadedSettings.handles.Settings.NumbersOfVariables;
end

try
    [NumberOfModules, MaxNumberVariables] = size(Settings.VariableValues);
    if (size(Settings.ModuleNames,2) ~= NumberOfModules)||(size(Settings.NumbersOfVariables,2) ~= NumberOfModules);
        errordlg(['The file ' SettingsPathname SettingsFileName ' is not a valid settings or output file. Settings can be extracted from an output file created when analyzing images with CellProfiler or from a small settings file saved using the "Save Settings" button.']); 
        errFlg = 1;
        return
    end
catch
    errordlg(['The file ' SettingsPathname SettingsFileName ' is not a valid settings or output file. Settings can be extracted from an output file created when analyzing images with CellProfiler or from a small settings file saved using the "Save Settings" button.']); 
    errFlg = 1;
    return
end

handles.Settings.ModuleNames = Settings.ModuleNames;
ModuleNamedotm = [char(Settings.ModuleNames{1}) '.m'];
%%% Checks to make sure that the modules have not changed
if exist(ModuleNamedotm,'file')
    FullPathname = which(ModuleNamedotm);
    [Pathname, filename, ext, versn] = fileparts(FullPathname);        
else
    %%% If the module.m file is not on the path, it won't be
    %%% found, so ask the user where the modules are.
    Pathname = uigetdir('','Please select directory where modules are located');
end

%%defaultVariableRevisionNumbers and revisionConfirm are both variables
%%used when asking user which variable revision number to save in settings
%%file
revisionConfirm = 0;
for ModuleNum=1:length(handles.Settings.ModuleNames),
    [defVariableValues defDescriptions handles.Settings.NumbersOfVariables(ModuleNum) DefVarRevNum] = LoadSettings_Helper(Pathname, char(handles.Settings.ModuleNames(ModuleNum)));
    if (isfield(Settings,'VariableRevisionNumbers')),
        SavedVarRevNum = Settings.VariableRevisionNumbers(ModuleNum);
    else
        SavedVarRevNum = 0;
    end
    if(SavedVarRevNum == DefVarRevNum)
        if(handles.Settings.NumbersOfVariables(ModuleNum) == Settings.NumbersOfVariables(ModuleNum))
            handles.Settings.VariableValues(ModuleNum,1:Settings.NumbersOfVariables(ModuleNum)) = Settings.VariableValues(ModuleNum,1:Settings.NumbersOfVariables(ModuleNum));
            handles.Settings.VariableRevisionNumbers(ModuleNum) = SavedVarRevNum;
            defaultVariableRevisionNumbers(ModuleNum) = DefVarRevNum;
            varChoice = 3;
        else
            errorString = 'Variable Revision Number same, but number of variables different for some reason';
            savedVariableValues = Settings.VariableValues(ModuleNum,1:Settings.NumbersOfVariables(ModuleNum));
            for i=1:(length(savedVariableValues)),
                if (iscellstr(savedVariableValues(i)) == 0)
                    savedVariableValues(i) = {''};
                end
            end
            varChoice = LoadSavedVariables(handles, savedVariableValues, defVariableValues, defDescriptions, errorString, char(handles.Settings.ModuleNames(ModuleNum)));
            revisionConfirm = 1;
        end
    else
        errorString = 'Variable Revision Numbers are not the same';
        savedVariableValues = Settings.VariableValues(ModuleNum,1:Settings.NumbersOfVariables(ModuleNum));
        for i=1:(length(savedVariableValues)),
            if (iscellstr(savedVariableValues(i)) == 0)
                savedVariableValues(i) = {''};
            end
        end
        varChoice = LoadSavedVariables(handles, savedVariableValues, defVariableValues,  defDescriptions, errorString, char(handles.Settings.ModuleNames(ModuleNum)));
        revisionConfirm = 1;
    end
    if (varChoice == 1),
        handles.Settings.VariableValues(ModuleNum,1:handles.Settings.NumbersOfVariables(ModuleNum)) = defVariableValues(1:handles.Settings.NumbersOfVariables(ModuleNum));
        handles.Settings.VariableValues(ModuleNum,1:Settings.NumbersOfVariables(ModuleNum)) = Settings.VariableValues(ModuleNum,1:Settings.NumbersOfVariables(ModuleNum));
        handles.Settings.VariableRevisionNumbers(ModuleNum) = DefVarRevNum;
        savedVariableRevisionNumbers(ModuleNum) = SavedVarRevNum;
    elseif (varChoice == 2),
        handles.Settings.VariableValues(ModuleNum,1:handles.Settings.NumbersOfVariables(ModuleNum)) = defVariableValues(1:handles.Settings.NumbersOfVariables(ModuleNum));
        handles.Settings.VariableRevisionNumbers(ModuleNum) = DefVarRevNum;
        savedVariableRevisionNumbers(ModuleNum) = SavedVarRevNum;
    elseif (varChoice == 0),
        break;
    end
end

if(varChoice == 0),
% CLEAR DOESN"T ACTUALLY WORK WHEN USED THIS WAY.
    %    clear handles.Settings.ModuleNames;
    %%% Update handles structure.
    guidata(hObject,handles);
    ModulePipelineListBox_Callback(hObject, eventdata, handles);
else
    try
        handles.Settings.PixelSize = Settings.PixelSize;
    end

    handles.Current.NumberOfModules = 0;
    handles.Current.NumberOfModules = length(handles.Settings.ModuleNames);
        
    contents = handles.Settings.ModuleNames;
    set(handles.ModulePipelineListBox,'String',contents);
    set(handles.ModulePipelineListBox,'Value',1);
    set(handles.PixelSizeEditBox,'string',handles.Settings.PixelSize);

    %%% Update handles structure.
    guidata(hObject,handles);
    ModulePipelineListBox_Callback(hObject, eventdata, handles);
    
    
    %%% If the user loaded settings from an output file, prompt them to
    %%% save it as a separate Settings file for future use.
    if isfield(LoadedSettings, 'handles'),
        Answer = CPquestdlg('The settings have been extracted from the output file you selected.  Would you also like to save these settings in a separate, smaller, settings-only file?','','Yes','No','Yes');
        if strcmp(Answer, 'Yes') == 1
            tempSettings = handles.Settings;
            if(revisionConfirm == 1)
                VersionAnswer = CPquestdlg('How should the settings file be saved?', 'Save Settings File', 'Exactly as found in output', 'As Loaded into CellProfiler window', 'Exactly as found in output');
                if strcmp(VersionAnswer, 'Exactly as found in output')
                    handles.Settings = Settings;
                end
            end
            SavePipelineButton_Callback(hObject, eventdata, handles);
            handles.Settings = tempSettings;
        end
    end
end

%%% After everything has been loaded, we make sure that any excess
%%% modules are removed.  The total number of modules used is
%%% retrieved from ModuleNum (see loop above).  There may not actually
%%% be any excess modules, so the following are each enclosed in a
%%% try/end.
try handles.Settings.VariableValues(ModuleNum+1:end,:) = [];
end
try handles.Settings.NumbersOfVariables(ModuleNum+1:end) = [];
end
try handles.Settings.VariableRevisionNumbers(ModuleNum+1:end) = [];
end
guidata(hObject,handles);

%%% SUBFUNCTION %%%
function [VariableValues VariableDescriptions NumbersOfVariables VarRevNum] = LoadSettings_Helper(Pathname, ModuleName)

VariableValues = {[]};
VariableDescriptions = {[]};
VarRevNum = 0;
NumbersOfVariables = 0;
try
    ModuleNamedotm = [ModuleName '.m'];
    fid=fopen(fullfile(Pathname,ModuleNamedotm));
    while 1;
        output = fgetl(fid); if ~ischar(output); break; end;
        if (strncmp(output,'%defaultVAR',11) == 1),
            displayval = output(17:end);
            istr = output(12:13);
            i = str2num(istr);
            VariableValues(i) = {displayval};
            NumbersOfVariables = i;
        elseif (strncmp(output,'%textVAR',8) == 1);
            displayval = output(13:end);
            if(length(displayval) > 8)
                if(strcmp(displayval(end-8:end),'#LongBox#'))
                    displayval = displayval(1:end-9);
                end
            end
            istr = output(9:10);
            i = str2num(istr);
            VariableDescriptions(i) = {displayval};
        elseif (strncmp(output,'%%%VariableRevisionNumber',25) == 1)
            try
                VarRevNum = str2num(output(29:30));
            catch
                VarRevNum = str2num(output(29:29));
            end
        end
    end
    fclose(fid);
catch
    errordlg('Module could not be found in directory specified','Error');
end

%%% SUBFUNCTION %%%
function varChoice = LoadSavedVariables(handles, savedVariables, defaultVariables, defaultDescriptions, errorString, algorithmName)
global variableChoice;
helpText = ['The settings contained within this file are based on an old version of the '...
    algorithmName ' module, as indicated by the Variable Revision Number of the'...
    ' module. As a result, it is possible that your old settings are no longer reasonable.'...
    '  Displayed below are the settings retrieved from your file (Saved settings) and the '...
    'default settings retrieved from the more recent version of the module (Default settings).'...
    '  Which do you want to try to load?"'];

savedbuttoncallback = 'LoadSavedWindowHandle = findobj(''name'',''LoadSavedWindow''); global variableChoice; variableChoice = 1; close(LoadSavedWindowHandle), clear LoadSavedWindowHandle';
defaultbuttoncallback = 'LoadSavedWindowHandle = findobj(''name'',''LoadSavedWindow''); global variableChoice; variableChoice = 2; close(LoadSavedWindowHandle), clear LoadSavedWindowHandle';
cancelbuttoncallback = 'LoadSavedWindowHandle = findobj(''name'',''LoadSavedWindow''); global variableChoice; variableChoice = 0; close(LoadSavedWindowHandle), clear LoadSavedWindowHandle';

%%% Creates the dialog box and its text, buttons, and edit boxes.
MainWinPos = get(handles.figure1,'Position');
Color = [0.925490196078431 0.913725490196078 0.847058823529412];

%%% Label we attach to figures (as UserData) so we know they are ours
userData.Application = 'CellProfiler';
LoadSavedWindowHandle = figure(...
'Units','pixels',...
'Color',Color,...
'DockControls','off',...
'MenuBar','none',...
'Name','LoadSavedWindow',...
'NumberTitle','off',...
'Position',[MainWinPos(1)+MainWinPos(3)/3 MainWinPos(2) MainWinPos(3)*4/5 MainWinPos(4)],...
'Resize','off',...
'HandleVisibility','on',...
'Tag','figure1',...
'UserData',userData);

savedbox = uicontrol(...
'Parent',LoadSavedWindowHandle,...
'Units','normalized',...
'Position',[0.41 0.155 0.23 0.464],...
'String',savedVariables,...
'Style','listbox',...
'Value',1,...
'Tag','savedbox');

defaultbox = uicontrol(...
'Parent',LoadSavedWindowHandle,...
'Units','normalized',...
'Position',[0.68 0.155 0.23 0.464],...
'String',defaultVariables,...
'Style','listbox',...
'Value',1,...
'Tag','defaultbox');

descriptionbox = uicontrol(...
'Parent',LoadSavedWindowHandle,...
'Units','normalized',...
'Position',[0.08 0.155 0.275 0.464],...
'String',defaultDescriptions,...
'Style','listbox',...
'Value',1,...
'Tag','descriptionbox');

cancelbutton = uicontrol(...
'Parent',LoadSavedWindowHandle,...
'Units','normalized',...
'Callback',cancelbuttoncallback,...
'Position',[0.42 0.077 0.202 0.06],...
'String','Cancel',...
'Tag','cancelbutton');

savedbutton = uicontrol(...
'Parent',LoadSavedWindowHandle,...
'Units','normalized',...
'Callback',savedbuttoncallback,...
'Position',[0.42 0.642 0.202 0.06],...
'String','Load Saved Settings',...
'Tag','savedbutton');

defaultbutton = uicontrol(...
'Parent',LoadSavedWindowHandle,...
'Units','normalized',...
'Callback',defaultbuttoncallback,...
'Position',[0.69 0.642 0.202 0.06],...
'String','Load Default Settings',...
'Tag','defaultbutton');

informtext = uicontrol(...
'Parent',LoadSavedWindowHandle,...
'Units','normalized',...
'Position',[0.112 0.70 0.76 0.21],...
'String',helpText,...
'Style','text',...
'Tag','informtext');

descriptiontext = uicontrol(...
'Parent',LoadSavedWindowHandle,...
'Units','normalized',...
'Position',[0.1 0.642 0.223 0.062],...
'String',{'Variable', 'Descriptions'},...
'Style','text',...
'Tag','descriptiontext');

uiwait(LoadSavedWindowHandle);
if exist('variableChoice','var') == 1
    if isempty(variableChoice) ~= 1
        varChoice = variableChoice;
        clear global variableChoice;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE PIPELINE BUTTON %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes on button press in SavePipelineButton.
function SavePipelineButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
%%% The "Settings" variable is saved to the file name the user chooses.
if exist(handles.Current.DefaultOutputDirectory, 'dir')
    [FileName,Pathname] = uiputfile(fullfile(handles.Current.DefaultOutputDirectory,'*.mat'), 'Save Settings As...');
else
    [FileName,Pathname] = uiputfile('*.mat', 'Save Settings As...');
end
%%% Allows canceling.
if FileName ~= 0
    %%% Allows user to save pipeline setting as a readable text file (.txt)
    SaveText = CPquestdlg('Do you want to save settings as a text file also?','Save as text?','No');
    if strcmp(SaveText,'Cancel') == 1
        return
    end
    %%% Checks if a field is present, and if it is, the value is stored in the
    %%% structure 'Settings' with the same name.
    if isfield(handles.Settings,'VariableValues'),
        Settings.VariableValues = handles.Settings.VariableValues;
    end
    if isfield(handles.Settings,'ModuleNames'),
        Settings.ModuleNames = handles.Settings.ModuleNames;
    end
    if isfield(handles.Settings,'NumbersOfVariables'),
        Settings.NumbersOfVariables = handles.Settings.NumbersOfVariables;
    end
    if isfield(handles.Settings,'PixelSize'),
        Settings.PixelSize = handles.Settings.PixelSize;
    end
    if isfield(handles.Settings,'VariableRevisionNumbers'),
        Settings.VariableRevisionNumbers = handles.Settings.VariableRevisionNumbers;
    end
    save(fullfile(Pathname,FileName),'Settings')
    %%% Writes settings into a readable text file.
    if strcmp(SaveText,'Yes') == 1
        VariableValues = handles.Settings.VariableValues;
        ModuleNames = handles.Settings.ModuleNames;
        ModuleNamedotm = [char(ModuleNames(1)) '.m'];
        %Prompt what to save file as, and where to save it.
        if exist(ModuleNamedotm,'file')
            FullPathname = which(ModuleNamedotm);
            [PathnameModules, filename, ext, versn] = fileparts(FullPathname);
        else
            %%% If the module.m file is not on the path, it won't be
            %%% found, so ask the user where the modules are.
            PathnameModules = uigetdir('','Please select directory where modules are located');
        end
        [filename,SavePathname] = uiputfile('*.txt', 'Save Settings As...');
        % make sure # of modules equals number of variable rows.
        VariableSize = size(VariableValues);
        if VariableSize(1) ~= max(size(ModuleNames))
            error('Your settings are not valid.')
        end
        display = ['Saved Settings, in file ' filename ', Saved on ' date];
        % Loop for each module loaded.
        for p = 1:VariableSize(1)
            Module = [char(ModuleNames(p))];
            display = strvcat(display, ['Module #' num2str(p) ': ' Module]);
            ModuleNamedotm = [Module '.m'];
            fid=fopen(fullfile(PathnameModules,ModuleNamedotm));
            while 1
                output = fgetl(fid);
                if ~ischar(output), break, end
                if (strncmp(output,'%textVAR',8) == 1);
                    displayval = output(13:end);
                    istr = output(9:10);
                    i = str2num(istr);
                    VariableDescriptions(i) = {displayval};
                end
            end
            fclose(fid);
            % Loop for each variable in the module.
            for q = 1:VariableSize(2)
                VariableDescrip = char(VariableDescriptions(q));
                try
                    VariableVal = char(VariableValues(p,q));
                catch
                    VariableVal = '  ';

                end
                display =strvcat(display, ['    ' VariableDescrip '    ' VariableVal]);
            end
        end
        %% tack on rest of Settings information.
        PixelSizeDisplay = ['Pixel Size: ' handles.Settings.PixelSize];
        RevisionNumbersDisplay = ['Variable Revision Numbers: ' num2str(handles.Settings.VariableRevisionNumbers)];
        display = strvcat(display, PixelSizeDisplay, RevisionNumbersDisplay);
        %% Save to a .txt file.
        dlmwrite(fullfile(SavePathname,filename), display, 'delimiter', '');
    end
  helpdlg('The settings file(s) has been written.');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% FIGURE DISPLAY BUTTONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% These buttons appear after analysis has begun, and disappear 
%%% when it is over.

function CloseFigureButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
global closeFigures;
ModuleHighlighted = get(handles.ModulePipelineListBox,'Value');
for i=1:length(ModuleHighlighted),
        closeFigures(length(closeFigures)+1) = ModuleHighlighted(i);
end
guidata(hObject, handles);

function OpenFigureButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
global openFigures;
ModuleHighlighted = get(handles.ModulePipelineListBox,'Value');
for i=1:length(ModuleHighlighted),
        openFigures(length(openFigures)+1) = ModuleHighlighted(i);
end
guidata(hObject, handles);

%%%%%%%%%%%%%%%%%%%%%%%%
%%% ADD MODULE BUTTON %%%
%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes on button press in AddModule.
function AddModule_Callback(hObject,eventdata,handles) %#ok We want to ignore MLint error checking for this line.

if handles.Current.NumberOfModules == 99
    errordlg('CellProfiler in its current state can only handle 99 modules. You have just attempted to load the 100th module. It should be fairly straightforward to modify the code in CellProfiler.m to expand its capabilities.');
    return
end

%%% 1. Opens a user interface to retrieve the .m file you want to use.
%%% Change to the default module directory. This line is within a
%%% try-end pair because the user may have changed the folder names
%%% leading up to this directory sometime after saving the
%%% Preferences.

if exist(handles.Preferences.DefaultModuleDirectory, 'dir')
    [ModuleNamedotm,Pathname] = uigetfile(fullfile(handles.Preferences.DefaultModuleDirectory,'.', '*.m'),...
    'Choose an image analysis module');
else
[ModuleNamedotm,Pathname] = uigetfile('*.m',...
    'Choose an image analysis module');
end

%%% 2. If the user presses "Cancel", the ModuleNamedotm = 0, and
%%% everything should be left as it was.
if ModuleNamedotm == 0,
else
    %%% The folder containing the desired .m file is added to Matlab's search path.
    addpath(Pathname);
    if(exist(ModuleNamedotm(1:end-2),'builtin') ~= 0)
        warningString = ['Your module has the same name as a builtin Matlab function.  Perhaps you should consider renaming your module.'];
        warndlg(warningString);
    end
    differentPaths = which(ModuleNamedotm, '-all');
    if length(differentPaths) == 0,
        %%% If the module's .m file is not found on the search path, the result
        %%% of exist is zero, and the user is warned.
        errordlg('Something is wrong; The .m file ', ModuleNamedotm, ' was not initially found by Matlab, so the folder containing it was added to the Matlab search path. But, Matlab still cannot find the .m file for the analysis module you selected. The module will not be added to the image analysis pipeline.');
        return
    elseif length(differentPaths) > 1,
        warndlg(['More than one file with this same module name exists in the Matlab search path.  The pathname from ' char(differentPaths{1}) ' will likely be used, but this is unpredictable.  Modules should have unique names that are not the same as already existing Matlab functions to avoid confusion.']);
    end

    %%% 3. The last two characters (=.m) are removed from the
    %%% ModuleName.m and called ModuleName.
    ModuleName = ModuleNamedotm(1:end-2);
    %%% The name of the module is shown in a text box in the GUI (the text
    %%% box is called ModuleName1.) and in a text box in the GUI which
    %%% displays the current module (whose settings are shown).
    
    %%% 4. The text description for each variable for the chosen module is
    %%% extracted from the module's .m file and displayed.
    ModuleNumber = TwoDigitString(handles.Current.NumberOfModules+1);
    ModuleNums = handles.Current.NumberOfModules+1;

    fid=fopen(fullfile(Pathname,ModuleNamedotm));
    lastVariableCheck = 0;
    while 1;
        output = fgetl(fid); if ~ischar(output); break; end;
        if strncmp(output,'%defaultVAR',11) == 1
            displayval = output(17:end);
            istr = output(12:13);
            lastVariableCheck = str2num(istr);
            handles.Settings.VariableValues(ModuleNums, lastVariableCheck) = {displayval};
            handles.Settings.NumbersOfVariables(str2double(ModuleNumber)) = lastVariableCheck;
        elseif strncmp(output,'%%%VariableRevisionNumber',25) == 1
            try
            handles.Settings.VariableRevisionNumbers(str2double(ModuleNumber)) = str2num(output(29:30));
            catch
            handles.Settings.VariableRevisionNumbers(str2double(ModuleNumber)) = str2num(output(29:29));
            end
        end
    end
    fclose(fid);
    if lastVariableCheck == 0
        errordlg(['The module you attempted to add, ', ModuleNamedotm,', is not a valid CellProfiler module because it does not appear to have any variables.  Sometimes this error occurs when you try to load a module that has the same name as a built-in Matlab function and the built in function is located in a directory higher up on the Matlab search path.']);
        return
    end
    
    try Contents = handles.Settings.VariableRevisionNumbers(str2double(ModuleNumber));
    catch Contents = [];
    end
    
    if isempty(Contents) == 1
        handles.Settings.VariableRevisionNumbers(str2double(ModuleNumber)) = 0;
    end
    
    %%% 5. Saves the ModuleName to the handles structure.
    % Find which module slot number this callback was called for.

    handles.Settings.ModuleNames{ModuleNums} = ModuleName;
    contents = get(handles.ModulePipelineListBox,'String');
    contents{ModuleNums} = ModuleName;
    set(handles.ModulePipelineListBox,'String',contents);
    
    %%% 6. Update handles.Current.NumberOfModules
    if str2double(ModuleNumber) > handles.Current.NumberOfModules,
        handles.Current.NumberOfModules = str2double(ModuleNumber);
    end

    %%% 7. Choose Loaded Module in Listbox
    set(handles.ModulePipelineListBox,'Value',handles.Current.NumberOfModules);

    %%% Updates the handles structure to incorporate all the changes.
    guidata(gcbo, handles);
    ModulePipelineListBox_Callback(hObject, eventdata, handles);
end

%%% SUBFUNCTION %%%
function twodigit = TwoDigitString(val)
%TwoDigitString is a function like num2str(int) but it returns a two digit
%representation of a string for our purposes.
if ((val > 99) || (val < 0)),
  error(['TwoDigitString: Can''t convert ' num2str(val) ' to a 2 digit number']);
end
twodigit = sprintf('%02d', val);

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% REMOVE MODULE BUTTON %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes on button press for RemoveModule button.
function RemoveModule_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
ModuleHighlighted = get(handles.ModulePipelineListBox,'Value');
RemoveModule_Helper(ModuleHighlighted, hObject, eventdata, handles, 'Confirm');
%%% SUBFUNCTION %%%
function RemoveModule_Helper(ModuleHighlighted, hObject, eventdata, handles, ConfirmOrNot) %#ok We want to ignore MLint error checking for this line.

if strcmp(ConfirmOrNot, 'Confirm') == 1
    %%% Confirms the choice to clear the module.
    Answer = CPquestdlg('Are you sure you want to clear this analysis module and its settings?','Confirm','Yes','No','Yes');
    if strcmp(Answer,'No') == 1
        return
    end
end
if isempty(handles.Settings.ModuleNames);
    return
end
%%% 1. Sets all 11 VariableBox edit boxes and all 11
%%% VariableDescriptions to be invisible.
for i = 1:99
    set(handles.(['VariableBox' TwoDigitString(i)]),'visible','off','String','n/a')
    set(handles.(['VariableDescription' TwoDigitString(i)]),'visible','off')
end

for ModuleDelete = 1:length(ModuleHighlighted);
    %%% 2. Removes the ModuleName from the handles structure.
    handles.Settings.ModuleNames(ModuleHighlighted(ModuleDelete)-ModuleDelete+1) = [];
    %%% 3. Clears the variable values in the handles structure.
    handles.Settings.VariableValues(ModuleHighlighted(ModuleDelete)-ModuleDelete+1,:) = [];
    %%% 4. Clears the number of variables in each module slot from handles structure.
    handles.Settings.NumbersOfVariables(ModuleHighlighted(ModuleDelete)-ModuleDelete+1) = [];
    %%% 4. Clears the Variable Revision Numbers in each module slot from handles structure.
    handles.Settings.VariableRevisionNumbers(ModuleHighlighted(ModuleDelete)-ModuleDelete+1) = [];
end

%%% 5. Update the number of modules loaded
handles.Current.NumberOfModules = 0;
handles.Current.NumberOfModules = length(handles.Settings.ModuleNames);

%%% 6. Sets the proper module name to "No analysis module loaded"
if(isempty(handles.Settings.ModuleNames))
    contents = {'No Modules Loaded'};
else
    contents = handles.Settings.ModuleNames;
end

set(handles.ModulePipelineListBox,'String',contents);

while((isempty(ModuleHighlighted)==0) && (ModuleHighlighted(length(ModuleHighlighted)) > handles.Current.NumberOfModules) )
    ModuleHighlighted(length(ModuleHighlighted)) = [];
end

if(handles.Current.NumberOfModules == 0)
    ModuleHighlighted = 1;
elseif (isempty(ModuleHighlighted))
    ModuleHighlighted = handles.Current.NumberOfModules;
end

set(handles.ModulePipelineListBox,'Value',ModuleHighlighted);

guidata(gcbo, handles);
ModulePipelineListBox_Callback(hObject, eventdata, handles);

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% MOVE UP/DOWN BUTTONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

function MoveUpButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
ModuleHighlighted = get(handles.ModulePipelineListBox,'Value');
if(handles.Current.NumberOfModules < 1 || ModuleHighlighted(1) == 1)
else
    for ModuleUp1 = 1:length(ModuleHighlighted);
        ModuleUp = ModuleHighlighted(ModuleUp1)-1;
        ModuleNow = ModuleHighlighted(ModuleUp1);
        %%% 1. Switches ModuleNames
        ModuleUpName = char(handles.Settings.ModuleNames(ModuleUp));
        ModuleName = char(handles.Settings.ModuleNames(ModuleNow));
        handles.Settings.ModuleNames{ModuleUp} = ModuleName;
        handles.Settings.ModuleNames{ModuleNow} = ModuleUpName;
        %%% 2. Copy then clear the variable values in the handles structure.
        copyVariables = handles.Settings.VariableValues(ModuleNow,:);
        handles.Settings.VariableValues(ModuleNow,:) = handles.Settings.VariableValues(ModuleUp,:);
        handles.Settings.VariableValues(ModuleUp,:) = copyVariables;
        %%% 3. Copy then clear the num of variables in the handles
        %%% structure.
        copyNumVariables = handles.Settings.NumbersOfVariables(ModuleNow);
        handles.Settings.NumbersOfVariables(ModuleNow) = handles.Settings.NumbersOfVariables(ModuleUp);
        handles.Settings.NumbersOfVariables(ModuleUp) = copyNumVariables;
        %%% 4. Copy then clear the variable revision numbers in the handles
        %%% structure.
        copyVarRevNums = handles.Settings.VariableRevisionNumbers(ModuleNow);
        handles.Settings.VariableRevisionNumbers(ModuleNow) = handles.Settings.VariableRevisionNumbers(ModuleUp);
        handles.Settings.VariableRevisionNumbers(ModuleUp) = copyVarRevNums;
    end
    %%% 5. Changes the Listbox to show the changes
    contents = handles.Settings.ModuleNames;
    ModuleHighlighted = ModuleHighlighted-1;
    set(handles.ModulePipelineListBox,'String',contents);
    set(handles.ModulePipelineListBox,'Value',ModuleHighlighted);
    %%% Updates the handles structure to incorporate all the changes.
    guidata(gcbo, handles);
    ModulePipelineListBox_Callback(hObject, eventdata, handles)
end

function MoveDownButton_Callback(hObject,eventdata,handles) %#ok We want to ignore MLint error checking for this line.
ModuleHighlighted = get(handles.ModulePipelineListBox,'Value');
if(handles.Current.NumberOfModules<1 || ModuleHighlighted(length(ModuleHighlighted)) >= handles.Current.NumberOfModules)
else
    for ModuleDown1 = 1:length(ModuleHighlighted);
        ModuleDown = ModuleHighlighted(ModuleDown1) + 1;
        ModuleNow = ModuleHighlighted(ModuleDown1);
        %%% 1. Saves the ModuleName
        ModuleDownName = char(handles.Settings.ModuleNames(ModuleDown));
        ModuleName = char(handles.Settings.ModuleNames(ModuleNow));
        handles.Settings.ModuleNames{ModuleDown} = ModuleName;
        handles.Settings.ModuleNames{ModuleNow} = ModuleDownName;
        %%% 2. Copy then clear the variable values in the handles structure.
        copyVariables = handles.Settings.VariableValues(ModuleNow,:);
        handles.Settings.VariableValues(ModuleNow,:) = handles.Settings.VariableValues(ModuleDown,:);
        handles.Settings.VariableValues(ModuleDown,:) = copyVariables;
        %%% 3. Copy then clear the num of variables in the handles
        %%% structure.
        copyNumVariables = handles.Settings.VariableRevisionNumbers(ModuleNow);
        handles.Settings.VariableRevisionNumbers(ModuleNow) = handles.Settings.VariableRevisionNumbers(ModuleDown);
        handles.Settings.VariableRevisionNumbers(ModuleDown) = copyNumVariables;
        %%% 4. Copy then clear the num of variables in the handles
        %%% structure.
        copyVarRevNums = handles.Settings.NumbersOfVariables(ModuleNow);
        handles.Settings.NumbersOfVariables(ModuleNow) = handles.Settings.NumbersOfVariables(ModuleDown);
        handles.Settings.NumbersOfVariables(ModuleDown) = copyVarRevNums;
    end
    %%% 5. Changes the Listbox to show the changes
    contents = handles.Settings.ModuleNames;
    set(handles.ModulePipelineListBox,'String',contents);
    set(handles.ModulePipelineListBox,'Value',ModuleHighlighted+1);
    ModuleHighlighted = ModuleHighlighted+1;
    %%% Updates the handles structure to incorporate all the changes.
    guidata(gcbo, handles);
    ModulePipelineListBox_Callback(hObject, eventdata, handles)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% MODULE PIPELINE LISTBOX %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes during object creation, after setting all properties.
function ModulePipelineListBox_CreateFcn(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.

% --- Executes on selection change in ModulePipelineListBox.
function ModulePipelineListBox_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
ModuleHighlighted = get(handles.ModulePipelineListBox,'Value');
if (length(ModuleHighlighted) > 0)
    ModuleNumber = ModuleHighlighted(1);
    if( handles.Current.NumberOfModules > 0 )
        %%% 2. Sets all VariableBox edit boxes and all
        %%% VariableDescriptions to be invisible.
        for i = 1:99,
            set(handles.(['VariableBox' TwoDigitString(i)]),'visible','off','String','n/a')
            set(handles.(['VariableDescription' TwoDigitString(i)]),'visible','off')
        end

        %%% 2.25 Removes slider and moves panel back to original
        %%% position.
        %%% If panel location gets changed in GUIDE, must change the
        %%% position values here as well.
        set(handles.variablepanel, 'position', [238 0 563 346]);
        set(handles.slider1,'visible','off');

        %%% 2.5 Checks whether a module is loaded in this slot.
        contents = get(handles.ModulePipelineListBox,'String');
        ModuleName = contents{ModuleNumber};

        %%% 3. Extracts and displays the variable descriptors from the .m file.
        lastVariableCheck = 0;
        ModuleNamedotm = strcat(ModuleName,'.m');
        if exist(ModuleNamedotm,'file') ~= 2
            errordlg(['The image analysis module named ', ModuleNamedotm, ' was not found. Is it stored in the folder with the other modules?  Has its name changed?  The settings stored for this module will be displayed, but this module will not run properly.']);
        else
            fid=fopen(ModuleNamedotm);
            while 1;
                output = fgetl(fid); if ~ischar(output); break; end;
                if (strncmp(output,'%textVAR',8) == 1);
                    set(handles.(['VariableDescription',output(9:10)]), 'string', output(13:end),'visible', 'on');
                    lastVariableCheck = str2num(output(9:10));
                end
            end
            fclose(fid);
            if lastVariableCheck == 0
                errordlg(['The module you attempted to add, ', ModuleNamedotm,', is not a valid CellProfiler module because it does not appear to have any variables.  Sometimes this error occurs when you try to load a module that has the same name as a built-in Matlab function and the built in function is located in a directory higher up on the Matlab search path.']);
                return  
            end
        end
        %%% 4. Extracts the stored values for the variables from the handles
        %%% structure and displays in the edit boxes.
        numberExtraLinesOfDescription = 0;
        numberOfLongBoxes = 0;
        varSpacing = 25;
        firstBoxLoc = 345; firstDesLoc = 343; normBoxHeight = 23; normDesHeight = 20;
        longBoxLength = 539; normBoxLength = 94;
        pixelSpacing = 2;
        if (lastVariableCheck < handles.Settings.NumbersOfVariables(ModuleNumber))
            lastVariableCheck = handles.Settings.NumbersOfVariables(ModuleNumber);
        end
        for i=1:lastVariableCheck,
            if(strcmp(get(handles.(['VariableDescription' TwoDigitString(i)]),'visible'), 'on'))
                descriptionString = get(handles.(['VariableDescription' TwoDigitString(i)]), 'string');
                flagExist = 0;
                if(length(descriptionString) > 8)
                    if(strcmp(descriptionString(end-8:end),'#LongBox#'))
                        flagExist = 1;
                        set(handles.(['VariableDescription' TwoDigitString(i)]), 'string', descriptionString(1:end-9))
                    end
                end
                linesVarDes = length(textwrap(handles.(['VariableDescription' TwoDigitString(i)]),{get(handles.(['VariableDescription' TwoDigitString(i)]),'string')}));
                numberExtraLinesOfDescription = numberExtraLinesOfDescription + linesVarDes - 1;
                VarDesPosition = get(handles.(['VariableDescription' TwoDigitString(i)]), 'Position');
                varXPos = VarDesPosition(1);
                varYPos = firstDesLoc+pixelSpacing*numberExtraLinesOfDescription-varSpacing*(i+numberOfLongBoxes+numberExtraLinesOfDescription);
                varXSize = VarDesPosition(3);
                varYSize = normDesHeight*linesVarDes + pixelSpacing*(linesVarDes-1);
                set(handles.(['VariableDescription' TwoDigitString(i)]),'Position', [varXPos varYPos varXSize varYSize]);
            end

            if (i <= handles.Settings.NumbersOfVariables(ModuleNumber))
                if iscellstr(handles.Settings.VariableValues(ModuleNumber, i));
                    VariableValuesString = char(handles.Settings.VariableValues{ModuleNumber, i});
                    if ( ( length(VariableValuesString) > 13) | (flagExist) )
                        numberOfLongBoxes = numberOfLongBoxes+1;
                        varXPos = 25;
                        varYPos = firstBoxLoc+pixelSpacing*numberExtraLinesOfDescription-varSpacing*(i+numberOfLongBoxes+numberExtraLinesOfDescription);
                        varXSize = longBoxLength;
                        varYSize = normBoxHeight;
                    else
                        varXPos = 470;
                        varYPos = firstBoxLoc+pixelSpacing*numberExtraLinesOfDescription-varSpacing*(i+numberOfLongBoxes+numberExtraLinesOfDescription-(linesVarDes-1)/2.0);
                        varXSize = normBoxLength;
                        varYSize = normBoxHeight;
                    end
                    set(handles.(['VariableBox' TwoDigitString(i)]), 'Position', [varXPos varYPos varXSize varYSize]);
                    set(handles.(['VariableBox' TwoDigitString(i)]),'string',VariableValuesString,'visible','on');
                else
                    set(handles.(['VariableBox' TwoDigitString(i)]),'string','n/a','visible','off');
                end
            end
        end

        %%% 5.  Sets the slider
        if((handles.Settings.NumbersOfVariables(ModuleNumber)+numberOfLongBoxes+numberExtraLinesOfDescription) > 14)
            set(handles.slider1,'visible','on');
            set(handles.slider1,'max',((handles.Settings.NumbersOfVariables(ModuleNumber)-14+numberOfLongBoxes+numberExtraLinesOfDescription)*25));
            set(handles.slider1,'value',get(handles.slider1,'min'));
        end
    else 
        helpdlg('No modules are loaded.');
    end
else 
    helpdlg('No module highlighted.');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% VARIABLE EDIT BOXES %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%

function storevariable(ModuleNumber, VariableNumber, UserEntry, handles)
%%% This function stores a variable's value in the handles structure, 
%%% when given the Module Number, the Variable Number, 
%%% the UserEntry (from the Edit box), and the initial handles
%%% structure.
handles.Settings.VariableValues(ModuleNumber, str2double(VariableNumber)) = {UserEntry};
guidata(gcbo, handles);

function [ModuleNumber] = whichactive(handles)
ModuleHighlighted = get(handles.ModulePipelineListBox,'Value');
ModuleNumber = ModuleHighlighted(1);
    
function VariableBox_CreateFcn(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.

function VariableBox_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
%%% Fetches the contents of the edit box, determines which module
%%% we are dealing with at the moment (by running the "whichactive"
%%% subfunction), and calls the storevariable function.
VariableName = get(hObject,'tag');
VariableNumberStr = VariableName(12:13);

UserEntry = get(handles.(['VariableBox' VariableNumberStr]),'string');
ModuleNumber = whichactive(handles);
if isempty(UserEntry)
  errordlg('Variable boxes must not be left blank');
  set(handles.(['VariableBox' VariableNumberStr]),'string', 'Fill in');
  storevariable(ModuleNumber,VariableNumberStr, 'Fill in', handles);
else
  if ModuleNumber == 0,     
    errordlg('Something strange is going on: none of the analysis modules are active right now but somehow you were able to edit a setting.','weirdness has occurred');
  else
    storevariable(ModuleNumber,VariableNumberStr,UserEntry, handles);
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% VARIABLE WINDOW SLIDER %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes on slider movement.
function slider1_Callback(hObject, eventdata, handles)
% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range
%        of slider
scrollPos = get(hObject, 'Value');
variablepanelPos = get(handles.variablepanel, 'position');
% Note:  The yPosition is 0 + scrollPos because 0 is the original Y
% Position of the variablePanel.  If the original location of the
% variablePanel gets changed, then the constant offset must be changed as
% well.
set(handles.variablepanel, 'position', [variablepanelPos(1) 0+scrollPos variablepanelPos(3) variablepanelPos(4)]);

function slider1_CreateFcn(hObject, eventdata, handles)

function handles = createVariablePanel(handles)
for i=1:99,
    handles.(['VariableBox' TwoDigitString(i)]) = uicontrol(...
        'Parent',handles.variablepanel,...
        'Units','pixels',...
        'BackgroundColor',[1 1 1],...
        'Callback','CellProfiler(''VariableBox_Callback'',gcbo,[],guidata(gcbo))',...
        'FontName','Times',...
        'FontSize',12,...
        'Position',[470 295-25*i 94 23],...
        'String','n/a',...
        'Style','edit',...
        'CreateFcn', 'CellProfiler(''VariableBox_CreateFcn'',gcbo,[],guidata(gcbo))',...
        'Tag',['VariableBox' TwoDigitString(i)],...
        'Behavior',get(0,'defaultuicontrolBehavior'),...
        'Visible','off');

    handles.(['VariableDescription' TwoDigitString(i)]) = uicontrol(...
        'Parent',handles.variablepanel,...
        'Units','pixels',...
        'BackgroundColor',[0.7 0.7 0.9],...
        'CData',[],...
        'FontName','Times',...
        'FontSize',12,...
        'FontWeight','bold',...
        'HorizontalAlignment','right',...
        'Position',[2 291-25*i 465 23],...
        'String','No analysis module has been loaded',...
        'Style','text',...
        'Tag',['VariableDescription' TwoDigitString(i)],...
        'UserData',[],...
        'Behavior',get(0,'defaultuicontrolBehavior'),...
        'Visible','off',...
        'CreateFcn', '');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PIXEL SIZE EDIT BOX %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes during object creation, after setting all properties.
function PixelSizeEditBox_CreateFcn(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.

function PixelSizeEditBox_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
%%% Checks to see whether the user input is a number, and generates an
%%% error message if it is not a number.
user_entry = str2double(get(hObject,'string'));
if isnan(user_entry)
    errordlg('You must enter a numeric value','Bad Input','modal');
    set(hObject,'string','0.25')
    %%% Checks to see whether the user input is positive, and generates an
    %%% error message if it is not.
elseif user_entry<=0
    errordlg('You entered a value less than or equal to zero','Bad Input','modal');
    set(hObject,'string', '0.25')
else
    %%% Gets the user entry and stores it in the handles structure.
    UserEntry = get(handles.PixelSizeEditBox,'string');
    handles.Settings.PixelSize = UserEntry;
    guidata(gcbo, handles);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SET PREFERENCES BUTTON %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes on button press in SetPreferencesButton.
function SetPreferencesButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.

%%% Creates a global variable to be used later.
global EnteredPreferences

%%% Opens a dialog box to retrieve input from the user.
%%% Sets the functions of the buttons and edit boxes in the dialog box.

PixelSizeEditBoxCallback = 'PixelSize = str2double(get(gco,''string'')); if isempty(PixelSize) == 1 | ~isnumeric(PixelSize), PixelSize = {''1''}, set(gco,''string'',PixelSize), end, clear';
ImageDirBrowseButtonCallback = 'EditBoxHandle = findobj(''Tag'',''ImageDirEditBox''); CurrentChoice = get(EditBoxHandle,''string''); if exist(CurrentChoice, ''dir''), tempdir = CurrentChoice; else, tempdir=pwd; end, DefaultImageDirectory = uigetdir(tempdir,''Select the default image directory''); if DefaultImageDirectory == 0, return, else set(EditBoxHandle,''string'', DefaultImageDirectory), end, clear';
ImageDirEditBoxCallback = 'DefaultImageDirectory = get(gco,''string''); if isempty(DefaultImageDirectory) == 1; DefaultImageDirectory = pwd; set(gco,''string'',DefaultImageDirectory); end, clear';
OutputDirBrowseButtonCallback = 'EditBoxHandle = findobj(''Tag'',''OutputDirEditBox''); CurrentChoice = get(EditBoxHandle,''string''); if exist(CurrentChoice, ''dir''), tempdir=CurrentChoice; else, tempdir=pwd; end, DefaultOutputDirectory = uigetdir(tempdir,''Select the default output directory''); if DefaultOutputDirectory == 0, return, else set(EditBoxHandle,''string'', DefaultOutputDirectory), end, clear';
OutputDirEditBoxCallback = 'DefaultOutputDirectory = get(gco,''string''); if isempty(DefaultOutputDirectory) == 1; DefaultOutputDirectory = pwd; set(gco,''string'',DefaultOutputDirectory), end, clear';
ModuleDirBrowseButtonCallback = 'EditBoxHandle = findobj(''Tag'',''ModuleDirEditBox''); CurrentChoice = get(EditBoxHandle,''string''); if exist(CurrentChoice, ''dir''), tempdir=CurrentChoice; else tempdir=pwd; end, DefaultModuleDirectory = uigetdir(tempdir,''Select the directory where modules are stored''); if DefaultModuleDirectory == 0, return, else set(EditBoxHandle,''string'', DefaultModuleDirectory), end, clear';
ModuleDirEditBoxCallback = 'DefaultModuleDirectory = get(gco,''string''); if isempty(DefaultModuleDirectory) == 1; DefaultModuleDirectory = pwd; set(gco,''string'',DefaultModuleDirectory), end, clear';

%%% TODO: Add error checking to each directory edit box (does pathname exist).
%%% TODO: Add error checking to pixel size box and font size box(is it a number).

SaveButtonCallback = 'SetPreferencesWindowHandle = findobj(''name'',''SetPreferences''); global EnteredPreferences, PixelSizeEditBoxHandle = findobj(''Tag'',''PixelSizeEditBox''); FontSizeEditBoxHandle = findobj(''Tag'',''FontSizeEditBox''); ImageDirEditBoxHandle = findobj(''Tag'',''ImageDirEditBox''); OutputDirEditBoxHandle = findobj(''Tag'',''OutputDirEditBox''); ModuleDirEditBoxHandle = findobj(''Tag'',''ModuleDirEditBox''); PixelSize = get(PixelSizeEditBoxHandle,''string''); PixelSize = PixelSize{1}; FontSize = get(FontSizeEditBoxHandle,''string''); DefaultImageDirectory = get(ImageDirEditBoxHandle,''string''); DefaultOutputDirectory = get(OutputDirEditBoxHandle,''string''); DefaultModuleDirectory = get(ModuleDirEditBoxHandle,''string''); EnteredPreferences.PixelSize = PixelSize; EnteredPreferences.FontSize = FontSize; EnteredPreferences.DefaultImageDirectory = DefaultImageDirectory; EnteredPreferences.DefaultOutputDirectory = DefaultOutputDirectory; EnteredPreferences.DefaultModuleDirectory = DefaultModuleDirectory; SavedPreferences = EnteredPreferences; CurrentDir = pwd; try, save(fullfile(matlabroot,''CellProfilerPreferences.mat''),''SavedPreferences''), clear SavedPreferences, helpdlg(''Your CellProfiler preferences were successfully set.  They are contained in a file called CellProfilerPreferences.mat in the Matlab root directory.''), catch, try save(fullfile(handles.Current.StartupDirectory, ''CellProfilerPreferences.mat''),''SavedPreferences''), clear SavedPreferences, helpdlg(''You do not have permission to write anything to the Matlab root directory, which is required to save your preferences permanently.  Instead, your preferences will only function properly when you start CellProfiler from the current directory.''), catch, helpdlg(''CellProfiler was unable to save your desired preferences, probably because you lack write permission for both the Matlab root directory as well as the current directory.  Your preferences will only be saved for the current session of CellProfiler. (Note: this feature is still a bit buggy if you do not have write access to the Matlab root directory).''); end, end, clear PixelSize* *Dir* , close(SetPreferencesWindowHandle), clear SetPreferencesWindowHandle FontSize FontSizeEditBoxHandle';
CancelButtonCallback = 'delete(gcf)';

%%% Creates the dialog box and its text, buttons, and edit boxes.
MainWinPos = get(handles.figure1,'Position');
Color = [0.7 0.7 0.7];

%%% Label we attach to figures (as UserData) so we know they are ours
userData.Application = 'CellProfiler';
SetPreferencesWindowHandle = figure(...
'Units','pixels',...
'Color',Color,...
'DockControls','off',...
'MenuBar','none',...
'Name','SetPreferences',...
'NumberTitle','off',...
'Position',[MainWinPos(1)+MainWinPos(3)/3 MainWinPos(2) MainWinPos(3)*2/3 MainWinPos(4)],...
'Resize','off',...
'HandleVisibility','on',...
'Tag','figure1',...
'UserData',userData);

InfoText = uicontrol(...
'Parent',SetPreferencesWindowHandle,...
'Units','normalized',...
'BackgroundColor',Color,...
'FontName','Times',...
'FontSize',handles.Current.FontSize,...
'FontWeight','bold',...
'HorizontalAlignment','left',...
'Position',[0.05 0.6 0.9 0.35],...
'String','Your preferences will be stored in a file called CellProfilerPreferences.mat which will be saved in the Matlab root directory, if you have write access there. Typing matlabroot at the command line will show you the Matlab root directory. The data will then be loaded every time you launch CellProfiler. If you do not have write access to the Matlab root directory, CellProfilerPreferences.mat will be saved in the current directory, and the data will only be used when you start CellProfiler from that directory. If you do not have write access to the current directory either, your preferences will be used only for the current session of CellProfiler.',...
'Style','text');

PixelSizeText = uicontrol(...
'Parent',SetPreferencesWindowHandle,...
'Units','normalized',...
'Callback',PixelSizeEditBoxCallback,...
'BackgroundColor',Color,...
'FontName','Times',...
'FontSize',handles.Current.FontSize,...
'FontWeight','bold',...
'HorizontalAlignment','left',...
'Position',[0.1 0.55 0.6 0.04],...
'String','Enter the default pixel size (in micrometers)',...
'Style','text');
PixelSizeEditBox = uicontrol(...
'Parent',SetPreferencesWindowHandle,...
'Units','normalized',...
'BackgroundColor',[1 1 1],...
'FontName','Times',...
'FontSize',handles.Current.FontSize,...
'FontWeight','bold',...
'Position',[0.6 0.55 0.1 0.04],...
'String',handles.Preferences.PixelSize,...
'Style','edit',...
'Tag','PixelSizeEditBox');

FontSizeText = uicontrol(...
'Parent',SetPreferencesWindowHandle,...
'Units','normalized',...
'BackgroundColor',Color,...
'FontName','Times',...
'FontSize',handles.Current.FontSize,...
'FontWeight','bold',...
'HorizontalAlignment','left',...
'Position',[0.1 0.5 0.6 0.04],...
'String','Enter the default font size',...
'Style','text');
FontSizeEditBox = uicontrol(...
'Parent',SetPreferencesWindowHandle,...
'Units','normalized',...
'BackgroundColor',[1 1 1],...
'FontName','Times',...
'FontSize',handles.Current.FontSize,...
'FontWeight','bold',...
'Position',[0.6 0.5 0.1 0.04],...
'String',num2str(round(handles.Current.FontSize)),...
'Style','edit',...
'Tag','FontSizeEditBox');

ImageDirTextBox = uicontrol(...
'Parent',SetPreferencesWindowHandle,...
'Units','normalized',...
'BackgroundColor',Color,...
'FontName','Times',...
'FontSize',handles.Current.FontSize,...
'FontWeight','bold',...
'HorizontalAlignment','left',...
'Position',[0.1 0.4 0.6 0.04],...
'String','Select the default image directory:',...
'Style','text');
ImageDirBrowseButton = uicontrol(...
'Parent',SetPreferencesWindowHandle,...
'Units','normalized',...
'Callback',ImageDirBrowseButtonCallback,...
'FontName','Times',...
'FontSize',handles.Current.FontSize,...
'FontWeight','bold',...
'Position',[0.74 0.36 0.12 0.05],...
'String','Browse...',...
'Tag','ImageDirBrowseButton');
ImageDirEditBox = uicontrol(...
'Parent',SetPreferencesWindowHandle,...
'Units','normalized',...
'BackgroundColor',[1 1 1],...
'Callback',ImageDirEditBoxCallback,...
'FontName','Times',...
'FontSize',handles.Current.FontSize,...
'FontWeight','bold',...
'Position',[0.1 0.36 0.6 0.05],...
'String',handles.Preferences.DefaultImageDirectory,...
'Style','edit',...
'Tag','ImageDirEditBox');

OutputDirTextBox = uicontrol(...
'Parent',SetPreferencesWindowHandle,...
'Units','normalized',...
'BackgroundColor',Color,...
'FontName','Times',...
'FontSize',handles.Current.FontSize,...
'FontWeight','bold',...
'HorizontalAlignment','left',...
'Position',[0.1 0.3 0.6 0.04],...
'String','Select the default directory for output:',...
'Style','text');
OutputDirBrowseButton = uicontrol(...
'Parent',SetPreferencesWindowHandle,...
'Units','normalized',...
'Callback',OutputDirBrowseButtonCallback,...
'FontName','Times',...
'FontSize',handles.Current.FontSize,...
'FontWeight','bold',...
'Position',[0.74 0.26 0.12 0.05],...
'String','Browse...',...
'Tag','OutputDirBrowseButton');
OutputDirEditBox = uicontrol(...
'Parent',SetPreferencesWindowHandle,...
'Units','normalized',...
'BackgroundColor',[1 1 1],...
'Callback',OutputDirEditBoxCallback,...
'FontName','Times',...
'FontSize',handles.Current.FontSize,...
'FontWeight','bold',...
'Position',[0.1 0.26 0.6 0.05],...
'String',handles.Preferences.DefaultOutputDirectory,...
'Style','edit',...
'Tag','OutputDirEditBox');

ModuleDirTextBox = uicontrol(...
'Parent',SetPreferencesWindowHandle,...
'Units','normalized',...
'BackgroundColor',Color,...
'FontName','Times',...
'FontSize',handles.Current.FontSize,...
'FontWeight','bold',...
'HorizontalAlignment','left',...
'Position',[0.1 0.2 0.6 0.04],...
'String','Select the directory where CellProfiler modules are stored:',...
'Style','text');
ModuleDirBrowseButton = uicontrol(...
'Parent',SetPreferencesWindowHandle,...
'Units','normalized',...
'Callback',ModuleDirBrowseButtonCallback,...
'FontName','Times',...
'FontSize',handles.Current.FontSize,...
'FontWeight','bold',...
'Position',[0.74 0.16 0.12 0.05],...
'String','Browse...',...
'Tag','ModuleDirBrowseButton');
ModuleDirEditBox = uicontrol(...
'Parent',SetPreferencesWindowHandle,...
'Units','normalized',...
'BackgroundColor',[1 1 1],...
'Callback',ModuleDirEditBoxCallback,...
'FontName','Times',...
'FontSize',handles.Current.FontSize,...
'FontWeight','bold',...
'Position',[0.1 0.16 0.6 0.05],...
'String',handles.Preferences.DefaultModuleDirectory,...
'Style','edit',...
'Tag','ModuleDirEditBox');

SaveButton = uicontrol(...
'Parent',SetPreferencesWindowHandle,...
'Units','normalized',...
'Callback',SaveButtonCallback,...
'FontName','Times',...
'FontSize',handles.Current.FontSize,...
'FontWeight','bold',...
'Position',[0.2 0.04 0.2 0.06],...
'String','Save preferences',...
'Tag','SaveButton');
CancelButton = uicontrol(...
'Parent',SetPreferencesWindowHandle,...
'Units','normalized',...
'Callback',CancelButtonCallback,...
'FontName','Times',...
'FontSize',handles.Current.FontSize,...
'FontWeight','bold',...
'Position',[0.6 0.04 0.2 0.06],...
'String','Cancel',...
'Tag','CancelButton');


%%% Waits for the user to respond to the window.
uiwait(SetPreferencesWindowHandle)
%%% Allows canceling by checking whether EnteredPreferences exists.
if exist('EnteredPreferences','var') == 1
    if isempty(EnteredPreferences) ~= 1
        %%% Retrieves the data that the user entered and saves it to the
        %%% handles structure.
        handles.Preferences.PixelSize = EnteredPreferences.PixelSize;
        handles.Preferences.FontSize  = EnteredPreferences.FontSize;
        handles.Preferences.DefaultImageDirectory = EnteredPreferences.DefaultImageDirectory;
        handles.Preferences.DefaultOutputDirectory = EnteredPreferences.DefaultOutputDirectory;
        handles.Preferences.DefaultModuleDirectory = EnteredPreferences.DefaultModuleDirectory;
        clear global EnteredPreferences
        
        %%% Now that handles.Preferences.(5 different variables) has been filled
        %%% in, the handles.Current values and edit box displays are set.
        handles.Current.DefaultOutputDirectory = handles.Preferences.DefaultOutputDirectory;
        handles.Current.DefaultImageDirectory = handles.Preferences.DefaultImageDirectory;
        handles.Current.PixelSize = handles.Preferences.PixelSize;
        handles.Current.FontSize  = str2num(handles.Preferences.FontSize);
        handles.Settings.PixelSize = handles.Preferences.PixelSize;

        %%% (No need to set a current module directory or display it in an
        %%% edit box; the one stored in preferences is the only one ever
        %%% used).
        set(handles.PixelSizeEditBox,'String',handles.Preferences.PixelSize)
        set(handles.DefaultOutputDirectoryEditBox,'String',handles.Preferences.DefaultOutputDirectory)
        set(handles.DefaultImageDirectoryEditBox,'String',handles.Preferences.DefaultImageDirectory)
        %%% Retrieves the list of image file names from the chosen directory,
        %%% stores them in the handles structure, and displays them in the
        %%% filenameslistbox, by faking a click on the DefaultImageDirectoryEditBox.
        handles = DefaultImageDirectoryEditBox_Callback(hObject, eventdata, handles);
        %%% Adds the default module directory to Matlab's search path.
        addpath(handles.Preferences.DefaultModuleDirectory)
        
        %%% Set new fontsize
        names = fieldnames(handles);
        for k = 1: length(names)
            if ishandle(handles.(names{k}))
                set(findobj(handles.(names{k}),'-property','FontSize'),'FontSize',handles.Current.FontSize,'FontName','Times')
            end
        end
        set(0,'UserData',handles.Current.FontSize)
        %%% Updates the handles structure to incorporate all the changes.
        guidata(gcbo, handles);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% TECHNICAL DIAGNOSIS BUTTON %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes on button press in TechnicalDiagnosisButton.
function TechnicalDiagnosisButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
%%% This button shows the handles structure in the main Matlab window.
%%% When running a GUI, typing these lines at the command line of
%%% Matlab is useless, because the CellProfiler GUI's workspace and
%%% the main workspace is not shared.
try MainHandles = handles, catch MainHandles = 'Does not exist', end %#ok We want to ignore MLint error checking for this line.
try Preferences = handles.Preferences, catch Preferences = 'Does not exist', end %#ok We want to ignore MLint error checking for this line.
try Current = handles.Current, catch Current = 'Does not exist', end %#ok We want to ignore MLint error checking for this line.
try Settings = handles.Settings, catch Settings = 'Does not exist', end %#ok We want to ignore MLint error checking for this line.
try Pipeline = handles.Pipeline, catch Pipeline = 'Does not exist', end %#ok We want to ignore MLint error checking for this line.
try Measurements = handles.Measurements, catch Measurements = 'Does not exist', end %#ok We want to ignore MLint error checking for this line.
CPmsgbox('The handles structure has been printed out at the command line of Matlab.')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% BROWSE DEFAULT IMAGE DIRECTORY BUTTON %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes on button press in BrowseImageDirectoryButton.
function BrowseImageDirectoryButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
%%% Opens a dialog box to allow the user to choose a directory and loads
%%% that directory name into the edit box.  Also, changes the current
%%% directory to the chosen directory.
if exist(handles.Current.DefaultImageDirectory, 'dir')
    pathname = uigetdir(handles.Current.DefaultImageDirectory,'Choose the directory of images to be analyzed');
else
    pathname = uigetdir('','Choose the directory of images to be analyzed');
end
%%% If the user presses "Cancel", the pathname will = 0 and nothing will
%%% happen.
if pathname == 0
else
    %%% Saves the pathname in the handles structure.
    handles.Current.DefaultImageDirectory = pathname;
    %%% Displays the chosen directory in the DefaultImageDirectoryEditBox.
    set(handles.DefaultImageDirectoryEditBox,'String',pathname);
    guidata(hObject,handles)
    %%% Retrieves the list of image file names from the chosen directory,
    %%% stores them in the handles structure, and displays them in the
    %%% filenameslistbox, by faking a click in the DefaultImageDirectoryEditBox.
    handles = DefaultImageDirectoryEditBox_Callback(hObject, eventdata, handles);
    guidata(hObject, handles);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% DEFAULT IMAGE DIRECTORY EDIT BOX %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes during object creation, after setting all properties.
function DefaultImageDirectoryEditBox_CreateFcn(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.

function handles = DefaultImageDirectoryEditBox_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
%%% Retrieves the text that was typed in.
pathname = get(handles.DefaultImageDirectoryEditBox,'string');
%%% Checks whether a directory with that name exists.
if exist(pathname,'dir') ~= 0
    %%% Saves the pathname in the handles structure.
    handles.Current.DefaultImageDirectory = pathname;
    guidata(hObject,handles)
    %%% Retrieves the list of image file names from the chosen directory and
    %%% stores them in the handles structure, using the function
    %%% RetrieveImageFileNames.
    handles = RetrieveImageFileNames(handles,pathname);
    %%% Test whether this is during CellProfiler launching or during
    %%% the image analysis run itself (by looking at some of the GUI
    %%% elements). If either is the case, the message is NOT
    %%% shown.
    ListBoxContents = get(handles.FilenamesListBox,'String');
    IsStartup = strcmp(ListBoxContents(1),'Listbox');
    IsAnalysisRun = strcmp(get(handles.PipelineOfModulesText,'visible'),'off');
    if any([IsStartup, IsAnalysisRun]) == 0 && isempty(handles.Current.FilenamesInImageDir) == 1;
        %%% Obtains the screen size and determines where the wait bar
        %%% will be displayed.
        ScreenSize = get(0,'ScreenSize');
        ScreenHeight = ScreenSize(4);
        PotentialBottom = [0, (ScreenHeight-720)];
        BottomOfMsgBox = max(PotentialBottom);
        PositionMsgBox = [500 BottomOfMsgBox 350 100];
        ErrorDlgHandle = CPmsgbox('Please note: There are no files in the default image directory, as specified in the main CellProfiler window.');
        set(ErrorDlgHandle, 'Position', PositionMsgBox)
        drawnow
    end
    guidata(hObject, handles);
    %%% If the directory entered in the box does not exist, give an error
    %%% message, change the contents of the edit box back to the
    %%% previously selected directory, and change the contents of the
    %%% filenameslistbox back to the previously selected directory.
else 
    errordlg('A directory with that name does not exist');
end
%%% Whether or not the directory exists and was updated, we want to
%%% update the GUI display to show the currrently stored information.
%%% Display the path in the edit box.
set(handles.DefaultImageDirectoryEditBox,'String',handles.Current.DefaultImageDirectory);
if isempty(handles.Current.FilenamesInImageDir)
    set(handles.FilenamesListBox,'String','No image files recognized',...
        'Value',1)
else
    %%% Loads these image names into the FilenamesListBox.
    set(handles.FilenamesListBox,'String',handles.Current.FilenamesInImageDir,...
        'Value',1)
end
%%% Updates the handles structure.
guidata(hObject,handles)

%%% SUBFUNCTION %%%
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

if isempty(FileNamesNoDir) == 1
    handles.Current.FilenamesInImageDir = [];
else
    %%% Makes a logical array that marks with a "1" all file names that start
    %%% with a period (hidden files):
    DiscardLogical1 = strncmp(FileNamesNoDir,'.',1);
    %%% Makes logical arrays that mark with a "1" all file names that have
    %%% particular suffixes (mat, m, m~, and frk). The dollar sign indicates
    %%% that the pattern must be at the end of the string in order to count as
    %%% matching.  The first line of each set finds the suffix and marks its
    %%% location in a cell array with the index of where that suffix begins;
    %%% the third line converts this cell array of numbers into a logical
    %%% array of 1's and 0's.   cellfun only works on arrays of class 'cell',
    %%% so there is a check to make sure the class is appropriate.  When there
    %%% are very few files in the directory (I think just one), the class is
    %%% not cell for some reason.
    DiscardsByExtension = regexpi(FileNamesNoDir, '\.(m|mat|m~|frk~|xls|doc|rtf|txt|csv)$', 'once');
    if strcmp(class(DiscardsByExtension), 'cell')
        DiscardsByExtension = cellfun('prodofsize',DiscardsByExtension);
    else
        DiscardsByExtension = [];
    end
    %%% Combines all of the DiscardLogical arrays into one.
    DiscardLogical = DiscardLogical1 | DiscardsByExtension;
    %%% Eliminates filenames to be discarded.
    if isempty(DiscardLogical)
        FileNames = FileNamesNoDir;
    else
        FileNames = FileNamesNoDir(~DiscardLogical);
    end
    %%% Checks whether any files are left.
    if isempty(FileNames)
        handles.Current.FilenamesInImageDir = [];
    else
        %%% Stores the final list of file names in the handles structure
        handles.Current.FilenamesInImageDir = FileNames;
        guidata(handles.figure1,handles);
    end
end
guidata(handles.figure1,handles);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% BROWSE DEFAULT OUTPUT DIRECTORY BUTTON %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes on button press in BrowseOutputDirectoryButton.
function BrowseOutputDirectoryButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.

%%% Opens a dialog box to allow the user to choose a directory and loads
%%% that directory name into the edit box.  Also, changes the current
%%% directory to the chosen directory.
pathname = uigetdir(handles.Current.DefaultOutputDirectory,'Choose the default output directory');
%%% If the user presses "Cancel", the pathname will = 0 and nothing will
%%% happen.
if pathname == 0
else
    %%% Saves the pathname in the handles structure.
    handles.Current.DefaultOutputDirectory = pathname;
    %%% Displays the chosen directory in the DefaultImageDirectoryEditBox.
    set(handles.DefaultOutputDirectoryEditBox,'String',pathname);
    guidata(hObject,handles)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% DEFAULT OUTPUT DIRECTORY EDIT BOX %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes during object creation, after setting all properties.
function DefaultOutputDirectoryEditBox_CreateFcn(hObject, eventdata, handles)

function DefaultOutputDirectoryEditBox_Callback(hObject, eventdata, handles)
%%% Retrieves the text that was typed in.
pathname = get(handles.DefaultOutputDirectoryEditBox,'string');
%%% Checks whether a directory with that name exists.
if exist(pathname,'dir') ~= 0
    %%% Saves the pathname in the handles structure.
    handles.Current.DefaultOutputDirectory = pathname;
    %%% If the directory entered in the box does not exist, give an error
    %%% message, change the contents of the edit box back to the
    %%% previously selected directory, and change the contents of the
    %%% filenameslistbox back to the previously selected directory.
else 
    errordlg('A directory with that name does not exist');
end
%%% Whether or not the directory exists and was updated, we want to
%%% update the GUI display to show the currrently stored information.
%%% Display the path in the edit box.
set(handles.DefaultOutputDirectoryEditBox,'String',handles.Current.DefaultOutputDirectory);
%%% Updates the handles structure.
guidata(hObject,handles)

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE LIST BOX %%%
%%%%%%%%%%%%%%%%%%%%%

% --- Executes during object creation, after setting all properties.
function FilenamesListBox_CreateFcn(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.

% --- Executes on selection change in FilenamesListBox.
function FilenamesListBox_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
%%% The list box has no function other than to display its contents,
%%% so there is no code here.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE AND DATA TOOLS POPUP MENUS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function ImageToolsPopUpMenu_CreateFcn(hObject, eventdata, handles)
function DataToolsPopUpMenu_CreateFcn(hObject, eventdata, handles)

function ImageToolsPopUpMenu_Callback(hObject, eventdata, handles)
handles = ToolsPopUpMenuCallbackSubfunction(handles, 'Image');

function DataToolsPopUpMenu_Callback(hObject, eventdata, handles)
handles = ToolsPopUpMenuCallbackSubfunction(handles, 'Data');

%%% SUBFUNCTION %%%
function handles = ToolsPopUpMenuCallbackSubfunction(handles, ImageOrData)
if strcmp(ImageOrData, 'Image') == 1
    ToolsFolder = 'ImageTools';
    NoneLoadedText = 'Image tools: none loaded';
    PopUpMenuLabel = 'Image tools';
    PopUpMenuHandle = 'ImageToolsPopUpMenu';
elseif strcmp(ImageOrData, 'Data') == 1
    ToolsFolder = 'DataTools';
    NoneLoadedText = 'Data tools: none loaded';
    PopUpMenuLabel = 'Data tools';
    PopUpMenuHandle = 'DataToolsPopUpMenu';
end
%%% Determines which Tool was selected from the list.
SelectedValue = get(handles.(PopUpMenuHandle),'Value');
ListOfTools = get(handles.(PopUpMenuHandle),'String');
SelectedTool = ListOfTools{SelectedValue};
%%% If the first entry is selected (which is just the title entry),
%%% nothing is done.
if strcmp(SelectedTool, PopUpMenuLabel) == 1
elseif strcmp(SelectedTool, NoneLoadedText) == 1
    errordlg(['There are no tools loaded, because CellProfiler could not find the ',ToolsFolder, ' directory, which should be located within the directory where the current CellProfiler.m resides.']);
else
    try eval(['handles = ', SelectedTool,'(handles);'])
    catch 
        %%% TODO: We should implement something where if
        %%% the last error was actually just canceling by the user
        %%% within the function, no error box is opened. Maybe this
        %%% already works automatically; not sure.
        errordlg(['An error occurred while attempting to run the tool you selected.  The error was "' lasterr '"']);
     end
end
%%% Resets the display to the first position (so "Data tools" is
%%% displayed).
 %%% TODO: Sometimes there is an error here - not sure why.
%%% handles.DataToolsPopUpMenu returns a numerical handle, but neither
%% the set line below, nor this line:
%%% get(handles.DataToolsPopUpMenu,'color')
%%% works. Both yield ? Invalid handle object errors.
try
set(handles.(PopUpMenuHandle),'Value',1)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% CLOSE WINDOWS BUTTON %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes on button press in CloseWindowsButton.
function CloseWindowsButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.

%%% Requests confirmation to really delete all the figure windows.
Answer = CPquestdlg('Are you sure you want to close all open figure windows, timers, and message boxes?','Confirm','Yes','No','Yes');
if strcmp(Answer, 'Yes') == 1
    %%% All CellProfiler windows are now marked with 
    %%%      UserData.Application = 'CellProfiler'
    %%% so they can be found and deleted. This will get rid both windows
    %%% from current CP session and leftover windows from previous CP runs
    %%% (e.g., if just close CP with windows still open)
    GraphicsHandles = findobj('-property','UserData');
    for k=1:length(GraphicsHandles)
        if (ishandle(GraphicsHandles(k)))
            userData = get(GraphicsHandles(k),'UserData');
            if (isfield(userData,'Application') && ...
                isstr(userData.Application) && ...
                strcmp(userData.Application, 'CellProfiler'))
                %%% Closes the figure windows.
                try
                    delete(GraphicsHandles(k));
                end
            end
        end
    end
    %%% Finds and closes timer windows, which have HandleVisibility off.
    TimerHandles = findall(findobj, 'Name', 'Timer');
    try
        delete(TimerHandles)
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% OUTPUT FILE NAME EDIT BOX %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes during object creation, after setting all properties.
function OutputFileNameEditBox_CreateFcn(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.

function OutputFileNameEditBox_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.

Pathname = handles.Current.DefaultOutputDirectory;
UserEntry = strtrim(get(handles.OutputFileNameEditBox,'string'));
if ~isempty(UserEntry)
    % Drop '.mat' if the user entered it
    if strfind(UserEntry,'.mat')
        UserEntry = UserEntry(1:end-4);
    end
    % If there is no 'OUT' in the filename, add it.
    if isempty(findstr(lower(UserEntry),'out'))
        UserEntry = [UserEntry 'OUT'];
    end
    % Find the files with the same base name and extract highest number
    % If the dir-command takes a long time when there are a lot of files
    % in a directory, another solution might be to try different numberings
    % until an unused number is encountered.
    if exist(fullfile(Pathname,[UserEntry '.mat']),'file')
        % Find base name
        index = findstr(UserEntry,'__');
        if ~isempty(index)
            UserEntry = UserEntry(1:index(end)-1);
        end
        d = dir(Pathname);
        numbers = [];
        for k = 1:length(d);
            index = findstr(d(k).name,[UserEntry '__']);
            if ~isempty(index)
                numbers = [numbers str2num(d(k).name(index(end)+length(UserEntry)+2:end-4))];
            end
        end
        if isempty(numbers)
            outputnumber = 1;
        else
            outputnumber = max(numbers) + 1;
        end
        set(handles.OutputFileNameEditBox,'string',sprintf('%s__%d.mat',UserEntry,outputnumber))
        Handle = CPmsgbox('The output file already exists. A new file name has been generated.','Output file name has changed');
    else
        set(handles.OutputFileNameEditBox,'string',[UserEntry '.mat'])
    end
    drawnow
end
guidata(gcbo, handles);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% ANALYZE IMAGES BUTTON %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes on button press in AnalyzeImagesButton.
function AnalyzeImagesButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
global closeFigures openFigures;
%%% Checks whether any modules are loaded.
sum = 0; %%% Initial value.
for i = 1:handles.Current.NumberOfModules;
    sum = sum + iscellstr(handles.Settings.ModuleNames(i));
end
if sum == 0
    errordlg('You do not have any analysis modules loaded');
else    
    %%% Call Callback function of FileNameEditBox to update filename
    tmp = get(handles.OutputFileNameEditBox,'string');
    OutputFileNameEditBox_Callback(hObject, eventdata, handles)
    if ~strcmp(tmp,get(handles.OutputFileNameEditBox,'string'))
        %%% Finds and closes the message box produced by the
        %%% OutputFileNameEditBox_Callback so that an alternate
        %%% warning can appear.
        HandleOfMsgBoxToDelete = findobj('name','Output file name has changed');
        close(HandleOfMsgBoxToDelete)
        Answer = CPquestdlg('The output file already exists. A new file name has been generated. Continue?','Output file exists','Yes','Cancel','Yes');
        if strcmp(Answer,'Cancel')
            return
        end
    end
        
    %%% Checks whether an output file name has been specified.
    if isempty(get(handles.OutputFileNameEditBox,'string'))
        errordlg('You have not entered an output file name in Step 2.');
    else
        %%% Checks whether the default output directory exists.
        if ~exist(handles.Current.DefaultOutputDirectory, 'dir')
            errordlg('The default output directory does not exist');
        end
        %%% Checks whether the default image directory exists
        if ~exist(handles.Current.DefaultImageDirectory, 'dir')
            errordlg('The default image directory does not exist');
        else
                %%% Retrieves the list of image file names from the
                %%% chosen directory, stores them in the handles
                %%% structure, and displays them in the filenameslistbox, by
                %%% faking a click on the DefaultImageDirectoryEditBox. This
                %%% should already have been done when the directory
                %%% was chosen, but in case some files were moved or
                %%% changed in the meantime, this will refresh the
                %%% list.
                set(handles.PipelineOfModulesText,'visible','off')
                handles = DefaultImageDirectoryEditBox_Callback(hObject, eventdata, handles);
                %%% Updates the handles structure.
                guidata(gcbo, handles);
                %%% Disables a lot of the buttons on the GUI so that the program doesn't
                %%% get messed up.  The Help buttons are left enabled.
                set(handles.PipelineOfModulesText,'visible','off')
                set(handles.LoadPipelineButton,'visible','off')
                set(handles.SavePipelineButton,'visible','off')
                set(handles.IndividualModulesText,'visible','off')
                set(handles.AddModule,'visible','off');
                set(handles.RemoveModule,'visible','off');
                set(handles.MoveUpButton,'visible','off');
                set(handles.MoveDownButton,'visible','off');
                set(handles.PixelSizeEditBox,'enable','inactive','foregroundcolor',[0.7,0.7,0.7])
                set(handles.SetPreferencesButton,'enable','off')
                set(handles.BrowseImageDirectoryButton,'enable','off')
                set(handles.DefaultImageDirectoryEditBox,'enable','inactive','foregroundcolor',[0.7,0.7,0.7])
                set(handles.BrowseOutputDirectoryButton,'enable','off')
                set(handles.DefaultOutputDirectoryEditBox,'enable','inactive','foregroundcolor',[0.7,0.7,0.7])
                set(handles.ImageToolsPopUpMenu,'enable','off')
                set(handles.DataToolsPopUpMenu,'enable','off')
                set(handles.CloseWindowsButton,'enable','off')
                set(handles.OutputFileNameEditBox,'enable','inactive','foregroundcolor',[0.7,0.7,0.7])
                set(handles.AnalyzeImagesButton,'enable','off')
                % FIXME: This should loop just over the number of actual variables in the display.
                for VariableNumber=1:99;
                    set(handles.(['VariableBox' TwoDigitString(VariableNumber)]),'enable','inactive','foregroundcolor',[0.7,0.7,0.7]);
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
                %%% Label we attach to figures (as UserData) so we know they are ours
                userData.Application = 'CellProfiler';
                timer_handle = figure('name','Timer','position',[0 BottomOfTimer 495 120],...
                    'menubar','none','NumberTitle','off','IntegerHandle','off', 'HandleVisibility', 'off', ...
                    'color',[0.7,0.7,0.9],'UserData',userData);
                %%% Sets initial text to be displayed in the text box within the timer window.
                timertext = 'First image set is being processed';
                %%% Creates the text box within the timer window which will display the
                %%% timer text.
                text_handle = uicontrol(timer_handle,'string',timertext,'style','text',...
                    'parent',timer_handle,'position', [0 40 494 74],'FontName','Times',...
                    'FontSize',handles.Current.FontSize,'FontWeight','bold','BackgroundColor',[0.7,0.7,0.9]);
                %%% Saves text handle to the handles structure.
                handles.timertexthandle = text_handle;
                %%% Creates the Cancel and Pause buttons.
                PauseButton_handle = uicontrol('Style', 'pushbutton', ...
                    'String', 'Pause', 'Position', [5 10 40 30], ...
                    'parent',timer_handle, 'BackgroundColor',[0.7,0.7,0.9],'FontName','Times','FontSize',handles.Current.FontSize);
                CancelAfterImageSetButton_handle = uicontrol('Style', 'pushbutton', ...
                    'String', 'Cancel after image set', 'Position', [50 10 120 30], ...
                    'parent',timer_handle, 'BackgroundColor',[0.7,0.7,0.9],'FontName','Times','FontSize',handles.Current.FontSize);
                CancelAfterModuleButton_handle = uicontrol('Style', 'pushbutton', ...
                    'String', 'Cancel after module', 'Position', [175 10 115 30], ...
                    'parent',timer_handle, 'BackgroundColor',[0.7,0.7,0.9],'FontName','Times','FontSize',handles.Current.FontSize);
                CancelNowCloseButton_handle = uicontrol('Style', 'pushbutton', ...
                    'String', 'Cancel & close CellProfiler', 'Position', [295 10 160 30], ...
                    'parent',timer_handle, 'BackgroundColor',[0.7,0.7,0.9],'FontName','Times','FontSize',handles.Current.FontSize);

                %%% Sets the functions to be called when the Cancel and Pause buttons
                %%% within the Timer window are pressed.
                PauseButtonFunction = 'h = CPmsgbox(''Image processing is paused without causing any damage. Processing will restart when you close the Pause window or click OK.''); waitfor(h); clear h;';
                set(PauseButton_handle,'Callback', PauseButtonFunction)
                CancelAfterImageSetButtonFunction = ['deleteme = CPquestdlg(''Paused. Are you sure you want to cancel after this image set? Processing will continue on the current image set, the data up to and including the current image set will be saved in the output file, and then the analysis will be canceled.'', ''Confirm cancel'',''Yes'',''No'',''Yes''); switch deleteme; case ''Yes''; set(',num2str(CancelAfterImageSetButton_handle*8192), '/8192,''enable'',''off''); set(', num2str(text_handle*8192), '/8192,''string'',''Canceling in progress; Waiting for the processing of current image set to be complete. You can press the Cancel after module button to cancel more quickly, but data relating to the current image set will not be saved in the output file.''); case ''No''; return; end; clear deleteme'];
                set(CancelAfterImageSetButton_handle, 'Callback', CancelAfterImageSetButtonFunction)
                CancelAfterModuleButtonFunction = ['deleteme = CPquestdlg(''Paused. Are you sure you want to cancel after this module? Processing will continue until the current image analysis module is completed, to avoid corrupting the current settings of CellProfiler. Data up to the *previous* image set are saved in the output file and processing is canceled.'', ''Confirm cancel'',''Yes'',''No'',''Yes''); switch deleteme; case ''Yes''; set(', num2str(CancelAfterImageSetButton_handle*8192), '/8192,''enable'',''off''); set(', num2str(CancelAfterModuleButton_handle*8192), '/8192,''enable'',''off''); set(', num2str(text_handle*8192), '/8192,''string'',''Immediate canceling in progress; Waiting for the processing of current module to be complete in order to avoid corrupting the current CellProfiler settings.''); case ''No''; return; end; clear deleteme'];
                set(CancelAfterModuleButton_handle,'Callback', CancelAfterModuleButtonFunction)
                CancelNowCloseButtonFunction = ['deleteme = CPquestdlg(''Paused. Are you sure you want to cancel immediately and close CellProfiler? The CellProfiler program will close, losing your current settings. The data up to the *previous* image set will be saved in the output file, but the current image set data will be stored incomplete in the output file, which might be confusing when using the output file.'', ''Confirm cancel & close'',''Yes'',''No'',''Yes''); helpdlg(''The CellProfiler program should have closed itself. Important: Go to the command line of Matlab and press Control-C to stop processes in progress. Then type clear and press the enter key at the command line.  Figure windows will not close properly: to close them, type delete(N) at the command line of Matlab, where N is the figure number. The data up to the *previous* image set will be saved in the output file, but the current image set data will be stored incomplete in the output file, which might be confusing when using the output file.''), switch deleteme; case ''Yes''; delete(', num2str((handles.figure1)*8192), '/8192); case ''No''; return; end; clear deleteme'];
                set(CancelNowCloseButton_handle,'Callback', CancelNowCloseButtonFunction)
                HelpButtonFunction = 'CPmsgbox(''Pause button: The current processing is immediately suspended without causing any damage. Processing restarts when you close the Pause window or click OK. Cancel after image set: Processing will continue on the current image set, the data up to and including the current image set will be saved in the output file, and then the analysis will be canceled.  Cancel after module: Processing will continue until the current image analysis module is completed, to avoid corrupting the current settings of CellProfiler. Data up to the *previous* image set are saved in the output file and processing is canceled. Cancel now & close CellProfiler: CellProfiler will immediately close itself. The data up to the *previous* image set will be saved in the output file, but the current image set data will be stored incomplete in the output file, which might be confusing when using the output file.'')';
                %%% HelpButton
                uicontrol('Style', 'pushbutton', ...
                    'String', '?', 'Position', [460 10 15 30], 'FontName','Times','FontSize', handles.Current.FontSize,...
                    'Callback', HelpButtonFunction, 'parent',timer_handle, 'BackgroundColor',[0.7,0.7,0.9]);
                
                %%% The timertext string is read by the analyze all images button's callback
                %%% at the end of each time around the loop (i.e. at the end of each image
                %%% set).  If it notices that the string says "Cancel...", it breaks out of
                %%% the loop and finishes up.

                %%% Update the handles structure. Not sure if it's necessary here.
                guidata(gcbo, handles);
                %%% Sets the timer window to show a warning box before allowing it to be
                %%% closed.
                CloseFunction = ['deleteme = CPquestdlg(''DO NOT CLOSE the Timer window while image processing is in progress!! Are you sure you want to close the timer?'', ''Confirm close'',''Yes'',''No'',''Yes''); switch deleteme; case ''Yes''; delete(',num2str(timer_handle*8192), '/8192); case ''No''; return; end; clear deleteme'];
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

                %%% If a module is chosen in this slot, assign it an output figure
                %%% window and write the figure window number to the handles structure so
                %%% that the modules know where to write to.  Each module should
                %%% resize the figure window appropriately.  The closing function of the
                %%% figure window is set to wait until an image set is done processing
                %%% before closing the window, to avoid unexpected results.              
                set(handles.CloseFigureButton,'visible','on')
                set(handles.OpenFigureButton,'visible','on')

                %%% Label we attach to figures (as UserData) so we know they are ours
                userData.Application = 'CellProfiler';
                for i=1:handles.Current.NumberOfModules;
                    if iscellstr(handles.Settings.ModuleNames(i)) == 1
                        handles.Current.(['FigureNumberForModule' TwoDigitString(i)]) = ...
                            figure('name',[char(handles.Settings.ModuleNames(i)), ' Display'],...
                            'Position',[(ScreenWidth*((i-1)/12)) (ScreenHeight-522) 560 442],...
                            'color',[0.7,0.7,0.7],'UserData',userData);
                    end
                end

                %%% For the first time through, the number of image sets
                %%% will not yet have been determined.  So, the Number of
                %%% image sets is set temporarily.
                handles.Current.NumberOfImageSets = 1;
                handles.Current.SetBeingAnalyzed = 1;
                %%% Marks the time that analysis was begun.
                handles.Current.TimeStarted = datestr(now);
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

                %%% This variable allows breaking out of nested loops.
                break_outer_loop = 0;

                startingImageSet = 1;
                handles.Current.StartingImageSet = startingImageSet;
                while handles.Current.SetBeingAnalyzed <= handles.Current.NumberOfImageSets
                    setbeinganalyzed = handles.Current.SetBeingAnalyzed;
                        a=clock;
                        begin_set=a(5:6);
                    for SlotNumber = 1:handles.Current.NumberOfModules,
                        %%% Variables for timer, time per module.
                        a=clock;
                        begin=a(5:6);
                        %%% If a module is not chosen in this slot, continue on to the next.
                        ModuleNumberAsString = TwoDigitString(SlotNumber);
                        ModuleName = char(handles.Settings.ModuleNames(SlotNumber));
                        if iscellstr(handles.Settings.ModuleNames(SlotNumber)) == 0
                        else
                            %%% Saves the current module number in the handles structure.
                            handles.Current.CurrentModuleNumber = ModuleNumberAsString;
                            %%% The try/catch/end set catches any errors that occur during the
                            %%% running of module 1, notifies the user, breaks out of the image
                            %%% analysis loop, and completes the refreshing process.
                            try
                                %%% Runs the appropriate module, with the handles structure as an
                                %%% input argument and as the output
                                %%% argument.p
                                eval(['handles = ',ModuleName,'(handles);'])
                            catch
                                if exist([ModuleName,'.m'],'file') ~= 2,
                                    errordlg(['Image processing was canceled because the image analysis module named ', ([ModuleName,'.m']), ' was not found. Is it stored in the folder with the other modules?  Has its name changed?']);
                                else
                                    %%% Runs the errorfunction function that catches errors and
                                    %%% describes to the user what to do.
                                    errorfunction(ModuleNumberAsString,handles.Current.FontSize)
                                end
                                %%% Causes break out of the image analysis loop (see below)
                                break_outer_loop = 1;
                                break;
                            end % Goes with try/catch.

                            %%% Check for a pending "Cancel after Module"
                            CancelWaiting = get(handles.timertexthandle,'string');
                            if (strncmp(CancelWaiting, 'Immediate', 9) == 1),
                                break_outer_loop = 1;
                                break
                            end
                        end
                                           
                        %%% If the module passed out a new value for
                        %%% StartingImageSet, then we set startingImageSet
                        %%% to be that value and break all the way our to
                        %%% the image set loop. The RestartImageSet in
                        %%% handles is deleted because we never want it in
                        %%% the output file.
                        startingImageSet = handles.Current.StartingImageSet;
                        if (setbeinganalyzed < startingImageSet)
                            handles.Current.SetBeingAnalyzed = startingImageSet;
                            guidata(gcbo,handles);
                            break;  %% break out of SlotNumber loop
                        end;
                        openFig = openFigures;
                        openFigures = [];
                        for i=1:length(openFig),
                            ModuleNumber = openFig(i);
                            try
                                ThisFigureNumber = handles.Current.(['FigureNumberForModule' TwoDigitString(ModuleNumber)]);
                                figure(ThisFigureNumber);
                                set(ThisFigureNumber, 'name',[(char(handles.Settings.ModuleNames(ModuleNumber))), ' Display']);
                                set(ThisFigureNumber, 'Position',[(ScreenWidth*((ModuleNumber-1)/12)) (ScreenHeight-522) 560 442]);
                                set(ThisFigureNumber,'color',[0.7,0.7,0.7]);
                                %%% Sets the closing function of the window appropriately. (See way
                                %%% above where 'ClosingFunction's are defined).
                                %set(ThisFigureNumber,'CloseRequestFcn',eval(['ClosingFunction' TwoDigitString(ModuleNumber)]));
                            catch
                            end
                        end
                        %%% Finds and records total time to run module.
                        a=clock;
                        finish=a(5:6);
                        TotalModuleTime = num2str(round(60*(finish(1)-begin(1))+(finish(2)-begin(2))));
                        while numel(TotalModuleTime) <=5
                            TotalModuleTime = [TotalModuleTime ' '];
                        end
                        
                        moduletime_text = ['Module' handles.Current.CurrentModuleNumber ': ' TotalModuleTime];
                        if (setbeinganalyzed) == startingImageSet %& num2str(handles.Current.CurrentModuleNumber) == 1
                            moduletime_text_all = moduletime_text;
                       ModuleTime(str2num(handles.Current.CurrentModuleNumber),:)= {moduletime_text_all};
                        else
                            moduletime_text_all = [char(ModuleTime(str2num(handles.Current.CurrentModuleNumber),:)) ' ' moduletime_text];
                            ModuleTime(str2num(handles.Current.CurrentModuleNumber), :)= {moduletime_text_all};
                        end
                        
                    end %%% ends loop over slot number

                    %%% Completes the breakout to the image loop.
                    if (setbeinganalyzed < startingImageSet)
                        continue;
                    end;
                    
                    closeFig = closeFigures;
                    closeFigures = [];
                    for i=1:length(closeFig),
                        ModuleNumber = closeFig(i);
                        try
                            ThisFigureNumber = handles.Current.(['FigureNumberForModule' TwoDigitString(ModuleNumber)]);
                            delete(ThisFigureNumber);
                        catch
                        end
                    end
                    
                    openFig = openFigures;
                    openFigures = [];
                    for i=1:length(openFig),
                        ModuleNumber = openFig(i);
                        try
                            ThisFigureNumber = handles.Current.(['FigureNumberForModule' TwoDigitString(ModuleNumber)]);
                            figure(ThisFigureNumber);
                            set(ThisFigureNumber, 'name',[(char(handles.Settings.ModuleNames(ModuleNumber))), ' Display']);
                            set(ThisFigureNumber, 'Position',[(ScreenWidth*((ModuleNumber-1)/12)) (ScreenHeight-522) 560 442]);
                            set(ThisFigureNumber,'color',[0.7,0.7,0.7]);
                            %%% Sets the closing function of the window appropriately. (See way
                            %%% above where 'ClosingFunction's are defined).
                            %set(ThisFigureNumber,'CloseRequestFcn',eval(['ClosingFunction' TwoDigitString(ModuleNumber)]));
                        catch
                        end
                    end
                    
                    if (break_outer_loop),
                        break;  %%% this break is out of the outer loop of image analysis
                    end

                    CancelWaiting = get(handles.timertexthandle,'string');
                    
                    
                    %%% Make calculations for the Timer window. Round to
                    %%% 1/10:th of seconds
                    time_elapsed = num2str(round(toc*10)/10);
                    %% Add variable to hold time elapsed for each image
                    %% set. 
                    %set_time_elapsed(handles.Current.SetBeingAnalyzed) = str2num(time_elapsed);
                    timer_elapsed_text =  ['Time elapsed (seconds) = ',time_elapsed];
                    number_analyzed = ['Number of image sets analyzed = ',...
                            num2str(setbeinganalyzed), ' of ', num2str(handles.Current.NumberOfImageSets)];
                    if setbeinganalyzed ~=0
                        time_per_set = ['Time per image set (seconds) = ', ...
                                num2str(round(10*toc/setbeinganalyzed)/10)];
                    else time_per_set = 'Time per image set (seconds) = none completed'; 
                    end
                    if setbeinganalyzed == startingImageSet+1
                    time_set1 = ['Time for first image set = ' num2str(TotalSetTime)];
                    elseif setbeinganalyzed <=startingImageSet+1
                        time_set1 = '  ';
                    end
                    timertext = {timer_elapsed_text; number_analyzed; time_per_set; time_set1};
                    %%% Display calculations in 
                    %%% the "Timer" window by changing the string property.
                    set(text_handle,'string',timertext)
                    drawnow    
                    %%% Save the time elapsed so far in the handles structure.
                    %%% Check first to see that the set being analyzed is not zero, or else an
                    %%% error will be produced when trying to do this.
                    if setbeinganalyzed ~= 0
                        handles.Measurements.GeneralInfo.TimeElapsed{setbeinganalyzed} = toc;
                        guidata(gcbo, handles)
                    end
                    %%% Save all data that is in the handles structure to the output file 
                    %%% name specified by the user.
                    
                    %%% Save everything, but don't want to write out
                    %%% StartingImageSet field.
                    handles.Current = rmfield(handles.Current,'StartingImageSet');
                    eval(['save ''',fullfile(handles.Current.DefaultOutputDirectory, ...
                        get(handles.OutputFileNameEditBox,'string')), ''' ''handles'';'])
                    %%% Restore StartingImageSet for those modules that
                    %%% need it.
                    handles.Current.StartingImageSet = startingImageSet;
                    %%% The setbeinganalyzed is increased by one and stored in the handles structure.
                    setbeinganalyzed = setbeinganalyzed + 1;
                    handles.Current.SetBeingAnalyzed = setbeinganalyzed;
                    guidata(gcbo, handles)

                    %%% If a "cancel" signal is waiting, break and go to the "end" that goes
                    %%% with the "while" loop.
                    if strncmp(CancelWaiting,'Cancel',6) == 1
                        break
                    end
                    
                     %%% Record time elapsed for each image set.
                     a=clock;
                        finish_set=a(5:6);
                        TotalSetTime=60*(finish_set(1)-begin_set(1))+(finish_set(2)-begin_set(2));
                        set_time_elapsed(handles.Current.SetBeingAnalyzed) = TotalSetTime;
                        ThisSet = handles.Current.SetBeingAnalyzed - 1;
                        if handles.Current.SetBeingAnalyzed-1 == startingImageSet
                            set_text = ['        Set' num2str(handles.Current.SetBeingAnalyzed-1) '           '];
                            show_set_text = set_text;
                        else
                            set_text = [show_set_text '       Set' num2str(handles.Current.SetBeingAnalyzed-1) '           '];
                            show_set_text = set_text;
                        end
                                                
                end %%% This "end" goes with the "while" loop (going through the image sets).
                
                %%% After all the image sets have been processed, the following checks to
                %%% be sure that the data loaded as "Sample Info" (Imported) has the proper number of
                %%% entries.  If not, the data is removed from the handles structure so
                %%% that the extract data button will work later on.
                
                %%% We are considering removing the ability to load
                %%% sample info into the handles structure prior to
                %%% running an analysis (and instead only allowing the
                %%% user to add it to an existing file.)  If that
                %%% changes and imported data will routinely be present in the handles
                %%% structure, we should adjust the following to have
                %%% more sophisticated error handling.  If the number
                %%% of entries of imported data does not equal the
                %%% number of image sets that were analyzed, the
                %%% current code forces that sample info to be deleted
                %%% altogether form the handles structure *and* the
                %%% output file.  It would be nice to at least allow
                %%% the user to choose to save the imported data in
                %%% the handles structure (for future runs), in case
                %%% they canceled this run prematurely but will then repeat
                %%% the analysis in full (which happens pretty
                %%% frequently).  It might also be nice to allow the
                %%% user to truncate the imported data to match how
                %%% many image sets were actually analyzed, although
                %%% we should show the user exactly what data will be
                %%% retained and deleted so they can verify that no
                %%% mistakes are made.
               
                
                %%% Create a vector that contains the length of each headings field.  In other
                %%% words, determine the number of entries for each column of Sample Info.
                if isfield(handles.Measurements,'GeneralInfo') == 1
                    Fieldnames = fieldnames(handles.Measurements.GeneralInfo);
                    ImportedFieldnames = Fieldnames(strncmp(Fieldnames,'Imported',8) == 1);
                    if isempty(ImportedFieldnames) == 0
                        for i = 1:length(ImportedFieldnames);
                            fieldname = char(ImportedFieldnames{i});
                            Lengths(i) = length(handles.Measurements.GeneralInfo.(fieldname));
                        end
                        %%% Create a logical array that indicates which headings do not have the
                        %%% same number of entries as the number of image sets analyzed.
                        IsWrongNumber = (Lengths ~= setbeinganalyzed - 1);
                        %%% Determine which heading names to remove.
                        HeadingsToBeRemoved = ImportedFieldnames(IsWrongNumber);
                        %%% Remove headings names from handles.headings and remove the sample
                        %%% info from the field named after the heading.
                        if isempty(HeadingsToBeRemoved) == 0
                            handles.Measurements.GeneralInfo = rmfield(handles.Measurements.GeneralInfo, HeadingsToBeRemoved);
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
                            %%% Save all data that is in the handles structure to the output file
                            %%% name specified by the user.
                            eval(['save ''',fullfile(handles.Current.DefaultOutputDirectory, ...
                                get(handles.OutputFileNameEditBox,'string')), ''' ''handles'';'])
                        end % This end goes with the "isempty" line.
                    end % This end goes with the 'isempty' line.
                end
                %%% Update the handles structure.
                guidata(gcbo, handles)

                %%% Calculate total time elapsed and display Complete in the Timer window.
                total_time_elapsed = ['Total time elapsed (seconds) = ',num2str(round(10*toc)/10)];
                number_analyzed = ['Number of image sets analyzed = ',...
                        num2str(setbeinganalyzed - 1)];
                    
                if setbeinganalyzed ~=1 
                    time_per_set = ['Time per image set (seconds) = ', ...
                            num2str(round(10*toc/(setbeinganalyzed - 1))/10)];                        
                else time_per_set = 'Time per image set (seconds) = none completed'; 
                end
                

                text_handle = uicontrol(timer_handle,'string',timertext,'style','text',...
                    'parent',timer_handle,'position', [0 40 494 74],'FontName','Times',... 
                    'FontSize',handles.Current.FontSize,'FontWeight','bold','backgroundcolor',[0.7,0.7,0.9]);
                timertext = {'IMAGE PROCESSING IS COMPLETE!';total_time_elapsed; number_analyzed; time_per_set};

                set(text_handle,'string',timertext)
                set(timer_handle,'CloseRequestFcn','closereq')
                
                %%% Show seperate calcualtion times for each image set.
                try
                set_time_elapsed = set_time_elapsed(set_time_elapsed ~=0);
                for i=1:handles.Current.NumberOfImageSets;
                    show_time_elapsed_text = ['Time elapsed for image set' num2str(i) '= ' num2str(set_time_elapsed(i)) ];
                    show_time_elapsed(i) = {show_time_elapsed_text};
                end
                show_time_elapsed = char(show_time_elapsed);    
                module_times = char(ModuleTime);
                split_time_elapsed = strvcat(show_time_elapsed, show_set_text, module_times);
                timebox = CPmsgbox(split_time_elapsed);
                end

                %%% Re-enable/disable appropriate buttons.
                set(handles.PipelineOfModulesText,'visible','on')
                set(handles.LoadPipelineButton,'visible','on')
                set(handles.SavePipelineButton,'visible','on')
                set(handles.IndividualModulesText,'visible','on')
                set(handles.AddModule,'visible','on');
                set(handles.RemoveModule,'visible','on');
                set(handles.MoveUpButton,'visible','on');
                set(handles.MoveDownButton,'visible','on');
                set(handles.PixelSizeEditBox,'enable','on','foregroundcolor','black')
                set(handles.SetPreferencesButton,'enable','on')
                set(handles.BrowseImageDirectoryButton,'enable','on')
                set(handles.DefaultImageDirectoryEditBox,'enable','on','foregroundcolor','black')
                set(handles.BrowseOutputDirectoryButton,'enable','on')
                set(handles.DefaultOutputDirectoryEditBox,'enable','on','foregroundcolor','black')
                set(handles.ImageToolsPopUpMenu,'enable','on')
                set(handles.DataToolsPopUpMenu,'enable','on')
                set(handles.CloseWindowsButton,'enable','on')
                set(handles.OutputFileNameEditBox,'enable','on','foregroundcolor','black')
                set(handles.AnalyzeImagesButton,'enable','on')
                for VariableNumber = 1:99
                    set(handles.(['VariableBox' TwoDigitString(VariableNumber)]),'enable','on','foregroundcolor','black');
                end
                %%% The following did not make sense: only some of the
                %%% variable boxes were re-enabled.  In fact, we want
                %%% them all to be enabled.  Perhaps this will change
                %%% if the lines which disabled these boxes changes
                %%% (see FIXME at the beginning of the AnalyzeImages
                %%% button).
%                 for ModuleNumber=1:handles.Current.NumberOfModules;
%                     for VariableNumber = 1:handles.Settings.NumbersOfVariables(ModuleNumber);
%                         set(handles.(['VariableBox' TwoDigitString(VariableNumber)]),'enable','on','foregroundcolor','black');
%                     end
%                 end
                set(handles.CloseFigureButton,'visible','off');
                set(handles.OpenFigureButton,'visible','off');
                set(CancelAfterModuleButton_handle,'enable','off')
                set(CancelAfterImageSetButton_handle,'enable','off')
                set(PauseButton_handle,'enable','off')
                set(CancelNowCloseButton_handle,'enable','off')
                %%% Sets the figure windows' Closing Functions back to normal, if 
                %%% the figure windows are still open.  If this is not done,
                %%% after the analysis is complete, these windows cannot be closed with the
                %%% ctrl-W or "X" buttons in the window, because the closing function would
                %%% wait for the image analysis to complete a loop (which would never
                %%% happen).  Has to check to see whether the figure exists first before
                %%% setting the close request function.  That requires looking up
                %%% handles.Current.FigureNumber1.  Before looking that up, you have to check to
                %%% see if it exists or else an error occurs.
    
                for i=1:handles.Current.NumberOfModules
                    ModuleNum = TwoDigitString(i);
                    if isfield(handles.Current,['FigureNumberForModule' ModuleNum]) ==1
                        if any(findobj == handles.Current.(['FigureNumberForModule' ModuleNum])) == 1;
                            properhandle = handles.Current.(['FigureNumberForModule' ModuleNum]);
                            set(properhandle,'CloseRequestFcn','delete(gcf)');
                        end
                    end
                end
                guidata(gcbo, handles)
                
                %%% This "end" goes with the error-detecting "You have no analysis modules
                %%% loaded". 
            end
            %%% This "end" goes with the error-detecting "You have not specified an
            %%% output file name".
       
    end
    %%% This "end" goes with the error-detecting "The chosen directory does not
    %%% exist."
end


%%% Note: an improvement I would like to make:
%%% Currently, it is possible to use the Zoom tool in the figure windows to
%%% zoom in on any of the subplots.  However, when new image data comes
%%% into the window, the Zoom factor is reset. If the processing is fairly
%%% rapid, there isn't really time to zoom in on an image before it
%%% refreshes. It would be nice if the
%%% Zoom factor was applied to the new incoming image.  I think that this
%%% would require redefining the Zoom tool's action, which is not likely to
%%% be a simple task.

function errorfunction(CurrentModuleNumber,FontSize)
Error = lasterr;
%%% If an error occurred in an image analysis module, the error message
%%% should begin with "Error using ==> ", which will be recognized here.
if strncmp(Error,'Error using ==> ', 16) == 1
    ErrorExplanation = ['There was a problem running the analysis module number ',CurrentModuleNumber, '.', Error];
    %%% The following are errors that may have occured within the analyze all
    %%% images callback itself.
elseif isempty(strfind(Error,'bad magic')) == 0
    ErrorExplanation = 'There was a problem running the image analysis. It seems likely that there are files in your image directory that are not images or are not the image format that you indicated. Probably the data for the image sets up to the one which generated this error are OK in the output file.';
else
    ErrorExplanation = ['There was a problem running the image analysis. Sorry, it is unclear what the problem is. It would be wise to close the entire CellProfiler program in case something strange has happened to the settings. The output file may be unreliable as well. Matlab says the error is: ', Error, ' in module ', CurrentModuleNumber];
end
errordlg(ErrorExplanation);

%%%%%%%%%%%%%%%%%%%
%%% HELP BUTTONS %%%
%%%%%%%%%%%%%%%%%%%

%%% --- Executes on button press in the Help buttons.
function PipelineModuleHelp_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
HelpText = help('HelpPipelineOfModules.m');
CPtextdisplaybox(HelpText,'CellProfiler Help');

function IndividualModuleHelp_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
%%% First, check to see whether there is a specific module loaded.
%%% If not, it opens a help dialog which explains how to pick one.
%%% The numeral 10 allows line breaks.
GeneralIndividualModuleHelpText = help('HelpIndividualModule');
NoModuleSelectedHelpMsg = ['You do not have an analysis module loaded.' 10 10 ...
    GeneralIndividualModuleHelpText];
ModuleNumber = whichactive(handles);
if ModuleNumber == 0
    CPtextdisplaybox(NoModuleSelectedHelpMsg,'Help for choosing an analysis module');
else
    try ModuleName = handles.Settings.ModuleNames(ModuleNumber);
        %%% This is the function that actually reads the module's help
        %%% data.
        HelpText = ['GENERAL HELP:' 10 ...
            GeneralIndividualModuleHelpText, 10, 10 ...
            'MODULE-SPECIFIC HELP:' 10 ...
            help(char(ModuleName))];
        DoesHelpExist = exist('HelpText','var');
        if DoesHelpExist == 1
            %%% Calls external subfunction: a nice text display box with a slider if the help is too long.
            CPtextdisplaybox(HelpText,'CellProfiler image analysis module help');
        else
            CPtextdisplaybox(['Sorry, there is no help information for this image analysis module.',GeneralIndividualModuleHelpText],'Image analysis module help');
        end
    catch
        CPtextdisplaybox(NoModuleSelectedHelpMsg,'Help for choosing an analysis module');
    end
end

function PixelPreferencesTechHelp_Callback(hObject, eventdata, handles)
HelpText = help('HelpPixelPreferencesTech.m');
CPtextdisplaybox(HelpText,'CellProfiler Help');

function DefaultImageDirectoryHelp_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
HelpText = help('HelpDefaultImageDirectory.m');
CPtextdisplaybox(HelpText,'CellProfiler Help');

function DefaultOutputDirectoryHelp_Callback(hObject, eventdata, handles)
HelpText = help('HelpDefaultOutputDirectory.m');
CPtextdisplaybox(HelpText,'CellProfiler Help');

function ImageToolsHelp_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
ListOfTools = get(handles.ImageToolsPopUpMenu, 'string');
ToolsHelpSubfunction(handles, 'Image', ListOfTools)

function DataToolsHelp_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
ListOfTools = get(handles.DataToolsPopUpMenu, 'string');
ToolsHelpSubfunction(handles, 'Data', ListOfTools)

%%% SUBFUNCTION %%%
function ToolsHelpSubfunction(handles, ImageOrData, ToolsCellArray)
global toolsChoice;
ToolsCellArray(1) = [];
okbuttoncallback = 'ToolsHelpWindowHandle = findobj(''name'',''ToolsHelpWindow''); toolsbox = findobj(''tag'',''toolsbox''); global toolsChoice; toolsChoice = get(toolsbox,''value''); close(ToolsHelpWindowHandle), clear ToolsHelpWindowHandle';
cancelbuttoncallback = 'ToolsHelpWindowHandle = findobj(''name'',''ToolsHelpWindow''); global toolsChoice; toolsChoice = 0; close(ToolsHelpWindowHandle), clear ToolsHelpWindowHandle';

MainWinPos = get(handles.figure1,'Position');
Color = [0.7 0.7 0.9];

%%% If there is a (are) ToolsHelpWindow(s) open, close it (them);
%%% otherwise, ok/cancel callbacks can get confused
ToolsHelpWindowHandles = findobj('name','ToolsHelpWindow');
if ~isempty(ToolsHelpWindowHandles)
    try
        close(ToolsHelpWindowHandles);
    end
end

%%% Label we attach to figures (as UserData) so we know they are ours
userData.Application = 'CellProfiler';
ToolsHelpWindowHandle = figure(...
'Units','pixels',...
'Color',Color,...
'DockControls','off',...
'MenuBar','none',...
'Name','ToolsHelpWindow',...
'NumberTitle','off',...
'Position',[MainWinPos(1)+MainWinPos(3)/3 MainWinPos(2) MainWinPos(3)/2 MainWinPos(4)*2/3],...
'Resize','off',...
'HandleVisibility','on',...
'Tag','ToolsHelpWindowHandle',...
'UserData',userData);

choosetext = uicontrol(...
'Parent',ToolsHelpWindowHandle,...
'BackGroundColor', Color,...
'Units','normalized',...
'Position',[0.10 0.6 0.80 0.31],...
'String', ['You can add your own tools by writing Matlab m-files, placing them in the ', ImageOrData, 'Tools folder, and restarting CellProfiler. To view help for an individual ' ImageOrData ' tool, choose one below:'],...
'Style','text',...
'Tag','informtext');

listboxcallback = 'ToolsHelpWindowHandle = findobj(''name'',''ToolsHelpWindow''); if (strcmpi(get(ToolsHelpWindowHandle,''SelectionType''),''open'')==1) toolsbox = findobj(''tag'',''toolsbox''); global toolsChoice; toolsChoice = get(toolsbox,''value''); close(ToolsHelpWindowHandle); clear toolsbox; end; clear ToolsHelpWindowHandle';
toolsbox = uicontrol(...
'Parent',ToolsHelpWindowHandle,...
'Units','normalized',...
'backgroundColor',Color,...
'Position',[0.30 0.18 0.45 0.464],...
'String',ToolsCellArray,...
'Style','listbox',...
'Callback',listboxcallback,...
'Value',1,...
'Tag','toolsbox');

okbutton = uicontrol(...
'Parent',ToolsHelpWindowHandle,...
'BackGroundColor', Color,...
'Units','normalized',...
'Callback',okbuttoncallback,...
'Position',[0.30 0.077 0.2 0.06],...
'String','Ok',...
'Tag','okbutton');

cancelbutton = uicontrol(...
'Parent',ToolsHelpWindowHandle,...
'BackGroundColor', Color,...
'Units','normalized',...
'Callback',cancelbuttoncallback,...
'Position',[0.55 0.077 0.2 0.06],...
'String','Cancel',...
'Tag','cancelbutton');

toolsChoice = 0; %%% Makes sure toolsChoice indicates no selection
                 %%% in case user closes window using x icon or Close Windows button
uiwait(ToolsHelpWindowHandle);

if(toolsChoice ~= 0)
    HelpText = handles.Current.([ImageOrData 'ToolHelp']){toolsChoice};
    CPtextdisplaybox(HelpText,['CellProfiler ' ImageOrData ' Tools Help'])
end
clear toolsChoice;

function AnalyzeImagesHelp_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
HelpText = help('HelpAnalyzeImages.m');
CPtextdisplaybox(HelpText,'CellProfiler Help');

%%% END OF HELP HELP HELP HELP HELP HELP BUTTONS %%%