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
%  ExampleFlyImages folder (downloaded from the CellProfiler website,
%  in a zipped file separate from the source code and manual), type in
%  the name of an output file (e.g. 'Temp1') in the appropriate box in
%  CellProfiler, click 'Load settings', choose 'ExampleFlySettings'
%  and click 'Analyze images'. An analysis run should begin.
%
%      H = CellProfiler returns the handle to a new CellProfiler or
%      the handle to the existing singleton*.

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


% Last Modified by GUIDE v2.5 14-Dec-2004 10:04:33
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

%%% Checks whether the user has the Image Processing Toolbox.
Answer = license('test','image_toolbox');
if Answer ~= 1
    warndlg('It appears that you do not have a license for the Image Processing Toolbox of Matlab.  Many of the image analysis modules of CellProfiler may not function properly. Typing ''ver'' or ''license'' at the Matlab command line may provide more information about your current license situation.')
end

%%% Determines the startup directory.
handles.Current.StartupDirectory = pwd;

%%% Retrieves preferences from CellProfilerPreferences.mat, if possible.
%%% Try loading CellProfilerPreferences.mat first from the matlabroot
%%% directory and then the current directory.  This is not necessary for
%%% CellProfiler to function; it just allows defaults to be
%%% pre-loaded.
try cd(matlabroot)
    load CellProfilerPreferences
    LoadedPreferences = SavedPreferences;
    clear SavedPreferences
catch
    try cd(handles.Current.StartupDirectory);
        load CellProfilerPreferences
        LoadedPreferences = SavedPreferences;
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

try handles.Preferences.DefaultModuleDirectory = LoadedPreferences.DefaultModuleDirectory;
        %%% Checks whether that pathname is valid.
    cd(handles.Preferences.DefaultModuleDirectory)
    cd(handles.Current.StartupDirectory)
catch
    %%% If the Default Module Directory is not present in the loaded
    %%% preferences or cannot be found, look at where the
    %%% CellProfiler.m file is located and see whether there is a
    %%% subdirectory within that directory, called "Modules".  If so,
    %%% use that subdirectory as the default module directory. If not,
    %%% use the current directory.
    try [CellProfilerPathname,FileName,ext,versn] = fileparts(which('CellProfiler'));
        CellProfilerModulePathname = fullfile(CellProfilerPathname,'Modules');
        handles.Preferences.DefaultModuleDirectory = CellProfilerModulePathname;
        %%% Checks whether that pathname is valid.
        cd(CellProfilerModulePathname)
        cd(handles.Current.StartupDirectory)
    catch handles.Preferences.DefaultModuleDirectory = handles.Current.StartupDirectory;
    end
end

try handles.Preferences.DefaultOutputDirectory = LoadedPreferences.DefaultOutputDirectory;
    %%% Checks whether that pathname is valid.
    cd(handles.Preferences.DefaultOutputDirectory)
    cd(handles.Current.StartupDirectory)
catch
    %%% If not present in the loaded preferences or not existent, the current
    %%% directory is used.
    handles.Preferences.DefaultOutputDirectory = handles.Current.StartupDirectory;
end

try handles.Preferences.DefaultImageDirectory = LoadedPreferences.DefaultImageDirectory;
    %%% Checks whether that pathname is valid.
    cd(handles.Preferences.DefaultImageDirectory)
    cd(handles.Current.StartupDirectory)
catch
    %%% If not present in the loaded preferences or not existent, the current
    %%% directory is used.
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

    %%%% Sets up the data tools popup menu.
%%% Initial value. maybe get rid of it.
ListOfTools = 'no data tools loaded';
set(handles.DataToolsPopUpMenu, 'string', ListOfTools)

% TODO: Add error checking if the folder is not found.

%%% Finds all available data tools, which are .m files residing in the
%%% DataTools folder.
%%% Specifies the DataTools folder.  CellProfilerPathname was defined
%%% upon launching CellProfiler.
Pathname = fullfile(handles.Current.CellProfilerPathname,'DataTools');
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
    errordlg('There are no data tools loaded, because CellProfiler could not find the DataTools directory, which should be located within the directory where the current CellProfiler.m resides.')
else
    %%% Looks for .m files.
    FileNames = FileNamesNoDir(strcmp(FileNamesNoDir(end-3:end),'.m'))
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

%%% Sets a suitable fontsize. 
%%% With the current setting, the fontsize is 10pts on a 
%%% screen with 90 pixels/inch resolution and 8pts on a 
%%% screen with 116 pixels/inch.
ScreenResolution = get(0,'ScreenPixelsPerInch');
FontSize = (220 - ScreenResolution)/13;       % 90 pix/inch => 10pts, 116 pix/inch => 8pts
handles.Current.FontSize = FontSize;
names = fieldnames(handles);
for k = 1:length(names)
    if ishandle(handles.(names{k}))
        set(findobj(handles.(names{k}),'-property','FontSize'),'FontSize',FontSize,'FontName','Times')
    end
end 

cd(handles.Current.StartupDirectory)

% Update handles structure
guidata(hObject, handles);

% --- Outputs from this function are returned to the command line.
function varargout = CellProfiler_OutputFcn(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
% Get default command line output from handles structure
varargout{1} = handles.output;

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% LOAD PIPELINE BUTTON %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes on button press in LoadPipelineButton.
function LoadPipelineButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.

cd(handles.Current.DefaultOutputDirectory)
[SettingsFileName, SettingsPathname] = uigetfile('*.mat','Choose a settings or output file');
%%% If the user presses "Cancel", the SettingsFileName.m will = 0 and
%%% nothing will happen.
if SettingsFileName == 0
    cd(handles.Current.StartupDirectory)
    return
end
%%% Loads the Settings file.
LoadedSettings = load([SettingsPathname SettingsFileName]);

if ~ (isfield(LoadedSettings, 'Settings') || isfield(LoadedSettings, 'handles')),
    errordlg(['The file ' SettingsPathname SettingsFilename ' does not appear to be a valid settings or output file. Settings can be extracted from an output file created when analyzing images with CellProfiler or from a small settings file saved using the "Save Settings" button.  Either way, this file must have the extension ".mat" and contain a variable named "Settings" or "handles".']);
    cd(handles.Current.StartupDirectory)
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

handles.Settings.ModuleNames = Settings.ModuleNames;
ModuleNamedotm = [char(handles.Settings.ModuleNames{1}) '.m'];
%%% Checks to make sure that the modules have not changed
if exist(ModuleNamedotm,'file')
    FullPathname = which(ModuleNamedotm);
    [Pathname, filename, ext, versn] = fileparts(FullPathname);        
else
    %%% If the module.m file is not on the path, it won't be
    %%% found, so ask the user where the modules are.
    Pathname = uigetdir('','Please select directory where modules are located');
end
for ModuleNum=1:length(handles.Settings.ModuleNames),
    [defVariableValues handles.Settings.NumbersOfVariables(ModuleNum) CurrentVarRevNum] = LoadSettings_Helper(Pathname, char(handles.Settings.ModuleNames(ModuleNum)));
    if (isfield(Settings,'VariableRevisionNumbers')),
        SavedVarRevNum = Settings.VariableRevisionNumbers(ModuleNum);
    else
        SavedVarRevNum = 0;
    end
    if( (SavedVarRevNum ~= 0) & (SavedVarRevNum == CurrentVarRevNum))
        if(handles.Settings.NumbersOfVariables(ModuleNum) == Settings.NumbersOfVariables(ModuleNum))
            handles.Settings.VariableValues = Settings.VariableValues;
            varChoice = 0;
        else
            errorString = 'Variable Revision Number same, but number of variables different for some reason';
            cd(Pathname);
            savedVariableValues = Settings.VariableValues(ModuleNum,1:Settings.NumbersOfVariables(ModuleNum));
            for i=1:(length(savedVariableValues)),
                if (iscellstr(savedVariableValues(i)) == 0)
                    savedVariableValues(i) = {''};
                end
            end
            varChoice = HelpLoadSavedVariables(savedVariableValues,defVariableValues, errorString, char(handles.Settings.ModuleNames(ModuleNum)));
            cd(handles.Current.StartupDirectory);
        end
    else
        errorString = 'Variable Revision Numbers are not the same';
        cd(Pathname);
        savedVariableValues = Settings.VariableValues(ModuleNum,1:Settings.NumbersOfVariables(ModuleNum));
        for i=1:(length(savedVariableValues)),
            if (iscellstr(savedVariableValues(i)) == 0)
                savedVariableValues(i) = {''};
            end
        end
        varChoice = HelpLoadSavedVariables(savedVariableValues,defVariableValues, errorString, char(handles.Settings.ModuleNames(ModuleNum)));
        cd(handles.Current.StartupDirectory);
    end
    if (varChoice == 1),
        handles.Settings.VariableValues(ModuleNum,1:handles.Settings.NumbersOfVariables(ModuleNum)) = defVariableValues(1:handles.Settings.NumbersOfVariables(ModuleNum));
        handles.Settings.VariableValues(ModuleNum,1:Settings.NumbersOfVariables(ModuleNum)) = Settings.VariableValues(ModuleNum,1:Settings.NumbersOfVariables(ModuleNum));
    elseif (varChoice == 2),
        handles.Settings.VariableValues(ModuleNum,1:handles.Settings.NumbersOfVariables(ModuleNum)) = defVariableValues(1:handles.Settings.NumbersOfVariables(ModuleNum));
    end
end

try 
    handles.Settings.PixelSize = Settings.PixelSize;
end

handles.Current.NumberOfModules = 0;
handles.Current.NumberOfModules = length(handles.Settings.ModuleNames);

if (isfield(Settings,'NumbersOfVariables')),
    handles.Settings.NumbersOfVariables = max(handles.Settings.NumbersOfVariables,Settings.NumbersOfVariables);
end

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
    Answer = questdlg('The settings have been extracted from the output file you selected.  Would you also like to save these settings in a separate, smaller, settings-only file?','','Yes','No','Yes');
    if strcmp(Answer, 'Yes') == 1
        SavePipelineButton_Callback(hObject, eventdata, handles);
    end
end
cd(handles.Current.StartupDirectory)

%%% SUBFUNCTION %%%
function [VariableValues NumbersOfVariables VarRevNum] = LoadSettings_Helper(Pathname, ModuleName)

VariableValues = {[]};
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
        elseif (strncmp(output,'%%%VariableRevisionNumber',25) == 1)
            VarRevNum = str2num(output(29:30));
        end
    end
    fclose(fid);
catch
    errordlg('Module could not be found in directory specified','Error')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE PIPELINE BUTTON %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes on button press in SavePipelineButton.
function SavePipelineButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
cd(handles.Current.DefaultOutputDirectory)
%%% The "Settings" variable is saved to the file name the user chooses.
[FileName,Pathname] = uiputfile('*.mat', 'Save Settings As...');
%%% Allows canceling.
if FileName ~= 0
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
  save([Pathname FileName],'Settings')
  helpdlg('The settings file has been written.')
end
cd(handles.Current.StartupDirectory)

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
% Find which module slot number this callback was called for.
ModuleNumber = TwoDigitString(handles.Current.NumberOfModules+1);
ModuleNums = handles.Current.NumberOfModules+1;

%%% 1. Opens a user interface to retrieve the .m file you want to use.
%%% Change to the default module directory. This line is within a
%%% try-end pair because the user may have changed the folder names
%%% leading up to this directory sometime after saving the
%%% Preferences.
try cd(handles.Preferences.DefaultModuleDirectory) %#ok We want to ignore MLint error checking for this line.
end
%%% Now, when the dialog box is opened to retrieve an module, the
%%% directory will be the default module directory.
[ModuleNamedotm,Pathname] = uigetfile('*.m',...
    'Choose an image analysis module');
%%% Change back to the original directory.
cd(handles.Current.StartupDirectory)

%%% 2. If the user presses "Cancel", the ModuleNamedotm = 0, and
%%% everything should be left as it was.
if ModuleNamedotm == 0,
else
    %%% The folder containing the desired .m file is added to Matlab's search path.
    addpath(Pathname)
    %%% If the module's .m file is not found on the search path, the result
    %%% of exist is zero, and the user is warned.
    if exist(ModuleNamedotm,'file') == 0
        %%% Doublecheck that the module exists on Matlab's search path.
        if exist(ModuleNamedotm,'file') == 0
            errordlg('Something is wrong; The .m file ', ModuleNamedotm, ' was not initially found by Matlab, so the folder containing it was added to the Matlab search path. But, Matlab still cannot find the .m file for the analysis module you selected. The module will not be added to the image analysis pipeline.');
            return
        else msgbox(['The .m file ', ModuleNamedotm, ...
                ' was not initially found by Matlab, so the folder containing it was added to the Matlab search path. If for some reason you did not want to add that folder to the path, go to Matlab > File > Set Path and remove the folder from the path.  If you have no idea what this means, don''t worry about it; the module was added to the image analysis pipeline just fine.'])
        end
    end
    %%% 3. The last two characters (=.m) are removed from the
    %%% ModuleName.m and called ModuleName.
    ModuleName = ModuleNamedotm(1:end-2);
    %%% The name of the module is shown in a text box in the GUI (the text
    %%% box is called ModuleName1.) and in a text box in the GUI which
    %%% displays the current module (whose settings are shown).

    %%% 4. Saves the ModuleName to the handles structure.
    handles.Settings.ModuleNames{ModuleNums} = ModuleName;
    contents = get(handles.ModulePipelineListBox,'String');
    contents{ModuleNums} = ModuleName;
    set(handles.ModulePipelineListBox,'String',contents);

    %%% 5. The text description for each variable for the chosen module is
    %%% extracted from the module's .m file and displayed.
    fid=fopen([Pathname ModuleNamedotm]);
    while 1;
        output = fgetl(fid); if ~ischar(output); break; end;

        if strncmp(output,'%defaultVAR',11) == 1
            displayval = output(17:end);
            istr = output(12:13);
            i = str2num(istr);
            handles.Settings.VariableValues(ModuleNums, i) = {displayval};
            handles.Settings.NumbersOfVariables(str2double(ModuleNumber)) = i;
        elseif strncmp(output,'%%%VariableRevisionNumber',25) == 1
            handles.Settings.VariableRevisionNumbers(str2double(ModuleNumber)) = str2num(output(29:30));
        end
    end
    fclose(fid);

    try Contents = handles.Settings.VariableRevisionNumbers(str2double(ModuleNumber));
    catch Contents = [];
    end
    
    if isempty(Contents) == 1
        handles.Settings.VariableRevisionNumbers(str2double(ModuleNumber)) = 0;
    end
    
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
    Answer = questdlg('Are you sure you want to clear this analysis module and its settings?','Confirm','Yes','No','Yes');
    if strcmp(Answer,'No') == 1
        return
    end
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
                errordlg(['The module you attempted to add, ', ModuleNamedotm,', is not a valid CellProfiler module because it does not appear to have any variables.  Sometimes this error occurs when you try to load a module that has the same name as a built-in Matlab function and the built in function is located in a directory higher up on the Matlab search path.'])
                return 
                % TODO: If this happens, we need to remove the module
                % name from the list box, etc. I.e. revert everything
                % as if we had not tried to load this module. As it
                % is, the module's name is displayed in the pipeline
                % list box, but it cannot be removed without causing
                % errors.
            end
        end
        %%% 4. Extracts the stored values for the variables from the handles
        %%% structure and displays in the edit boxes.
        numberExtraLinesOfDescription = 0;
        numberOfLongBoxes = 0;
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
                set(handles.(['VariableDescription' TwoDigitString(i)]), 'Position', [2 346+linesVarDes-25*(i+numberOfLongBoxes+numberExtraLinesOfDescription) 464 23*(linesVarDes)-3*linesVarDes]);
            end

            if (i <= handles.Settings.NumbersOfVariables(ModuleNumber))
                if iscellstr(handles.Settings.VariableValues(ModuleNumber, i));
                    VariableValuesString = char(handles.Settings.VariableValues{ModuleNumber, i});
                    if ( ( length(VariableValuesString) > 13) | (flagExist) )
                        numberOfLongBoxes = numberOfLongBoxes+1;
                        set(handles.(['VariableBox' TwoDigitString(i)]), 'Position', [25 346-25*(i+numberOfLongBoxes+numberExtraLinesOfDescription) 539 23]);
                    else
                        set(handles.(['VariableBox' TwoDigitString(i)]), 'Position', [470 346-25*(i+numberOfLongBoxes+numberExtraLinesOfDescription) 94 23]);
                    end
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
            set(handles.slider1,'value',get(handles.slider1,'max'));
        end
    else helpdlg('No modules are loaded.');
    end
else helpdlg('No module highlighted.');
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
  errordlg('Variable boxes must not be left blank')
  set(handles.(['VariableBox' VariableNumberStr]),'string', 'Fill in');
  storevariable(ModuleNumber,VariableNumberStr, 'Fill in', handles);
else
  if ModuleNumber == 0,     
    errordlg('Something strange is going on: none of the analysis modules are active right now but somehow you were able to edit a setting.','weirdness has occurred')
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
scrollPos = get(hObject,'max') - get(hObject, 'Value');
variablepanelPos = get(handles.variablepanel, 'position');
set(handles.variablepanel, 'position', [235 80+scrollPos 563 297]);

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

%Note, it doesn't seem like EditBox callbacks are ever executed...
PixelSizeEditBoxCallback = 'PixelSize = str2double(get(gco,''string'')); if isempty(PixelSize) == 1 | ~isnumeric(PixelSize), PixelSize = {''1''}, set(gco,''string'',PixelSize), end, clear';
ImageDirBrowseButtonCallback = 'EditBoxHandle = findobj(''Tag'',''ImageDirEditBox''); CurrentChoice = get(EditBoxHandle,''string''); try cd(CurrentChoice), end, DefaultImageDirectory = uigetdir(cd,''Select the default image directory''); if DefaultImageDirectory == 0, return, else set(EditBoxHandle,''string'', DefaultImageDirectory), end, clear';
ImageDirEditBoxCallback = 'DefaultImageDirectory = get(gco,''string''); if isempty(DefaultImageDirectory) == 1; DefaultImageDirectory = cd; set(gco,''string'',DefaultImageDirectory); end, clear';
OutputDirBrowseButtonCallback = 'EditBoxHandle = findobj(''Tag'',''OutputDirEditBox''); CurrentChoice = get(EditBoxHandle,''string''); try cd(CurrentChoice), end, DefaultOutputDirectory = uigetdir(cd,''Select the default output directory''); if DefaultOutputDirectory == 0, return, else set(EditBoxHandle,''string'', DefaultOutputDirectory), end, clear';
OutputDirEditBoxCallback = 'DefaultOutputDirectory = get(gco,''string''); if isempty(DefaultOutputDirectory) == 1; DefaultOutputDirectory = cd; set(gco,''string'',DefaultOutputDirectory), end, clear';
ModuleDirBrowseButtonCallback = 'EditBoxHandle = findobj(''Tag'',''ModuleDirEditBox''); CurrentChoice = get(EditBoxHandle,''string''); try cd(CurrentChoice), end, DefaultModuleDirectory = uigetdir(cd,''Select the directory where modules are stored''); if DefaultModuleDirectory == 0, return, else set(EditBoxHandle,''string'', DefaultModuleDirectory), end, clear';
ModuleDirEditBoxCallback = 'DefaultModuleDirectory = get(gco,''string''); if isempty(DefaultModuleDirectory) == 1; DefaultModuleDirectory = cd; set(gco,''string'',DefaultModuleDirectory), end, clear';

%%% TODO: Add error checking to each directory edit box (does pathname exist).
%%% TODO: Add error checking to pixel size box and font size box(is it a number).

SaveButtonCallback = 'SetPreferencesWindowHandle = findobj(''name'',''SetPreferences''); global EnteredPreferences, PixelSizeEditBoxHandle = findobj(''Tag'',''PixelSizeEditBox''); FontSizeEditBoxHandle = findobj(''Tag'',''FontSizeEditBox''); ImageDirEditBoxHandle = findobj(''Tag'',''ImageDirEditBox''); OutputDirEditBoxHandle = findobj(''Tag'',''OutputDirEditBox''); ModuleDirEditBoxHandle = findobj(''Tag'',''ModuleDirEditBox''); PixelSize = get(PixelSizeEditBoxHandle,''string''); PixelSize = PixelSize{1}; FontSize = get(FontSizeEditBoxHandle,''string''); DefaultImageDirectory = get(ImageDirEditBoxHandle,''string''); DefaultOutputDirectory = get(OutputDirEditBoxHandle,''string''); DefaultModuleDirectory = get(ModuleDirEditBoxHandle,''string''); EnteredPreferences.PixelSize = PixelSize; EnteredPreferences.FontSize = FontSize; EnteredPreferences.DefaultImageDirectory = DefaultImageDirectory; EnteredPreferences.DefaultOutputDirectory = DefaultOutputDirectory; EnteredPreferences.DefaultModuleDirectory = DefaultModuleDirectory; SavedPreferences = EnteredPreferences; CurrentDir = pwd; try cd(matlabroot), save CellProfilerPreferences SavedPreferences, clear SavedPreferences, helpdlg(''Your CellProfiler preferences were successfully set.  They are contained in a file called CellProfilerPreferences.mat in the Matlab root directory.''), cd(CurrentDir), catch, try save CellProfilerPreferences SavedPreferences, clear SavedPreferences, helpdlg(''You do not have permission to write anything to the Matlab root directory, which is required to save your preferences permanently.  Instead, your preferences will only function properly when you start CellProfiler from the current directory.''), catch, helpdlg(''CellProfiler was unable to save your desired preferences, probably because you lack write permission for both the Matlab root directory as well as the current directory.  Your preferences will only be saved for the current session of CellProfiler.''); end, end, clear PixelSize* *Dir* , close(SetPreferencesWindowHandle), clear SetPreferencesWindowHandle';
CancelButtonCallback = 'delete(gcf)';


%%% Creates the dialog box and its text, buttons, and edit boxes.
MainWinPos = get(handles.figure1,'Position');
Color = [0.7 0.7 0.7];

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
'Tag','figure1');

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
msgbox('The handles structure has been printed out at the command line of Matlab.')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% BROWSE DEFAULT IMAGE DIRECTORY BUTTON %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes on button press in BrowseImageDirectoryButton.
function BrowseImageDirectoryButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
cd(handles.Current.DefaultImageDirectory)
%%% Opens a dialog box to allow the user to choose a directory and loads
%%% that directory name into the edit box.  Also, changes the current
%%% directory to the chosen directory.
pathname = uigetdir('','Choose the directory of images to be analyzed');
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
cd(handles.Current.StartupDirectory)

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
    guidata(hObject, handles);
    %%% If the directory entered in the box does not exist, give an error
    %%% message, change the contents of the edit box back to the
    %%% previously selected directory, and change the contents of the
    %%% filenameslistbox back to the previously selected directory.
else errordlg('A directory with that name does not exist')
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

if isempty(FileNamesNoDir)
    handles.Current.FilenamesInImageDir = [];
    %%% Test whether this is during CellProfiler launching, in which case
    %%% the following error is unnecessary.
    if strcmp(get(handles.FilenamesListBox,'String'),'Listbox') ~= 1
        errordlg('There are no files in the chosen directory')
    end
else
    DiscardsHidden = strncmp(FileNamesNoDir,'.',1);
    DiscardsByExtension = regexpi(FileNamesNoDir, '\.(m|mat|m~|frk~|xls|doc|txt|csv)$', 'once');
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
        handles.Current.FilenamesInImageDir = [];
        %%% Test whether this is during CellProfiler launching, in which case
        %%% the following error is unnecessary.
        if strcmp(get(handles.FilenamesListBox,'String'),'Listbox') ~= 1
            errordlg('There are no files in the chosen directory')
        end
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

cd(handles.Current.DefaultOutputDirectory)
%%% Opens a dialog box to allow the user to choose a directory and loads
%%% that directory name into the edit box.  Also, changes the current
%%% directory to the chosen directory.
pathname = uigetdir('','Choose the default output directory');
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
cd(handles.Current.StartupDirectory)

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
    guidata(hObject,handles) %%% TODO: Is this necessary?
    %%% If the directory entered in the box does not exist, give an error
    %%% message, change the contents of the edit box back to the
    %%% previously selected directory, and change the contents of the
    %%% filenameslistbox back to the previously selected directory.
else errordlg('A directory with that name does not exist')
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE TOOLS POPUP MENU %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function ImageToolsPopUpMenu_CreateFcn(hObject, eventdata, handles)

set(gcbo, 'string', ListOfTools)

function ImageToolsPopUpMenu_Callback(hObject, eventdata, handles)
% Hints: contents = get(hObject,'String') returns ImageToolsPopUpMenu contents as cell array
%        contents{get(hObject,'Value')} returns selected item from ImageToolsPopUpMenu



%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% DATA TOOLS POPUP MENU %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function DataToolsPopUpMenu_CreateFcn(hObject, eventdata, handles)


function DataToolsPopUpMenu_Callback(hObject, eventdata, handles)
% Hints: contents = get(hObject,'String') returns DataToolsPopUpMenu contents as cell array
%        contents{get(hObject,'Value')} returns selected item from DataToolsPopUpMenu
val = get(hObject,'Value');
string_list = get(hObject,'String');
selected_string = string_list{val}; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% CLOSE WINDOWS BUTTON %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes on button press in CloseWindowsButton.
function CloseWindowsButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% OUTPUT FILE NAME EDIT BOX %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes during object creation, after setting all properties.
function OutputFileNameEditBox_CreateFcn(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.

function OutputFileNameEditBox_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
Pathname = handles.Current.DefaultOutputDirectory;
%%% Gets the user entry and stores it in the handles structure.
InitialUserEntry = get(handles.OutputFileNameEditBox,'string');
if isempty(InitialUserEntry)
    handles.Current.OutputFilename =[];
    guidata(gcbo, handles); %%% TODO: Is this necessary?
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
    guidata(gcbo, handles);  %%% TODO: Is this necessary?
    %%% Checks whether a file with that name already exists, to warn the user
    %%% that the file will be overwritten.
    if exist([Pathname,'/',UserEntry],'file') ~= 0   %%% TODO: Fix filename construction.
        errordlg(['A file already exists at ', [Pathname,'/',UserEntry],... %%% TODO: Fix filename construction.
            '. Enter a different name. Click the help button for an explanation of why you cannot just overwrite an existing file.'], 'Warning!');
        set(handles.OutputFileNameEditBox,'string',[])
    else guidata(gcbo, handles); %%% TODO: Is this necessary?
        handles.Current.OutputFilename = UserEntry;
        set(handles.OutputFileNameEditBox,'string',UserEntry)
    end
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
if sum == 0, errordlg('You do not have any analysis modules loaded')
else
    %%% Checks whether an output file name has been specified.
    if isfield(handles.Current, 'OutputFilename') == 0
        errordlg('You have not entered an output file name in Step 2.')
    elseif isempty(handles.Current.OutputFilename)
        errordlg('You have not entered an output file name in Step 2.')
    else
    %%% Checks whether the default output directory exists.
        DirDoesNotExist = 0; %%% Initial value.
        try cd(handles.Current.DefaultOutputDirectory)
        catch DirDoesNotExist == 1;
        end
        %%% If the image directory exists, change to that directory.
        try cd(handles.Current.DefaultImageDirectory)
        catch DirDoesNotExist = 2;
        end
        if DirDoesNotExist == 1
            errordlg('The default output directory does not exist')
        elseif DirDoesNotExist == 2
            errordlg('The default image directory does not exist')
        else           
            %%% Checks whether the specified output file name will overwrite an
            %%% existing file.
            
            %%% TODO: Use fullfile to make the following
            %%% multi-platform compatible.
            OutputFileOverwrite = exist([cd,'/',handles.Current.OutputFilename],'file'); %%% TODO: Fix filename construction.
            if OutputFileOverwrite ~= 0
                errordlg('An output file with the name you entered in Step 2 already exists. Overwriting can cause errors and is disallowed, so please enter a new filename.')
            else
                %%% Retrieves the list of image file names from the
                %%% chosen directory, stores them in the handles
                %%% structure, and displays them in the filenameslistbox, by
                %%% faking a click on the DefaultImageDirectoryEditBox. This
                %%% should already have been done when the directory
                %%% was chosen, but in case some files were moved or
                %%% changed in the meantime, this will refresh the
                %%% list.
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
                set(handles.ImageToolsPopUpMenu,'enable','inactive')
                set(handles.DataToolsPopUpMenu,'enable','inactive')
                set(handles.CloseWindowsButton,'enable','off')
                set(handles.OutputFileNameEditBox,'enable','inactive','foregroundcolor',[0.7,0.7,0.7])
                set(handles.AnalyzeImagesButton,'enable','off')
                % FIXME: This should loop just over the number of actual variables in the display.
                for VariableNumber=1:99;
                    set(handles.(['VariableBox' TwoDigitString(VariableNumber)]),'enable','inactive','foregroundcolor',[0.7,0.7,0.7]);
                end
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
                    'String', '?', 'Position', [460 10 15 30], 'FontSize', 12,...
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

                %%% If a module is chosen in this slot, assign it an output figure
                %%% window and write the figure window number to the handles structure so
                %%% that the modules know where to write to.  Each module should
                %%% resize the figure window appropriately.  The closing function of the
                %%% figure window is set to wait until an image set is done processing
                %%% before closing the window, to avoid unexpected results.              
                set(handles.CloseFigureButton,'visible','on')
                set(handles.OpenFigureButton,'visible','on')

                for i=1:handles.Current.NumberOfModules;
                    if iscellstr(handles.Settings.ModuleNames(i)) == 1
                        handles.Current.(['FigureNumberForModule' TwoDigitString(i)]) = ...
                            figure('name',[char(handles.Settings.ModuleNames(i)), ' Display'], 'Position',[(ScreenWidth*((i-1)/12)) (ScreenHeight-522) 560 442],'color',[0.7,0.7,0.7]);
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

                while handles.Current.SetBeingAnalyzed <= handles.Current.NumberOfImageSets
                    setbeinganalyzed = handles.Current.SetBeingAnalyzed;

                    for SlotNumber = 1:handles.Current.NumberOfModules,
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
                                %%% input argument and as the output argument.
                                eval(['handles = ',ModuleName,'(handles);'])
                            catch
                                if exist([ModuleName,'.m'],'file') ~= 2,
                                    errordlg(['Image processing was canceled because the image analysis module named ', ([ModuleName,'.m']), ' was not found. Is it stored in the folder with the other modules?  Has its name changed?'])
                                else
                                    %%% Runs the errorfunction function that catches errors and
                                    %%% describes to the user what to do.
                                    errorfunction(ModuleNumberAsString)
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
                    end %%% ends loop over slot number

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

                    %%% Make calculations for the Timer window.
                    time_elapsed = num2str(toc);
                    timer_elapsed_text =  ['Time elapsed (seconds) = ',time_elapsed];
                    number_analyzed = ['Number of image sets analyzed = ',...
                            num2str(setbeinganalyzed), ' of ', num2str(handles.Current.NumberOfImageSets)];
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
                    cd(handles.Current.DefaultOutputDirectory)
                    eval(['save ',handles.Current.OutputFilename, ' handles;'])                   
                    %%% The setbeinganalyzed is increased by one and stored in the handles structure.
                    setbeinganalyzed = setbeinganalyzed + 1;
                    handles.Current.SetBeingAnalyzed = setbeinganalyzed;
                    guidata(gcbo, handles)

                    %%% If a "cancel" signal is waiting, break and go to the "end" that goes
                    %%% with the "while" loop.
                    if strncmp(CancelWaiting,'Cancel',6) == 1
                        break
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
                Fieldnames = fieldnames(handles.Measurements);
                ImportedFieldnames = Fieldnames(strncmp(Fieldnames,'Imported',8) == 1);
                if isempty(ImportedFieldnames) == 0
                    for i = 1:length(ImportedFieldnames);
                        fieldname = char(ImportedFieldnames{i});
                        Lengths(i) = length(handles.Measurements.(fieldname));
                    end   
                    %%% Create a logical array that indicates which headings do not have the
                    %%% same number of entries as the number of image sets analyzed.
                    IsWrongNumber = (Lengths ~= setbeinganalyzed - 1);
                    %%% Determine which heading names to remove.
                    HeadingsToBeRemoved = ImportedFieldnames(IsWrongNumber);
                    %%% Remove headings names from handles.headings and remove the sample
                    %%% info from the field named after the heading.
                    if isempty(HeadingsToBeRemoved) == 0
                        handles.Measurements = rmfield(handles.Measurements, HeadingsToBeRemoved);
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
                        eval(['save ',handles.Current.OutputFilename, ' handles;'])
                    end % This end goes with the "isempty" line.
                end % This end goes with the 'isempty' line.    
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
                for ModuleNumber=1:handles.Current.NumberOfModules;
                    for VariableNumber = 1:handles.Settings.NumbersOfVariables(ModuleNumber);
                        set(handles.(['VariableBox' TwoDigitString(VariableNumber)]),'enable','on','foregroundcolor','black');
                    end
                end
                set(handles.CloseFigureButton,'visible','off');
                set(handles.OpenFigureButton,'visible','off');
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
                %%% Clears the output file name to prevent it from being reused.
                set(handles.OutputFileNameEditBox,'string',[])
                handles.Current = rmfield(handles.Current,'OutputFilename');
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
cd(handles.Current.StartupDirectory);

%%% Note: an improvement I would like to make:
%%% Currently, it is possible to use the Zoom tool in the figure windows to
%%% zoom in on any of the subplots.  However, when new image data comes
%%% into the window, the Zoom factor is reset. If the processing is fairly
%%% rapid, there isn't really time to zoom in on an image before it
%%% refreshes. It would be nice if the
%%% Zoom factor was applied to the new incoming image.  I think that this
%%% would require redefining the Zoom tool's action, which is not likely to
%%% be a simple task.

function errorfunction(CurrentModuleNumber)
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
errordlg(ErrorExplanation)

%%%%%%%%%%%%%%%%%%%
%%% HELP BUTTONS %%%
%%%%%%%%%%%%%%%%%%%

%%% --- Executes on button press in the permanent Help buttons.
%%% (The permanent Help buttons are the ones that don't change 
%%% depending on the module loaded.) 
function PipelineModuleHelp_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
HelpText = help('Help3.m');
helpdlg(HelpText,'CellProfiler Help #3')

function IndividualModuleHelp_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
%%% First, check to see whether there is a specific module loaded.
%%% If not, it opens a help dialog which explains how to pick one.
ModuleNumber = whichactive(handles);
if ModuleNumber == 0
    helpdlg('You do not have an analysis module selected.  Click "?" next to "Image analysis settings" to get help in choosing an analysis module, or click "View" next to an analysis module that has been loaded already.','Help for choosing an analysis module')
else
    ModuleName = handles.Settings.ModuleNames(ModuleNumber);
    IsItNotChosen = strncmp(ModuleName,'No a',4);
    if IsItNotChosen == 1
        helpdlg('You do not have an analysis module selected.  Click "?" next to "Image analysis settings" to get help in choosing an analysis module, or click "View" next to an analysis module that has been loaded already.','Help for choosing an analysis module')
    else
        %%% This is the function that actually reads the module's help
        %%% data.
        HelpText = help(char(ModuleName));
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

function PixelPreferencesTechHelp_Callback(hObject, eventdata, handles)

function DefaultImageDirectoryHelp_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
HelpText = help('Help1.m');
helpdlg(HelpText,'CellProfiler Help #1')

function DefaultOutputDirectoryHelp_Callback(hObject, eventdata, handles)

function ImageToolsHelp_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
HelpText = help('Help4.m');
helpdlg(HelpText,'CellProfiler Help #4')

function DataToolsHelp_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
HelpText = help('Help2.m');
msgbox(HelpText,'CellProfiler Help #2')

function AnalyzeImagesHelp_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
HelpText = help('HelpAnalyzeImages.m');
helpdlg(HelpText,'CellProfiler Help: Analyze images')

%%% ^ END OF HELP HELP HELP HELP HELP HELP BUTTONS ^ %%%











%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% LOAD SAMPLE INFO BUTTON %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes on button press in LoadSampleInfo.
function LoadSampleInfo_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.

cd(handles.Current.DefaultOutputDirectory)
ExistingOrMemory = questdlg('Do you want to add sample info into an existing output file or into memory so that it is incorporated into future output files?', 'Load Sample Info', 'Existing', 'Memory', 'Cancel', 'Existing');
if strcmp(ExistingOrMemory, 'Cancel') == 1 | isempty(ExistingOrMemory) ==1
    %%% Allows canceling.
    return
elseif strcmp(ExistingOrMemory, 'Memory') == 1
    OutputFile = []; pOutName = []; fOutName = [];
else [fOutName,pOutName] = uigetfile('*.mat','Add sample info to which existing output file?');
    %%% Allows canceling.
    if fOutName == 0
        return
    else
        try OutputFile = load([pOutName fOutName]);
        catch error('Sorry, the file could not be loaded for some reason.')
        end
    end
end

%%% Opens a dialog box to retrieve a file name that contains a list of
%%% sample descriptions, like gene names or sample numbers.
[fname,pname] = uigetfile({'*.csv;*.txt','Readable files: .txt (Plain text) or .csv (Comma-separated)'},'Choose sample info file');
%%% If the user presses "Cancel", the fname will = 0 and nothing will
%%% happen.
if fname == 0
else extension = fname(end-2:end);
    HeadingsPresent = questdlg('Does the first row of your file contain headings?', 'Are headings present?', 'Yes', 'No', 'Cancel', 'Yes');
    %%% Allows canceling.
    if strcmp(HeadingsPresent, 'Cancel') == 1 | isempty(HeadingsPresent) == 1
        return
    end
    %%% Determines the file type.
    if strcmp(extension,'csv') == 1
        try fid = fopen([pname fname]);
            FirstLineOfFile = fgetl(fid);
            LocationOfCommas = strfind(FirstLineOfFile,',');
            NumberOfColumns = size(LocationOfCommas,2) + 1;
            Format = repmat('%s',1,NumberOfColumns);
            %%% Returns to the beginning of the file so that textscan
            %%% reads the entire contents.
            frewind(fid);
            ImportedData = textscan(fid,Format,'delimiter',',');
            for i = 1:NumberOfColumns
                ColumnOfData = ImportedData{i};
                ColumnOfData = ColumnOfData';
                %%% Sends the heading and the sample info to a
                %%% subfunction to be previewed and saved.
                if i == 1
                   Newhandles = handles; 
                end
                [Newhandles,CancelOption,OutputFile] = PreviewAndSaveColumnOfSampleInfo(Newhandles,ColumnOfData,ExistingOrMemory,HeadingsPresent,OutputFile);
                if CancelOption == 1
                    fclose(fid);
                    warndlg('None of the sample info was saved.')
                    return
                end
            end
            fclose(fid);
            if strcmp(ExistingOrMemory,'Memory') == 1
                %%% For future output files:
                %%% Saves the new sample info to the handles
                %%% structure.
                handles = Newhandles;
                h = msgbox(['The sample info is successfully stored in memory and will be added to future output files']);
                waitfor(h)
            else 
                %%% For existing output files:
                %%% Saves the output file with this new sample info.
                save([pOutName,fOutName],'-struct','OutputFile');
                h = msgbox(['The sample info was successfully added to output file']);
                waitfor(h)
            end
        catch lasterr
            fclose(fid)
            if CancelOption == 1
                fclose(fid);
            else error('Sorry, the sample info could not be imported for some reason.')
                fclose(fid);
            end
        end
    elseif strcmp(extension,'txt') == 1
       try fid = fopen([pname fname]);
           ImportedData = textscan(fid,'%s','delimiter','\r');
           ColumnOfData = ImportedData{1};
           ColumnOfData = ColumnOfData';
           %%% Sends the heading and the sample info to a
           %%% subfunction to be previewed and saved.
           [Newhandles,CancelOption,OutputFile] = PreviewAndSaveColumnOfSampleInfo(handles,ColumnOfData,ExistingOrMemory,HeadingsPresent,OutputFile);
           if CancelOption == 1
               fclose(fid);
               warndlg('None of the sample info was saved.')
                return
            end
            fclose(fid);
            if strcmp(ExistingOrMemory,'Memory') == 1
                %%% For future output files:
                %%% Saves the new sample info to the handles
                %%% structure.
                handles = Newhandles;
                h = msgbox(['The sample info will be added to future output files']);
                waitfor(h)
            else
                %%% For existing output files:
                %%% Saves the output file with this new sample info.
                save([pOutName,fOutName],'-struct','OutputFile');
                h = msgbox(['The sample info was successfully added to output file']);
                waitfor(h)
            end
        catch lasterr
            fclose(fid)
            if CancelOption == 1
                fclose(fid);
            else error('Sorry, the sample info could not be imported for some reason.')
                fclose(fid);
            end
        end
    else errordlg('Sorry, the list of sample descriptions must be in a text file (.txt) or comma delimited file (.csv).');
    end
end
cd(handles.Current.StartupDirectory)

%%% SUBFUNCTION %%%
function [handles,CancelOption,OutputFile] = PreviewAndSaveColumnOfSampleInfo(handles,ColumnOfData,ExistingOrMemory,HeadingsPresent,OutputFile);
%%% Sets the initial value to zero.
CancelOption = 0;
%%% Extracts the sample info and the headings from the first row, if they are present.
if strcmp(HeadingsPresent, 'Yes') == 1
    SingleHeading = ColumnOfData(1);
    %%% Converts to char in order to perform the following lines.
    SingleHeading = char(SingleHeading);
    %%% Replaces spaces with underscores, because spaces
    %%% are forbidden in fieldnames.
    SingleHeading(strfind(SingleHeading,' ')) = '_';
    %%% Strips weird characters (e.g. punctuation) out, because such
    %%% characters are forbidden in fieldnames.  The user is still
    %%% responsible for making sure their heading begins with a letter
    %%% rather than a number or underscore.
    PermittedCharacterLocations = regexp(SingleHeading, '[A-Za-z0-9_]');
    SingleHeading = SingleHeading(PermittedCharacterLocations);
    if isempty(SingleHeading) == 1
        SingleHeading = 'Heading not yet entered';
    end
    %%% Converts back to cell array.
    SingleHeading = {SingleHeading};
    ColumnOfSampleInfo = ColumnOfData(2:end);
else SingleHeading = {'Heading not yet entered'};
    ColumnOfSampleInfo = ColumnOfData(1:end);
end
NumberSamples = length(ColumnOfSampleInfo);
%%% Displays a notice.  The buttons don't do anything except proceed.
Notice = {['You have ', num2str(NumberSamples), ' lines of sample information with the heading:']; ...
    char(SingleHeading); ...
    ''; ...
    'The next window will show you a preview of the sample'; ...
    'info you have loaded, and you will then have the'; ...
    'opportunity to enter or change the heading name (Disallowed';...
    'characters have been removed). Press ''OK'' to continue.';...
    '-------'; ...
    'Please note:';...
    '(1) For text files, any spaces or punctuation characters'; ...
    'may split the entry into two entries.';...
    '(2) For csv files, entries containing commas ';...
    'will split the entry into two entries.';...
    '(3) Check that the order of the image files within Matlab is';...
    'as expected.  For example, If you are running Matlab within';...
    'X Windows on a Macintosh, the Mac will show the files as: ';...
    '(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11) whereas the X windows ';...
    'system that Matlab uses will show them as ';...
    '(1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9) and so on.  So be sure that ';...
    'the order of your sample info matches the order that Matlab ';...
    'is using.  You can see the order of image files within';...
    'CellProfiler, or in the Current Directory window of';...
    'Matlab. Go to View > Current Directory to open the window';...
    'if it is not already visible.'};
[Selection,OK] = listdlg('ListString',Notice,'ListSize', [300 600],'Name','Imported sample info',...
    'PromptString','Press any button to continue.',...
    'SelectionMode','single');
%%% Allows canceling.
if OK == 0
    CancelOption = 1;
else
    %%% Displays a filenameslistbox so the user can preview the data.  The buttons in
    %%% this filenameslistbox window don't do anything except proceed.
    [Selection,OK] = listdlg('ListString',ColumnOfSampleInfo, 'ListSize', [300 600],...
        'Name','Sample info preview',...
        'PromptString','Press ''OK'' to continue.',...
        'SelectionMode','single');
    %%% Allows canceling.
    if OK == 0
        CancelOption = 1;
    else
        %%% Sets the initial value.
        HeadingApproved = 0;
        while HeadingApproved ~= 1
            if strcmp(SingleHeading, 'Heading not yet entered') == 1;
                SingleHeading = {''};
            end
            %%% The input dialog displays the current candidate for
            %%% the heading, or it is blank if nothing has been
            %%% entered.
            SingleHeading = inputdlg('Enter the heading for these sample descriptions (e.g. GeneNames                 or SampleNumber). Your entry must be one word with letters and                   numbers only, and must begin with a letter.','Name the Sample Info',1,SingleHeading);
            %%% Allows canceling.
            if isempty(SingleHeading) == 1
                CancelOption = 1;
                return
            elseif strcmp(SingleHeading,'') == 1
                errordlg('No heading was entered. Please try again.');
                %%% For future output files:
            elseif strcmp(ExistingOrMemory, 'Memory') == 1
                %%% Checks to see if the heading exists already.
                    if isfield(handles.Measurements, ['Imported',char(SingleHeading)]) == 1
                        Answer = questdlg('Sample info with that heading already exists in memory.  Do you want to overwrite?');
                        %%% Allows canceling.
                        if isempty(Answer) == 1 | strcmp(Answer,'Cancel') == 1
                            CancelOption = 1;
                            return
                        end
                    else Answer = 'Newfield';
                    end
                %%% If the user does not want to overwrite, try again.
                if strcmp(Answer,'No')

                elseif strcmp(Answer,'Yes') == 1 | strcmp(Answer, 'Newfield') == 1
                    if strcmp(Answer,'Yes') == 1
                        handles.Measurements = rmfield(handles.Measurements, ['Imported',char(SingleHeading)]);
                        guidata(gcbo,handles)
                    end
                    %%% Tries to make a field with that name.
                    try handles.Measurements.(['Imported',char(SingleHeading)]) = [];
                        HeadingApproved = 1;
                    catch
                        MessageHandle = errordlg(['The heading name ',char(SingleHeading),' is not acceptable for some reason. Please try again.']);
                        waitfor(MessageHandle)
                    end
                end
            else %%% For existing output files:
                %%% Checks to see if the heading exists already. Some
                %%% old output files may not have the 'Measurements'
                %%% substructure, so we check for that field first.
                if isfield(OutputFile.handles, 'Measurements') == 1
                    if isfield(OutputFile.handles.Measurements, ['Imported',char(SingleHeading)]) == 1
                        Answer = questdlg(['Sample info with the heading ',char(SingleHeading),' already exists in the output file.  Do you want to overwrite?']);
                        %%% Allows canceling.
                        if isempty(Answer) == 1 | strcmp(Answer,'Cancel') == 1
                            CancelOption = 1;
                            return
                        end
                    else Answer = 'Newfield';
                    end
                else Answer = 'Newfield';
                end
                %%% If the user does not want to overwrite, try again.
                if strcmp(Answer,'No')

                elseif strcmp(Answer,'Yes') == 1 | strcmp(Answer, 'Newfield') == 1
                    if strcmp(Answer,'Yes') == 1
                        OutputFile.handles.Measurements = rmfield(OutputFile.handles.Measurements,['Imported',char(SingleHeading)]);
                    end
                    %%% Tries to make a field with that name.
                    try OutputFile.handles.Measurements.(['Imported',char(SingleHeading)]) = [];
                        HeadingApproved = 1;
                    catch
                        MessageHandle = errordlg(['The heading name ',char(SingleHeading),' is not acceptable for some reason. Please try again.']);
                        waitfor(MessageHandle)
                    end
                end
            end
        end
        %%% Saves the sample info to the handles structure or existing output
        %%% file.
        if strcmp(ExistingOrMemory, 'Memory') == 1
            %%% For future files:
            %%% Saves the column of sample info to the handles.
            handles.Measurements.(['Imported',char(SingleHeading)]) = ColumnOfSampleInfo;
            guidata(gcbo,handles)
        else
            %%% For an existing file:
            %%% Saves the column of sample info to the handles structure from an existing output file.
            OutputFile.handles.Measurements.(['Imported',char(SingleHeading)]) = ColumnOfSampleInfo;
        end
    end
end

% Some random advice from Ganesh:
% SampleNames is a n x m (n - number of rows, m - 1 column) cell array
% If you want to make it into a 1x1 cell array where the cell element
% contains the text, you can do the following
%
% cell_data = {strvcat(SampleNames)};
%
% This will assign the string matrix being created into a single cell
% element.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% CLEAR SAMPLE INFO BUTTON %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes on button press in ClearSampleInfo.
function ClearSampleInfo_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
%%% The Clear Sample Info button allows deleting any list of
%%% sample info, specified by its heading, from the handles structure.

cd(handles.Current.DefaultOutputDirectory)

ExistingOrMemory = questdlg('Do you want to delete sample info or data in an existing output file or do you want to delete the sample info or data stored in memory to be placed into future output files?', 'Delete Sample Info', 'Existing', 'Memory', 'Cancel', 'Existing');
if strcmp(ExistingOrMemory, 'Cancel') == 1 | isempty(ExistingOrMemory) ==1
    %%% Allows canceling.
    cd(handles.Current.StartupDirectory)
    return
elseif strcmp(ExistingOrMemory, 'Memory') == 1
    %%% Checks whether any headings are loaded yet.
    Fieldnames = fieldnames(handles.Measurements);
    ImportedFieldnames = Fieldnames(strncmp(Fieldnames,'Imported',8) == 1);
    if isempty(ImportedFieldnames) == 1
        errordlg('No sample info has been loaded.')
    else
        %%% Opens a filenameslistbox which displays the list of headings so that one can be
        %%% selected.  The OK button has been assigned to mean "Delete".
        [Selected,Action] = listdlg('ListString',ImportedFieldnames, 'ListSize', [300 600],...
            'Name','Current sample info loaded',...
            'PromptString','Select the sample descriptions you would like to delete',...
            'OKString','Delete','CancelString','Cancel','SelectionMode','single');
        %%% Extracts the actual heading name.
        SelectedFieldName = ImportedFieldnames(Selected);
        % Action = 1 if the user pressed the OK (DELETE) button.  If they pressed
        % the cancel button or closed the window Action == 0 and nothing happens.
        if Action == 1
            %%% Delete the selected heading (with its contents, the sample data)
            %%% from the structure.
            handles.Measurements = rmfield(handles.Measurements,SelectedFieldName);
            %%% Handles structure is updated
            guidata(gcbo,handles)
            h = msgbox(['The sample info was successfully deleted from memory']);
        end
        %%% This end goes with the error-detecting - "Do you have any sample info
        %%% loaded?"
    end
elseif strcmp(ExistingOrMemory, 'Existing') == 1
    [fOutName,pOutName] = uigetfile('*.mat','Choose the output file');
    %%% Allows canceling.
    if fOutName == 0
        cd(handles.Current.StartupDirectory)
        return
    else
        try OutputFile = load([pOutName fOutName]);
        catch error('Sorry, the file could not be loaded for some reason.')
        end
    end
    %%% Checks whether any sample info is contained within the file.
    Fieldnames = fieldnames(OutputFile.handles.Measurements);
    ImportedFieldnames = Fieldnames(strncmp(Fieldnames,'Imported',8) == 1 | strncmp(Fieldnames,'Image',5) == 1);
    if isempty(ImportedFieldnames) == 1
        errordlg('The output file you selected does not contain any sample info or data. It would be in a field called handles.Measurements, and would be prefixed with either ''Image'' or ''Imported''.')
    else
        %%% Opens a filenameslistbox which displays the list of headings so that one can be
        %%% selected.  The OK button has been assigned to mean "Delete".
        [Selected,Action] = listdlg('ListString',ImportedFieldnames, 'ListSize', [300 600],...
            'Name','Current sample info loaded',...
            'PromptString','Select the sample descriptions you would like to delete',...
            'OKString','Delete','CancelString','Cancel','SelectionMode','single');
        %%% Extracts the actual heading name.
        SelectedFieldName = ImportedFieldnames(Selected);
        % Action = 1 if the user pressed the OK (DELETE) button.  If they pressed
        % the cancel button or closed the window Action == 0 and nothing happens.
        if Action == 1
            %%% Delete the selected heading (with its contents, the sample data)
            %%% from the structure.
            OutputFile.handles.Measurements = rmfield(OutputFile.handles.Measurements,SelectedFieldName);
                %%% Saves the output file with this new sample info.
                save([pOutName,fOutName],'-struct','OutputFile');
                h = msgbox(['The sample info was successfully deleted from the output file']);
        end
        %%% This end goes with the error-detecting - "Do you have any sample info
        %%% loaded?"
    end
end
cd(handles.Current.StartupDirectory)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% VIEW SAMPLE INFO BUTTON %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes on button press in ViewSampleInfo.
function ViewSampleInfo_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
%%% The View Sample Info button allows viewing any list of
%%% sample info, specified by its heading, taken from the handles structure.

cd(handles.Current.DefaultOutputDirectory)
ExistingOrMemory = questdlg('Do you want to view sample info or data in an existing output file or do you want to view the sample info or data stored in memory to be placed into future output files?', 'View Sample Info', 'Existing', 'Memory', 'Cancel', 'Existing');
if strcmp(ExistingOrMemory, 'Cancel') == 1 | isempty(ExistingOrMemory) ==1
    %%% Allows canceling.
    cd(handles.Current.StartupDirectory)
    return
elseif strcmp(ExistingOrMemory, 'Memory') == 1
    %%% Checks whether any headings are loaded yet.
    Fieldnames = fieldnames(handles.Measurements);
    ImportedFieldnames = Fieldnames(strncmp(Fieldnames,'Imported',8) == 1);
    if isempty(ImportedFieldnames) == 1
        errordlg('No sample info or data is currently stored in memory.')
        %%% Opens a filenameslistbox which displays the list of headings so that one can be
        %%% selected.  The OK button has been assigned to mean "View".
    else
        [Selected,Action] = listdlg('ListString',ImportedFieldnames, 'ListSize', [300 600],...
            'Name','Current sample info loaded',...
            'PromptString','Select the sample descriptions you would like to view.',...
            'OKString','View','CancelString','Cancel','SelectionMode','single');

        %%% Extracts the actual heading name.
        SelectedFieldName = ImportedFieldnames(Selected);

        % Action = 1 if the user pressed the OK (VIEW) button.  If they pressed
        % the cancel button or closed the window Action == 0.
        if Action == 1
            ListToShow = handles.Measurements.(char(SelectedFieldName));
            listdlg('ListString',ListToShow, 'ListSize', [300 600],...
                'Name','Preview your sample info/data','PromptString',...
                char(SelectedFieldName),'SelectionMode','single');
            %%% The OK buttons within this window don't do anything.
        else
            %%% If the user pressed "cancel" or closes the window, Action = 0, so
            %%% nothing happens.
        end
        %%% This "end" goes with the "isempty" if no sample info is loaded.
    end
elseif strcmp(ExistingOrMemory, 'Existing') == 1
    [fOutName,pOutName] = uigetfile('*.mat','Choose the output file');
    %%% Allows canceling.
    if fOutName == 0
        cd(handles.Current.StartupDirectory)
        return
    else
        try OutputFile = load([pOutName fOutName]);
        catch error('Sorry, the file could not be loaded for some reason.')
        end
    end
    %%% Checks whether any sample info is contained within the file. Some
    %%% old output files may not have the 'Measurements'
    %%% substructure, so we check for that field first.
    if isfield(OutputFile.handles,'Measurements') == 1
        Fieldnames = fieldnames(OutputFile.handles.Measurements);
        ImportedFieldnames = Fieldnames(strncmp(Fieldnames,'Imported',8) == 1 | strncmp(Fieldnames,'Image',5) == 1);
    else ImportedFieldnames = [];
    end
    if isempty(ImportedFieldnames) == 1
        errordlg('The output file you selected does not contain any sample info or data. It would be in a field called handles.Measurements, and would be prefixed with either ''Image'' or ''Imported''.')
        %%% Opens a filenameslistbox which displays the list of headings so that one can be
        %%% selected.  The OK button has been assigned to mean "View".
    else
        [Selected,Action] = listdlg('ListString',ImportedFieldnames, 'ListSize', [300 600],...
            'Name','Current sample info loaded',...
            'PromptString','Select the sample descriptions you would like to view.',...
            'OKString','View','CancelString','Cancel','SelectionMode','single');

        %%% Extracts the actual heading name.
        SelectedFieldName = ImportedFieldnames(Selected);

        % Action = 1 if the user pressed the OK (VIEW) button.  If they pressed
        % the cancel button or closed the window Action == 0.
        if Action == 1
            try
                ListToShow = OutputFile.handles.Measurements.(char(SelectedFieldName));
                if strcmp(class(ListToShow{1}),'double') == 1
                    ListToShow = sprintf('%d\n',cell2mat(ListToShow));
                end
                listdlg('ListString',ListToShow, 'ListSize', [300 600],...
                    'Name','Preview your sample info/data','PromptString',...
                    char(SelectedFieldName),'SelectionMode','single');
                %%% The OK buttons within this window don't do anything.
            catch errordlg('Sorry, there was an error displaying this sample info or data. This function may not yet work properly on mixed numerical and text data.')
            end
        else
            %%% If the user pressed "cancel" or closes the window, Action = 0, so
            %%% nothing happens.
        end
        %%% This "end" goes with the "isempty" if no sample info is loaded.
    end
end
cd(handles.Current.StartupDirectory)

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE IMAGE AS BUTTON %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes on button press in SaveImageAsButton.
function SaveImageAsButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.

MsgboxHandle = msgbox('Click twice on the image you wish to save. This window will be closed automatically - do not close it or click OK.');
%%% TODO: Should allow canceling.
waitforbuttonpress
ClickedImage = getimage(gca);
delete(MsgboxHandle)
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
    ProposedFileAndPathname = [handles.Current.DefaultOutputDirectory,'/',CompleteFileName];%%% TODO: Fix filename construction.
    OutputFileOverwrite = exist(ProposedFileAndPathname,'file'); 
    if OutputFileOverwrite ~= 0 
        Answer = questdlg(['A file with the name ', CompleteFileName, ' already exists at ', handles.Current.DefaultOutputDirectory,'. Do you want to overwrite it?'],'Confirm file overwrite','Yes','No','No');
        if strcmp(Answer,'Yes') == 1;
            imwrite(ClickedImage, ProposedFileAndPathname, Extension)
            msgbox(['The file ', CompleteFileName, ' has been saved to the default output directory.']);
        end
    else
        imwrite(ClickedImage, ProposedFileAndPathname, Extension)
        msgbox(['The file ', CompleteFileName, ' has been saved to the default output directory.']);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%
%%% SHOW IMAGE BUTTON %%%
%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes on button press in ShowImageButton.
function ShowImageButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
cd(handles.Current.SelectedImageDirectory)
%%% Opens a user interface window which retrieves a file name and path 
%%% name for the image to be shown.
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
    Image = im2double(imread([Pathname,'/',FileName])); %%% TODO: Fix filename construction.
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
cd(handles.Current.StartupDirectory)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SHOW PIXEL DATA BUTTON %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes on button press in ShowPixelDataButton.
function ShowPixelDataButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
FigureNumber = inputdlg('In which figure number would like to see pixel data?','',1);
if ~isempty(FigureNumber)
    FigureNumber = str2double(FigureNumber{1});
    pixval(FigureNumber,'on')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% EXPORT MEAN DATA BUTTON %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes on button press in ExportDataButton.
function ExportDataButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.

cd(handles.Current.DefaultOutputDirectory)
%%% Ask the user to choose the file from which to extract measurements.
[RawFileName, RawPathname] = uigetfile('*.mat','Select the raw measurements file');
if RawFileName == 0
else
    load(fullfile(RawPathname, RawFileName));
    %%% Extract the fieldnames of measurements from the handles structure.
    Fieldnames = fieldnames(handles.Measurements);
    MeasFieldnames = Fieldnames(strncmp(Fieldnames,'Image',5) == 1);
    FileFieldNames = Fieldnames(strncmp(Fieldnames, 'Filename', 8) == 1);
    ImportedFieldnames = Fieldnames(strncmp(Fieldnames,'Imported',8) == 1);
    TimeElapsedFieldNames = Fieldnames(strncmp(Fieldnames, 'TimeElapsed', 11) == 1);
    
    %%% Error detection.
    if isempty(MeasFieldnames) && isempty(FileFieldNames) && isempty(ImportedFieldnames) && isempty(TimeElapsedFieldNames)
        errordlg('No measurements were found in the file you selected. In the handles structure contained within the output file, the Measurements substructure must have fieldnames prefixed by ''Image'', ''Imported'', ''Filename'', or ''TimeElapsed''.')
    else
        %%% Tries to determine the number of image sets for which there are data.
        if isempty(FileFieldNames{1}) == 0
        fieldname = FileFieldNames{1};
        elseif isempty(MeasFieldnames{1}) == 0
        fieldname = MeasFieldnames{1};
        elseif isempty(ImportedFieldnames{1}) == 0
        fieldname = ImportedFieldnames{1};
        elseif isempty(TimeElapsedFieldNames{1}) == 0
        fieldname = TimeElapsedFieldNames{1};
        end
        TotalNumberImageSets = num2str(length(handles.Measurements.(fieldname)));
        TotalNumberImageSetsMsg = ['As a shortcut,                     type the numeral 0 to extract data from all ', TotalNumberImageSets, ' image sets.'];
        %%% Ask the user to specify the number of image sets to extract.
        NumberOfImages = inputdlg({['How many image sets do you want to extract?  ',TotalNumberImageSetsMsg]},'Specify number of image sets',1,{'0';' '});
        %%% If the user presses the Cancel button, the program goes to the end.
        if isempty(NumberOfImages)
        else
            %%% Calculate the appropriate number of image sets.
            NumberOfImages = str2double(NumberOfImages{1});
            if NumberOfImages == 0
                NumberOfImages = str2double(TotalNumberImageSets);
            elseif NumberOfImages > length(handles.Measurements.(char(MeasFieldnames(1))));
                errordlg(['There are only ', length(handles.Measurements.(char(MeasFieldnames(1)))), ' image sets total.'])
                %%% TODO: This error checking is only for the first field of
                %%% measurements.  Should make it more comprehensive.
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
                OutputFileOverwrite = exist([cd,'/',FileName],'file'); %%% TODO: Fix filename construction.
                if OutputFileOverwrite ~= 0
                    errordlg('A file with that name already exists in the directory containing the raw measurements file.  Repeat and choose a different file name.')
                else
                    %%% Extract the measurements.  Waitbar shows the percentage of image sets
                    %%% remaining.
                    WaitbarHandle = waitbar(0,'Extracting measurements...');
                    %%% TODO: Combine all the fieldnames into a single
                    %%% variable to speed this processing.
                    
                    %%% Preallocate the variable Measurements.
                    NumberOfMeasFieldnames = length(MeasFieldnames);
                    NumberOfFileFieldNames = length(FileFieldNames);
                    NumberOfImportedFieldnames = length(ImportedFieldnames);
                    NumberOfTimeElapsedFieldNames = length(TimeElapsedFieldNames);
                    
                    NumberOfFields = NumberOfMeasFieldnames + NumberOfFileFieldNames + NumberOfImportedFieldnames + NumberOfTimeElapsedFieldNames;
                    Measurements(NumberOfImages,NumberOfFields) = {[]};
                    %%% Finished preallocating the variable Measurements.
                    TimeStart = clock;
                    for imagenumber = 1:NumberOfImages
                        for FileNameFieldNumber = 1:NumberOfFileFieldNames
                            Fieldname = cell2mat(FileFieldNames(FileNameFieldNumber));
                            FieldNumber = FieldNumber + 1;
                            Measurements(imagenumber,FieldNumber) = {handles.Pipeline.(Fieldname){imagenumber}};
                        end
                        for ImportedFieldNumber = 1:NumberOfImportedFieldnames
                            Fieldname = cell2mat(ImportedFieldnames(ImportedFieldNumber));
                            FieldNumber = FieldNumber + 1;
                            Measurements(imagenumber, FieldNumber) = {handles.Measurements.(Fieldname){imagenumber}};
                        end
                        for FieldNumber = 1:NumberOfMeasFieldnames
                            Fieldname = cell2mat(MeasFieldnames(FieldNumber));
                            Measurements(imagenumber,FieldNumber) = {handles.Measurements.(Fieldname){imagenumber}};
                        end
                        for TimeElapsedFieldNumber = 1:NumberOfTimeElapsedFieldNames
                            Fieldname = cell2mat(TimeElapsedFieldNames(TimeElapsedFieldNumber));
                            FieldNumber = FieldNumber + 1;
                            Measurements(imagenumber, FieldNumber) = {handles.Measurements.(Fieldname){imagenumber}};
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
                    for i = 1:NumberOfMeasFieldnames
                        fwrite(fid, char(MeasFieldnames(i)), 'char');
                        fwrite(fid, sprintf('\t'), 'char');
                    end
                    for i = 1:NumberOfFileFieldNames
                        fwrite(fid, char(FileFieldNames(i)), 'char');
                        fwrite(fid, sprintf('\t'), 'char');
                    end
                    for i = 1:NumberOfImportedFieldnames
                        fwrite(fid, char(ImportedFieldnames(i)), 'char');
                        fwrite(fid, sprintf('\t'), 'char');
                    end
                    for i = 1:NumberOfTimeElapsedFieldNames
                        fwrite(fid, char(TimeElapsedFieldNames(i)), 'char');
                        fwrite(fid, sprintf('\t'), 'char');
                    end

                    fwrite(fid, sprintf('\n'), 'char');
                    %%% Write the Measurements.
                    WaitbarHandle = waitbar(0,'Writing the measurements file...');
                    NumberMeasurements = size(Measurements,1);
                    TimeStart = clock;
                    for i = 1:NumberMeasurements
                        for measure = 1:NumberOfFields
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

cd(handles.Current.StartupDirectory);
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% EXPORT CELL BY CELL DATA BUTTON %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes on button press in ExportCellByCellButton.
function ExportCellByCellButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.

cd(handles.Current.DefaultOutputDirectory)
%%% Ask the user to choose the file from which to extract measurements.
[RawFileName, RawPathname] = uigetfile('*.mat','Select the raw measurements file');
if RawFileName == 0
    cd(handles.Current.StartupDirectory);
    return
end
load(fullfile(RawPathname, RawFileName));

Answer = questdlg('Do you want to export cell by cell data for all measurements from one image, or data from all images for one measurement?','','All measurements','All images','All measurements');

if strcmp(Answer, 'All images') == 1
    %%% Extract the fieldnames of cell by cell measurements from the handles structure. 
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
        'PromptString','Choose a measurement to export','CancelString','Cancel',...
        'SelectionMode','single');
    if ok == 0
        cd(handles.Current.StartupDirectory);
        return
    end
    EditedMeasurementToExtract = char(EditedMeasFieldnames(Selection));
    MeasurementToExtract = ['Object', EditedMeasurementToExtract];
    TotalNumberImageSets = length(handles.Measurements.(MeasurementToExtract));
    Measurements = handles.Measurements.(MeasurementToExtract);
    
    %%% Extract the fieldnames of non-cell by cell measurements from the
    %%% handles structure. This will be used as headings for each column of
    %%% measurements.
    Fieldnames = fieldnames(handles.Measurements);
    HeadingFieldnames = Fieldnames(strncmp(Fieldnames,'Filename',8) == 1 | strncmp(Fieldnames,'Imported',8) == 1 | strncmp(Fieldnames,'TimeElapsed',11) == 1);
    %%% Error detection.
    if isempty(HeadingFieldnames)
        errordlg('No headings were found in the file you selected.  They would be found within the output file''s handles.Pipeline or handles.Measurements structure preceded by ''Filename'', ''Imported'', or ''TimeElapsed''.')
        cd(handles.Current.StartupDirectory);
        return
    end

    %%% Allows the user to select a heading name from the list.
    [Selection, ok] = listdlg('ListString',HeadingFieldnames, 'ListSize', [300 600],...
        'Name','Select measurement',...
        'PromptString','Choose a field to label each column of data with','CancelString','Cancel',...
        'SelectionMode','single');
    if ok == 0
        cd(handles.Current.StartupDirectory);
        return
    end
    HeadingToDisplay = char(HeadingFieldnames(Selection));

    %%% Have the user choose which of image/cells should be rows/columns
    RowColAnswer = questdlg('Which layout do you want images and cells to follow in the exported data?  WARNING: Excel spreadsheets can only have 256 columns.','','Rows = Cells, Columns = Images','Rows = Images, Columns = Cells','Rows = Cells, Columns = Images');
    %%% Extracts the headings.
    try ListOfHeadings = handles.Pipeline.(HeadingToDisplay);
    catch ListOfHeadings = handles.Measurements.(HeadingToDisplay);
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
        cd(handles.Current.StartupDirectory);
        return
    end
    FileName = FileName{1};
    OutputFileOverwrite = exist([cd,'/',FileName],'file'); %%% TODO: Fix filename construction.
    if OutputFileOverwrite ~= 0
        Answer = questdlg('A file with that name already exists in the directory containing the raw measurements file. Do you wish to overwrite?','Confirm overwrite','Yes','No','No');
        if strcmp(Answer, 'No') == 1
            cd(handles.Current.StartupDirectory);
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
    TotalNumberImageSets = handles.Current.SetBeingAnalyzed;
    %%% Asks the user to specify which image set to export.
    Answers = inputdlg({['Enter the sample number to export. There are ', num2str(TotalNumberImageSets), ' total.']},'Choose samples to export',1,{'1'});
    if isempty(Answers{1})
        cd(handles.Current.StartupDirectory);
        return
    end
    try ImageNumber = str2double(Answers{1});
    catch errordlg('The text entered was not a number.')
        cd(handles.Current.StartupDirectory);
        return
    end
    if ImageNumber > TotalNumberImageSets
        errordlg(['There are only ', num2str(TotalNumberImageSets), ' image sets total.'])
        cd(handles.Current.StartupDirectory);
        return
    end
    
    %%% Extract the fieldnames of cell by cell measurements from the handles structure. 
    Fieldnames = fieldnames(handles.Measurements);
    MeasFieldnames = Fieldnames(strncmp(Fieldnames,'Object',6)==1);
    %%% Error detection.
    if isempty(MeasFieldnames)
        errordlg('No measurements were found in the file you selected.  They would be found within the output file''s handles.Measurements structure preceded by ''Object''.')
        cd(handles.Current.StartupDirectory);
        return
    end
    
    %%% Extract the fieldnames of non-cell by cell measurements from the
    %%% handles structure. This will be used as headings for each column of
    %%% measurements.
    Fieldnames = fieldnames(handles.Measurements);
    HeadingFieldnames = Fieldnames(strncmp(Fieldnames,'Filename',8)==1 | strncmp(Fieldnames,'Imported',8) == 1);
    %%% Error detection.
    if isempty(HeadingFieldnames)
        errordlg('No headings were found in the file you selected.  They would be found within the output file''s handles.Pipeline structure preceded by ''Filename''.')
        cd(handles.Current.StartupDirectory);
        return
    end

    %%% Allows the user to select a heading name from the list.
    [Selection, ok] = listdlg('ListString',HeadingFieldnames, 'ListSize', [300 600],...
        'Name','Select measurement',...
        'PromptString','Choose a field to label this data.','CancelString','Cancel',...
        'SelectionMode','single');
    if ok == 0
        cd(handles.Current.StartupDirectory);
        return
    end
    HeadingToDisplay = char(HeadingFieldnames(Selection));
    %%% Extracts the headings.
    try ImageNamesToDisplay = handles.Pipeline.(HeadingToDisplay);
    catch ImageNamesToDisplay = handles.Measurements.(HeadingToDisplay);
    end
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
        cd(handles.Current.StartupDirectory);
        return
    end
    FileName = FileName{1};
    OutputFileOverwrite = exist([cd,'/',FileName],'file'); %%% TODO: Fix filename construction.
    if OutputFileOverwrite ~= 0
        Answer = questdlg('A file with that name already exists in the directory containing the raw measurements file. Do you wish to overwrite?','Confirm overwrite','Yes','No','No');
        if strcmp(Answer, 'No') == 1
            cd(handles.Current.StartupDirectory);
            return    
        end
    end
    
    %%% Opens the file and names it appropriately.
    fid = fopen(FileName, 'wt');
    %%% Writes ImageNameToDisplay as the heading for the first column/row.
    fwrite(fid, char(ImageNameToDisplay), 'char');
    fwrite(fid, sprintf('\n'), 'char');
    %%% Writes the data, row by row: one row for each measurement type.

    %%% Writes the headings
    for MeasNumber = 1:length(MeasFieldnames)
      FieldName = char(MeasFieldnames(MeasNumber));
      %%% Writes the measurement heading in the first column.
      fwrite(fid, FieldName, 'char');
      %%% Tabs to the next column.
      fwrite(fid, sprintf('\t'), 'char');
    end

    %%% Find the largest measurement set
    maxlength = 0;
    for MeasNumber = 1:length(MeasFieldnames)
      FieldName = char(MeasFieldnames(MeasNumber));
      Measurements = handles.Measurements.(FieldName);
      maxlength = max(maxlength, length(Measurements{ImageNumber}));
    end

    %%% Returns to the next row.
    fwrite(fid, sprintf('\n'), 'char');
      
    %%% Writes the data row-by-row
    for idx = 1:maxlength,
      for MeasNumber = 1:length(MeasFieldnames)
        FieldName = char(MeasFieldnames(MeasNumber));
        Measurements = handles.Measurements.(FieldName){ImageNumber};
        if (length(Measurements) >= idx),
          %%% Writes the measurements for that measurement type in successive columns.
          fprintf(fid,'%d\t',Measurements(idx));
        else
          fwrite(fid, sprintf('\t'), 'char');
        end
      end
      %%% Returns to the next row.
      fwrite(fid, sprintf('\n'), 'char');
    end

    %%% Closes the file
    fclose(fid);
    helpdlg(['The file ', FileName, ' has been written to the directory where the raw measurements file is located.'])
end
cd(handles.Current.StartupDirectory);

%%%%%%%%%%%%%%%%%%%%%%%
%%% PLOT DATA BUTTON %%%
%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes on button press in PlotDataButton.
function PlotDataButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.

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

%%%%%%%%%%%%%%%%%%%%%%%%%
%%% DATA LAYOUT BUTTON %%%
%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes on button press in DataLayoutButton.
function DataLayoutButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.

cd(handles.Current.DefaultOutputDirectory)
%%% Ask the user to choose the file from which to extract measurements.
[RawFileName, RawPathname] = uigetfile('*.mat','Select the raw measurements file');
if RawFileName == 0
    cd(handles.Current.StartupDirectory);
    return
end
    load(fullfile(RawPathname, RawFileName));
%%% Extract the fieldnames of measurements from the handles structure.
Fieldnames = fieldnames(handles.Measurements);
MeasFieldnames = Fieldnames(strncmp(Fieldnames,'Image',5)==1);
%%% Error detection.
if isempty(MeasFieldnames)
    errordlg('No measurements were found in the file you selected.  They would be found within the output file''s handles.Measurements structure preceded by ''Image''.')
    cd(handles.Current.StartupDirectory);
    return
end
%%% Removes the 'Object' prefix from each name for display purposes.
for Number = 1:length(MeasFieldnames)
    EditedMeasFieldnames{Number} = MeasFieldnames{Number}(6:end);
end
%%% Allows the user to select a measurement from the list.
[Selection, ok] = listdlg('ListString',EditedMeasFieldnames, 'ListSize', [300 600],...
    'Name','Select measurement',...
    'PromptString','Choose a measurement to display','CancelString','Cancel',...
    'SelectionMode','single');
if ok == 0
    return
end
EditedMeasurementToExtract = char(EditedMeasFieldnames(Selection));
MeasurementToExtract = ['Image', EditedMeasurementToExtract];

AllMeasurementsCellArray = handles.Measurements.(MeasurementToExtract);

Prompts = {'Enter the number of rows','Enter the number of columns'};
Defaults = {'24','16'};
Answers = inputdlg(Prompts,'Describe Array/Slide Format',1,Defaults);
if isempty(Answers)
    return
end
NumberRows = str2double(Answers{1});
NumberColumns = str2double(Answers{2});
TotalSamplesToBeGridded = NumberRows*NumberColumns;
NumberSamplesImported = length(AllMeasurementsCellArray);
if TotalSamplesToBeGridded > NumberSamplesImported
    h = warndlg(['You have specified a layout of ', num2str(TotalSamplesToBeGridded), ' samples in the layout, but only ', num2str(NumberSamplesImported), ' measurements were imported. The remaining spaces in the layout will be filled in with the value of the last sample.']);
    waitfor(h)
    AllMeasurementsCellArray(NumberSamplesImported+1:TotalSamplesToBeGridded) = AllMeasurementsCellArray(NumberSamplesImported);
elseif TotalSamplesToBeGridded < NumberSamplesImported
    h = warndlg(['You have specified a layout of ', num2str(TotalSamplesToBeGridded), ' samples in the layout, but ', num2str(NumberSamplesImported), ' measurements were imported. The imported measurements at the end will be ignored.']);
    waitfor(h)
    AllMeasurementsCellArray(TotalSamplesToBeGridded+1:NumberSamplesImported) = [];
end
MeanImage = reshape(cell2mat(AllMeasurementsCellArray),NumberRows,NumberColumns);

%%% Shows the results.
figure, imagesc(MeanImage), title(EditedMeasurementToExtract), colorbar

% % --- Executes on button press in DataLayoutButton.
% %%% THIS WAS A VERY SPECIALIZED VERSION OUR LAB USED TO NORMALIZE OUR
% %%% DATA>>>>
% function DataLayoutButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
% h = msgbox('Copy your data to the clipboard then press OK');
% waitfor(h)
% 
% uiimport('-pastespecial');
% h = msgbox('After importing your data and pressing "Finish", click OK');
% waitfor(h)
% if exist('clipboarddata','var') == 0
%     return
% end
% IncomingData = clipboarddata;
% 
% Prompts = {'Enter the number of rows','Enter the number of columns','Enter the percentile below which values will be excluded from fitting the normalization function.','Enter the percentile above which values will be excluded from fitting the normalization function.'};
% Defaults = {'24','16','.05','.95'};
% Answers = inputdlg(Prompts,'Describe Array/Slide Format',1,Defaults);
% if isempty(Answers)
%     return
% end
% NumberRows = str2double(Answers{1});
% NumberColumns = str2double(Answers{2});
% LowPercentile = str2double(Answers{3});
% HighPercentile = str2double(Answers{4});
% TotalSamplesToBeGridded = NumberRows*NumberColumns;
% NumberSamplesImported = length(IncomingData);
% if TotalSamplesToBeGridded > NumberSamplesImported
%     h = warndlg(['You have specified a layout of ', num2str(TotalSamplesToBeGridded), ' samples in the layout, but only ', num2str(NumberSamplesImported), ' measurements were imported. The remaining spaces in the layout will be filled in with the value of the last sample.']); 
%     waitfor(h)
%     IncomingData(NumberSamplesImported+1:TotalSamplesToBeGridded) = IncomingData(NumberSamplesImported);
% elseif TotalSamplesToBeGridded < NumberSamplesImported
%     h = warndlg(['You have specified a layout of ', num2str(TotalSamplesToBeGridded), ' samples in the layout, but ', num2str(NumberSamplesImported), ' measurements were imported. The imported measurements at the end will be ignored.']); 
%     waitfor(h)
%     IncomingData(TotalSamplesToBeGridded+1:NumberSamplesImported) = [];
% end
% 
% %%% The data is shaped into the appropriate grid.
% MeanImage = reshape(IncomingData,NumberRows,NumberColumns);
% 
% %%% The data are listed in ascending order.
% AscendingData = sort(IncomingData);
% 
% %%% The percentiles are calculated. (Statistics Toolbox has a percentile
% %%% function, but many users may not have that function.)
% %%% The values to be ignored are set to zero in the mask.
% mask = MeanImage;
% if LowPercentile ~= 0
% RankOrderOfLowThreshold = floor(LowPercentile*NumberSamplesImported);
% LowThreshold = AscendingData(RankOrderOfLowThreshold);
% mask(mask <= LowThreshold) = 0;
% end
% if HighPercentile ~= 1
% RankOrderOfHighThreshold = ceil(HighPercentile*NumberSamplesImported);
% HighThreshold = AscendingData(RankOrderOfHighThreshold);
% mask(mask >= HighThreshold) = 0;
% end
% ThrownOutDataForDisplay = mask;
% ThrownOutDataForDisplay(mask > 0) = 1;
% 
% %%% Fits the data to a third-dimensional polynomial 
% % [x,y] = meshgrid(1:size(MeanImage,2), 1:size(MeanImage,1));
% % x2 = x.*x;
% % y2 = y.*y;
% % xy = x.*y;
% % x3 = x2.*x;
% % x2y = x2.*y;
% % xy2 = y2.*x;
% % y3 = y2.*y;
% % o = ones(size(MeanImage));
% % ind = find((MeanImage > 0) & (mask > 0));
% % coeffs = [x3(ind) x2y(ind) xy2(ind) y3(ind) x2(ind) y2(ind) xy(ind) x(ind) y(ind) o(ind)] \ double(MeanImage(ind));
% % IlluminationImage = reshape([x3(:) x2y(:) xy2(:) y3(:) x2(:) y2(:) xy(:) x(:) y(:) o(:)] * coeffs, size(MeanImage));
% % 
% 
% %%% Fits the data to a fourth-dimensional polynomial 
% [x,y] = meshgrid(1:size(MeanImage,2), 1:size(MeanImage,1));
% x2 = x.*x;
% y2 = y.*y;
% xy = x.*y;
% x3 = x2.*x;
% x2y = x2.*y;
% xy2 = y2.*x;
% y3 = y2.*y;
% x4 = x2.*x2;
% y4 = y2.*y2;
% x3y = x3.*y;
% x2y2 = x2.*y2;
% xy3 = x.*y3;
% o = ones(size(MeanImage));
% ind = find((MeanImage > 0) & (mask > 0));
% coeffs = [x4(ind) x3y(ind) x2y2(ind) xy3(ind) y4(ind) ...
%           x3(ind) x2y(ind) xy2(ind) y3(ind) ...
%           x2(ind) xy(ind) y2(ind)  ...
%           x(ind) y(ind) ...
%           o(ind)] \ double(MeanImage(ind));
% IlluminationImage = reshape([x4(:) x3y(:) x2y2(:) xy3(:) y4(:) ...
%           x3(:) x2y(:) xy2(:) y3(:) ...
%           x2(:) xy(:) y2(:)  ...
%           x(:) y(:) ...
%           o(:)] * coeffs, size(MeanImage));
% CorrFactorsRaw = reshape(IlluminationImage,TotalSamplesToBeGridded,1);
% IlluminationImage2 = IlluminationImage ./ mean(CorrFactorsRaw);
%   
% %%% Shows the results.
% figure, subplot(1,3,1), imagesc(MeanImage), title('Imported Data'), colorbar
% subplot(1,3,2), imagesc(ThrownOutDataForDisplay), title('Ignored Samples'),
% subplot(1,3,3), imagesc(IlluminationImage2), title('Correction Factors'), colorbar
% 
% %%% Puts the results in a column and displays in the main Matlab window.
% OrigData = reshape(MeanImage,TotalSamplesToBeGridded,1) %#ok We want to ignore MLint error checking for this line.
% CorrFactors = reshape(IlluminationImage2,TotalSamplesToBeGridded,1);
% CorrectedData = OrigData./CorrFactors %#ok We want to ignore MLint error checking for this line.
% 
% msgbox('The original data and the corrected data are now displayed in the Matlab window. You can cut and paste from there.')
% 
% % %%% Exports the results to the clipboard.
% % clipboard('copy',CorrFactors);
% % h = msgbox('The correction factors are now on the clipboard. Paste them where desired and press OK.  The data is also displayed in column format in the main Matlab window, so you can copy and paste from there as well.');
% % waitfor(h)
% % clipboard('copy',OrigData);
% % h = msgbox('The original data used to generate those normalization factors is now on the clipboard. Paste them where desired (if desired) and press OK.  The data is also displayed in column format in the main Matlab window, so you can copy and paste from there as well.');
% % waitfor(h)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SHOW DATA ON IMAGE BUTTON %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes on button press in ShowDataOnImageButton.
function ShowDataOnImageButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.

cd(handles.Current.DefaultOutputDirectory)
%%% Asks the user to choose the file from which to extract measurements.
[RawFileName, RawPathname] = uigetfile('*.mat','Select the raw measurements file');
if RawFileName ~= 0
    load(fullfile(RawPathname,RawFileName));
    %%% Extracts the fieldnames of measurements from the handles structure. 
    Fieldnames = fieldnames(handles.Measurements);
    MeasFieldnames = Fieldnames(strncmp(Fieldnames,'Object',6)==1);
    %%% Error detection.
    if isempty(MeasFieldnames)
        errordlg('No measurements were found in the file you selected.  They would be found within the output file''s handles.Measurements structure preceded by ''Object''.')
        cd(handles.Current.StartupDirectory);
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
                        cd(handles.Current.StartupDirectory);
                        return
                    end
                    SampleNumber = str2double(Answer{1});
                    TotalNumberImageSets = length(handles.Measurements.(MeasurementToExtract));
                    if SampleNumber > TotalNumberImageSets
                        cd(handles.Current.StartupDirectory);
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
                        cd(handles.Current.SelectedImageDirectory)
                        [FileName,Pathname] = uigetfile('*.*','Select the image to view');
                        delete(h)
                        %%% If the user presses "Cancel", the FileName will = 0 and nothing will
                        %%% happen.
                        if FileName == 0
                            cd(handles.Current.StartupDirectory);
                            return
                        else
                            %%% Opens and displays the image, with pixval shown.
                            ImageToDisplay = im2double(imread([Pathname,'/',FileName])); %%% TODO: Fix filename construction.
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
cd(handles.Current.StartupDirectory);