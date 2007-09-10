function varargout = CellProfiler(varargin)

% CellProfilerTM cell image analysis software
%
% CellProfiler cell image analysis software is designed for biologists
% without training in computer vision or programming to quantitatively
% measure phenotypes from thousands of images automatically.
%
% CellProfiler Developer's version allows you to write your own modules and
% tools for CellProfiler using Matlab.

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
% $Revision$

%%% Do not remove the following line.  It is used by CompileWizard.m.
%%% Compiler: INSERT FUNCTIONS HERE

% Begin initialization code - DO NOT EDIT
if ~nargin
    SplashScreen;
    tic
    if ~isdeployed
        try
            addpath(genpath(fileparts(which('CellProfiler.m'))))
            savepath
        catch CPerrordlg('You changed the name of CellProfiler.m file. Consequences of this are unknown.');
        end
    end
end
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @CellProfiler_OpeningFcn, ...
    'gui_OutputFcn',  @CellProfiler_OutputFcn, ...
    'gui_LayoutFcn',  @CellProfiler_LayoutFcn, ...
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

%%%%%%%%%%%%%%%%%%%%%%%%
%%% INITIAL SETTINGS %%%
%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes just before CellProfiler is made visible.
function CellProfiler_OpeningFcn(hObject, eventdata, handles, varargin) %#ok We want to ignore MLint error checking for this line.

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
if ~isdeployed
    addpath(pwd);
end
handles.FunctionHandles.LoadPipelineCallback = @LoadPipeline_Callback;
%%% Retrieves preferences from CellProfilerPreferences.mat, if possible.
%%% Try loading CellProfilerPreferences.mat first from the matlabroot
%%% directory and then the current directory.  This is not necessary for
%%% CellProfiler to function; it just allows defaults to be
%%% pre-loaded.
if isdeployed
    try
        load(fullfile(handles.Current.StartupDirectory, 'CellProfilerPreferences.mat'))
        LoadedPreferences = SavedPreferences;
        clear SavedPreferences
    end
    handles.Preferences.DefaultModuleDirectory = fullfile(pwd,'Modules');
else
    try
        load(fullfile(matlabroot,'CellProfilerPreferences.mat'))
        LoadedPreferences = SavedPreferences;
        clear SavedPreferences
    catch
        try
            load(fullfile(handles.Current.StartupDirectory, 'CellProfilerPreferences.mat'))
            LoadedPreferences = SavedPreferences;
            clear SavedPreferences
        end
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

%%% Set the default color for CP dialogs.
set(0, 'defaultuicontrolbackgroundcolor', [0.7 0.7 0.9]);

if ~isdeployed
    %%% If the Default Module Directory has not yet been successfully
    %%% identified (i.e., it is not present in the loaded preferences or
    %%% the directory does not exist), look at where the CellProfiler.m
    %%% file is located and see whether there is a subdirectory within
    %%% that directory, called "Modules".  If so, use that subdirectory as
    %%% the default module directory. If not, use the current directory.
    if ~isfield(handles.Preferences,'DefaultModuleDirectory')
        CellProfilerPathname = fileparts(which('CellProfiler'));
        %%% Checks whether the Modules subdirectory exists.
        if exist(fullfile(CellProfilerPathname,'Modules'), 'dir')
            CellProfilerModulePathname = fullfile(CellProfilerPathname,'Modules');
            handles.Preferences.DefaultModuleDirectory = CellProfilerModulePathname;
        else
            handles.Preferences.DefaultModuleDirectory = handles.Current.StartupDirectory;
        end
    end
end

%%% Similar approach for the DefaultOutputDirectory.
try
    if exist(LoadedPreferences.DefaultOutputDirectory, 'dir')
        handles.Preferences.DefaultOutputDirectory = LoadedPreferences.DefaultOutputDirectory;
    end
end
if ~isfield(handles.Preferences,'DefaultOutputDirectory')
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

try
    handles.Preferences.IntensityColorMap = LoadedPreferences.IntensityColorMap;
catch
    handles.Preferences.IntensityColorMap = 'gray';
end

try
    handles.Preferences.LabelColorMap = LoadedPreferences.LabelColorMap;
catch
    handles.Preferences.LabelColorMap = 'jet';
end

try
    handles.Preferences.StripPipeline = LoadedPreferences.StripPipeline;
catch
    handles.Preferences.StripPipeline = 'Yes';
end

try
    handles.Preferences.SkipErrors = LoadedPreferences.SkipErrors;
catch
    handles.Preferences.SkipErrors = 'No';
end

try
    handles.Preferences.DisplayModeValue = LoadedPreferences.DisplayModeValue;
catch
    handles.Preferences.DisplayModeValue = 1;
end

handles.Preferences.DisplayWindows =[];

%%% Now that handles.Preferences.(10 different variables) has been filled
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
if ~isdeployed
    addpath(handles.Preferences.DefaultModuleDirectory)
    CellProfilerPathname = fileparts(which('CellProfiler'));
    CellProfilerModulePathname = fullfile(CellProfilerPathname,'Modules');
    handles.Current.CellProfilerPathname = CellProfilerPathname;
    try
        addpath(CellProfilerModulePathname)
    end
end

%%% Sets a suitable fontsize. An automatic font size is calculated,
%%% but it is overridden if the user has set a default font size.
%%% The fontsize is also saved in the main window's (i.e. "0") UserData property so that
%%% it can be used for setting the fontsize in dialog boxes.
if exist('LoadedPreferences','var') && isfield(LoadedPreferences,'FontSize') && ~isempty(str2double(LoadedPreferences.FontSize))
    handles.Preferences.FontSize = str2double(LoadedPreferences.FontSize);
else
    ScreenResolution = get(0,'ScreenPixelsPerInch');
    handles.Preferences.FontSize = (220 - ScreenResolution)/13;       % 90 pix/inch => 10pts, 116 pix/inch => 8pts
end
names = fieldnames(handles);
for k = 1:length(names)
    if ishandle(handles.(names{k}))
        set(findobj(handles.(names{k}),'-property','FontSize'),'FontSize',handles.Preferences.FontSize,'FontName','helvetica')
    end
end

%%% Set the default font size for CP dialogs, based on the preferences.
set(0, 'defaultuicontrolfontsize', handles.Preferences.FontSize);
set(0, 'defaultuicontrolfontname', 'helvetica');

%%% Checks whether the user has the Image Processing Toolbox.
Answer = license('test','image_toolbox');
if Answer ~= 1
    CPwarndlg('It appears that you do not have a license for the Image Processing Toolbox of Matlab.  Many of the image analysis modules of CellProfiler may not function properly. Typing ''ver'' or ''license'' at the Matlab command line may provide more information about your current license situation.');
end

if ~isdeployed
    %%% Adds the Help folder to Matlab's search path.
    try Pathname = fullfile(handles.Current.CellProfilerPathname,'Help');
        addpath(Pathname)
    catch
        CPerrordlg('CellProfiler could not find its help files, which should be located in a folder called Help within the folder containing CellProfiler.m. The help buttons will not be functional.');
    end
end

%%% Checks figure handles for current open windows.
handles.Current.CurrentHandles = findobj;

%%% Note on the use of "8192" when retrieving handles...
%%% Apparently, referring to a handle in a callback by converting the numerical
%%% handle value to a string does not work, due to a precision problem in
%%% Matlab, according to this newsgroup:
%%% http://groups.google.com/groups?hl=en&lr=&safe=off&selm=3lglao%242pi%40borg.svpal.org
%%% Here's what that site says:
% I have successfully used the following technique to convert a handle to
% a string:  str_h =  sprintf('%d',8192*handle)
% Then to retreive it:  handle = eval(str_h) / 8192
% This works (at least under MATLAB for Windows) because handles are always
% multiples of 8192. I found that Duane's method is not reliable due to
% accuracy problems with MATLAB's decimal to binary conversion.
%
% Duane Hanselman (dua...@eece.maine.edu) wrote:
% : In article <3l6dm0$...@hardy.ee.gatech.edu>, "Mark A. Yoder" <Mark.A.Yo...@Rose-Hulman.edu> says:
% : Mark:
% : Have you tried sprintf(%.15f,handle) to convert the handle to string?
% : Then you can use handle=eval(string_handle) to convert the string back
% : to a number.

%%% Sets up the main program window (Main GUI window) so that it asks for
%%% confirmation prior to closing.
%%% First, obtains the handle for the main GUI window (aka figure1).
ClosingFunction = ...
    ['deleteme = CPquestdlg(''Do you really want to quit?'', ''Confirm quit'',''Yes'',''No'',''Yes''); switch deleteme; case ''Yes'';  CellProfiler(''CloseWindows_Helper'',gcbo,[],guidata(gcbo)), delete(', num2str((handles.figure1)*8192), '/8192); case ''No''; end; clear deleteme'];
%%% Sets the closing function of the Main GUI window to be the line
%%% above.
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

%%% Adds the CellProfiler logo to the top left of the main GUI.
handles.CPlogoAxes = axes('Position',[.01 .84 .1 .15],...
    'Parent',handles.figure1,...
    'Tag','CPlogoAxes');
Logo = CPlogo;
handles.CPlogoImage = image(Logo,...
    'parent',handles.CPlogoAxes,...
    'Tag','CPlogoImage');
%%% The axes are turned off so there are not pixel markings on the logo.
%%% Visibility must apparently be turned off after the image() command
%%% because otherwise the axes reappear.
set(handles.CPlogoAxes,'visible','off')

%%% Finds all available tools, which are .m files residing in the
%%% Modules folder.
%%%
%%% Do not remove the BEGIN line below.  It is used by CompileWizard.m.
%%% Compiler: BEGIN HELP
Pathname = fullfile(handles.Current.CellProfilerPathname,'Modules');
ListOfTools{1} = 'Modules: none loaded';
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
    %%% I don't think we really want to display this text for each tools'
    %%% help. This info is already provided at the stage where the user
    %%% chooses which tool; it's confusing to show it again.
    %    ToolHelpInfo = 'Help information from individual modules files, which are Matlab m-files located within the Modules directory:';
    ToolHelpInfo = '';
    if ~isempty(FileNamesNoDir)
        %%% Looks for .m files.
        for i = 1:length(FileNamesNoDir),
            if strncmp(FileNamesNoDir{i}(end-1:end),'.m',2)
                if ~strcmp(FileNamesNoDir{i},'ShowHelpForThisMenu.m')
                    ListOfTools(length(ListOfTools)+1) = {FileNamesNoDir{i}(1:end-2)};
                    ToolHelp{length(ListOfTools)-1} = help(char(FileNamesNoDir{i}(1:end-2))); %#ok Ignore MLint
                else
                    helpnum = i;
                end
            end
        end
        if exist('helpnum','var')
            ListOfTools(length(ListOfTools)+1) = {FileNamesNoDir{helpnum}(1:end-2)};
            ToolHelp{length(ListOfTools)-1} = help(char(FileNamesNoDir{helpnum}(1:end-2)));
        end
        if length(ListOfTools) > 1
            ListOfTools(1) = {'Modules'};
        else ToolHelp = 'No modules were loaded upon starting up CellProfiler. Modules are Matlab m-files ending in ''.m'', and should be located in a folder called Modules within the folder containing CellProfiler.m';
        end
    end
end
handles.Current.ModulesFilenames = ListOfTools;
handles.Current.ModulesHelp = ToolHelp;

%%% Finds all available tools, which are .m files residing in the
%%% ImageTools folder.
ListOfTools = {};
Pathname = fullfile(handles.Current.CellProfilerPathname,'ImageTools');
ListOfTools{1} = 'Image tools: none loaded';
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
    %%% I don't think we really want to display this text for each tools'
    %%% help. This info is already provided at the stage where the user
    %%% chooses which tool; it's confusing to show it again.
    %    ToolHelpInfo = 'Help information from individual image tool files, which are Matlab m-files located within the ImageTools directory:';
    ToolHelpInfo = '';
    if ~isempty(FileNamesNoDir)
        %%% Looks for .m files.
        for i = 1:length(FileNamesNoDir),
            if strncmp(FileNamesNoDir{i}(end-1:end),'.m',2)
                if ~strcmp(FileNamesNoDir{i},'ShowHelpForThisMenu.m')
                    ListOfTools(length(ListOfTools)+1) = {FileNamesNoDir{i}(1:end-2)};
                    ToolHelp{length(ListOfTools)-1} = help(char(FileNamesNoDir{i}(1:end-2)));
                else
                    helpnum = i;
                end
            end
        end
        if exist('helpnum','var')
            ListOfTools(length(ListOfTools)+1) = {FileNamesNoDir{helpnum}(1:end-2)};
            ToolHelp{length(ListOfTools)-1} = help(char(FileNamesNoDir{helpnum}(1:end-2)));
        end
        if length(ListOfTools) > 1
            ListOfTools(1) = {'Image tools'};
        else ToolHelp = 'No image tools were loaded upon starting up CellProfiler. Image tools are Matlab m-files ending in ''.m'', and should be located in a folder called ImageTools within the folder containing CellProfiler.m';
        end
    end
end
handles.Current.ImageToolsFilenames = ListOfTools;
handles.Current.ImageToolHelp = ToolHelp;

ListOfTools = {};
%%% Finds all available tools, which are .m files residing in the
%%% DataTools folder.
Pathname = fullfile(handles.Current.CellProfilerPathname,'DataTools');
ListOfTools{1} = 'Data tools: none loaded';
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
    ToolHelpInfo = 'Help information from individual data tool files, which are Matlab m-files located within the DataTools directory:';
    if isempty(FileNamesNoDir) ~= 1
        %%% Looks for .m files.
        for i = 1:length(FileNamesNoDir),
            if strncmp(FileNamesNoDir{i}(end-1:end),'.m',2)
                ListOfTools(length(ListOfTools)+1) = {FileNamesNoDir{i}(1:end-2)};
                ToolHelp{length(ListOfTools)-1} = [ToolHelpInfo, '-----------' 10 help(char(FileNamesNoDir{i}(1:end-2)))];
            end
        end
        if length(ListOfTools) > 1
            ListOfTools(1) = {'Data tools'};
        else ToolHelp = 'No data tools were loaded upon starting up CellProfiler. Data tools are Matlab m-files ending in ''.m'', and should be located in a folder called DataTools within the folder containing CellProfiler.m';
        end
    end
end
handles.Current.DataToolsFilenames = ListOfTools;
handles.Current.DataToolHelp = ToolHelp;

ListOfTools = {};
GSListOfTools = {};
%%% Finds all available tools, which are .m files residing in the
%%% Help folder.
Pathname = fullfile(handles.Current.CellProfilerPathname,'Help');
ListOfTools{1} = 'Help: none loaded';
GSListOfTools{1} = 'Help: none loaded';
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
            if strncmp(FileNamesNoDir{i}(end-1:end),'.m',2)
                if strncmp(FileNamesNoDir{i}(1:2),'GS',2)
                    GSListOfTools(length(GSListOfTools)+1) = {FileNamesNoDir{i}(3:end-2)};
                    GSToolHelp{length(GSListOfTools)-1} = help(char(FileNamesNoDir{i}(1:end-2))); %#ok Ignore MLint
                else
                    ListOfTools(length(ListOfTools)+1) = {FileNamesNoDir{i}(1:end-2)};
                    ToolHelp{length(ListOfTools)-1} = help(char(FileNamesNoDir{i}(1:end-2)));
                end
            end
        end
        if length(GSListOfTools) > 1
            GSListOfTools(1) = {'GettingStarted'};
        else ToolHelp = 'No help files were loaded upon starting up CellProfiler. Help files are Matlab m-files ending in ''.m'', and should be located in a folder called Help within the folder containing CellProfiler.m';
        end
        if length(ListOfTools) > 1
            ListOfTools(1) = {'Help'};
        else ToolHelp = 'No help files were loaded upon starting up CellProfiler. Help files are Matlab m-files ending in ''.m'', and should be located in a folder called Help within the folder containing CellProfiler.m';
        end
    end
end
handles.Current.GSFilenames = GSListOfTools;
handles.Current.GS = GSToolHelp;
handles.Current.HelpFilenames = ListOfTools;
handles.Current.Help = ToolHelp;

clear GSListOfTools GSToolHelp

% Update handles structure
%%% Do not remove the END line below.  It is used by CompileWizard.m.
%%% Compiler: END HELP
guidata(hObject, handles);



FileMenu=uimenu(hObject,'Label','File');
DataToolsMenu=uimenu(hObject,'Label','Data Tools');
WindowsMenu=uimenu(hObject,'Label','Windows','Tag','WindowsMenu','Callback',[...
    'WindowsMenu = findobj(''Tag'',''WindowsMenu'');',...
    'children = allchild(WindowsMenu);',...
    'for i=1:length(children),',...
    'if isempty(get(children(i), ''Tag'')),',...
    'delete(children(i));',...
    'end;',...
    'end;',...
    'OpenWindows = findobj(''NumberTitle'',''on'',''-and'',''-property'',''UserData'');',...
    'OpenWindows = sort(OpenWindows);',...
    'for k = 1:length(OpenWindows),',...
    'if ishandle(OpenWindows(k)),',...
    'userData = get(OpenWindows(k),''UserData'');',...
    'name = get(OpenWindows(k),''Name'');',...
    'if (~isempty(userData) && isfield(userData,''Application'') && ',...
    'isstr(userData.Application) && strcmp(userData.Application,''CellProfiler'') && ',...
    '~isempty(strfind(name,''Display, cycle #''))),',...
    'try,',...
    'h = uimenu(WindowsMenu,''Label'', [''Figure '' num2str(OpenWindows(k)) '': '' name],''UserData'',OpenWindows(k),''Callback'',''figure(get(gcbo,''''UserData''''))'');',...
    'if k==1,',...
    'set(h,''Separator'',''on'');',...
    'end;',...
    'end;',...
    'end;',...
    'end;',...
    'end;',...
    'clear WindowsMenu OpenWindows children i k h userData name;']);
HelpMenu=uimenu(hObject,'Label','Help');

uimenu(FileMenu,'Label','Open Image','Callback','CellProfiler(''OpenImage_Callback'',gcbo,[],guidata(gcbo));');
uimenu(FileMenu,'Label','Clear Pipeline','Callback','CellProfiler(''ClearPipeline_Callback'',gcbo,[],guidata(gcbo));');
uimenu(FileMenu,'Label','Save Pipeline','Callback','CellProfiler(''SavePipeline_Callback'',gcbo,[],guidata(gcbo));');
uimenu(FileMenu,'Label','Load Pipeline','Callback','CellProfiler(''LoadPipeline_Callback'',gcbo,[],guidata(gcbo));');
uimenu(FileMenu,'Label','Set Preferences','Callback','CellProfiler(''SetPreferences_Callback'',gcbo,[],guidata(gcbo));');
uimenu(FileMenu,'Label','Load Preferences','Callback','CellProfiler(''LoadPreferences_Callback'',gcbo,[],guidata(gcbo));');
if ~isdeployed
    uimenu(FileMenu,'Label','Save current CellProfiler code','Callback','CellProfiler(''ZipFiles_Callback'',gcbo,[],guidata(gcbo));');
    uimenu(FileMenu,'Label','Tech Diagnosis','Callback','CellProfiler(''TechnicalDiagnosis_Callback'',gcbo,[],guidata(gcbo));');
end
uimenu(FileMenu,'Label','Exit','Callback',ClosingFunction);

ListOfDataTools=handles.Current.DataToolsFilenames;
for j=2:length(ListOfDataTools)
    uimenu(DataToolsMenu,'Label',char(ListOfDataTools(j)),'Callback',[char(ListOfDataTools(j))  '(guidata(gcbo));clear ans']);
end

uimenu(WindowsMenu,'Label','Close All','Tag','Close All','Callback','CellProfiler(''CloseWindows_Callback'',gcbo,[],guidata(gcbo));');

uimenu(HelpMenu,'Label','Getting Started','Callback','CellProfiler(''HelpFiles_Callback'',gcbo,''GS'',guidata(gcbo))');
uimenu(HelpMenu,'Label','General Help','Callback','CellProfiler(''HelpFiles_Callback'',gcbo,''Help'',guidata(gcbo))');
uimenu(HelpMenu,'Label','Modules Help','Callback','CellProfiler(''ModulesHelp_Callback'',gcbo,[],guidata(gcbo))');
uimenu(HelpMenu,'Label','Image Tools Help','Callback','CellProfiler(''ImageToolsHelp_Callback'',gcbo,[],guidata(gcbo))');
uimenu(HelpMenu,'Label','Data Tools Help','Callback','CellProfiler(''DataToolsHelp_Callback'',gcbo,[],guidata(gcbo))');
%uimenu(HelpMenu,'Label','Report Bugs','Callback','CellProfiler(''ReportBugs_Callback'',gcbo,[],guidata(gcbo));');
% if ~isdeployed
%     uimenu(HelpMenu,'Label','Download New Modules','Callback','CellProfiler(''DownloadModules_Callback'',gcbo,[],guidata(gcbo));');
% end

% Set default output filename
set(handles.OutputFileNameEditBox,'string','DefaultOUT.mat')

% --- Outputs from this function are returned to the command line.
function varargout = CellProfiler_OutputFcn(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
% Get default command line output from handles structure
varargout{1} = handles.output;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% CLEAR PIPELINE BUTTON %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function ClearPipeline_Callback(hObject, eventdata, handles) %#ok Ignore MLint

if isempty(eventdata)
    Answer = CPquestdlg('Are you sure you want to clear the existing pipeline?','Confirm','Yes','No','Yes');
else
    Answer = 'Yes';
end

if strcmp(Answer,'Yes')
    handles.Settings.ModuleNames = {};
    handles.Settings.VariableValues = {};
    handles.Settings.VariableInfoTypes = {};
    handles.Settings.VariableRevisionNumbers = [];
    handles.Settings.ModuleRevisionNumbers = [];
    delete(get(handles.variablepanel,'children'));
    set(handles.slider1,'visible','off');
    handles.VariableBox = {};
    handles.VariableDescription = {};
    set(handles.ModulePipelineListBox,'Value',1);
    handles.Settings.NumbersOfVariables = [];
    handles.Current.NumberOfModules = 0;
    contents = {'No Modules Loaded'};
    set(handles.ModulePipelineListBox,'String',contents);
    guidata(hObject,handles);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% LOAD PIPELINE BUTTON %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [SettingsPathname, SettingsFileName, errFlg, handles] = ...
    LoadPipeline_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.

if isempty(eventdata)
    errFlg = 0;
    [SettingsFileName, SettingsPathname] = ...
	CPuigetfile('*.mat', 'Choose a pipeline file', ...
		    handles.Current.DefaultOutputDirectory); 
    pause(.1);
    figure(handles.figure1);
else
    SettingsFileName = eventdata.SettingsFileName;
    SettingsPathname = eventdata.SettingsPathname;
end

%%% If the user presses "Cancel", the SettingsFileName.m will = 0 and
%%% nothing will happen.
if SettingsFileName == 0
    return
end

drawnow
%%% Loads the Settings file.
try
    LoadedSettings = load(fullfile(SettingsPathname,SettingsFileName));
catch
    error(['CellProfiler was unable to load ',fullfile(SettingsPathname,SettingsFileName),'. The file may be corrupt.']);
end
%%% Error Checking for valid settings file.
if ~(isfield(LoadedSettings, 'Settings') || isfield(LoadedSettings, 'handles'))
    CPerrordlg(['The file ' SettingsPathname SettingsFileName ' does not appear to be a valid settings or output file. Settings can be extracted from an output file created when analyzing images with CellProfiler or from a small settings file saved using the "Save Settings" button.  Either way, this file must have the extension ".mat" and contain a variable named "Settings" or "handles".']);
    errFlg = 1;
    return
end
%%% Figures out whether we loaded a Settings or Output file, and puts
%%% the correct values into Settings. Splices the subset of variables
%%% from the "settings" structure into the handles structure.
if (isfield(LoadedSettings, 'Settings')),
    Settings = LoadedSettings.Settings;
else
    try Settings = LoadedSettings.handles.Settings;
        Settings.NumbersOfVariables = LoadedSettings.handles.Settings.NumbersOfVariables;
    end
end

try
    [NumberOfModules, MaxNumberVariables] = size(Settings.VariableValues); %#ok Ignore MLint
    if (size(Settings.ModuleNames,2) ~= NumberOfModules)||(size(Settings.NumbersOfVariables,2) ~= NumberOfModules);
        CPerrordlg(['The file ' SettingsPathname SettingsFileName ' is not a valid settings or output file. Settings can be extracted from an output file created when analyzing images with CellProfiler or from a small settings file saved using the "Save Settings" button.']);
        errFlg = 1;
        return
    end
catch
    CPerrordlg(['The file ' SettingsPathname SettingsFileName ' is not a valid settings or output file. Settings can be extracted from an output file created when analyzing images with CellProfiler or from a small settings file saved using the "Save Settings" button.']);
    errFlg = 1;
    return
end

%%% Hide stuff in the background, but keep old values in case of errors.
OldValue = get(handles.ModulePipelineListBox,'Value');
OldString = get(handles.ModulePipelineListBox,'String');
set(handles.ModulePipelineListBox,'Value',1);
set(handles.ModulePipelineListBox,'String','Loading...');
set(get(handles.variablepanel,'children'),'visible','off');
set(handles.slider1,'visible','off');

%%% Check to make sure that the module files can be found and get paths
ModuleNames = Settings.ModuleNames;
Skipped = 0;
for k = 1:NumberOfModules
    if ~isdeployed
        CurrentModuleNamedotm = [char(ModuleNames{k}) '.m'];
        if exist(CurrentModuleNamedotm,'file')
            Pathnames{k-Skipped} = fileparts(which(CurrentModuleNamedotm)); %#ok Ignore MLint
        else
            %%% If the module.m file is not on the path, it won't be
            %%% found, so ask the user where the modules are.
            Choice = CPquestdlg(['The module ', CurrentModuleNamedotm, ' cannot be found. Either its name has changed or it was moved or deleted. What do you want to do? Note: You can also choose another module to replace ' CurrentModuleNamedotm ' if you select Search Module. It will be loaded with its default settings and you will also be able to see the saved settings of ' CurrentModuleNamedotm '.'],'Module not found','Skip Module','Search Module','Abort','Skip Module');
            switch Choice
                case 'Skip Module'
                    %%% Check if this was the only module in the pipeline or if
                    %%% all previous modules have been skipped too
                    if Skipped+1 == NumberOfModules
                        CPerrordlg('All modules in this pipeline were skipped. Loading will be canceled.','Loading Pipeline Error')
                        Abort = 1;
                    else
                        %%% Remove module info from the settings
                        View = CPquestdlg(['The pipeline will be loaded without ' CurrentModuleNamedotm ', but keep in mind that it might not work properly. Would you like to see the saved settings ' CurrentModuleNamedotm ' had?'], 'Module Skipped', 'Yes', 'No', 'Yes');
                        if strcmp(View,'Yes')
                            FailedModule(handles,Settings.VariableValues(k-Skipped,:),'Sorry, variable descriptions could not be retrieved from this file',CurrentModuleNamedotm,k-Skipped);
                        end
                        %%% Notice that if the skipped module is the one that
                        %%% had the most variables, then the VariableValues
                        %%% will have some empty columns at the end. I guess it
                        %%% doesn't matter, but it could be fixed if necessary.
                        Settings.VariableValues(k-Skipped,:) = [];
                        Settings.VariableInfoTypes(k-Skipped,:) = [];
                        Settings.ModuleNames(k-Skipped) = [];
                        Settings.NumbersOfVariables(k-Skipped) = [];
                        Settings.VariableRevisionNumbers(k-Skipped) = [];
                        Settings.ModuleRevisionNumbers(k-Skipped) = [];
                        Skipped = Skipped+1;
                        Abort = 0;
                    end
                case 'Search Module'
                    if ~isdeployed
		      filter = '*.m';
		    else
		      filter = '*.txt';
		    end
		    [Filename Pathname] = CPuigetfile(filter, ['Find ' CurrentModuleNamedotm ' or Choose Another Module'], handles.Preferences.DefaultModuleDirectory);
                    pause(.1);
                    figure(handles.figure1);
                    if Filename == 0
                        Abort = 1;
                    else
                        Pathnames{k-Skipped} = Pathname;
                        if ~isdeployed
                            Settings.ModuleNames{k-Skipped} = Filename(1:end-2);
                        else
                            Settings.ModuleNames{k-Skipped} = Filename(1:end-4);
                        end
                        Abort = 0;
                    end
                otherwise
                    Abort = 1;
            end
            if Abort
                %%% Restore whatever the user had before attempting to load
                set(handles.ModulePipelineListBox,'String',OldString);
                set(handles.ModulePipelineListBox,'Value',OldValue);
                ModulePipelineListBox_Callback(hObject,[],handles);
                errFlg = 1;
                return
            end
        end
    else
        Pathnames{k-Skipped} = handles.Preferences.DefaultModuleDirectory;
    end
end

%%% Save old settings in case of error
OldValue = get(handles.ModulePipelineListBox,'Value');
OldString = get(handles.ModulePipelineListBox,'String');
OldSettings = handles.Settings;
try
    OldVariableBox = handles.VariableBox;
    OldVariableDescription = handles.VariableDescription;
catch
    OldVariableBox = {};
    OldVariableDescription = {};
end

%%% Update handles structure
handles.Settings.ModuleNames = Settings.ModuleNames;
handles.Settings.VariableValues = {};
handles.Settings.VariableInfoTypes = {};
handles.Settings.VariableRevisionNumbers = [];
handles.Settings.ModuleRevisionNumbers = [];
handles.Settings.NumbersOfVariables = [];
handles.VariableBox = {};
handles.VariableDescription = {};

%%% For each module, extract its settings and check if they seem alright
revisionConfirm = 0;
Skipped = 0;
for ModuleNum=1:length(handles.Settings.ModuleNames)
    CurrentModuleName = handles.Settings.ModuleNames{ModuleNum-Skipped};
    %%% Replace names of modules whose name changed
    if strcmp('CreateBatchScripts',CurrentModuleName) || strcmp('CreateClusterFiles',CurrentModuleName)
        handles.Settings.ModuleNames(ModuleNum-Skipped) = {'CreateBatchFiles'};
    elseif strcmp('WriteSQLFiles',CurrentModuleName)
        handles.Settings.ModuleNames(ModuleNum-Skipped) = {'ExportToDatabase'};
    end
    %%% Load the module's settings

    try
        %%% First load the module with its default settings
        [defVariableValues defVariableInfoTypes defDescriptions handles.Settings.NumbersOfVariables(ModuleNum-Skipped) DefVarRevNum ModuleRevNum] = LoadSettings_Helper(Pathnames{ModuleNum-Skipped}, CurrentModuleName);
        %%% If no VariableRevisionNumber was extracted, default it to 0
        if isfield(Settings,'VariableRevisionNumbers')
            SavedVarRevNum = Settings.VariableRevisionNumbers(ModuleNum-Skipped);
        else
            SavedVarRevNum = 0;
        end

        %%% Using the VariableRevisionNumber and the number of variables,
        %%% check if the loaded module and the module the user is trying to
        %%% load is the same
        if SavedVarRevNum == DefVarRevNum && handles.Settings.NumbersOfVariables(ModuleNum-Skipped) == Settings.NumbersOfVariables(ModuleNum-Skipped)
            %%% If so, replace the default settings with the saved ones
            handles.Settings.VariableValues(ModuleNum-Skipped,1:Settings.NumbersOfVariables(ModuleNum-Skipped)) = Settings.VariableValues(ModuleNum-Skipped,1:Settings.NumbersOfVariables(ModuleNum-Skipped));
            %%% save module revision number
            handles.Settings.ModuleRevisionNumbers(ModuleNum-Skipped) = ModuleRevNum;
        else
            %%% If not, show the saved settings. Note: This will always
            %%% appear if user selects another module when they search for
            %%% the missing module, but the user is appropriately warned
            savedVariableValues = Settings.VariableValues(ModuleNum-Skipped,1:Settings.NumbersOfVariables(ModuleNum-Skipped));
            FailedModule(handles, savedVariableValues, defDescriptions, char(handles.Settings.ModuleNames(ModuleNum-Skipped)),ModuleNum-Skipped);
            %%% Go over each variable
            for k = 1:handles.Settings.NumbersOfVariables(ModuleNum-Skipped)
                if strcmp(defVariableValues(k),'Pipeline Value')
                    %%% Create FixList, which will later be used to replace
                    %%% pipeline-dependent variable values in the loaded modules
                    handles.Settings.VariableValues(ModuleNum-Skipped,k) = {''};
                    if exist('FixList','var')
                        FixList(end+1,1) = ModuleNum-Skipped;
                        FixList(end,2) = k;
                    else
                        FixList(1,1) = ModuleNum-Skipped;
                        FixList(1,2) = k;
                    end
                else
                    %%% If no need to change, save the default loaded variables
                    handles.Settings.VariableValues(ModuleNum-Skipped,k) = defVariableValues(k);
                end
            end
            %%% Save the infotypes and VariableRevisionNumber
             handles.Settings.VariableInfoTypes(ModuleNum-Skipped,1:numel(defVariableInfoTypes)) = defVariableInfoTypes;
             handles.Settings.VariableRevisionNumbers(ModuleNum-Skipped) = DefVarRevNum;
             handles.Settings.ModuleNames{ModuleNum-Skipped} = CurrentModuleName;
             handles.Settings.ModuleRevisionNumbers(ModuleNum-Skipped) = ModuleRevNum;
            revisionConfirm = 1;
        end
        clear defVariableInfoTypes;
    catch
        %%% It is very unlikely to get here, because this means the
        %%% pathname was incorrect, but we had checked this before
        Choice = CPquestdlg(['The ' CurrentModuleName ' module could not be found in the directory specified or an error occured while extracting its variable settings. This error is not common; the module might be corrupt or, if running on the non-developers version of CellProfiler, the module might not be located in the default Module folder. The module will be skipped and the rest of the pipeline will be loaded. Would you like to see the module''s saved settings? (' lasterr ')'],'Error','Yes','No','Abort','Yes');
        switch Choice
            case 'Yes'
                FailedModule(handles,Settings.VariableValues(ModuleNum-Skipped,:),'Sorry, variable descriptions could not be retrieved from this file',CurrentModuleName,ModuleNum-Skipped);
                Abort = 0;
            case 'No'
                Abort = 0;
            otherwise
                Abort = 1;
        end
        if Skipped+1 == length(handles.Settings.ModuleNames)
            CPerrordlg('All modules in this pipeline were skipped. Loading will be canceled.','Loading Pipeline Error')
            Abort = 1;
        else
            %%% Remove module info from the settings and handles
            handles.Settings.ModuleNames(ModuleNum-Skipped) = [];
            Pathnames(ModuleNum-Skipped) = [];
            Settings.VariableValues(ModuleNum-Skipped,:) = [];
            Settings.VariableInfoTypes(ModuleNum-Skipped,:) = [];
            Settings.ModuleNames(ModuleNum-Skipped) = [];
            Settings.NumbersOfVariables(ModuleNum-Skipped) = [];
            try Settings.VariableRevisionNumbers(ModuleNum-Skipped) = []; end
            try Settings.ModuleRevisionNumbers(ModuleNum-Skipped) = []; end
            Skipped = Skipped+1;
        end
        if Abort
            %%% Reset initial handles settings
            handles.Settings = OldSettings;
            handles.VariableBox = OldVariableBox;
            handles.VariableDescription = OldVariableDescription;
            set(handles.ModulePipelineListBox,'String',OldString);
            set(handles.ModulePipelineListBox,'Value',OldValue);
            guidata(hObject,handles);
            ModulePipelineListBox_Callback(hObject,[],handles);
            errFlg = 1;
            return
        end
    end
end

delete(get(handles.variablepanel,'children'));
try
    handles.Settings.PixelSize = Settings.PixelSize;
    handles.Preferences.PixelSize = Settings.PixelSize;
    set(handles.PixelSizeEditBox,'String',handles.Preferences.PixelSize)
end
handles.Current.NumberOfModules = 0;
contents = handles.Settings.ModuleNames;
guidata(hObject,handles);

WaitBarHandle = CPwaitbar(0,'Loading Pipeline...');
for i=1:length(handles.Settings.ModuleNames)
    if isdeployed
        PutModuleInListBox([contents{i} '.txt'], Pathnames{i}, handles, 1);
    else
        PutModuleInListBox([contents{i} '.m'], Pathnames{i}, handles, 1);
    end
    handles=guidata(handles.figure1);
    handles.Current.NumberOfModules = i;
    CPwaitbar(i/length(handles.Settings.ModuleNames),WaitBarHandle,'Loading Pipeline...');
end

if exist('FixList','var')
    for k = 1:size(FixList,1)
        PipeList = get(handles.VariableBox{FixList(k,1)}(FixList(k,2)),'string');
        FirstValue = PipeList(1);
        handles.Settings.VariableValues(FixList(k,1),FixList(k,2)) = FirstValue;
    end
end

guidata(hObject,handles);
set(handles.ModulePipelineListBox,'String',contents);
set(handles.ModulePipelineListBox,'Value',1);
ModulePipelineListBox_Callback(hObject, eventdata, handles);
close(WaitBarHandle);

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
        SavePipeline_Callback(hObject, eventdata, handles);
        handles.Settings = tempSettings;
    end
end

%%% SUBFUNCTION %%%
function [VariableValues VariableInfoTypes VariableDescriptions NumbersOfVariables VarRevNum ModuleRevNum] = LoadSettings_Helper(Pathname, ModuleName)

VariableValues = {[]};
VariableInfoTypes = {[]};
VariableDescriptions = {[]};
VarRevNum = 0;
ModuleRevNum = 0;
NumbersOfVariables = 0;
if isdeployed
    ModuleNamedotm = [ModuleName '.txt'];
else
    ModuleNamedotm = [ModuleName '.m'];
end
fid=fopen(fullfile(Pathname,ModuleNamedotm));
while 1
    output = fgetl(fid);
    if ~ischar(output)
        break
    end
    if strncmp(output,'%defaultVAR',11)
        displayval = output(17:end);
        istr = output(12:13);
        i = str2double(istr);
        VariableValues(i) = {displayval};
    elseif strncmp(output,'%choiceVAR',10)
        if ~iscellstr(VariableValues(i))
            displayval = output(16:end);
            istr = output(11:12);
            i = str2double(istr);
            VariableValues(i) = {displayval};
        end
    elseif strncmp(output,'%textVAR',8)
        displayval = output(13:end);
        istr = output(9:10);
        i = str2double(istr);
        VariableDescriptions(i) = {displayval};
        VariableValues(i) = {[]};
        NumbersOfVariables = i;
    elseif strncmp(output,'%pathnametextVAR',16)
        displayval = output(21:end);
        istr = output(17:18);
        i = str2double(istr);
        VariableDescriptions(i) = {displayval};
        VariableValues(i) = {[]};
        NumbersOfVariables = i;
    elseif strncmp(output,'%filenametextVAR',16)
        displayval = output(21:end);
        istr = output(17:18);
        i = str2double(istr);
        VariableDescriptions(i) = {displayval};
        VariableValues(i) = {[]};
        NumbersOfVariables = i;
    elseif strncmp(output,'%infotypeVAR',12)
        displayval = output(18:end);
        istr = output(13:14);
        i = str2double(istr);
        VariableInfoTypes(i) = {displayval};
        if ~strcmp(output((length(output)-4):end),'indep') && isempty(VariableValues{i})
            VariableValues(i) = {'Pipeline Value'};
        end
    elseif strncmp(output,'%%%VariableRevisionNumber',25)
        try
            VarRevNum = str2double(output(29:30));
        catch
            VarRevNum = str2double(output(29:29));
        end
    elseif strncmp(output,'% $Revision:', 12)
        try
            ModuleRevNum = str2double(output(14:17));
        catch
            ModuleRevNum = str2double(output(14:18));
        end
    end
end
fclose(fid);

%%% SUBFUNCTION %%%
function FailedModule(handles, savedVariables, defaultDescriptions, ModuleName, ModuleNum)
helpText = ['The settings contained within the selected file are based on an old version of the ',ModuleName,...
    ' module. As a result, it is possible that your old settings are no longer reasonable. '...
    'Displayed below are the settings retrieved from your file. You can use the saved settings '...
    'to attempt to set up the module again. Sorry for the inconvenience.'];

%%% Creates the dialog box and its text, buttons, and edit boxes.
MainWinPos = get(handles.figure1,'Position');
ScreenSize = get(0,'ScreenSize');
FigWidth = MainWinPos(3)*4/5;
FigHeight = MainWinPos(4);
LeftPos = .5*(ScreenSize(3)-FigWidth);
BottomPos = .5*(ScreenSize(4)-FigHeight);
FigPosition = [LeftPos BottomPos FigWidth FigHeight];
Color = [0.7 .7 .9];

%%% Label we attach to figures (as UserData) so we know they are ours
userData.Application = 'CellProfiler';
LoadSavedWindowHandle = figure(...
    'Units','pixels',...
    'Color',Color,...
    'DockControls','off',...
    'MenuBar','none',...
    'Name',['Saved Variables for Module ',num2str(ModuleNum)],...
    'NumberTitle','off',...
    'Position',FigPosition,...
    'Resize','off',...
    'HandleVisibility','on',...
    'Tag','savedwindow',...
    'UserData',userData);

informtext = uicontrol(...
    'Parent',LoadSavedWindowHandle,...
    'BackgroundColor',Color',...
    'Units','normalized',...
    'Position',[0.05 0.70 0.9 0.25],...
    'String',helpText,...
    'Style','text',...
    'FontName','helvetica',...
    'HorizontalAlignment','left',...
    'FontSize',handles.Preferences.FontSize,...
    'Tag','informtext'); %#ok Ignore MLint

savedbox = uicontrol(...
    'Parent',LoadSavedWindowHandle,...
    'BackgroundColor', Color,...
    'Units','normalized',...
    'Position',[0.7 0.1 0.25 0.55],...
    'String',savedVariables,...
    'Style','listbox',...
    'Value',1,...
    'FontName','helvetica',...
    'FontSize',handles.Preferences.FontSize,...
    'Tag','savedbox'); %#ok Ignore MLint

descriptionbox = uicontrol(...
    'Parent',LoadSavedWindowHandle,...
    'BackgroundColor',Color,...
    'Units','normalized',...
    'Position',[0.05 0.1 0.6 0.55],...
    'String',defaultDescriptions,...
    'Style','listbox',...
    'Value',1,...
    'FontName','helvetica',...
    'FontSize',handles.Preferences.FontSize,...
    'Tag','descriptionbox'); %#ok Ignore MLint

savedtext = uicontrol(...
    'Parent',LoadSavedWindowHandle,...
    'BackgroundColor',Color,...
    'Units','normalized',...
    'Position',[0.665 0.65 0.2 0.05],...
    'String','Saved Variables:',...
    'Style','text',...
    'FontName','helvetica',...
    'FontSize',handles.Preferences.FontSize,...
    'Tag','descriptiontext'); %#ok Ignore MLint

descriptiontext = uicontrol(...
    'Parent',LoadSavedWindowHandle,...
    'BackgroundColor',Color,...
    'Units','normalized',...
    'Position',[0.015 0.65 0.25 0.05],...
    'String','Variable Descriptions:',...
    'Style','text',...
    'FontName','helvetica',...
    'FontSize',handles.Preferences.FontSize,...
    'Tag','descriptiontext'); %#ok Ignore MLint

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE PIPELINE BUTTON %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function SavePipeline_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.

if handles.Current.NumberOfModules == 0
    warndlg('Pipeline saving was canceled because there are no modules in the current pipeline.','Warning')
    return
end

if length(handles.Settings.NumbersOfVariables) ~= length(handles.Settings.ModuleNames)
    CPmsgbox('STOP! Somehow the NumbersOfVariable was not set correctly! Please record EVERYTHING you have done in the past few minutes and send to mrl@wi.mit.edu! Be sure to include what modules are in your pipeline and what you tried to do to cause this error (Adding, Subtracting, Moving modules, how many?). PLEASE NOTE YOUR PIPELINE WILL STILL BE SAVED AND THIS ERROR WILL BE CORRECTED. REPORTING HOW THIS ERROR OCCURRED WILL HELP US DETERMINE HOW TO FIX IT!');
    handles.Settings.NumbersOfVariables((length(handles.Settings.ModuleNames)+1):end) = [];
end

%%% The "Settings" variable is saved to the file name the user chooses.
if exist(handles.Current.DefaultOutputDirectory, 'dir')
    [FileName,Pathname] = uiputfile(fullfile(handles.Current.DefaultOutputDirectory,'*.mat'), 'Save Pipeline As...');
else
    [FileName,Pathname] = uiputfile('*.mat', 'Save Pipeline As...');
end
%%% Allows canceling.
if FileName ~= 0
    [Temp,FileNom,FileExt] = fileparts(FileName); %#ok Ignore MLint
    %%% search for 'pipe' in the filename
    LocatePipe = strfind(FileName,'pipe');
    if isempty(LocatePipe)
        LocatePipe = strfind(FileName,'Pipe');
    end
    if isempty(LocatePipe)
        LocatePipe = strfind(FileName,'PIPE');
    end
    if isempty(LocatePipe)
        AutoName = CPquestdlg(['Do you want to rather name the file as ', FileNom, 'PIPE', FileExt, ' in order to prevent confusion with output files?'],'Rename file?','Yes');
        if strcmp(AutoName,'Yes')
            FileName = [FileNom,'PIPE',FileExt];
            CPhelpdlg('The pipeline file has been saved.');
        elseif strcmp(AutoName,'No')
            CPhelpdlg('The pipeline file has been saved.');
        elseif strcmp(AutoName,'Cancel')
            return
        end
    end
    %%% Allows user to save pipeline setting as a readable text file (.txt)
    SaveText = CPquestdlg('Do you want to save the pipeline as a text file also?','Save as text?','No');
    if strcmp(SaveText,'Cancel')
        return
    end
    %%% Checks if a field is present, and if it is, the value is stored in the
    %%% structure 'Settings' with the same name.
    if isfield(handles.Settings,'VariableValues'),
        Settings.VariableValues = handles.Settings.VariableValues;
    end
    if isfield(handles.Settings,'VariableInfoTypes'),
        Settings.VariableInfoTypes = handles.Settings.VariableInfoTypes;
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
    if isfield(handles.Settings,'ModuleRevisionNumbers'),
        Settings.ModuleRevisionNumbers = handles.Settings.ModuleRevisionNumbers;
    end
    save(fullfile(Pathname,FileName),'Settings')
    %%% Writes settings into a readable text file.
    if strcmp(SaveText,'Yes')
        CPtextpipe(handles,0,0,0);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% FIGURE DISPLAY BUTTONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

%%%%%%%%%%%%%%%%%%%%%%%%%
%%% ADD MODULE BUTTON %%%
%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes on button press in AddModule.
function AddModule_Callback(hObject,eventdata,handles) %#ok We want to ignore MLint error checking for this line.

obj=findobj('Tag','AddModuleWindow');
if(~isempty(obj))  %Window already exists
    figure(obj);
    return
end

if handles.Current.NumberOfModules == 99
    CPerrordlg('CellProfiler in its current state can only handle 99 modules. You have just attempted to load the 100th module. It should be fairly straightforward to modify the code in CellProfiler.m to expand its capabilities.');
    return
end

%%% 1. Opens a user interface to retrieve the .m file you want to use.
%%% Change to the default module directory. This line is within a
%%% try-end pair because the user may have changed the folder names
%%% leading up to this directory sometime after saving the
%%% Preferences.

if exist(handles.Preferences.DefaultModuleDirectory, 'dir')
    AddModuleWindow_OpeningFcn(hObject, eventdata, AddModuleWindow_LayoutFcn(handles.figure1));
    obj1=findobj('Tag','ModulesListBox');
    obj2=findobj('Tag','ModuleCategoryListBox');
    if ~isempty(obj1)
        set(obj1,'value',1);
    end
    if ~isempty(obj2)
        set(obj2,'value',1);
    end
else
    if isdeployed
      filter = '*.txt';
    else
      filter = '*.m';
    end
    [ModuleNamedotm,Pathname] = uigetfile(filter, 'Choose an image analysis module');
    pause(.1);
    figure(handles.figure1);
    PutModuleInListBox(ModuleNamedotm,Pathname,handles,0);
end

function PutModuleInListBox(ModuleNamedotm, Pathname, handles, RunInBG)
if ModuleNamedotm ~= 0,
    %%% The folder containing the desired .m file is added to Matlab's search path.
    if ~isdeployed
        addpath(Pathname);
        differentPaths = which(ModuleNamedotm, '-all');
        if isempty(differentPaths)
            %%% If the module's .m file is not found on the search path, the result
            %%% of exist is zero, and the user is warned.
            CPerrordlg(['Something is wrong; The .m file ', ModuleNamedotm, ' was not initially found by Matlab, so the folder containing it was added to the Matlab search path. But, Matlab still cannot find the .m file for the analysis module you selected. The module will not be added to the image analysis pipeline.'],'Error');
            return
        elseif length(differentPaths) > 1
            CPwarndlg(['More than one file with this same module name exists in the Matlab search path.  The pathname from ' char(differentPaths{1}) ' will likely be used, but this is unpredictable.  Modules should have unique names that are not the same as already existing Matlab functions to avoid confusion.'],ModuleNamedotm,'modal');
        end
    end
    if(exist(ModuleNamedotm(1:end-2),'builtin') ~= 0)
        warningString = 'Your module has the same name as a builtin Matlab function.  Perhaps you should consider renaming your module.';
        warndlg(warningString);
    end

    %%% 3. The last two characters (=.m) are removed from the
    %%% ModuleName.m and called ModuleName. If we are using the compiled
    %%% version (isdeployed), we must remove 4 characters (=.txt) instead.
    if isdeployed
        ModuleName = ModuleNamedotm(1:end-4);
	if ~ strcmp(ModuleNamedotm(end-3:end), '.txt'),			       
	        CPwarndlg('Only compiled modules (.txt files) can be added to the pipeline in the compiled version of CellProfiler. If you load .m files, your pipeline will not function correctly.');
	end
    else
        ModuleName = ModuleNamedotm(1:end-2);
    end
    %%% The name of the module is shown in a text box in the GUI (the text
    %%% box is called ModuleName1.) and in a text box in the GUI which
    %%% displays the current module (whose settings are shown).

    %%% 4. The text description for each variable for the chosen module is
    %%% extracted from the module's .m file and displayed.
    if handles.Current.NumberOfModules == 0
        ModuleNums = 1;
    elseif RunInBG
        ModuleNums = handles.Current.NumberOfModules+1;
    else
        ModuleNums = get(handles.ModulePipelineListBox,'value')+1;
    end
    ModuleNumber = TwoDigitString(ModuleNums);

    for ModuleCurrent = handles.Current.NumberOfModules:-1:ModuleNums;
        %%% 1. Switches ModuleNames
        handles.Settings.ModuleNames{ModuleCurrent+1} = handles.Settings.ModuleNames{ModuleCurrent};
        contents = get(handles.ModulePipelineListBox,'String');
        contents{ModuleCurrent+1} = handles.Settings.ModuleNames{ModuleCurrent};
        set(handles.ModulePipelineListBox,'String',contents);
        %%% 2. Copy then clear the variable values in the handles
        %%% structure.
        handles.Settings.VariableValues(ModuleCurrent+1,:) = handles.Settings.VariableValues(ModuleCurrent,:);
        %%% 3. Copy then clear the num of variables in the handles
        %%% structure.
        handles.Settings.NumbersOfVariables(ModuleCurrent+1) = handles.Settings.NumbersOfVariables(ModuleCurrent);
        %%% 4. Copy then clear the variable revision numbers in the handles
        %%% structure.
        handles.Settings.VariableRevisionNumbers(ModuleCurrent+1) = handles.Settings.VariableRevisionNumbers(ModuleCurrent);
        %%% 5. Copy then clear the module revision numbers in the handles
        %%% structure.
        handles.Settings.ModuleRevisionNumbers(ModuleCurrent+1) = handles.Settings.ModuleRevisionNumbers(ModuleCurrent);
        %%% 6. Copy then clear the variable infotypes in the handles
        %%% structure.
        if size(handles.Settings.VariableInfoTypes,1) >= ModuleCurrent
            handles.Settings.VariableInfoTypes(ModuleCurrent+1,:) = handles.Settings.VariableInfoTypes(ModuleCurrent,:);
        end
        contents = get(handles.ModulePipelineListBox,'String');
        contents{ModuleCurrent+1} = handles.Settings.ModuleNames{ModuleCurrent};
        set(handles.ModulePipelineListBox,'String',contents);
    end

    if ModuleNums <= handles.Current.NumberOfModules
        handles.VariableDescription = [handles.VariableDescription(1:ModuleNums-1) {[]} handles.VariableDescription(ModuleNums:end)];
        handles.VariableBox = [handles.VariableBox(1:ModuleNums-1) {[]} handles.VariableBox(ModuleNums:end)];
        if isfield(handles,'BrowseButton')
            if length(handles.BrowseButton) >= ModuleNums
                handles.BrowseButton = [handles.BrowseButton(1:ModuleNums-1) {[]} handles.BrowseButton(ModuleNums:end)];
            end
        end
    end

    fid=fopen(fullfile(Pathname,ModuleNamedotm));
    lastVariableCheck = 0;

    numberExtraLinesOfDescription = 0;
    varSpacing = 25;
    firstBoxLoc = 345; firstDesLoc = 343; normBoxHeight = 23; normDesHeight = 20;
    normBoxLength = 94;
    pixelSpacing = 2;

    while 1;
        output = fgetl(fid);
        if ~ischar(output)
            break
        end
        if strncmp(output,'%defaultVAR',11)
            displayval = output(17:end);
            istr = output(12:13);
            lastVariableCheck = str2double(istr);
            handles.Settings.NumbersOfVariables(str2double(ModuleNumber)) = lastVariableCheck;
            set(handles.VariableBox{ModuleNums}(lastVariableCheck),'String',displayval);
            CheckVal = handles.Settings.VariableValues(ModuleNums,lastVariableCheck);
            if isempty(CheckVal{1})
                handles.Settings.VariableValues(ModuleNums,lastVariableCheck) = {displayval};
            end
        elseif strncmp(output,'%textVAR',8)
            lastVariableCheck = str2double(output(9:10));
            if ~RunInBG
                handles.Settings.VariableValues(ModuleNums, lastVariableCheck) = {''};
            end
            handles.Settings.NumbersOfVariables(str2double(ModuleNumber)) = lastVariableCheck;
            descriptionString = output(14:end);

            handles.VariableBox{ModuleNums}(lastVariableCheck) = uicontrol(...
                'Parent',handles.variablepanel,...
                'Units','pixels',...
                'BackgroundColor',[1 1 1],...
                'Callback','CellProfiler(''VariableBox_Callback'',gcbo,[],guidata(gcbo))',...
                'FontName','helvetica',...
                'FontSize',handles.Preferences.FontSize,...
                'Position',[470 295-25*lastVariableCheck 94 23],...
                'String','n/a',...
                'Style','edit',...
                'CreateFcn', 'CellProfiler(''VariableBox_CreateFcn'',gcbo,[],guidata(gcbo))',...
                'Tag',['VariableBox' TwoDigitString(lastVariableCheck)],...
                'Behavior',get(0,'defaultuicontrolBehavior'),...
                'UserData','undefined',...
                'Visible','off');

            handles.VariableDescription{ModuleNums}(lastVariableCheck) = uicontrol(...
                'Parent',handles.variablepanel,...
                'Units','pixels',...
                'BackgroundColor',[0.7 0.7 0.9],...
                'CData',[],...
                'FontName','helvetica',...
                'FontSize',handles.Preferences.FontSize,...
                'FontWeight','bold',...
                'HorizontalAlignment','right',...
                'Position',[2 291-25*lastVariableCheck 465 23],...
                'String','',...
                'Style','text',...
                'Tag',['VariableDescription' TwoDigitString(lastVariableCheck)],...
                'UserData',[],...
                'Behavior',get(0,'defaultuicontrolBehavior'),...
                'Visible','off',...
                'CreateFcn', '');

            set(handles.VariableDescription{ModuleNums}(lastVariableCheck),'string',descriptionString);

            linesVarDes = length(textwrap(handles.VariableDescription{ModuleNums}(lastVariableCheck),{descriptionString}));
            numberExtraLinesOfDescription = numberExtraLinesOfDescription + linesVarDes - 1;
            VarDesPosition = get(handles.VariableDescription{ModuleNums}(lastVariableCheck), 'Position');
            varXPos = VarDesPosition(1);
            varYPos = firstDesLoc+pixelSpacing*numberExtraLinesOfDescription-varSpacing*(lastVariableCheck+numberExtraLinesOfDescription);
            varXSize = VarDesPosition(3);
            varYSize = normDesHeight*linesVarDes + pixelSpacing*(linesVarDes-1);
            set(handles.VariableDescription{ModuleNums}(lastVariableCheck),'Position', [varXPos varYPos varXSize varYSize]);
            varXPos = 470;
            varYPos = firstBoxLoc+pixelSpacing*numberExtraLinesOfDescription-varSpacing*(lastVariableCheck+numberExtraLinesOfDescription-(linesVarDes-1)/2.0);
            varXSize = normBoxLength;
            varYSize = normBoxHeight;
            set(handles.VariableBox{ModuleNums}(lastVariableCheck), 'Position', [varXPos varYPos varXSize varYSize]);

        elseif strncmp(output,'%filenametextVAR',16)

            lastVariableCheck = str2double(output(17:18));
            if ~RunInBG
                handles.Settings.VariableValues(ModuleNums, lastVariableCheck) = {''};
            end
            handles.Settings.NumbersOfVariables(str2double(ModuleNumber)) = lastVariableCheck;
            descriptionString = output(22:end);

            handles.VariableBox{ModuleNums}(lastVariableCheck) = uicontrol(...
                'Parent',handles.variablepanel,...
                'Units','pixels',...
                'BackgroundColor',[1 1 1],...
                'Callback','CellProfiler(''VariableBox_Callback'',gcbo,[],guidata(gcbo))',...
                'FontName','helvetica',...
                'FontSize',handles.Preferences.FontSize,...
                'Position',[305 295-25*lastVariableCheck 195 23],...
                'String','NO FILE LOADED',...
                'Style','edit',...
                'CreateFcn', 'CellProfiler(''VariableBox_CreateFcn'',gcbo,[],guidata(gcbo))',...
                'Tag',['VariableBox' TwoDigitString(lastVariableCheck)],...
                'Behavior',get(0,'defaultuicontrolBehavior'),...
                'UserData','undefined',...
                'Visible','off');

            handles.BrowseButton{ModuleNums}(lastVariableCheck) = uicontrol(...
                'Parent',handles.variablepanel,...
                'Units','pixels',...
                'BackgroundColor',[.7 .7 .9],...
                'Callback','handles = guidata(findobj(''tag'',''figure1'')); VariableBoxHandle = get(gco,''UserData''); CurrentChoice = get(VariableBoxHandle,''String''); if exist(CurrentChoice,''file''), if ~isdeployed, Pathname = fileparts(which(CurrentChoice)); end; else, Pathname = handles.Current.DefaultImageDirectory; end; [Filename Pathname] = CPuigetfile([],''Choose a file'',Pathname); pause(.1); figure(handles.figure1); if Pathname == 0, else, set(VariableBoxHandle,''String'',Filename); ModuleHighlighted = get(handles.ModulePipelineListBox,''Value''); ModuleNumber = ModuleHighlighted(1); VariableName = get(VariableBoxHandle,''Tag''); VariableNumberStr = VariableName(12:13); VarNum = str2num(VariableNumberStr); handles.Settings.VariableValues(ModuleNumber,VarNum) = {Filename}; guidata(handles.figure1,handles); end; clear handles VariableBoxHandle CurrentChoice Pathname Filename ModuleHighlighted ModuleNumber VariableName VariableNumberStr VarNum;',...
                'FontName','helvetica',...
                'FontSize',handles.Preferences.FontSize,...
                'FontWeight','bold',...
                'Position',[501 296-25*lastVariableCheck 63 20],...
                'String','Browse...',...
                'Style','pushbutton',...
                'CreateFcn', 'CellProfiler(''VariableBox_CreateFcn'',gcbo,[],guidata(gcbo))',...
                'Tag',['BrowseButton' TwoDigitString(lastVariableCheck)],...
                'Behavior',get(0,'defaultuicontrolBehavior'),...
                'UserData',handles.VariableBox{ModuleNums}(lastVariableCheck),...
                'Visible','off');
            %%% In the callback function, if the user inputs an invalid filename, uigetfile goes to the default image directory instead of the current one as in other browse button callbacks because as of now (7/06), all modules use filenametextVAR only to get input files that should probably be there, so it's more convenient this way. In the future, code could be replaced to look more like the one for the pathnametextVAR browse button.

            handles.VariableDescription{ModuleNums}(lastVariableCheck) = uicontrol(...
                'Parent',handles.variablepanel,...
                'Units','pixels',...
                'BackgroundColor',[0.7 0.7 0.9],...
                'CData',[],...
                'FontName','helvetica',...
                'FontSize',handles.Preferences.FontSize,...
                'FontWeight','bold',...
                'HorizontalAlignment','right',...
                'Position',[2 295-25*lastVariableCheck 300 28],...
                'String','',...
                'Style','text',...
                'Tag',['VariableDescription' TwoDigitString(lastVariableCheck)],...
                'UserData',[],...
                'Behavior',get(0,'defaultuicontrolBehavior'),...
                'Visible','off',...
                'CreateFcn', '');

            set(handles.VariableDescription{ModuleNums}(lastVariableCheck),'string',descriptionString);

            linesVarDes = length(textwrap(handles.VariableDescription{ModuleNums}(lastVariableCheck),{descriptionString}));
            numberExtraLinesOfDescription = numberExtraLinesOfDescription + linesVarDes - 1;
            VarDesPosition = get(handles.VariableDescription{ModuleNums}(lastVariableCheck), 'Position');
            varXPos = VarDesPosition(1);
            varYPos = firstDesLoc+pixelSpacing*numberExtraLinesOfDescription-varSpacing*(lastVariableCheck+numberExtraLinesOfDescription);
            varXSize = VarDesPosition(3);
            varYSize = normDesHeight*linesVarDes + pixelSpacing*(linesVarDes-1);
            set(handles.VariableDescription{ModuleNums}(lastVariableCheck),'Position', [varXPos varYPos varXSize varYSize]);
            varYPos = firstBoxLoc+pixelSpacing*numberExtraLinesOfDescription-varSpacing*(lastVariableCheck+numberExtraLinesOfDescription-(linesVarDes-1)/2.0);
            set(handles.VariableBox{ModuleNums}(lastVariableCheck), 'Position', [305 varYPos 195 23]);
            set(handles.BrowseButton{ModuleNums}(lastVariableCheck), 'Position', [501 varYPos 63 20]);

        elseif strncmp(output,'%pathnametextVAR',16)

            lastVariableCheck = str2double(output(17:18));
            if ~RunInBG
                handles.Settings.VariableValues(ModuleNums, lastVariableCheck) = {''};
            end
            handles.Settings.NumbersOfVariables(str2double(ModuleNumber)) = lastVariableCheck;
            descriptionString = output(22:end);

            handles.VariableBox{ModuleNums}(lastVariableCheck) = uicontrol(...
                'Parent',handles.variablepanel,...
                'Units','pixels',...
                'BackgroundColor',[1 1 1],...
                'Callback','CellProfiler(''VariableBox_Callback'',gcbo,[],guidata(gcbo))',...
                'FontName','helvetica',...
                'FontSize',handles.Preferences.FontSize,...
                'Position',[305 295-25*lastVariableCheck 195 23],...
                'String','.',...
                'Style','edit',...
                'CreateFcn', 'CellProfiler(''VariableBox_CreateFcn'',gcbo,[],guidata(gcbo))',...
                'Tag',['VariableBox' TwoDigitString(lastVariableCheck)],...
                'Behavior',get(0,'defaultuicontrolBehavior'),...
                'UserData','undefined',...
                'Visible','off');

            handles.BrowseButton{ModuleNums}(lastVariableCheck) = uicontrol(...
                'Parent',handles.variablepanel,...
                'Units','pixels',...
                'BackgroundColor',[.7 .7 .9],...
                'Callback','handles = guidata(findobj(''tag'',''figure1'')); VariableBoxHandle = get(gco,''UserData''); CurrentChoice = get(VariableBoxHandle,''String''); if ~exist(CurrentChoice, ''dir''), CurrentChoice = pwd; end; Pathname = uigetdir(CurrentChoice,''Pick the directory you want.''); pause(.1); figure(handles.figure1); if Pathname == 0, else, set(VariableBoxHandle,''String'',Pathname); ModuleHighlighted = get(handles.ModulePipelineListBox,''Value''); ModuleNumber = ModuleHighlighted(1); VariableName = get(VariableBoxHandle,''Tag''); VariableNumberStr = VariableName(12:13); VarNum = str2num(VariableNumberStr); handles.Settings.VariableValues(ModuleNumber,VarNum) = {Pathname}; guidata(handles.figure1,handles); end; clear handles VariableBoxHandle CurrentChoice Pathname ModuleHighlighted ModuleNumber VariableName VariableNumberStr VarNum;',...
                'FontName','helvetica',...
                'FontSize',handles.Preferences.FontSize,...
                'FontWeight','bold',...
                'Position',[501 295-25*lastVariableCheck 63 20],...
                'String','Browse...',...
                'Style','pushbutton',...
                'CreateFcn', 'CellProfiler(''VariableBox_CreateFcn'',gcbo,[],guidata(gcbo))',...
                'Tag',['BrowseButton' TwoDigitString(lastVariableCheck)],...
                'Behavior',get(0,'defaultuicontrolBehavior'),...
                'UserData',handles.VariableBox{ModuleNums}(lastVariableCheck),...
                'Visible','off');

            handles.VariableDescription{ModuleNums}(lastVariableCheck) = uicontrol(...
                'Parent',handles.variablepanel,...
                'Units','pixels',...
                'BackgroundColor',[0.7 0.7 0.9],...
                'CData',[],...
                'FontName','helvetica',...
                'FontSize',handles.Preferences.FontSize,...
                'FontWeight','bold',...
                'HorizontalAlignment','right',...
                'Position',[2 295-25*lastVariableCheck 300 28],...
                'String','',...
                'Style','text',...
                'Tag',['VariableDescription' TwoDigitString(lastVariableCheck)],...
                'UserData',[],...
                'Behavior',get(0,'defaultuicontrolBehavior'),...
                'Visible','off',...
                'CreateFcn', '');

            set(handles.VariableDescription{ModuleNums}(lastVariableCheck),'string',descriptionString);

            linesVarDes = length(textwrap(handles.VariableDescription{ModuleNums}(lastVariableCheck),{descriptionString}));
            numberExtraLinesOfDescription = numberExtraLinesOfDescription + linesVarDes - 1;
            VarDesPosition = get(handles.VariableDescription{ModuleNums}(lastVariableCheck), 'Position');
            varXPos = VarDesPosition(1);
            varYPos = firstDesLoc+pixelSpacing*numberExtraLinesOfDescription-varSpacing*(lastVariableCheck+numberExtraLinesOfDescription);
            varXSize = VarDesPosition(3);
            varYSize = normDesHeight*linesVarDes + pixelSpacing*(linesVarDes-1);
            set(handles.VariableDescription{ModuleNums}(lastVariableCheck),'Position', [varXPos varYPos varXSize varYSize]);
            varYPos = firstBoxLoc+pixelSpacing*numberExtraLinesOfDescription-varSpacing*(lastVariableCheck+numberExtraLinesOfDescription-(linesVarDes-1)/2.0);
            set(handles.VariableBox{ModuleNums}(lastVariableCheck), 'Position', [305 varYPos 195 23]);
            set(handles.BrowseButton{ModuleNums}(lastVariableCheck), 'Position', [501 varYPos 63 20]);

        elseif strncmp(output,'%choiceVAR',10)
            if ~(exist('StrSet','var'))
                StrSet = cell(1);
                StrSet{1} = output(16:end);
            else
                StrSet{numel(StrSet)+1} = output(16:end);
            end
            if isempty(handles.Settings.VariableValues(ModuleNums,lastVariableCheck))
                handles.Settings.VariableValues(ModuleNums,lastVariableCheck) = StrSet(1);
            end
        elseif strncmp(output,'%infotypeVAR',12)
            lastVariableCheck = str2double(output(13:14));
            try
                set(handles.VariableBox{ModuleNums}(lastVariableCheck),'UserData', output(18:end));
            catch
                keyboard;
            end
            handles.Settings.VariableInfoTypes(ModuleNums,lastVariableCheck) = {output(18:end)};

            if strcmp(output((length(output)-4):end),'indep')
                UserEntry = char(handles.Settings.VariableValues(ModuleNums,lastVariableCheck));
                if ~strcmp(UserEntry,'n/a') && ~strcmp(UserEntry,'/') && ~isempty(UserEntry)
                    storevariable(ModuleNums,output(13:14),UserEntry,handles);
                end
            end
            guidata(handles.figure1,handles);
        elseif strncmp(output,'%inputtypeVAR',13)
            lastVariableCheck = str2double(output(14:15));
            set(handles.VariableBox{ModuleNums}(lastVariableCheck),'style', output(19:27));
            VersionCheck = version;
            if strcmp(output(19:27),'popupmenu') && ~ispc && str2double(VersionCheck(1:3)) >= 7.1
                set(handles.VariableBox{ModuleNums}(lastVariableCheck),'BackgroundColor',[.7 .7 .9]);
            end
            if ~(exist('StrSet','var'))
                StrSet = cell(1);
                Count = 1;
            else
                Count = size(StrSet,2)+1;
            end
            for i=1:handles.Current.NumberOfModules
                for j=1:size(handles.Settings.VariableInfoTypes,2)
                    if size(handles.Settings.VariableInfoTypes,1) >= i
                        if ~strcmp(get(handles.VariableBox{ModuleNums}(lastVariableCheck),'UserData'),'undefined') && strcmp(handles.Settings.VariableInfoTypes{i,j},[get(handles.VariableBox{ModuleNums}(lastVariableCheck),'UserData'),' indep'])
                            if  (~isempty(handles.Settings.VariableValues{i,j})) && ( Count == 1 || (ischar(handles.Settings.VariableValues{i,j}) && isempty(strmatch(handles.Settings.VariableValues{i,j}, StrSet, 'exact')))) && ~strcmp(handles.Settings.VariableValues{i,j},'/') && ~strcmp(handles.Settings.VariableValues{i,j},'Do not save') && ~strcmp(handles.Settings.VariableValues{i,j},'n/a')
                                TestStr = 0;
                                for m=1:length(StrSet)
                                    if strcmp(StrSet(m),handles.Settings.VariableValues(i,j))
                                        TestStr = TestStr + 1;
                                    end
                                end
                                if TestStr == 0
                                    StrSet(Count) = handles.Settings.VariableValues(i,j);
                                    Count = Count + 1;
                                end
                            end
                        end
                    end
                end
            end

            if strcmp(output(29:end),'custom')
                if  (~isempty(handles.Settings.VariableValues{ModuleNums,lastVariableCheck})) && ( Count == 1 || (ischar(handles.Settings.VariableValues{ModuleNums,lastVariableCheck}) && isempty(strmatch(handles.Settings.VariableValues{ModuleNums,lastVariableCheck}, StrSet, 'exact'))))
                    StrSet(Count) = handles.Settings.VariableValues(ModuleNums,lastVariableCheck);
                    Count = Count + 1;
                end
                StrSet(Count) = {'Other..'};
                Count = Count + 1;
            end

            set(handles.VariableBox{ModuleNums}(lastVariableCheck),'string',StrSet);
            guidata(handles.figure1,handles);

            if Count == 1
                set(handles.VariableBox{ModuleNums}(lastVariableCheck),'enable','off')
                guidata(handles.figure1,handles);
            end

            clear StrSet
        elseif strncmp(output,'% $Revision:', 12)
            try
                handles.Settings.ModuleRevisionNumbers(ModuleNums) = str2double(output(14:17));
            catch
                handles.Settings.ModuleRevisionNumbers(ModuleNums) = str2double(output(14:18));
            end
        elseif strncmp(output,'%%%VariableRevisionNumber',25)
            try
                handles.Settings.VariableRevisionNumbers(ModuleNums) = str2double(output(29:30));
            catch
                handles.Settings.VariableRevisionNumbers(ModuleNums) = str2double(output(29:29));
            end
            break;
        end
    end

    fclose(fid);
    if ~isfield(handles.Settings,'VariableInfoTypes')||size(handles.Settings.VariableInfoTypes,1)==size(handles.Settings.VariableValues,1)-1
        handles.Settings.VariableInfoTypes(size(handles.Settings.VariableValues,1),:)={[]};
    end
    
    for i=1:lastVariableCheck
        if strcmp(get(handles.VariableBox{ModuleNums}(i),'style'),'edit')
            if ~RunInBG
                handles.Settings.VariableValues{ModuleNums, i} = get(handles.VariableBox{ModuleNums}(i),'String');
            else
                set(handles.VariableBox{ModuleNums}(i),'String',handles.Settings.VariableValues{ModuleNums,i});
            end
        else
            OptList = get(handles.VariableBox{ModuleNums}(i),'String');
            if ~RunInBG
                handles.Settings.VariableValues{ModuleNums, i} = OptList{1};
            else
                PPos = find(strcmp(handles.Settings.VariableValues{ModuleNums,i},OptList));
                if isempty(PPos)
                    if ~strcmp(handles.Settings.VariableValues{ModuleNums,i},'Pipeline Value')
                        %%% [Kyungnam & Anne, Aug-08-2007]
                        %%% Here is the place where the values of VariableBoxes are set
                        %%% based on the Settings of the pipeline loaded by the user.
                        %%% We are adding a catch to help deal with old versions of the Modules using Median Filtering
                        %%% e.g., CorrectIllumination_Calculate, Smooth. Median Filtering is automatically converted to
                        %%% Gaussian Filter.
                        if strcmp(handles.Settings.VariableValues{ModuleNums,i}, 'Median Filtering')
                            handles.Settings.VariableValues{ModuleNums,i} = 'Gaussian Filter';
                            CPwarndlg('Your pipeline uses Modules(s) including old ''Median Filtering'' which is actually ''Gaussian Filter''. We automatically convert Median Filtering to Gaussian Filter for your convenience');
                        else
                            set(handles.VariableBox{ModuleNums}(i),'String',[OptList;handles.Settings.VariableValues(ModuleNums,i)]);                            
                        end                                            
                        PPos = find(strcmp(handles.Settings.VariableValues{ModuleNums,i},OptList));
                        set(handles.VariableBox{ModuleNums}(i),'Value',PPos);
                    end
                else
                    set(handles.VariableBox{ModuleNums}(i),'Value',PPos);
                end
            end
        end
    end
    
    if lastVariableCheck == 0
        CPerrordlg(['The module you attempted to add, ', ModuleNamedotm,', is not a valid CellProfiler module because it does not appear to have any variables.  Sometimes this error occurs when you try to load a module that has the same name as a built-in Matlab function and the built in function is located in a directory higher up on the Matlab search path.']);
        return
    end

    try Contents = handles.Settings.VariableRevisionNumbers(str2double(ModuleNumber));
    catch handles.Settings.VariableRevisionNumbers(str2double(ModuleNumber)) = 0;
    end

    %blah
    try ModuleRevContents = handles.Settings.ModuleRevisionNumbers(str2double(ModuleNumber));
    catch handles.Settings.ModuleRevisionNumbers(str2double(ModuleNumber)) = 0;
    end

    %%% 5. Saves the ModuleName to the handles structure.
    % Find which module slot number this callback was called for.

    handles.Settings.ModuleNames{ModuleNums} = ModuleName;
    if ~RunInBG
        contents = get(handles.ModulePipelineListBox,'String');
        if iscell(contents)
            contents{ModuleNums} = ModuleName;
        else
            contents = {ModuleName};
        end
        set(handles.ModulePipelineListBox,'String',contents);
    end

    handles.Current.NumberOfModules = numel(handles.Settings.ModuleNames);

    %%% 6. Choose Loaded Module in Listbox
    if ~RunInBG
        set(handles.ModulePipelineListBox,'Value',ModuleNums);
    else
        set(findobj('Parent',handles.variablepanel,'Visible','On'),'Visible','Off');
    end

    MaxInfo = get(handles.slider1,'UserData');
    MaxInfo = [MaxInfo(1:ModuleNums-1) ((handles.Settings.NumbersOfVariables(ModuleNums)-12+numberExtraLinesOfDescription)*25) MaxInfo(ModuleNums:end)];
    set(handles.slider1,'UserData',MaxInfo);

    %%% Updates the handles structure to incorporate all the changes.
    guidata(handles.figure1,handles);

    if ~RunInBG
        ModulePipelineListBox_Callback(gcbo, [], handles);
        slider1_Callback(handles.slider1,0,handles);
    end
end

%%% SUBFUNCTION %%%
function twodigit = TwoDigitString(val)
%TwoDigitString is a function like num2str(int) but it returns a two digit
%representation of a string for our purposes.
if ((val > 99) || (val < 0)),
    error(['TwoDigitString: Can''t convert ' num2str(val) ' to a 2 digit number']);
end
twodigit = sprintf('%02d', val);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% REMOVE MODULE BUTTON %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes on button press for RemoveModule button.
function RemoveModule_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
ModuleHighlighted = get(handles.ModulePipelineListBox,'Value');
RemoveModule_Helper(ModuleHighlighted, hObject, eventdata, handles, 'Confirm');
%%% SUBFUNCTION %%%
function RemoveModule_Helper(ModuleHighlighted, hObject, eventdata, handles, ConfirmOrNot) %#ok We want to ignore MLint error checking for this line.

if strcmp(ConfirmOrNot, 'Confirm') == 1
    %%% Confirms the choice to clear the module.
    Answer = CPquestdlg('Are you sure you want to clear this analysis module and its settings?','Confirm','Yes','No','Yes');
    if strcmp(Answer,'No')
        return
    end
end
if isempty(handles.Settings.ModuleNames);
    return
end
%%% 1. Sets all VariableBox edit boxes and all VariableDescriptions to be invisible.

MaxInfo = get(handles.slider1,'UserData');

for ModuleDelete = 1:length(ModuleHighlighted);
    handles = RemoveVariables(handles,ModuleHighlighted(ModuleDelete)-ModuleDelete+1);
    %%% Remove variable names from other modules
    for VariableNumber = 1:length(handles.VariableBox{ModuleHighlighted(ModuleDelete)-ModuleDelete+1})
        delete(handles.VariableBox{ModuleHighlighted(ModuleDelete)-ModuleDelete+1}(VariableNumber));
    end
    for VariableNumber = 1:length(handles.VariableDescription{ModuleHighlighted(ModuleDelete)-ModuleDelete+1})
        delete(handles.VariableDescription{ModuleHighlighted(ModuleDelete)-ModuleDelete+1}(VariableNumber));
    end
    if isfield(handles,'BrowseButton')
        if length(handles.BrowseButton) >= (ModuleHighlighted(ModuleDelete)-ModuleDelete+1)
            if ~isempty(handles.BrowseButton{ModuleHighlighted(ModuleDelete)-ModuleDelete+1})
                for VariableNumber = 1:length(handles.BrowseButton{ModuleHighlighted(ModuleDelete)-ModuleDelete+1})
                    if (handles.BrowseButton{ModuleHighlighted(ModuleDelete)-ModuleDelete+1}(VariableNumber) ~= 0) && ishandle(handles.BrowseButton{ModuleHighlighted(ModuleDelete)-ModuleDelete+1}(VariableNumber))
                        delete(handles.BrowseButton{ModuleHighlighted(ModuleDelete)-ModuleDelete+1}(VariableNumber));
                    end
                end
            end
        end
    end
    %%% 2. Removes the ModuleName from the handles structure.
    handles.Settings.ModuleNames(ModuleHighlighted(ModuleDelete)-ModuleDelete+1) = [];
    %%% 3. Clears the variable values in the handles structure.
    handles.Settings.VariableValues(ModuleHighlighted(ModuleDelete)-ModuleDelete+1,:) = [];
    %%% 4. Clears the number of variables in each module slot from handles structure.
    handles.Settings.NumbersOfVariables(ModuleHighlighted(ModuleDelete)-ModuleDelete+1) = [];
    %%% 5. Clears the Variable Revision Numbers in each module slot from handles structure.
    handles.Settings.VariableRevisionNumbers(ModuleHighlighted(ModuleDelete)-ModuleDelete+1) = [];
    %%% 6. Clears the Module Revision Numbers in each module slot from handles structure.
    handles.Settings.ModuleRevisionNumbers(ModuleHighlighted(ModuleDelete)-ModuleDelete+1) = [];
    if size(handles.Settings.VariableInfoTypes,1) >= (ModuleHighlighted(ModuleDelete)-ModuleDelete+1)
        handles.Settings.VariableInfoTypes(ModuleHighlighted(ModuleDelete)-ModuleDelete+1,:) = [];
    end
    handles.VariableDescription(ModuleHighlighted(ModuleDelete)-ModuleDelete+1)=[];
    handles.VariableBox(ModuleHighlighted(ModuleDelete)-ModuleDelete+1)=[];
    if isfield(handles,'BrowseButton')
        if length(handles.BrowseButton) >= (ModuleHighlighted(ModuleDelete)-ModuleDelete+1)
            handles.BrowseButton(ModuleHighlighted(ModuleDelete)-ModuleDelete+1)=[];
        end
    end
    MaxInfo = [MaxInfo(1:(ModuleHighlighted(ModuleDelete)-ModuleDelete)) MaxInfo((ModuleHighlighted(ModuleDelete)-ModuleDelete+2):end)];
end

if length(handles.Settings.NumbersOfVariables) ~= length(handles.Settings.ModuleNames)
    CPmsgbox('STOP! Somehow the NumbersOfVariable was not set correctly! Please record EVERYTHING you have done in the past few minutes and send to mrl@wi.mit.edu! Be sure to include what modules are in your pipeline and what you tried to do to cause this error (Adding, Subtracting, Moving modules, how many?).');
end

set(handles.slider1,'UserData',MaxInfo);

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
    set(handles.slider1,'visible','off');
elseif (isempty(ModuleHighlighted))
    ModuleHighlighted = handles.Current.NumberOfModules;
end

set(handles.ModulePipelineListBox,'Value',ModuleHighlighted);

guidata(gcbo, handles);
ModulePipelineListBox_Callback(hObject, eventdata, handles);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% MOVE UP/DOWN BUTTONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function MoveUpButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
ModuleHighlighted = get(handles.ModulePipelineListBox,'Value');
if~(handles.Current.NumberOfModules < 1 || ModuleHighlighted(1) == 1)
    MaxInfo = get(handles.slider1,'UserData');
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
        %%% 5. Copy then clear the module revision numbers in the handles
        %%% structure.
        copyModRevNums = handles.Settings.ModuleRevisionNumbers(ModuleNow);
        handles.Settings.ModuleRevisionNumbers(ModuleNow) = handles.Settings.ModuleRevisionNumbers(ModuleUp);
        handles.Settings.ModuleRevisionNumbers(ModuleUp) = copyModRevNums;
        %%% 6. Copy then clear the variable infotypes in the handles
        %%% structure.
        copyVarInfoTypes = handles.Settings.VariableInfoTypes(ModuleNow,:);
        handles.Settings.VariableInfoTypes(ModuleNow,:) = handles.Settings.VariableInfoTypes(ModuleUp,:);
        handles.Settings.VariableInfoTypes(ModuleUp,:) = copyVarInfoTypes;

        CopyVariableDescription = handles.VariableDescription(ModuleNow);
        handles.VariableDescription(ModuleNow) = handles.VariableDescription(ModuleUp);
        handles.VariableDescription(ModuleUp) = CopyVariableDescription;

        CopyVariableBox = handles.VariableBox(ModuleNow);
        handles.VariableBox(ModuleNow) = handles.VariableBox(ModuleUp);
        handles.VariableBox(ModuleUp) = CopyVariableBox;

        if isfield(handles,'BrowseButton')
            if length(handles.BrowseButton) >= ModuleNow
                CopyBrowseButton = handles.BrowseButton(ModuleNow);
                handles.BrowseButton(ModuleNow) = handles.BrowseButton(ModuleUp);
                handles.BrowseButton(ModuleUp) = CopyBrowseButton;
            else
                if length(handles.BrowseButton) >= ModuleUp
                    handles.BrowseButton(ModuleNow) = handles.BrowseButton(ModuleUp);
                    handles.BrowseButton(ModuleUp) = {[]};
                end
            end
        end

        CopyMaxInfo = MaxInfo(ModuleNow);
        MaxInfo(ModuleNow) = MaxInfo(ModuleUp);
        MaxInfo(ModuleUp) = CopyMaxInfo;
    end
    %%% 7. Changes the Listbox to show the changes
    contents = handles.Settings.ModuleNames;
    set(handles.ModulePipelineListBox,'String',contents);
    set(handles.ModulePipelineListBox,'Value',ModuleHighlighted-1);
    set(handles.slider1,'UserData',MaxInfo);
    %%% Updates the handles structure to incorporate all the changes.
    guidata(gcbo, handles);
    ModulePipelineListBox_Callback(hObject, eventdata, handles)
end

if length(handles.Settings.NumbersOfVariables) ~= length(handles.Settings.ModuleNames)
    CPmsgbox('STOP! Somehow the NumbersOfVariable was not set correctly! Please record EVERYTHING you have done in the past few minutes and send to mrl@wi.mit.edu! Be sure to include what modules are in your pipeline and what you tried to do to cause this error (Adding, Subtracting, Moving modules, how many?).');
end

function MoveDownButton_Callback(hObject,eventdata,handles) %#ok We want to ignore MLint error checking for this line.
ModuleHighlighted = get(handles.ModulePipelineListBox,'Value');
if~(handles.Current.NumberOfModules<1 || ModuleHighlighted(length(ModuleHighlighted)) >= handles.Current.NumberOfModules)
    MaxInfo = get(handles.slider1,'UserData');
    for ModuleDown1 = length(ModuleHighlighted):-1:1;
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
        copyNumVariables = handles.Settings.NumbersOfVariables(ModuleNow);
        handles.Settings.NumbersOfVariables(ModuleNow) = handles.Settings.NumbersOfVariables(ModuleDown);
        handles.Settings.NumbersOfVariables(ModuleDown) = copyNumVariables;
        %%% 4. Copy then clear the variable revision numbers in the handles
        %%% structure.
        copyVarRevNums = handles.Settings.VariableRevisionNumbers(ModuleNow);
        handles.Settings.VariableRevisionNumbers(ModuleNow) = handles.Settings.VariableRevisionNumbers(ModuleDown);
        handles.Settings.VariableRevisionNumbers(ModuleDown) = copyVarRevNums;
        %%% 5. Copy then clear the module revision numbers in the handles
        %%% structure.
        copyModRevNums = handles.Settings.ModuleRevisionNumbers(ModuleNow);
        handles.Settings.ModuleRevisionNumbers(ModuleNow) = handles.Settings.ModuleRevisionNumbers(ModuleDown);
        handles.Settings.ModuleRevisionNumbers(ModuleDown) = copyModRevNums;
        %%% 6. Copy then clear the variable infotypes in the handles
        %%% structure.
        copyVarInfoTypes = handles.Settings.VariableInfoTypes(ModuleNow,:);
        handles.Settings.VariableInfoTypes(ModuleNow,:) = handles.Settings.VariableInfoTypes(ModuleDown,:);
        handles.Settings.VariableInfoTypes(ModuleDown,:) = copyVarInfoTypes;

        CopyVariableDescription = handles.VariableDescription(ModuleNow);
        handles.VariableDescription(ModuleNow) = handles.VariableDescription(ModuleDown);
        handles.VariableDescription(ModuleDown) = CopyVariableDescription;

        CopyVariableBox = handles.VariableBox(ModuleNow);
        handles.VariableBox(ModuleNow) = handles.VariableBox(ModuleDown);
        handles.VariableBox(ModuleDown) = CopyVariableBox;

        if isfield(handles,'BrowseButton')
            if length(handles.BrowseButton) >= ModuleNow
                CopyBrowseButton = handles.BrowseButton(ModuleNow);
                if length(handles.BrowseButton) >= ModuleDown
                    handles.BrowseButton(ModuleNow) = handles.BrowseButton(ModuleDown);
                else
                    handles.BrowseButton(ModuleNow) = [];
                end
                handles.BrowseButton(ModuleDown) = CopyBrowseButton;
            end
        end

        CopyMaxInfo = MaxInfo(ModuleNow);
        MaxInfo(ModuleNow) = MaxInfo(ModuleDown);
        MaxInfo(ModuleDown) = CopyMaxInfo;
    end
    %%% 7. Changes the Listbox to show the changes
    contents = handles.Settings.ModuleNames;
    set(handles.ModulePipelineListBox,'String',contents);
    set(handles.ModulePipelineListBox,'Value',ModuleHighlighted+1);
    set(handles.slider1,'UserData',MaxInfo);
    %%% Updates the handles structure to incorporate all the changes.
    guidata(gcbo, handles);
    ModulePipelineListBox_Callback(hObject, eventdata, handles)
end

if length(handles.Settings.NumbersOfVariables) ~= length(handles.Settings.ModuleNames)
    CPmsgbox('STOP! Somehow the NumbersOfVariable was not set correctly! Please record EVERYTHING you have done in the past few minutes and send to mrl@wi.mit.edu! Be sure to include what modules are in your pipeline and what you tried to do to cause this error (Adding, Subtracting, Moving modules, how many?).');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% MODULE PIPELINE LISTBOX %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes on selection change in ModulePipelineListBox.
function ModulePipelineListBox_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
ModuleHighlighted = get(handles.ModulePipelineListBox,'Value');
if (length(ModuleHighlighted) > 0)
    ModuleNumber = ModuleHighlighted(1);
    if( handles.Current.NumberOfModules > 0 )
        %%% 2. Sets all VariableBox edit boxes and all
        %%% VariableDescriptions to be invisible.
        set(findobj('Parent',handles.variablepanel,'Visible','On'),'Visible','Off');
        set(handles.VariableDescription{ModuleNumber},'Visible','On');

        if length(handles.VariableBox) == 1
            set(handles.VariableBox{ModuleNumber}(~strcmp({get(handles.VariableBox{ModuleNumber},'string')},'n/a')),'Visible','On'); %only makes the boxes without n/a as the string visible
        else
            set(handles.VariableBox{ModuleNumber}(~strcmp(get(handles.VariableBox{ModuleNumber},'string'),'n/a')),'Visible','On'); %only makes the boxes without n/a as the string visible
        end
        try
            set(handles.BrowseButton{ModuleNumber},'Visible','On')
        end
        %%% 2.25 Removes slider and moves panel back to original
        %%% position.
        %%% If panel location gets changed in GUIDE, must change the
        %%% position values here as well.
        set(handles.variablepanel, 'position', [238 0 563 346]);
        MatlabVersion = version;
        MatlabVersion = str2double(MatlabVersion(1:3));
        if ispc || (MatlabVersion >= 7.1)
            set(handles.slider1,'value',get(handles.slider1,'max'));
        else
            set(handles.slider1,'value',get(handles.slider1,'min'));
        end

        set(handles.slider1,'visible','off');
        %%% 2.5 Checks whether a module is loaded in this slot.
        % contents = get(handles.ModulePipelineListBox,'String');
        % ModuleName = contents{ModuleNumber};

        %%% 5.  Sets the slider
        MaxInfo = get(handles.slider1,'UserData');
        MaxInfo = MaxInfo(ModuleNumber);
        if(MaxInfo > 0)
            set(handles.slider1,'visible','on');
            set(handles.slider1,'max',MaxInfo);
            if ispc || (MatlabVersion >= 7.1)
                set(handles.slider1,'value',get(handles.slider1,'max'));
            else
                set(handles.slider1,'value',get(handles.slider1,'min'));
            end
            set(handles.slider1,'SliderStep',[max(.2,1/MaxInfo) min(1,5/MaxInfo)]);
        end
        slider1_Callback(handles.slider1,0,handles);
    else
        % Anne 7/11/06 Nice idea to have a confirmation, but I commented out the
        % dialog, because I think almost always
        % the user will have clicked intentionally, and the consequences of opening
        % up the Add module window are pretty minimal (that is, it is an easily
        % reversible choice). Also, because this can only happen right when the
        % user starts up CellProfiler, the chances for random clicking are fairly
        % minimal as well.
        % Rodrigo 7/13/06 - Maybe we can check the selection type, and open the Add
        % module window only if the user double-clicked. Nothing will happen
        % otherwise. What do you prefer?
        % Rodrigo 7/20/06 - We may have another problem with this. When
        % ModulePipelineListBox_Callback gets called by ClearPipeline_Callback, it
        % opens the AddModule window because ClearPipeline sets
        % handles.Current.NumberOfModules to 0 right before making the call. I
        % removed the call from ClearPipeline, but the same happens when you use
        % RemoveModule when you only have 1 module in the listbox. Obviously I
        % can't remove the call from RemoveModule_Callback because it's necessary,
        % so we might need to add some kind of test here. I checked every other
        % call to ModulePipelineListBox_Callback and I think they're all ok.
        % Rodrigo 7/21/06 - I just added another call to this function. I needed to
        % call it in the LoadPipeline_Callback function to refresh the variable
        % panel if the loading was aborted. It also opens the AddModule window.

        %        Answer = CPquestdlg('No modules are loaded. Do you want to add one?','No modules are loaded','Yes','No','Yes');
        %       if strcmp(Answer,'Yes')
        %%%%%  if strcmp(get(gcf,'SelectionType'),'open') %% these two lines should make it work
        %
        % Mike 9/7/06 I think opening the add module window should not
        % be done here, since it happens on occasions such as trying to
        % plot a histogram. Commented out.
        % AddModule_Callback(findobj('tag','AddModule'),[],handles);
        %%%%%  end
        %      end
    end
else
    % Mike 9/7/06 This is also very annoying. Feel free to discuss with me.
    %CPhelpdlg('No module highlighted.');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% VARIABLE EDIT BOXES %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

function handles = RemoveVariables(handles,ModuleNumber)
%%% This function removes all variables of a specified Module from the
%%% handles structure.

for i = 1:length(handles.VariableBox{ModuleNumber})
    InfoType = get(handles.VariableBox{ModuleNumber}(i),'UserData');
    StrSet = get(handles.VariableBox{ModuleNumber}(i),'string');
    if length(InfoType) >= 5 && strcmp(InfoType(end-4:end),'indep')
        ModList = findobj('UserData',InfoType(1:end-6));
        ModList2 = findobj('UserData',InfoType);
        ModList2 = ModList2(ModList2 ~= handles.VariableBox{ModuleNumber}(i));
        TestVars = get(ModList2,'string');
        OtherIndepWithSameValue = [];
        for k = 1:length(TestVars)
            if isempty(OtherIndepWithSameValue)
                if iscell(TestVars)
                    OtherIndepWithSameValue = strmatch(StrSet,TestVars{k});
                else
                    OtherIndepWithSameValue = strmatch(StrSet,TestVars(k));
                end
            end
        end

        if isempty(OtherIndepWithSameValue)
            for m=1:numel(ModList)
                PrevList = get(ModList(m),'string');
                VarVal = get(ModList(m),'value');
                BoxTag = get(ModList(m),'tag');
                BoxNum = str2double(BoxTag((length(BoxTag)-1):end));
                ModNum = [];
                for j = 1:length(handles.VariableBox)
                    if length(handles.VariableBox{j}) >= BoxNum
                        if ModList(m) == handles.VariableBox{j}(BoxNum)
                            ModNum = j;
                        end
                    end
                end
                if strcmp(get(ModList(m),'style'),'popupmenu')
                    if strcmp(PrevList(VarVal),StrSet)
                        if size(PrevList,1) == 1
                            NewStrSet = PrevList;
                        else
                            NewStrSet = cat(1,PrevList(1:(VarVal-1)),PrevList((VarVal+1):end));
                        end
                        set(ModList(m),'string',NewStrSet);
                        set(ModList(m),'value',1);
                        handles.Settings.VariableValues(ModNum,BoxNum) = NewStrSet(1);
                    else
                        OldPos = strmatch(StrSet,PrevList);
                        if ~isempty(OldPos)
                            OldStr = PrevList(VarVal);
                            NewStrSet = cat(1,PrevList(1:(OldPos-1)),PrevList((OldPos+1):end));
                            CorrectValue = strmatch(OldStr,NewStrSet,'exact');
                            set(ModList(m),'string',NewStrSet);
                            set(ModList(m),'value',CorrectValue);
                            handles.Settings.VariableValues(ModNum,BoxNum) = NewStrSet(CorrectValue);
                        end
                    end
                end
            end
        end
    end
end
guidata(handles.figure1,handles);

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% VARIABLE EDIT BOXES %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
function storevariable(ModuleNumber, VariableNumber, UserEntry, handles)
%%% This function stores a variable's value in the handles structure,
%%% when given the Module Number, the Variable Number,
%%% the UserEntry (from the Edit box), and the initial handles
%%% structure.

InfoType = get(handles.VariableBox{ModuleNumber}(str2double(VariableNumber)),'UserData');
StrSet = get(handles.VariableBox{ModuleNumber}(str2double(VariableNumber)),'string');
% Type = get(handles.VariableBox{ModuleNumber}(str2double(VariableNumber)),'Style');

if length(InfoType) >= 5 && strcmp(InfoType(end-4:end),'indep')
    PrevValue = handles.Settings.VariableValues(ModuleNumber, str2double(VariableNumber));
    ModList = findobj('UserData',InfoType(1:end-6));
    %Filter out objects that are over this one
    ModList2 = findobj('UserData',InfoType(1:end));
    ModList2 = ModList2(ModList2 ~= handles.VariableBox{ModuleNumber}(str2double(VariableNumber)));
    %ModList3 = nonzeros(ModList2(strcmp(get(ModList2,'String'),PrevValue)));
    for i = 1:length(ModList2)
        Values = get(ModList2(i),'value');
        PrevStrSet = get(ModList2(i),'string');
        if Values == 0
            if strcmp(PrevStrSet,PrevValue)
                if exist('ModList3','var')
                    ModList3(end+1) = ModList2(i); %#ok Ignore MLint
                else
                    ModList3 = ModList2(i);
                end
            end
        else
            if strcmp(PrevStrSet(Values),PrevValue)
                if exist('ModList3','var')
                    ModList3(end+1) = ModList2(i); %#ok
                else
                    ModList3 = ModList2(i);
                end
            end
        end
    end
    if ischar(UserEntry)
        if size(StrSet,1) == 1
            for i = 1:length(ModList2)
                Values = get(ModList2(i),'value');
                PrevStrSet = get(ModList2(i),'string');
                if Values == 0
                    if strcmp(PrevStrSet,StrSet)
                        if exist('ModList4','var')
                            ModList4(end+1) = ModList2(i); %#ok Ignore MLint
                        else
                            ModList4 = ModList2(i);
                        end
                    end
                else
                    if strcmp(PrevStrSet(Values),StrSet)
                        if exist('ModList4','var')
                            ModList4(end+1) = ModList2(i); %#ok
                        else
                            ModList4 = ModList2(i);
                        end
                    end
                end
            end
        else
            OrigValues = get(handles.VariableBox{ModuleNumber}(str2double(VariableNumber)),'value');
            for i = 1:length(ModList2)
                Values = get(ModList2(i),'value');
                PrevStrSet = get(ModList2(i),'string');
                if Values == 0
                    if strcmp(PrevStrSet,StrSet(OrigValues))
                        if exist('ModList4','var')
                            ModList4(end+1) = ModList2(i); %#ok
                        else
                            ModList4 = ModList2(i);
                        end
                    end
                else
                    if strcmp(PrevStrSet(Values),StrSet(OrigValues))
                        if exist('ModList4','var')
                            ModList4(end+1) = ModList2(i); %#ok
                        else
                            ModList4 = ModList2(i);
                        end
                    end
                end
            end
        end
    else
        for i = 1:length(ModList2)
            Values = get(ModList2(i),'value');
            PrevStrSet = get(ModList2(i),'string');
            if Values == 0
                if strcmp(PrevStrSet,StrSet(UserEntry))
                    if exist('ModList4','var')
                        ModList4(end+1) = ModList2(i); %#ok
                    else
                        ModList4 = ModList2(i);
                    end
                end
            else
                if strcmp(PrevStrSet(Values),StrSet(UserEntry))
                    if exist('ModList4','var')
                        ModList4(end+1) = ModList2(i); %#ok
                    else
                        ModList4 = ModList2(i);
                    end
                end
            end
        end
    end

    if ~exist('ModList4','var')
        ModList4 = [];
    end
    if ~exist('ModList3','var')
        ModList3 = [];
    end

    for i=1:numel(ModList)
        BoxTag = get(ModList(i),'tag');
        BoxNum = str2double(BoxTag((length(BoxTag)-1):end));
        ModNum = [];
        for m = 1:handles.Current.NumberOfModules
            if length(handles.VariableBox{m}) >= BoxNum
                if ModList(i) == handles.VariableBox{m}(BoxNum)
                    ModNum = m;
                end
            end
        end
        if isempty(ModNum)
            m = handles.Current.NumberOfModules + 1;
            if ModList(i) == handles.VariableBox{m}(BoxNum)
                ModNum = m;
            end
        end
        CurrentString = get(ModList(i),'String');
        try
            if isempty(CurrentString{1})
                CurrentString = StrSet;
                set(ModList(i),'Enable','on');
            end
        catch
            if isempty(CurrentString)
                CurrentString = StrSet;
                set(ModList(i),'Enable','on');
            end
        end
        MatchedIndice = strmatch(PrevValue,CurrentString);
        if ~isempty(MatchedIndice) && isempty(ModList3)
            if isempty(ModList4)
                if ~iscell(CurrentString)
                    set(ModList(i),'String',{UserEntry});
                else
                    if length(CurrentString) == 1
                        set(ModList(i),'String',cat(1,CurrentString,{UserEntry}));
                    else
                        if ischar(UserEntry)
                            set(ModList(i),'String',cat(1,CurrentString(1:(MatchedIndice-1)),{UserEntry},CurrentString((MatchedIndice+1):end)));
                            VarVal = get(ModList(i),'value');
                            SetStr = get(ModList(i),'string');
                            handles.Settings.VariableValues(ModNum,BoxNum) = SetStr(VarVal);
                            clear VarVal SetStr
                        else
                            set(ModList(i),'String',cat(1,CurrentString(1:(MatchedIndice-1)),StrSet(UserEntry),CurrentString((MatchedIndice+1):end)));
                        end
                    end
                end
            else
                set(ModList(i),'String',cat(1,CurrentString(1:(MatchedIndice-1)),CurrentString((MatchedIndice+1):end)));
                if get(ModList(i),'Value')==MatchedIndice
                    set(ModList(i),'Value',1);
                    VarVals = get(ModList(i),'string');
                    if iscell(VarVals)
                        handles.Settings.VariableValues(ModNum, BoxNum) = VarVals(1);
                    else
                        handles.Settings.VariableValues(ModNum, BoxNum) = VarVals;
                    end
                else
                    OldVal = get(ModList(i),'Value');
                    if (OldVal ~= 0) && OldVal > MatchedIndice
                        set(ModList(i),'Value',(OldVal-1));
                    end
                end
            end
        elseif isempty(ModList4)
            if numel(CurrentString) == 0
                CurrentString = {UserEntry};
                set(ModList(i),'String',CurrentString);
            elseif ~iscell(CurrentString)
                CurrentString = {CurrentString};
                set(ModList(i),'String',CurrentString);
            else
                if ischar(UserEntry)
                    if size(StrSet,1) == 1
                        if ~strcmp(StrSet,'n/a') && ~strcmp(StrSet,'/')
                            CurrentString(numel(CurrentString)+1) = {StrSet};
                        end
                        set(ModList(i),'String',CurrentString);
                    else
                        OrigValues = get(handles.VariableBox{ModuleNumber}(str2double(VariableNumber)),'value');
                        if ~strcmp(StrSet{OrigValues},'n/a') && ~strcmp(StrSet{OrigValues},'/')
                            CurrentString(numel(CurrentString)+1) = {StrSet{OrigValues}};
                        end
                        set(ModList(i),'String',CurrentString);
                    end
                else
                    if ~strcmp(StrSet(UserEntry),'n/a') && ~strcmp(StrSet(UserEntry),'/')
                        CurrentString(numel(CurrentString)+1) = StrSet(UserEntry);
                    end
                    set(ModList(i),'String',CurrentString);
                end
            end
        end
    end
end

if strcmp(get(handles.VariableBox{ModuleNumber}(str2double(VariableNumber)),'style'),'edit')
    handles.Settings.VariableValues(ModuleNumber, str2double(VariableNumber)) = {UserEntry};
else
    if ischar(UserEntry)
        handles.Settings.VariableValues(ModuleNumber, str2double(VariableNumber)) = {UserEntry};
    else
        handles.Settings.VariableValues(ModuleNumber, str2double(VariableNumber)) = StrSet(UserEntry);
    end
end
guidata(handles.figure1, handles);

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
ModuleNumber = whichactive(handles);
InputType = get(hObject,'style');

if strcmp(InputType, 'edit')
    UserEntry = get(hObject,'string');
elseif strcmp(InputType, 'popupmenu')
    UserEntry = get(hObject,'value');
    ChoiceList = get(hObject,'string');
    if strcmp('Other..', ChoiceList{UserEntry})
        CustomInput = CPinputdlg('Enter your custom input: ');
        if isempty(CustomInput) | isempty(CustomInput{1}) %#ok
            set(hObject,'value',1);
        else
            ChoiceList(numel(ChoiceList)) = CustomInput;
            ChoiceList(numel(ChoiceList)+1) = {'Other..'};
            set(hObject,'string',ChoiceList);
        end
    end
end

if isempty(UserEntry)
    CPerrordlg('Variable boxes must not be left blank');
    set(handles.VariableBox{ModuleNumber}(str2double(VariableNumberStr)),'string','Fill in');
    storevariable(ModuleNumber,VariableNumberStr, 'Fill in',handles);
else
    if ModuleNumber == 0,
        CPerrordlg('Something strange is going on: none of the analysis modules are active right now but somehow you were able to edit a setting.','weirdness has occurred');
    else
        storevariable(ModuleNumber,VariableNumberStr,UserEntry, handles);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% VARIABLE WINDOW SLIDER %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes on slider movement.
function slider1_Callback(hObject, eventdata, handles) %#ok Ignore MLint
% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range
%        of slider
scrollPos = get(hObject, 'Value');
variablepanelPos = get(handles.variablepanel, 'position');
ModuleHighlighted = get(handles.ModulePipelineListBox,'Value');
ModuleNumber = ModuleHighlighted(1);

if isempty(handles.Settings.NumbersOfVariables)
    set(handles.slider1,'visible','off');
    guidata(handles.figure1,handles);
    return
end
% Note:  The yPosition is 0 + scrollPos because 0 is the original Y
% Position of the variablePanel.  If the original location of the
% variablePanel gets changed, then the constant offset must be changed as
% well.
MatlabVersion = version;
MatlabVersion = str2double(MatlabVersion(1:3));
if ispc || (MatlabVersion >= 7.1)
    Ypos = get(handles.slider1,'max') - get(handles.slider1,'Value');
    set(handles.variablepanel, 'position', [variablepanelPos(1) Ypos variablepanelPos(3) variablepanelPos(4)]);
    for i=1:handles.Settings.NumbersOfVariables(ModuleNumber)
        tempPos=get(handles.VariableDescription{ModuleNumber}(i),'Position');
        if(tempPos(2)+Ypos)>-20
            set(handles.VariableDescription{ModuleNumber}(i),'visible','on');
            VarDesOn=1;
        else
            set(handles.VariableDescription{ModuleNumber}(i),'visible','off');
            VarDesOn=0;
        end
        tempPos=get(handles.VariableBox{ModuleNumber}(i),'Position');
        if ((tempPos(2)+Ypos)>-20) && VarDesOn  && (size(get(handles.VariableBox{ModuleNumber}(i),'String'),1)~=1 || ~strcmp(get(handles.VariableBox{ModuleNumber}(i),'String'),'n/a'))
            set(handles.VariableBox{ModuleNumber}(i),'visible','on');
        else
            set(handles.VariableBox{ModuleNumber}(i),'visible','off');
        end
        try
            tempPos=get(handles.BrowseButton{ModuleNumber}(i),'Position');
            if ((tempPos(2)+Ypos)>-20) && VarDesOn
                set(handles.BrowseButton{ModuleNumber}(i),'visible','on');
            else
                set(handles.BrowseButton{ModuleNumber}(i),'visible','off');
            end
        end
    end
    guidata(handles.figure1,handles);
else
    set(handles.variablepanel, 'position', [variablepanelPos(1) 0+scrollPos variablepanelPos(3) variablepanelPos(4)]);
    for i=1:handles.Settings.NumbersOfVariables(ModuleNumber)
        tempPos=get(handles.VariableDescription{ModuleNumber}(i),'Position');
        if(tempPos(2)+scrollPos)>-20
            set(handles.VariableDescription{ModuleNumber}(i),'visible','on');
            VarDesOn=1;
        else
            set(handles.VariableDescription{ModuleNumber}(i),'visible','off');
            VarDesOn=0;
        end
        tempPos=get(handles.VariableBox{ModuleNumber}(i),'Position');
        if ((tempPos(2)+scrollPos)>-20) && VarDesOn  && (size(get(handles.VariableBox{ModuleNumber}(i),'String'),1)~=1 || ~strcmp(get(handles.VariableBox{ModuleNumber}(i),'String'),'n/a'))
            set(handles.VariableBox{ModuleNumber}(i),'visible','on');
        else
            set(handles.VariableBox{ModuleNumber}(i),'visible','off');
        end
        try
            tempPos=get(handles.BrowseButton{ModuleNumber}(i),'Position');
            if ((tempPos(2)+scrollPos)>-20) && VarDesOn
                set(handles.BrowseButton{ModuleNumber}(i),'visible','on');
            else
                set(handles.BrowseButton{ModuleNumber}(i),'visible','off');
            end
        end
    end
    guidata(handles.figure1,handles);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PIXEL SIZE EDIT BOX %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

function PixelSizeEditBox_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
%%% Checks to see whether the user input is a number, and generates an
%%% error message if it is not a number.
user_entry = str2double(get(hObject,'string'));
if isnan(user_entry)
    CPerrordlg('You must enter a numeric value','Bad Input','modal');
    set(hObject,'string','0.25')
    %%% Checks to see whether the user input is positive, and generates an
    %%% error message if it is not.
elseif user_entry<=0
    CPerrordlg('You entered a value less than or equal to zero','Bad Input','modal');
    set(hObject,'string', '0.25')
else
    %%% Gets the user entry and stores it in the handles structure.
    UserEntry = get(handles.PixelSizeEditBox,'string');
    handles.Settings.PixelSize = UserEntry;
    handles.Preferences.PixelSize = UserEntry;
    guidata(gcbo, handles);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SET PREFERENCES BUTTON %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function SaveButton_Callback (hObject, eventdata, handles)  %#ok Ignore MLint

Answer = CPquestdlg('Do you want to save these as the default preferences? If not, you will be asked to name your preferences file, which can be loaded by File -> Load Preferences.','Save as default?','Yes','No','Yes');
if strcmp(Answer, 'No')
    [FileName,Pathname] = uiputfile(fullfile(matlabroot,'*.mat'), 'Save Preferences As...');
    if isequal(FileName,0) || isequal(Pathname,0)
        Pathname = matlabroot;
        FileName = 'CellProfilerPreferences.mat';
        FullFileName = fullfile(Pathname,FileName);
        DefaultVal = 1;
        CPwarndlg('Since you did not specify a file name, the file was saved as CellProfilerPreferences.mat in the matlab root folder.');
    else
        FullFileName = fullfile(Pathname,FileName);
        DefaultVal = 0;
    end
else
    if isdeployed
        FullFileName = fullfile(pwd,'CellProfilerPreferences.mat');
        DefaultVal = 1;
    else
        FullFileName = fullfile(matlabroot,'CellProfilerPreferences.mat');
        DefaultVal = 1;
    end
end

SetPreferencesWindowHandle = findobj('name','SetPreferences');
global EnteredPreferences
PixelSizeEditBoxHandle = findobj('Tag','PixelSizeEditBox');
FontSizeEditBoxHandle = findobj('Tag','FontSizeEditBox');
ImageDirEditBoxHandle = findobj('Tag','ImageDirEditBox');
OutputDirEditBoxHandle = findobj('Tag','OutputDirEditBox');
ModuleDirEditBoxHandle = findobj('Tag','ModuleDirEditBox');
IntensityColorMapHandle = findobj('Tag','IntensityColorMapEditBox');
StripPipelineCheckboxHandle = findobj('Tag','StripPipelineCheckbox');
SkipErrorsCheckboxHandle = findobj('Tag','SkipErrorCheckbox');
LabelColorMapHandle = findobj('Tag','LabelColorMapEditBox');
SelectDisplayModeHandle = findobj('Tag','SelectDisplay');
PixelSize = get(PixelSizeEditBoxHandle,'string');
PixelSize = PixelSize{1};
FontSize = get(FontSizeEditBoxHandle,'string');
DefaultImageDirectory = get(ImageDirEditBoxHandle,'string');
DefaultOutputDirectory = get(OutputDirEditBoxHandle,'string');
DefaultModuleDirectory = get(ModuleDirEditBoxHandle,'string');
IntensityColorMap = get(IntensityColorMapHandle,'string');
LabelColorMap = get(LabelColorMapHandle,'string');
DisplayModeValue = get(SelectDisplayModeHandle,'value');
if get(StripPipelineCheckboxHandle,'Value') == get(StripPipelineCheckboxHandle,'Max')
    StripPipeline = 'Yes';
else
    StripPipeline = 'No';
end
if get(SkipErrorsCheckboxHandle,'Value') == get(SkipErrorsCheckboxHandle,'Max')
    SkipErrors = 'Yes';
else
    SkipErrors = 'No';
end

EnteredPreferences.PixelSize = PixelSize;
EnteredPreferences.FontSize = FontSize;
EnteredPreferences.DefaultImageDirectory = DefaultImageDirectory;
EnteredPreferences.DefaultOutputDirectory = DefaultOutputDirectory;
EnteredPreferences.DefaultModuleDirectory = DefaultModuleDirectory;
EnteredPreferences.IntensityColorMap = IntensityColorMap;
EnteredPreferences.LabelColorMap = LabelColorMap;
EnteredPreferences.StripPipeline = StripPipeline;
EnteredPreferences.SkipErrors = SkipErrors;
EnteredPreferences.DisplayModeValue = DisplayModeValue;
SavedPreferences = EnteredPreferences; %#ok ignore MLint
CurrentDir = pwd;
try
    save(FullFileName,'SavedPreferences')
    clear SavedPreferences
    if DefaultVal == 1;
        CPhelpdlg(['Your CellProfiler preferences were successfully set.  They are contained in a file called CellProfilerPreferences.mat in the directory ', fileparts(FullFileName)])
    else
        CPhelpdlg('Your CellProfiler preferences were successfully set.')
    end
catch
    try
        save(fullfile(CurrentDir,FileName),'SavedPreferences')
        clear SavedPreferences
        CPhelpdlg('You do not have permission to write anything to the Matlab root directory.  Instead, your default preferences will only function properly when you start CellProfiler from the current directory.')
    catch
        CPhelpdlg('CellProfiler was unable to save your desired preferences, probably because you lack write permission for both the Matlab root directory as well as the current directory.  Your preferences will only be saved for the current session of CellProfiler.');
    end
end
clear PixelSize* *Dir* , close(SetPreferencesWindowHandle);
clear SetPreferencesWindowHandle FontSize FontSizeEditBoxHandle;

function SetPreferences_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.

%%% Creates a global variable to be used later.
global EnteredPreferences

%%% Opens a dialog box to retrieve input from the user.
%%% Sets the functions of the buttons and edit boxes in the dialog box.

ImageDirBrowseButtonCallback = 'EditBoxHandle = findobj(''Tag'',''ImageDirEditBox''); CurrentChoice = get(EditBoxHandle,''string''); if exist(CurrentChoice, ''dir''), tempdir = CurrentChoice; else, tempdir=pwd; end, DefaultImageDirectory = uigetdir(tempdir,''Select the default image directory''); pause(.1);figure(findobj(''Tag'',''figure1''));figure(findobj(''Tag'',''SetPreferenceWindow''));if DefaultImageDirectory == 0, else set(EditBoxHandle,''string'', DefaultImageDirectory), end, clear EditBoxHandle CurrentChoice tempdir DefaultImageDirectory';
ImageDirEditBoxCallback = 'DefaultImageDirectory = get(gco,''string''); if(~isdir(DefaultImageDirectory)); warndlg(''That is not a valid directory'');end;if isempty(DefaultImageDirectory); DefaultImageDirectory = pwd; set(gco,''string'',DefaultImageDirectory); end, clear';
OutputDirBrowseButtonCallback = 'EditBoxHandle = findobj(''Tag'',''OutputDirEditBox''); CurrentChoice = get(EditBoxHandle,''string''); if exist(CurrentChoice, ''dir''), tempdir=CurrentChoice; else, tempdir=pwd; end, DefaultOutputDirectory = uigetdir(tempdir,''Select the default output directory''); pause(.1);figure(findobj(''Tag'',''figure1''));figure(findobj(''Tag'',''SetPreferenceWindow''));if DefaultOutputDirectory == 0, else set(EditBoxHandle,''string'', DefaultOutputDirectory), end, clear EditBoxHandle CurrentChoice tempdir DefaultOutputDirectory';
OutputDirEditBoxCallback = 'DefaultOutputDirectory = get(gco,''string''); if(~isdir(DefaultOutputDirectory)); warndlg(''That is not a valid directory'');end;if isempty(DefaultOutputDirectory) == 1; DefaultOutputDirectory = pwd; set(gco,''string'',DefaultOutputDirectory), end, clear';
ModuleDirBrowseButtonCallback = 'EditBoxHandle = findobj(''Tag'',''ModuleDirEditBox''); CurrentChoice = get(EditBoxHandle,''string''); if exist(CurrentChoice, ''dir''), tempdir=CurrentChoice; else tempdir=pwd; end, DefaultModuleDirectory = uigetdir(tempdir,''Select the directory where modules are stored''); pause(.1);figure(findobj(''Tag'',''figure1''));figure(findobj(''Tag'',''SetPreferenceWindow''));if DefaultModuleDirectory == 0, else set(EditBoxHandle,''string'', DefaultModuleDirectory), end, clear EditBoxHandle CurrentChoice tempdir DefaultModuleDirectory';
ModuleDirEditBoxCallback = 'DefaultModuleDirectory = get(gco,''string''); if(~isdir(DefaultModuleDirectory)); warndlg(''That is not a valid directory'');end;if isempty(DefaultModuleDirectory) == 1; DefaultModuleDirectory = pwd; set(gco,''string'',DefaultModuleDirectory), end, clear';

CancelButtonCallback = 'delete(gcf)';

%%% Creates the dialog box and its text, buttons, and edit boxes.
MainWinPos = get(handles.figure1,'Position');
Color = [0.7 0.7 0.9];

%%% Label we attach to figures (as UserData) so we know they are ours
userData.Application = 'CellProfiler';
userData.MyHandles=handles;
SetPreferencesWindowHandle = figure(...
    'Units','pixels',...
    'Color',Color,...
    'DockControls','off',...
    'MenuBar','none',...
    'Name','SetPreferences',...
    'NumberTitle','off',...
    'Position',[MainWinPos(1)+MainWinPos(3)/10 MainWinPos(2) MainWinPos(3)*(4/5) MainWinPos(4)+50],...
    'Resize','off',...
    'HandleVisibility','on',...
    'Tag','SetPreferenceWindow',...
    'UserData',userData);

Option = [];

for i = 1:length(handles.Current.HelpFilenames),
    if strfind(handles.Current.HelpFilenames{i},'HelpPreferences')
        Option = i;
    end
end
if ~isempty(Option)
    StringForInfoText = handles.Current.Help{Option-1};
else
    StringForInfoText = 'See Help > General Help > Help Preferences for more information';
end

InfoText = uicontrol(...
    'Parent',SetPreferencesWindowHandle,...
    'Units','normalized',...
    'BackgroundColor',Color,...
    'FontName','Helvetica',...
    'FontSize',handles.Preferences.FontSize,...
    'FontWeight','bold',...
    'HorizontalAlignment','left',...
    'Position',[0.025 0.63 0.95 0.35],...
    'String',StringForInfoText,...
    'Style','text'); %#ok Ignore MLint

IntensityColorMapText = uicontrol(...
    'Parent',SetPreferencesWindowHandle,...
    'Units','normalized',...
    'BackgroundColor',Color,...
    'FontName','Helvetica',...
    'FontSize',handles.Preferences.FontSize,...
    'FontWeight','bold',...
    'HorizontalAlignment','left',...
    'Position',[.2 .7 .6 .04],...
    'String','Enter the default colormap for intensity images',...
    'Style','text'); %#ok Ignore MLint

IntensityColorMapEditBox = uicontrol(...
    'Parent',SetPreferencesWindowHandle,...
    'Units','normalized',...
    'BackgroundColor',[1 1 1],...
    'Callback','cmap=get(gcbo,''String''); try, colormap(cmap);catch, warndlg(''That is not a valid entry'');end;clear cmap',...
    'FontName','Helvetica',...
    'FontSize',handles.Preferences.FontSize,...
    'FontWeight','bold',...
    'Position',[0.7 0.7 0.1 0.04],...
    'String',handles.Preferences.IntensityColorMap,...
    'Style','edit',...
    'Tag','IntensityColorMapEditBox'); %#ok Ignore MLint

LabelColorMapText = uicontrol(...
    'Parent',SetPreferencesWindowHandle,...
    'Units','normalized',...
    'BackgroundColor',Color,...
    'FontName','Helvetica',...
    'FontSize',handles.Preferences.FontSize,...
    'FontWeight','bold',...
    'HorizontalAlignment','left',...
    'Position',[.2 .65 .6 .04],...
    'String','Enter the default colormap for objects',...
    'Style','text'); %#ok Ignore MLint

LabelColorMapEditBox = uicontrol(...
    'Parent',SetPreferencesWindowHandle,...
    'Units','normalized',...
    'BackgroundColor',[1 1 1],...
    'Callback','cmap=get(gcbo,''String''); try, colormap(cmap);catch, warndlg(''That is not a valid entry'');end;clear cmap',...
    'FontName','Helvetica',...
    'FontSize',handles.Preferences.FontSize,...
    'FontWeight','bold',...
    'Position',[0.7 0.65 0.1 0.04],...
    'String',handles.Preferences.LabelColorMap,...
    'Style','edit',...
    'Tag','LabelColorMapEditBox'); %#ok Ignore MLint

ColorMapHelp = uicontrol(...
    'Parent',SetPreferencesWindowHandle,...
    'Units','normalized',...
    'BackgroundColor',Color,...
    'Callback','handles = guidata(findobj(''tag'',''figure1''));for i = 1:length(handles.Current.HelpFilenames), if strfind(handles.Current.HelpFilenames{i},''HelpColormaps''), Option = i;end,end,if ~isempty(Option),CPtextdisplaybox(handles.Current.Help{Option-1},''Colormaps Help'');end;clear ans handles Option i;',...
    'FontWeight','bold',...
    'Position',[0.18 0.71 0.02 0.04],...
    'String','?',...
    'Tag','ColorMapHelp',...
    'Behavior',get(0,'defaultuicontrolBehavior')); %#ok Ignore MLint

PixelSizeHelp = uicontrol(...
    'Parent',SetPreferencesWindowHandle,...
    'Units','normalized',...
    'BackgroundColor',Color,...
    'Callback','handles = guidata(findobj(''tag'',''figure1''));for i = 1:length(handles.Current.HelpFilenames), if strfind(handles.Current.HelpFilenames{i},''HelpPixelSize''), Option = i;end,end,if ~isempty(Option),CPtextdisplaybox(handles.Current.Help{Option-1},''Pixel Size Help'');end;clear ans handles Option i;',...
    'FontWeight','bold',...
    'Position',[0.18 0.61 0.02 0.04],...
    'String','?',...
    'Tag','PixelSizeHelp',...
    'Behavior',get(0,'defaultuicontrolBehavior')); %#ok Ignore MLint

PixelSizeText = uicontrol(...
    'Parent',SetPreferencesWindowHandle,...
    'Units','normalized',...
    'BackgroundColor',Color,...
    'FontName','Helvetica',...
    'FontSize',handles.Preferences.FontSize,...
    'FontWeight','bold',...
    'HorizontalAlignment','left',...
    'Position',[0.2 0.6 0.6 0.04],...
    'String','Enter the default pixel size (in micrometers)',...
    'Style','text'); %#ok Ignore MLint

PixelSizeEditBox = uicontrol(...
    'Parent',SetPreferencesWindowHandle,...
    'Units','normalized',...
    'BackgroundColor',[1 1 1],...
    'Callback','val=str2double(get(gcbo,''String'')); if(isnan(val)||(val<=0)); warndlg(''That is not a valid entry'');end;clear val',...
    'FontName','Helvetica',...
    'FontSize',handles.Preferences.FontSize,...
    'FontWeight','bold',...
    'Position',[0.7 0.6 0.1 0.04],...
    'String',handles.Preferences.PixelSize,...
    'Style','edit',...
    'Tag','PixelSizeEditBox'); %#ok Ignore MLint

FontSizeText = uicontrol(...
    'Parent',SetPreferencesWindowHandle,...
    'Units','normalized',...
    'BackgroundColor',Color,...
    'FontName','Helvetica',...
    'FontSize',handles.Preferences.FontSize,...
    'FontWeight','bold',...
    'HorizontalAlignment','left',...
    'Position',[0.2 0.55 0.6 0.04],...
    'String','Enter the default font size',...
    'Style','text'); %#ok Ignore MLint

FontSizeEditBox = uicontrol(...
    'Parent',SetPreferencesWindowHandle,...
    'Units','normalized',...
    'BackgroundColor',[1 1 1],...
    'Callback','val=str2double(get(gcbo,''String'')); if(isnan(val)||(val<=5)||(val>=18)); warndlg(''That is not a valid entry'');end;clear val',...
    'FontName','Helvetica',...
    'FontSize',handles.Preferences.FontSize,...
    'FontWeight','bold',...
    'Position',[0.7 0.55 0.1 0.04],...
    'String',num2str(round(handles.Preferences.FontSize)),...
    'Style','edit',...
    'Tag','FontSizeEditBox'); %#ok Ignore MLint

FastModeText = uicontrol(...
    'Parent',SetPreferencesWindowHandle,...
    'Units','normalized',...
    'BackgroundColor',Color,...
    'FontName','Helvetica',...
    'FontSize',handles.Preferences.FontSize,...
    'FontWeight','bold',...
    'HorizontalAlignment','left',...
    'Position',[0.2 0.5 0.6 0.04],...
    'String','Run in fast mode:',...
    'Style','text'); %#ok Ignore MLint

FastModeHelp = uicontrol(...
    'Parent',SetPreferencesWindowHandle,...
    'Units','normalized',...
    'BackgroundColor',Color,...
    'Callback','handles = guidata(findobj(''tag'',''figure1''));for i = 1:length(handles.Current.HelpFilenames), if strfind(handles.Current.HelpFilenames{i},''HelpFastMode''), Option = i;end,end,if ~isempty(Option),CPtextdisplaybox(handles.Current.Help{Option-1},''Fast Mode Help'');end;clear ans handles Option i;',...
    'FontWeight','bold',...
    'Position',[0.18 0.51 0.02 0.04],...
    'String','?',...
    'Tag','FastModeHelp',...
    'Behavior',get(0,'defaultuicontrolBehavior')); %#ok Ignore MLint

FastModeCheckbox = uicontrol(...
    'Parent',SetPreferencesWindowHandle,...
    'Units','normalized',...
    'BackgroundColor',Color,...
    'Min',0,...
    'Max',1,...
    'Position',[.7 .5 .04 .04],...
    'Style','checkbox',...
    'Tag','StripPipelineCheckbox',...
    'Value',strcmp(handles.Preferences.StripPipeline,'Yes')); %#ok Ignore MLint

SkipErrorText = uicontrol(...
    'Parent',SetPreferencesWindowHandle,...
    'Units','normalized',...
    'BackgroundColor',Color,...
    'FontName','Helvetica',...
    'FontSize',handles.Preferences.FontSize,...
    'FontWeight','bold',...
    'HorizontalAlignment','left',...
    'Position',[0.2 0.45 0.6 0.04],...
    'String','Skip modules which fail:',...
    'Style','text'); %#ok Ignore MLint

SkipErrorHelp = uicontrol(...
    'Parent',SetPreferencesWindowHandle,...
    'Units','normalized',...
    'BackgroundColor',Color,...
    'Callback','handles = guidata(findobj(''tag'',''figure1''));for i = 1:length(handles.Current.HelpFilenames), if strfind(handles.Current.HelpFilenames{i},''HelpSkipErrors''), Option = i;end,end,if ~isempty(Option),CPtextdisplaybox(handles.Current.Help{Option-1},''Skip Errors Help'');end;clear ans handles Option i;',...
    'FontWeight','bold',...
    'Position',[0.18 0.46 0.02 0.04],...
    'String','?',...
    'Tag','SkipErrorHelp',...
    'Behavior',get(0,'defaultuicontrolBehavior')); %#ok Ignore MLint

SkipErrorCheckbox = uicontrol(...
    'Parent',SetPreferencesWindowHandle,...
    'Units','normalized',...
    'BackgroundColor',Color,...
    'Min',0,...
    'Max',1,...
    'Position',[.7 .45 .04 .04],...
    'Style','checkbox',...
    'Tag','SkipErrorCheckbox',...
    'Value',strcmp(handles.Preferences.SkipErrors,'Yes')); %#ok Ignore MLint

SelectDisplayModeText = uicontrol(...
    'Parent',SetPreferencesWindowHandle,...
    'Units','normalized',...
    'BackgroundColor',Color,...
    'FontName','Helvetica',...
    'FontSize',handles.Preferences.FontSize,...
    'FontWeight','bold',...
    'HorizontalAlignment','left',...
    'Position',[0.2 0.4 0.6 0.04],...
    'String','Display Mode:',...
    'Style','text'); %#ok Ignore MLint

SelectDisplayMode = uicontrol(...
    'Parent',SetPreferencesWindowHandle,...
    'Style', 'popupmenu',...
    'BackgroundColor',[1 1 1],...
    'String',{'Display all windows', 'Do not display any windows', 'Specify windows to display'},...
    'Units','normalized',...
    'FontName','Helvetica',...
    'FontSize',handles.Preferences.FontSize,...
    'FontWeight','bold',...
    'Position',[0.5 0.4 0.4 0.05],...
    'Tag','SelectDisplay',...
    'value',handles.Preferences.DisplayModeValue); %#ok Ignore MLint

ImageDirTextBox = uicontrol(...
    'Parent',SetPreferencesWindowHandle,...
    'Units','normalized',...
    'BackgroundColor',Color,...
    'FontName','Helvetica',...
    'FontSize',handles.Preferences.FontSize,...
    'FontWeight','bold',...
    'HorizontalAlignment','left',...
    'Position',[0.025 0.35 0.6 0.04],...
    'String','Select the default image folder:',...
    'Style','text'); %#ok Ignore MLint

ImageDirBrowseButton = uicontrol(...
    'Parent',SetPreferencesWindowHandle,...
    'Units','normalized',...
    'Callback',ImageDirBrowseButtonCallback,...
    'FontName','Helvetica',...
    'FontSize',handles.Preferences.FontSize,...
    'FontWeight','bold',...
    'Position',[0.85 0.31 0.12 0.05],...
    'String','Browse...',...
    'Tag','ImageDirBrowseButton',...
    'BackgroundColor',Color); %#ok Ignore MLint

ImageDirEditBox = uicontrol(...
    'Parent',SetPreferencesWindowHandle,...
    'Units','normalized',...
    'BackgroundColor',[1 1 1],...
    'Callback',ImageDirEditBoxCallback,...
    'FontName','Helvetica',...
    'FontSize',handles.Preferences.FontSize,...
    'FontWeight','bold',...
    'Position',[0.025 0.31 0.8 0.05],...
    'String',handles.Preferences.DefaultImageDirectory,...
    'Style','edit',...
    'Tag','ImageDirEditBox'); %#ok Ignore MLint

OutputDirTextBox = uicontrol(...
    'Parent',SetPreferencesWindowHandle,...
    'Units','normalized',...
    'BackgroundColor',Color,...
    'FontName','Helvetica',...
    'FontSize',handles.Preferences.FontSize,...
    'FontWeight','bold',...
    'HorizontalAlignment','left',...
    'Position',[0.025 0.25 0.6 0.04],...
    'String','Select the default output folder:',...
    'Style','text'); %#ok Ignore MLint

OutputDirBrowseButton = uicontrol(...
    'Parent',SetPreferencesWindowHandle,...
    'Units','normalized',...
    'Callback',OutputDirBrowseButtonCallback,...
    'FontName','Helvetica',...
    'FontSize',handles.Preferences.FontSize,...
    'FontWeight','bold',...
    'Position',[0.85 0.21 0.12 0.05],...
    'String','Browse...',...
    'Tag','OutputDirBrowseButton',...
    'BackgroundColor',Color); %#ok Ignore MLint

OutputDirEditBox = uicontrol(...
    'Parent',SetPreferencesWindowHandle,...
    'Units','normalized',...
    'BackgroundColor',[1 1 1],...
    'Callback',OutputDirEditBoxCallback,...
    'FontName','Helvetica',...
    'FontSize',handles.Preferences.FontSize,...
    'FontWeight','bold',...
    'Position',[0.025 0.21 0.8 0.05],...
    'String',handles.Preferences.DefaultOutputDirectory,...
    'Style','edit',...
    'Tag','OutputDirEditBox'); %#ok Ignore MLint

ModuleDirTextBox = uicontrol(...
    'Parent',SetPreferencesWindowHandle,...
    'Units','normalized',...
    'BackgroundColor',Color,...
    'FontName','Helvetica',...
    'FontSize',handles.Preferences.FontSize,...
    'FontWeight','bold',...
    'HorizontalAlignment','left',...
    'Position',[0.025 0.15 0.6 0.04],...
    'String','Select the folder where CellProfiler modules are stored:',...
    'Style','text'); %#ok Ignore MLint

ModuleDirBrowseButton = uicontrol(...
    'Parent',SetPreferencesWindowHandle,...
    'Units','normalized',...
    'Callback',ModuleDirBrowseButtonCallback,...
    'FontName','Helvetica',...
    'FontSize',handles.Preferences.FontSize,...
    'FontWeight','bold',...
    'Position',[0.85 0.11 0.12 0.05],...
    'String','Browse...',...
    'Tag','ModuleDirBrowseButton',...
    'BackgroundColor',Color); %#ok Ignore MLint

ModuleDirEditBox = uicontrol(...
    'Parent',SetPreferencesWindowHandle,...
    'Units','normalized',...
    'BackgroundColor',[1 1 1],...
    'Callback',ModuleDirEditBoxCallback,...
    'FontName','Helvetica',...
    'FontSize',handles.Preferences.FontSize,...
    'FontWeight','bold',...
    'Position',[0.025 0.11 0.8 0.05],...
    'String',handles.Preferences.DefaultModuleDirectory,...
    'Style','edit',...
    'Tag','ModuleDirEditBox'); %#ok Ignore MLint

SaveButton = uicontrol(...
    'Parent',SetPreferencesWindowHandle,...
    'Units','normalized',...
    'Callback','CellProfiler(''SaveButton_Callback'',gcbo,[],guidata(gcbo))',...
    'FontName','Helvetica',...
    'FontSize',handles.Preferences.FontSize,...
    'FontWeight','bold',...
    'Position',[0.2 0.02 0.2 0.06],...
    'String','Save preferences',...
    'Tag','SaveButton',...
    'BackgroundColor',Color); %#ok Ignore MLint

CancelButton = uicontrol(...
    'Parent',SetPreferencesWindowHandle,...
    'Units','normalized',...
    'Callback',CancelButtonCallback,...
    'FontName','Helvetica',...
    'FontSize',handles.Preferences.FontSize,...
    'FontWeight','bold',...
    'Position',[0.6 0.02 0.2 0.06],...
    'String','Cancel',...
    'Tag','CancelButton',...
    'BackgroundColor',Color); %#ok Ignore MLint

%%% Waits for the user to respond to the window.
uiwait(SetPreferencesWindowHandle)
%%% Allows canceling by checking whether EnteredPreferences exists.
LoadPreferences_Helper(hObject,eventdata,handles,EnteredPreferences);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% TECHNICAL DIAGNOSIS BUTTON %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function TechnicalDiagnosis_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
%%% This pauses execution and allows the user to type things in at the
%%% matlab prompt.  You can check the current variables and they will show
%%% up in the workspace.

CPmsgbox('Type ''return'' in the MATLAB prompt (where the K>> is) to stop diagnosis mode');
display('Type ''return'' in the MATLAB prompt (where the K>> is) to stop diagnosis mode');
%%% TYPE "return" TO STOP.
keyboard;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% BROWSE DEFAULT IMAGE DIRECTORY BUTTON %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes on button press in BrowseImageDirectoryButton.
function BrowseImageDirectoryButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
%%% Opens a dialog box to allow the user to choose a directory and loads
%%% that directory name into the edit box.  Also, changes the current
%%% directory to the chosen directory.
% if exist(handles.Current.DefaultImageDirectory, 'dir')
%     pathname = uigetdir(handles.Current.DefaultImageDirectory,'Choose the directory of images to be analyzed');
%     pause(.1);
%     figure(handles.figure1);
% else
%     pathname = uigetdir('','Choose the directory of images to be analyzed');
%     pause(.1);
%     figure(handles.figure1);
% end
%%% The if/else statement is removed because if the directory doesn't
%%% exist, matlab automatically recovers (see uigetdir for details).
%%% By contrast, using uigetdir with the starting path '' (empty
%%% string) failed on the mac platform, even though it's not supposed
%%% to.
pathname = uigetdir(handles.Current.DefaultImageDirectory,'Choose the directory of images to be analyzed');
pause(.1);
figure(handles.figure1);

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% DEFAULT IMAGE DIRECTORY EDIT BOX %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function handles = DefaultImageDirectoryEditBox_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
%%% Retrieves the text that was typed in.
pathname = get(handles.DefaultImageDirectoryEditBox,'string');
%%% Checks whether a directory with that name exists.
if exist(pathname,'dir')
    %%% Saves the pathname in the handles structure.
    handles.Current.DefaultImageDirectory = pathname;
    guidata(hObject,handles)
    %%% Retrieves the list of image file names from the chosen directory and
    %%% stores them in the handles structure, using the function
    %%% RetrieveImageFileNames.
    FileNames = CPretrievemediafilenames(pathname,'','No','Exact','Both');
    %%% Test whether this is during CellProfiler launching or during
    %%% the image analysis run itself (by looking at some of the GUI
    %%% elements). If either is the case, the message is NOT
    %%% shown.
    handles.Current.FilenamesInImageDir = FileNames;
    ListBoxContents = get(handles.FilenamesListBox,'String');
    IsStartup = strcmp(ListBoxContents(1),'Listbox');
    IsAnalysisRun = strcmp(get(handles.AnalyzeImagesButton,'enable'),'off');
    if any([IsStartup, IsAnalysisRun]) == 0 && isempty(handles.Current.FilenamesInImageDir) == 1;
        CPmsgbox('Please note: there are no recognizable files in the default image folder.','Default Image Folder');
    end
    guidata(hObject, handles);
    %%% If the directory entered in the box does not exist, give an error
    %%% message, change the contents of the edit box back to the
    %%% previously selected directory, and change the contents of the
    %%% filenameslistbox back to the previously selected directory.
else
    CPerrordlg('A directory with that name does not exist');
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% BROWSE DEFAULT OUTPUT DIRECTORY BUTTON %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes on button press in BrowseOutputDirectoryButton.
function BrowseOutputDirectoryButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.

%%% Opens a dialog box to allow the user to choose a directory and loads
%%% that directory name into the edit box.  Also, changes the current
%%% directory to the chosen directory.
pathname = uigetdir(handles.Current.DefaultOutputDirectory,'Choose the default output directory');
pause(.1);
figure(handles.figure1);
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% DEFAULT OUTPUT DIRECTORY EDIT BOX %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function DefaultOutputDirectoryEditBox_Callback(hObject, eventdata, handles) %#ok Ignore MLint
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
    CPerrordlg('A directory with that name does not exist');
end
%%% Whether or not the directory exists and was updated, we want to
%%% update the GUI display to show the currrently stored information.
%%% Display the path in the edit box.
set(handles.DefaultOutputDirectoryEditBox,'String',handles.Current.DefaultOutputDirectory);
%%% Updates the handles structure.
guidata(hObject,handles)

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE LIST BOX %%%
%%%%%%%%%%%%%%%%%%%%%%

% --- Executes on selection change in FilenamesListBox.
function FilenamesListBox_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.

if strcmp(get(gcf,'SelectionType'),'open')
    Val = get(handles.FilenamesListBox,'value');
    String = get(handles.FilenamesListBox,'string');
    %%% Check if there are images
    if strcmpi(String,'No image files recognized')
        return
    end
    FileName = char(String(Val));
    PathName = get(handles.DefaultImageDirectoryEditBox,'string');
    if strcmpi(FileName(end-3:end),'.mat')
        test = load(fullfile(PathName,FileName));
        if isfield(test,'Settings') || isfield(test,'handles')
            Answer = CPquestdlg('Would you like to load the pipeline from this file?','Confirm','Yes','No','Yes');
            if strcmp(Answer,'Yes')
                eventdata.SettingsPathname = PathName;
                eventdata.SettingsFileName = FileName;
                LoadPipeline_Callback(hObject,eventdata,handles);
            end
        elseif isfield(test,'Image')
            try
                %%% Reads the image.
                Image = CPimread(fullfile(PathName, FileName));
                CPfigure(handles,'image','name',FileName);
                CPimagesc(Image,handles);
                colormap(gray); % is this needed/correct? CPfigure sets the default intensity colormap. CPimagesc does too. What if it's a label image?
                FileName = strrep(FileName,'_','\_');
                title(FileName);
            catch CPerrordlg('There was an error opening this file. It is possible that it is not an image, figure, pipeline file, or output file.');
            end
        elseif isfield(test,'SavedPreferences')
            EnteredPreferences = test.SavedPreferences;
            LoadPreferences_Helper(hObject,eventdata,handles,EnteredPreferences);
        else
            CPerrordlg('This mat file is not a proper pipeline or output file.');
        end
    elseif strcmpi(FileName(end-3:end),'.fig')
        open(fullfile(PathName,FileName));
    else
        try
            %%% Reads the image.
            Image = CPimread(fullfile(PathName, FileName));
            CPfigure(handles,'image','name',FileName);
            CPimagesc(Image,handles);
            colormap(gray); % is this needed/correct? CPfigure sets the default intensity colormap. CPimagesc does too. What if it's a label image?
            FileName = strrep(FileName,'_','\_');
            title(FileName);
        catch CPerrordlg('There was an error opening this file. It is possible that it is not an image, figure, pipeline file, or output file.');
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% CLOSE WINDOWS BUTTON %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function CloseWindows_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.

%%% Requests confirmation to really delete all the figure windows.
Answer = CPquestdlg('Are you sure you want to close all figure windows, timers, and message boxes that CellProfiler created?','Confirm','Yes','No','Yes');
if strcmp(Answer, 'Yes')
    %%% Run the CloseWindows_Helper function
    CloseWindows_Helper(hObject, eventdata, handles);
end


% --- CloseWindows_Helper function was called because it is called from two
% separate places...from the close windows button and when the user quits
% CellProfiler
function CloseWindows_Helper(hObject, eventdata, handles) %#ok Ignore MLint
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
                ischar(userData.Application) && ...
                strcmp(userData.Application, 'CellProfiler'))
            %%% Closes the figure windows.
            try
                delete(GraphicsHandles(k));
            catch
                CPmsgbox('There was a problem closing some windows.');
            end
        end
    end
end
%%% Finds and closes timer windows, which have HandleVisibility off.
TimerHandles = findall(findobj, 'Name', 'Status');
try
    delete(TimerHandles);
    delete(timerfind);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% OUTPUT FILE NAME EDIT BOX %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
                numbers = [numbers str2double(d(k).name(index(end)+length(UserEntry)+2:end-4))];
            end
        end
        if isempty(numbers)
            outputnumber = 1;
        else
            outputnumber = max(numbers) + 1;
        end
        set(handles.OutputFileNameEditBox,'string',sprintf('%s__%d.mat',UserEntry,outputnumber))
    else
        set(handles.OutputFileNameEditBox,'string',[UserEntry '.mat'])
    end
    drawnow
end
guidata(gcbo, handles);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% ANALYZE IMAGES BUTTON %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes on button press in AnalyzeImagesButton.
function AnalyzeImagesButton_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
global closeFigures openFigures;
%%% Checks whether any modules are loaded.
total = 0; %%% Initial value.
for i = 1:handles.Current.NumberOfModules;
    total = total + iscellstr(handles.Settings.ModuleNames(i));
end
if total == 0
    CPerrordlg('You do not have any analysis modules loaded');
else
    %%% Call Callback function of FileNameEditBox to update filename
    tmp = get(handles.OutputFileNameEditBox,'string');
    OutputFileNameEditBox_Callback(hObject, eventdata, handles)
    if ~strcmp(tmp,get(handles.OutputFileNameEditBox,'string'))
        Answer = CPquestdlg('The output file already exists. A new file name has been generated. Continue?','Output file exists','Yes','Cancel','Yes'); %When closing this dialog box, it assumes 'Yes' was chosen
        if ~strcmp(Answer,'Yes')
            set(handles.OutputFileNameEditBox,'string',tmp)
            return
        end
    end

    %%% Checks whether an output file name has been specified.
    if isempty(get(handles.OutputFileNameEditBox,'string'))
        CPerrordlg('You have not entered an output file name in Step 2.');
    else
        %%% Checks whether the default output directory exists.
        if ~exist(handles.Current.DefaultOutputDirectory, 'dir')
            CPerrordlg('The default output folder does not exist');
        end
        %%% Checks whether the default image directory exists
        if ~exist(handles.Current.DefaultImageDirectory, 'dir')
            CPerrordlg('The default image folder does not exist');
        else
            try
                handles.Preferences = rmfield(handles.Preferences,'DisplayWindows');
            end
            if handles.Preferences.DisplayModeValue == 2
                handles.Preferences.DisplayWindows = zeros(handles.Current.NumberOfModules,1);
            elseif handles.Preferences.DisplayModeValue == 3
                try
                    handles.Preferences.DisplayWindows = CPselectmodules(handles.Settings.ModuleNames);
                catch
                    CPerrordlg('An error occurred while selecting the modules you wanted to display. All modules will be displayed.');
                    handles.Preferences.DisplayWindows = ones(handles.Current.NumberOfModules,1);
                end
            else
                handles.Preferences.DisplayWindows= ones(handles.Current.NumberOfModules,1);
            end

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
            set(handles.IndividualModulesText,'visible','off')
            set(handles.AddModule,'visible','off');
            set(handles.RemoveModule,'visible','off');
            set(handles.MoveUpButton,'visible','off');
            set(handles.MoveDownButton,'visible','off');
            set(handles.PixelSizeEditBox,'enable','inactive','foregroundcolor',[0.7,0.7,0.7])
            set(handles.BrowseImageDirectoryButton,'enable','off')
            set(handles.DefaultImageDirectoryEditBox,'enable','inactive','foregroundcolor',[0.7,0.7,0.7])
            set(handles.BrowseOutputDirectoryButton,'enable','off')
            set(handles.DefaultOutputDirectoryEditBox,'enable','inactive','foregroundcolor',[0.7,0.7,0.7])
            set(handles.OutputFileNameEditBox,'enable','inactive','foregroundcolor',[0.7,0.7,0.7])
            set(handles.AnalyzeImagesButton,'enable','off')
            set(cat(2,handles.VariableBox{:}),'enable','inactive','foregroundcolor',[0.7,0.7,0.7]);

            %%% In the following code, the Timer window and
            %%% timer_text is created.  Each time around the loop,
            %%% the text will be updated using the string property.

            %%% Obtains the screen size.
            ScreenSize = get(0,'ScreenSize');
            ScreenWidth = ScreenSize(3);
            ScreenHeight = ScreenSize(4);

            %%% Determines where to place the timer window: We want it below the image
            %%% windows, which means at about 800 pixels from the top of the screen,
            %%% but in case the screen doesn't have that many pixels, we don't want it
            %%% to be below zero.
            PotentialBottom = [0, (ScreenHeight-800)];
            BottomOfTimer = max(PotentialBottom);
            %%% Creates the Timer window.
            %%% Label we attach to figures (as UserData) so we know they are ours
            userData.Application = 'CellProfiler';
            userData.MyHandles = handles;
            timerFig = figure('name','Status','position',[0 BottomOfTimer 350 120],...
                'menubar','none','NumberTitle','off','IntegerHandle','off', 'HandleVisibility', 'off', ...
                'color',[0.7,0.7,0.9],'UserData',userData,'Resize','off');
            TimerData.timerFig = timerFig;
            TimerData.SetBeingAnalyzed = 1;
            TimerData.NumberOfImageSets = 1;
            TimerData.StartingImageSet = 1;
            TimerData.TimerTime = 0;
            TimerData.FontSize = handles.Preferences.FontSize;
            TimerData.NumberOfModules = handles.Current.NumberOfModules;

            delete(timerfind);
            timer_handle = timer('StartFcn','tic','period',1,'ExecutionMode','fixedRate','tag','CellProfilerTimer');

            Timer_Callback = ['Timers=timerfind(''Tag'',''CellProfilerTimer'');',...
                'TimerData = get(Timers(1),''UserData'');',...
                'if strncmpi(get(TimerData.timertexthandle,''string''),''Cancel'',6),',...
                'return;',...
                'end;'...
                'time_elapsed = round(toc*10)/10;',...
                'if (time_elapsed > 60),',...
                'if (time_elapsed > 3600),',...
                'timer_elapsed_text =  [''Time elapsed = '',num2str(floor(time_elapsed/3600)),'':'',sprintf(''%02.0f'',abs(mod(time_elapsed,3600))/60),'':'',sprintf(''%02.0f'',abs(mod(time_elapsed,60)))];',...
                'else,',...
                'timer_elapsed_text =  [''Time elapsed = '',num2str(floor(time_elapsed/60)),'':'',sprintf(''%02.0f'',abs(mod(time_elapsed,60)))];',...
                'end;',...
                'else,',...
                'timer_elapsed_text =  [''Time elapsed = '',num2str(floor(time_elapsed))];',...
                'end;',...
                'number_analyzed = [''Number of cycles completed = '',num2str(TimerData.SetBeingAnalyzed-1), '' of '', num2str(TimerData.NumberOfImageSets)];',...
                'if TimerData.SetBeingAnalyzed > TimerData.StartingImageSet,',...
                'time_set1 = [''Time for first cycle (seconds) = '', num2str(round(10*sum(TimerData.TimerTime(:,1)))/10)];',...
                'else,',...
                'time_set1 = '' '';',...
                'end;',...
                'if TimerData.SetBeingAnalyzed > TimerData.StartingImageSet + 1,',...
                'time_per_set = [''Time per cycle (seconds) = '', num2str(round(10*(sum(sum(TimerData.TimerTime(:,2:(TimerData.SetBeingAnalyzed-1)))))/(TimerData.SetBeingAnalyzed-TimerData.StartingImageSet-1))/10)];',...
                'else, time_per_set = '' '';',...
                'end;',...
                'timertext = {timer_elapsed_text; number_analyzed; time_set1; time_per_set};',...
                'set(TimerData.timertexthandle,''string'',timertext);',...
                'clear Timers TimerData number_analyzed time_elapsed time_per_set time_set1 timer_elapsed_text timertext;'];

            TimerStopFcn = 'Timers=timerfind(''Tag'',''CellProfilerTimer'');TimerData = get(Timers(1),''UserData'');set(TimerData.timertexthandle,''String'',''Image analysis is complete'');set(TimerData.timerFig,''Color'',[.5 .5 .7]);set(TimerData.timertexthandle,''BackgroundColor'',[.5 .5 .7]);figure(TimerData.timerFig);clear Timers TimerData';

            set(timer_handle,'TimerFcn',Timer_Callback,'StopFcn',TimerStopFcn);

            %%% Sets initial text to be displayed in the text box within the timer window.
            timertext = 'Timer is starting';
            %%% Creates the text box within the timer window which will display the
            %%% timer text.
            text_handle = uicontrol(timerFig,'string',timertext,'style','text',...
                'parent',timerFig,'position', [10 40 260 74],'FontName','Helvetica','HorizontalAlignment','left',...
                'FontSize',handles.Preferences.FontSize,'FontWeight','bold','BackgroundColor',[0.7,0.7,0.9]);
            %%% Saves text handle to the handles structure.
            handles.timertexthandle = text_handle;
            TimerData.timertexthandle = text_handle;
            set(timer_handle,'UserData',TimerData);
            %%% Creates the Cancel and Pause buttons.
            PauseButton_handle = uicontrol('Style', 'pushbutton', ...
                'String', 'Pause', 'Position', [280 60 60 25], ...
                'parent',timerFig, 'BackgroundColor',[0.7,0.7,0.9],'FontName','Helvetica','FontSize',handles.Preferences.FontSize,'UserData',0);
            CancelAfterCycleButton_handle = uicontrol('Style', 'pushbutton', ...
                'String', 'Cancel after cycle', 'Position', [10 10 120 25], ...
                'parent',timerFig, 'BackgroundColor',[0.7,0.7,0.9],'FontName','Helvetica','FontSize',handles.Preferences.FontSize,'UserData',0);
            CancelAfterModuleButton_handle = uicontrol('Style', 'pushbutton', ...
                'String', 'Cancel after module', 'Position', [140 10 120 25], ...
                'parent',timerFig, 'BackgroundColor',[0.7,0.7,0.9],'FontName','Helvetica','FontSize',handles.Preferences.FontSize,'UserData',0);

            uicontrol('Style','pushbutton','String','Close','Position',[300 90 40 25],'BackgroundColor',[.7 .7 .9],...
                'parent',timerFig,'FontName','Helvetica','FontSize',handles.Preferences.FontSize,'Callback',...
                ['close(' num2str(timerFig*8192) '/8192)']);

            %%% Sets the functions to be called when the Cancel and Pause buttons
            %%% within the Timer window are pressed.
            PauseButtonFunction = 'if ~exist(''h''); h = CPmsgbox(''Image processing is paused without causing any damage. Processing will restart when you close the Pause window or click OK.''); waitfor(h); clear h; end';
            set(PauseButton_handle,'Callback', PauseButtonFunction)
            CancelAfterCycleButtonFunction = ['if ~exist(''delme''); delme=1; deleteme = CPquestdlg(''Paused. Are you sure you want to cancel after this cycle? Processing will continue on the current cycle, the data up to and including this cycle will be saved in the output file, and then the analysis will be canceled.'', ''Confirm cancel'',''Yes'',''No'',''Yes''); switch deleteme; case ''Yes''; set(',num2str(CancelAfterCycleButton_handle*8192), '/8192,''enable'',''off''); set(', num2str(text_handle*8192), '/8192,''string'',''Canceling in progress; Waiting for the processing of current cycle to be complete. You can press the Cancel after module button or cancel now button to cancel more quickly, but data relating to the current cycle will not be saved in the output file.''); case ''No''; clear deleteme; clear delme; return; end; clear deleteme; clear delme; end'];
            set(CancelAfterCycleButton_handle, 'Callback', CancelAfterCycleButtonFunction)
            CancelAfterModuleButtonFunction = ['if ~exist(''delme2''); delme2=1; deleteme = CPquestdlg(''Paused. Are you sure you want to cancel after this module? Processing will continue until the current image analysis module is completed, to avoid corrupting the current settings of CellProfiler. Data up to the *previous* cycle are saved in the output file and processing is canceled.'', ''Confirm cancel'',''Yes'',''No'',''Yes''); switch deleteme; case ''Yes''; set(', num2str(CancelAfterCycleButton_handle*8192), '/8192,''enable'',''off''); set(', num2str(CancelAfterModuleButton_handle*8192), '/8192,''enable'',''off''); set(', num2str(text_handle*8192), '/8192,''string'',''Canceling after current module in progress; Waiting for the processing of current module to be complete in order to avoid corrupting the current CellProfiler settings.''); case ''No''; clear deleteme; clear delme2; return; end; clear deleteme; clear delme2; end'];
            set(CancelAfterModuleButton_handle,'Callback', CancelAfterModuleButtonFunction)
            HelpButtonFunction = 'CPmsgbox(''Pause button: The current processing is immediately suspended without causing any damage. Processing restarts when you close the Pause window or click OK. Cancel after cycle: Processing will continue on the current cycle, the data up to and including this cycle will be saved in the output file, and then the analysis will be canceled.  Cancel after module: Processing will continue until the current image analysis module is completed, to avoid corrupting the current settings of CellProfiler. Data up to the *previous* cycle are saved in the output file and processing is canceled. Cancel now: The data up to the *previous* cycle will be saved in the output file, but the current cycle data will be stored incomplete in the output file, which might be confusing or corrupt when using the output file.'')';
            %%% HelpButton
            uicontrol('Style', 'pushbutton', ...
                'String', '?', 'Position', [280 90 15 25], 'FontName','Helvetica','FontSize', handles.Preferences.FontSize,...
                'Callback', HelpButtonFunction, 'parent',timerFig, 'BackgroundColor',[0.7,0.7,0.9]);
            DetailButton=uicontrol('Style','pushbutton','String','Details','Position',[280 30 60 25],'FontName','Helvetica','FontSize',handles.Preferences.FontSize,...
                'parent',timerFig,'Tag','DetailButton','BackgroundColor',[.7 .7 .9],'Callback',[...
                'Timers = timerfind(''Tag'',''CellProfilerTimer'');',...
                'TimerFcn = get(Timers(1),''TimerFcn'');',...
                'set(Timers(1),''TimerFcn'','''');',...
                'TimerData = get(Timers(1),''UserData'');',...
                'DetWin = findobj(''tag'',''DetailWindow'');',...
                'if isempty(DetWin),',...
                'DetWin = CPfigure(''tag'',''DetailWindow'',''Name'',''Status Details'',''NumberTitle'',''off'',''menubar'',''none'',''position'',[400 100 250 13*(TimerData.NumberOfModules+6)+50]);',...
                'pos = get(DetWin,''Position'');',...
                'DetWinheight = pos(4);',...
                'uicontrol(DetWin, ''Tag'', ''DetailWindowHeader'', ''Style'', ''text'', ''Position'', [10 DetWinheight-17 78 13],', ...
                '''HorizontalAlignment'', ''left'', ''BackgroundColor'', [0.7 0.7 0.9], ''FontSize'', TimerData.FontSize,', ...
                '''FontWeight'', ''bold'', ''String'', ''Time(sec) for:'');',...
                'uicontrol(DetWin, ''Tag'', ''DetailWindowHeader'', ''Style'', ''text'', ''Position'', [100 DetWinheight-17 51 13],', ...
                '''HorizontalAlignment'', ''center'', ''BackgroundColor'', [0.7 0.7 0.9], ''FontSize'', TimerData.FontSize,', ...
                '''FontWeight'', ''bold'', ''String'', ''1st Cycle'');',...
                'uicontrol(DetWin, ''Tag'', ''DetailWindowHeader'', ''Style'', ''text'', ''Position'', [160 DetWinheight-17 63 13],', ...
                '''HorizontalAlignment'', ''center'', ''BackgroundColor'', [0.7 0.7 0.9], ''FontSize'', TimerData.FontSize,', ...
                '''FontWeight'', ''bold'', ''String'', ''Avg Others'');',...
                'ModuleStr = {};',...
                'for k = 1:TimerData.NumberOfModules,',...
                'ModuleStr{end+1} = [''Module '', num2str(k), '':''];',...
                'end;',...
                'ModuleStr{end+1} = '''';',...
                'ModuleStr{end+1} = ''Avg Totals:'';',...
                'ModuleStr{end+1} = '''';',...
                'ModuleStr{end+1} = ''TOTAL TIME:'';',...
                'uicontrol(DetWin, ''Tag'', ''DetailWindowText'', ''Style'', ''text'', ''Position'', [10 DetWinheight-30-13*(TimerData.NumberOfModules+6) 78 13*(TimerData.NumberOfModules+6)],', ...
                '''HorizontalAlignment'', ''left'', ''BackgroundColor'', [0.7 0.7 0.9], ''FontSize'', TimerData.FontSize,', ...
                '''String'', ModuleStr);',...
                'else,',...
                'figure(DetWin);',...
                'pos = get(DetWin,''Position'');',...
                'DetWinheight = pos(4);',...
                'end;',...
                'Time1stCycleStr = {};',...
                'TimeAvgOthersStr = {};',...
                'for i = [1:size(TimerData.TimerTime,1)],',...
                'Time1stCycleStr{end+1} = sprintf(''%1.1f'',TimerData.TimerTime(i,1));',...
                'if TimerData.SetBeingAnalyzed == 1 | TimerData.NumberOfImageSets==1,',...
                'TimeAvgOthersStr{end+1} = ''N/A'';',...
                'elseif TimerData.SetBeingAnalyzed > TimerData.NumberOfImageSets,',...
                'TimeAvgOthersStr{end+1} = sprintf(''%1.1f'',sum(TimerData.TimerTime(i,2:end))/(TimerData.NumberOfImageSets-1));',...
                'else,',...
                'count=0;',...
                'total=0;',...
                'for j=[2:size(TimerData.TimerTime,2)],',...
                'if TimerData.TimerTime(i,j)~=0,',...
                'count=count+1;',...
                'total=total+TimerData.TimerTime(i,j);',...
                'end;',...
                'end;',...
                'if count~=0,',...
                'TimeAvgOthersStr{end+1} = sprintf(''%1.1f'',total/count);',...
                'else,',...
                'TimeAvgOthersStr{end+1} = ''N/A'';',...
                'end;',...
                'end;',...
                'end;',...
                'for m = 1:(TimerData.NumberOfModules+1 - length(Time1stCycleStr)),',...
                'Time1stCycleStr{end+1} = '''';',...
                'TimeAvgOthersStr{end+1} = '''';',...
                'end;',...
                'Time1stCycleStr{end+1} = sprintf(''%2.1f'',sum(TimerData.TimerTime(:,1)));',...
                'Time1stCycleStr{end+1} = '''';',...
                'Time1stCycleStr{end+1} = sprintf(''%1.1f'',sum(sum(TimerData.TimerTime(:))));',...
                'if TimerData.SetBeingAnalyzed == 1 | TimerData.NumberOfImageSets==1,',...
                'TimeAvgOthersStr{end+1} = ''N/A'';',...
                'else,',...
                'TotalAvgOthers = 0;',...
                'for n = 1:(length(TimeAvgOthersStr)-1),',...
                'if ~(str2num(TimeAvgOthersStr{n})==0 | TimeAvgOthersStr{n}(1) == ''N''),',...
                'TotalAvgOthers = TotalAvgOthers + str2num(TimeAvgOthersStr{n});',...
                'end;',...
                'end;',...
                'TimeAvgOthersStr{end+1} = sprintf(''%1.1f'',TotalAvgOthers);',...
                'end;',...
                'Time1stCycle = findobj(DetWin,''Tag'',''Time1stCycle'');',...
                'if ~isempty(Time1stCycle),',...
                'set(Time1stCycle, ''String'', Time1stCycleStr);',...
                'else,',...
                'uicontrol(DetWin, ''Tag'', ''Time1stCycle'', ''Style'', ''text'', ''Position'', [100 DetWinheight-30-13*(TimerData.NumberOfModules+6) 51 13*(TimerData.NumberOfModules+6)],', ...
                '''HorizontalAlignment'', ''center'', ''BackgroundColor'', [0.7 0.7 0.9], ''FontSize'', TimerData.FontSize,', ...
                '''String'', Time1stCycleStr);',...
                'end;',...
                'TimeAvgOthers = findobj(DetWin,''Tag'',''TimeAvgOthers'');',...
                'if ~isempty(TimeAvgOthers),',...
                'set(TimeAvgOthers, ''String'', TimeAvgOthersStr);',...
                'else,',...
                'uicontrol(DetWin, ''Tag'', ''TimeAvgOthers'', ''Style'', ''text'', ''Position'', [160 DetWinheight-30-13*(TimerData.NumberOfModules+6) 63 13*(TimerData.NumberOfModules+6)],', ...
                '''HorizontalAlignment'', ''center'', ''BackgroundColor'', [0.7 0.7 0.9], ''FontSize'', TimerData.FontSize,', ...
                '''String'', TimeAvgOthersStr);',...
                'end;',...
                'set(Timers(1),''TimerFcn'',TimerFcn);',...
                'clear DetWin DetWinheight ModuleStr Time1stCycle Time1stCycleStr TimeAvgOthers TimeAvgOthersStr TimerData TimerFcn Timers TotalAvgOthers count i j k m n pos total;']);

            %%% The timertext string is read by the analyze all images button's callback
            %%% at the end of each time around the loop (i.e. at the end of each image
            %%% set).  If it notices that the string says "Cancel...", it breaks out of
            %%% the loop and finishes up.

            %%% Update the handles structure. Not sure if it's necessary here.
            guidata(gcbo, handles);
            %%% Sets the timer window to show a warning box before allowing it to be
            %%% closed.
            CloseFunction = ['deleteme = CPquestdlg(''DO NOT CLOSE the Timer window while image processing is in progress!! Are you sure you want to close the timer?'', ''Confirm close'',''Yes'',''No'',''Yes''); switch deleteme; case ''Yes''; delete(',num2str(timerFig*8192), '/8192); clear deleteme; case ''No''; clear deleteme; return; end;'];
            set(timerFig,'CloseRequestFcn',CloseFunction)
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
            %%% figure window is set to wait until a cycle is done processing
            %%% before closing the window, to avoid unexpected results.
            set(handles.CloseFigureButton,'visible','on')
            set(handles.OpenFigureButton,'visible','on')

            %%% For the first time through, the number of cycles
            %%% will not yet have been determined.  So, the Number of
            %%% cycles is set temporarily.
            handles.Current.NumberOfImageSets = 1;
            handles.Current.SetBeingAnalyzed = 1;
            handles.Current.SaveOutputHowOften = 1;
            %%% Marks the time that analysis was begun.
            handles.Current.TimeStarted = datestr(now);
            %%% Clear the buffers (Pipeline and Measurements)
            handles.Pipeline = struct;
            handles.Measurements = struct;
            %%% Start the timer.
            start(timer_handle)
            %%% Update the handles structure.
            guidata(gcbo, handles);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%% Begin loop (going through all the cycles). %%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %%% This variable allows breaking out of nested loops.
            break_outer_loop = 0;
            startingImageSet = 1;
            handles.Current.StartingImageSet = startingImageSet;
            while handles.Current.SetBeingAnalyzed <= handles.Current.NumberOfImageSets
                setbeinganalyzed = handles.Current.SetBeingAnalyzed;
                NumberofWindows = 0;
                %%% This is written as a while loop (rather than a for loop) to allow fixes and restarts.
                SlotNumber = 1;
                while SlotNumber <= handles.Current.NumberOfModules
                    %%% If a module is not chosen in this slot, continue on to the next.
                    ModuleNumberAsString = TwoDigitString(SlotNumber);
                    ModuleName = char(handles.Settings.ModuleNames(SlotNumber));
                    if ~iscellstr(handles.Settings.ModuleNames(SlotNumber))
                    else
                        %%% Saves the current module number in the handles structure.
                        handles.Current.CurrentModuleNumber = ModuleNumberAsString;
                        %%% The try/catch/end set catches any errors that occur during the
                        %%% running of module 1, notifies the user, breaks out of the image
                        %%% analysis loop, and completes the refreshing
                        %%% process.
                        try
                            if handles.Current.SetBeingAnalyzed == 1
                                if handles.Preferences.DisplayWindows(SlotNumber) == 0
                                    handles.Current.(['FigureNumberForModule' TwoDigitString(SlotNumber)]) = ceil(max(findobj))+1;
                                else
                                    NumberofWindows = NumberofWindows+1;
                                    if iscellstr(handles.Settings.ModuleNames(SlotNumber))
                                        LeftPos = (ScreenWidth*((NumberofWindows-1)/12));
                                        if LeftPos >= ScreenWidth
                                            LeftPos = LeftPos - ScreenWidth;
                                        end
                                        handles.Current.(['FigureNumberForModule' TwoDigitString(SlotNumber)]) = ...
                                            CPfigure(handles,'','name',[char(handles.Settings.ModuleNames(SlotNumber)), ' Display, cycle # '],...
                                            'Position',[LeftPos (ScreenHeight-522) 560 442]);
                                    end
                                    TempFigHandle = handles.Current.(['FigureNumberForModule' TwoDigitString(SlotNumber)]);
                                    if exist('FigHandleList','var')
                                        if any(TempFigHandle == FigHandleList)
                                            for z = 1:length(FigHandleList)
                                                if TempFigHandle == FigHandleList(z)
                                                    handles.Current.(['FigureNumberForModule' TwoDigitString(z)]) = ceil(max(findobj))+z;
                                                end
                                            end
                                        end
                                    end
                                    FigHandleList(SlotNumber) = handles.Current.(['FigureNumberForModule' TwoDigitString(SlotNumber)]); %#ok
                                end
                            end
                            %%% This is used to check for errors and allow restart.
                            CanMoveToNextModule = false;
                            

                            %%% Runs the appropriate module, with the handles structure as an
                            %%% input argument and as the output
                            %%% argument.
                            handles.Measurements.Image.ModuleErrorFeatures(str2double(TwoDigitString(SlotNumber))) = {ModuleName};
                            handles = feval(ModuleName,handles);

                            %%% If the call to feval succeeded, then the module succeeded and we can move to the next module.
                            %%% (if there is an error, it will be caught below, at the point marked MODULE ERROR)
                            CanMoveToNextModule = true;

                            %%% Store the handles back to the figure.
                            guidata(handles.figure1,handles);
                            try
                                TimerData = get(timer_handle,'UserData');
                                TimerData.NumberOfImageSets = handles.Current.NumberOfImageSets;
                                TimerData.StartingImageSet = handles.Current.StartingImageSet;
                                set(timer_handle,'UserData',TimerData);
                            end
                            FigHandle = -1;
                            try
                                FigHandle = handles.Current.(['FigureNumberForModule' TwoDigitString(SlotNumber)]);
                            end
                            if ishandle(FigHandle)
                                OldText = get(FigHandle,'name');
                                NewNum = handles.Current.SetBeingAnalyzed;
                                set(FigHandle,'name',[OldText(1:(end-length(num2str(NewNum-1)))) num2str(NewNum)]);
                            end
                            handles.Measurements.Image.ModuleError{handles.Current.SetBeingAnalyzed}(1,str2double(ModuleNumberAsString)) = 0;
                        catch
                            handles.Measurements.Image.ModuleError{handles.Current.SetBeingAnalyzed}(1,str2double(ModuleNumberAsString)) = 1;
                            if strcmp(handles.Preferences.SkipErrors,'No')
                                if isdeployed
                                    errorfunction(ModuleNumberAsString,handles.Preferences.FontSize,ModuleName)
                                else
                                    if exist([ModuleName,'.m'],'file') ~= 2,
                                        CPerrordlg(['Image processing was canceled because the image analysis module named ', ([ModuleName,'.m']), ' was not found. Is it stored in the folder with the other modules?  Has its name changed?']);
                                    else
                                        %%% MODULE ERROR
                                        %%% Runs the errorfunction function that catches errors and
                                        %%% describes to the user what to do.
                                        errorfunction(ModuleNumberAsString,handles.Preferences.FontSize,ModuleName)
                                        %%% Give the user a chance to fix the bug and retry the module.
                                        if strcmp(getenv('CPDEBUG'), 'yes'),
                                            if strcmp(CPquestdlg('Edit code and retry module?  (note: breakpoints will be lost)', 'Retry pipeline?', 'Yes', 'No', 'Yes'), 'Yes'),
                                                %%% If we get an error in the retry code, below, we skip the retry.
                                                give_up = 0;
                                                try
                                                    %%% To force code to be reloaded, we clear functions on the error stack, up to the called module.
                                                    err = lasterror;
                                                    stack = err.stack;
                                                    for i = 1:length(stack),
                                                        clear(stack(i).name);
                                                        %%% Stop at the called module.  (Hopefully none of them recurse.(?))
                                                        if strcmp(stack(i).name, ModuleName),
                                                            break;
                                                        end
                                                    end
                                                catch
                                                    %%% If there was an error in the retry code, report it, then revert to not retrying.
                                                    CPerrordlg(['Could not retry: (' lasterr ')']);
                                                    give_up = 1;
                                                end
                                                if ~ give_up,
                                                    %%% This continue binds to the while loop over SlotNumber.
                                                    continue;
                                                end
                                                %%% The implicit else clause is to fall through to the break below.
                                            end
                                        end
                                        %%% This will cause the image analysis loop to break out of the loop over images.
                                        break_outer_loop = 1;
                                    end
                                end
                                %%% Got to here with an error, so break outer loop.
                                break;
                            else
                                errorfunction(ModuleNumberAsString,handles.Preferences.FontSize,ModuleName)
                            end
                        end % Goes with try/catch.

                        %%% Check for a pending "Cancel after Module"
                        CancelWaiting = get(handles.timertexthandle,'string');
                        if strncmpi(CancelWaiting, 'Canceling after current module', 30)
                            break_outer_loop = 1;
                            break
                        end
                    end

                    %%% If the module passed out a new value for
                    %%% StartingImageSet, then we set startingImageSet
                    %%% to be that value and break all the way our to
                    %%% the cycle loop. The RestartImageSet in
                    %%% handles is deleted because we never want it in
                    %%% the output file.
                    startingImageSet = handles.Current.StartingImageSet;
                    if (setbeinganalyzed < startingImageSet)
                        handles.Current.SetBeingAnalyzed = startingImageSet;
                        guidata(gcbo,handles);
                        break  %% break out of SlotNumber loop
                    end

                    openFig = openFigures;
                    openFigures = [];
                    for i=1:length(openFig),
                        ModuleNumber = openFig(i);
                        try
                            LeftPos = (ScreenWidth*((ModuleNumber-1)/12));
                            if LeftPos >= ScreenWidth
                                LeftPos = LeftPos - ScreenWidth;
                            end
                            handles.Current.(['FigureNumberForModule' TwoDigitString(ModuleNumber)]) = ...
                                CPfigure(handles,'','name',[char(handles.Settings.ModuleNames(ModuleNumber)), ' Display, cycle # '],...
                                'Position',[LeftPos (ScreenHeight-522) 560 442]);
                            %%% Sets the closing function of the window appropriately. (See way
                            %%% above where 'ClosingFunction's are defined).
                        catch
                        end
                    end

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

                    %%% Finds and records total to run module.
                    TimerData = get(timer_handle,'UserData');
                    if SlotNumber==1 && handles.Current.SetBeingAnalyzed==handles.Current.StartingImageSet
                        TimerData.TimerTime(SlotNumber,handles.Current.SetBeingAnalyzed) = toc;
                    else
                        TimerData.TimerTime(SlotNumber,handles.Current.SetBeingAnalyzed) = toc - sum(TimerData.TimerTime(:));
                    end
                    set(timer_handle,'UserData',TimerData);
                    if ~isempty(findobj('Tag','DetailWindow'))
                        eval(get(DetailButton,'callback'));
                    end

                    %%% if we can move to the next module, do so
                    if CanMoveToNextModule,
                        SlotNumber = SlotNumber + 1;
                    end
                end %%% ends loop over slot number

                %%% Completes the breakout to the image loop.
                if (setbeinganalyzed < startingImageSet)
                    if startingImageSet ==2
                        handles.Current.StartingImageSet = 1;
                        guidata(gcbo,handles);
                    end
                    continue;
                end;

                if (break_outer_loop),
                    break;  %%% this break is out of the outer loop of image analysis
                end

                CancelWaiting = get(handles.timertexthandle,'string');

                %%% Save all data that is in the handles structure to the output file
                %%% name specified by the user, but only save it
                %%% in the increments that the user has specified
                %%% (e.g. every 5th cycle, every 10th image
                %%% set, as set by the SpeedUpCellProfiler
                %%% module), or if it is the last cycle.  If
                %%% the user has not used the SpeedUpCellProfiler
                %%% module, then
                %%% handles.Current.SaveOutputHowOften is the
                %%% number 1, so the output file will be saved
                %%% every time.
                %%% Save everything, but don't want to write out
                %%% StartingImageSet field.
                handles.Current = rmfield(handles.Current,'StartingImageSet');
                if (rem(handles.Current.SetBeingAnalyzed,handles.Current.SaveOutputHowOften) == 0) || (handles.Current.SetBeingAnalyzed == 1) || (handles.Current.SetBeingAnalyzed == handles.Current.NumberOfImageSets)
                    %Removes images from the Pipeline
                    if strcmp(handles.Preferences.StripPipeline,'Yes')
                        ListOfFields = fieldnames(handles.Pipeline);
                        restorePipe = handles.Pipeline;
                        tempPipe = handles.Pipeline;
                        for i = 1:length(ListOfFields)
                            if all(size(tempPipe.(ListOfFields{i}))~=1)
                                tempPipe = rmfield(tempPipe,ListOfFields(i));
                            end
                        end
                        handles.Pipeline = tempPipe;
                    end
                    try eval(['save ''',fullfile(handles.Current.DefaultOutputDirectory, ...
                            get(handles.OutputFileNameEditBox,'string')), ''' ''handles'';'])
                    catch CPerrordlg('There was an error saving the output file. Please check whether you have permission and space to write to that location.');
                        break;
                    end
                    if strcmp(handles.Preferences.StripPipeline,'Yes')
                        %%% restores the handles.Pipeline structure if
                        %%% it was removed above.
                        handles.Pipeline = restorePipe;
                    end
                end
                %%% Restore StartingImageSet for those modules that
                %%% need it.
                handles.Current.StartingImageSet = startingImageSet;
                %%% If a "cancel" signal is waiting, break and go to the "end" that goes
                %%% with the "while" loop.
                if strncmpi(CancelWaiting,'Cancel',6)
                    break
                end
                drawnow
                %%% The setbeinganalyzed is increased by one and stored in the handles structure.
                setbeinganalyzed = setbeinganalyzed + 1;
                handles.Current.SetBeingAnalyzed = setbeinganalyzed;
                TimerData = get(timer_handle,'UserData');
                TimerData.SetBeingAnalyzed = setbeinganalyzed;
                set(timer_handle,'UserData',TimerData);
                guidata(gcbo, handles)

            end %%% This "end" goes with the "while" loop (going through the cycles).


            %%% Update the handles structure.
            guidata(gcbo, handles)

            set(timerFig,'CloseRequestFcn','closereq')
            stop(timer_handle);

            %%% Re-enable/disable appropriate buttons.
            set(handles.IndividualModulesText,'visible','on')
            set(handles.AddModule,'visible','on');
            set(handles.RemoveModule,'visible','on');
            set(handles.MoveUpButton,'visible','on');
            set(handles.MoveDownButton,'visible','on');
            set(handles.PixelSizeEditBox,'enable','on','foregroundcolor','black')
            set(handles.BrowseImageDirectoryButton,'enable','on')
            set(handles.DefaultImageDirectoryEditBox,'enable','on','foregroundcolor','black')
            set(handles.BrowseOutputDirectoryButton,'enable','on')
            set(handles.DefaultOutputDirectoryEditBox,'enable','on','foregroundcolor','black')
            set(handles.OutputFileNameEditBox,'enable','on','foregroundcolor','black')
            set(handles.AnalyzeImagesButton,'enable','on')

            set(cat(2,handles.VariableBox{:}),'enable','on','foregroundcolor','black');

            set(handles.CloseFigureButton,'visible','off');
            set(handles.OpenFigureButton,'visible','off');
            set(CancelAfterModuleButton_handle,'enable','off')
            set(CancelAfterCycleButton_handle,'enable','off')
            set(PauseButton_handle,'enable','off')
            %set(CancelNowButton_handle,'enable','off')
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
                if isfield(handles.Current,['FigureNumberForModule' TwoDigitString(i)])
                    if any(findobj == handles.Current.(['FigureNumberForModule' TwoDigitString(i)]))
                        properhandle = handles.Current.(['FigureNumberForModule' TwoDigitString(i)]);
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

function errorfunction(CurrentModuleNumber,FontSize,ModuleName)
%%% lasterr is an old MATLAB function; lasterror is a new MATLAB function
%%% that will eventually replace it. Most likely sometime we can simplify
%%% the following code, but we should think carefully about how it affects
%%% old vs new versions of MATLAB.
Error = lasterr;
%%% If an error occurred in an image analysis module, the error message
%%% should begin with "Error using ==> ", which will be recognized here.
if strncmp(Error,'Error using ==> ',16)
    ErrorExplanation = ['There was a problem running the analysis module ',ModuleName,' which is number ',CurrentModuleNumber, '. ', Error];
    %%% The following are errors that may have occured within the analyze all
    %%% images callback itself.
elseif ~isempty(strfind(Error,'bad magic'))
    ErrorExplanation = ['There was a problem running the image analysis. It seems likely that there are files in your image directory that are not images or are not the image format that you indicated. Probably the data for the cycles up to the one which generated this error are OK in the output file.'];
else
    ErrorExplanation = ['There was a problem running the image analysis. Sorry, it is unclear what the problem is. It would be wise to close the entire CellProfiler program in case something strange has happened to the settings. The output file may be unreliable as well. Matlab says the error is: ', Error, ' in the ', ModuleName, ' module, which is module #', CurrentModuleNumber, ' in the pipeline.'];
end
CPerrordlg(ErrorExplanation);

%%%%%%%%%%%%%%%%%%%%
%%% HELP BUTTONS %%%
%%%%%%%%%%%%%%%%%%%%

function IndividualModuleHelp_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.

NoModuleSelectedHelpMsg = 'You do not have an analysis module loaded. Add a module to the pipeline using the + button. Clicking the "?" button below the pipeline when a module is selected will then reveal help for that module.';
ModuleNumber = whichactive(handles);
if ModuleNumber == 0
    CPtextdisplaybox(NoModuleSelectedHelpMsg,'Help for choosing an analysis module');
else
    try ModuleName = handles.Settings.ModuleNames(ModuleNumber);
        %%% This is the function that actually reads the module's help
        %%% data.
        if isdeployed
            for i = 1:length(handles.Current.ModulesFilenames)
                if strmatch(ModuleName,handles.Current.ModulesFilenames{i},'exact')
                    Option = i;
                    break
                end
            end
            if ~isempty(Option)
                HelpText = handles.Current.ModulesHelp{Option};
            end
        else
            HelpText = help(char(ModuleName));
        end
        DoesHelpExist = exist('HelpText','var');
        if DoesHelpExist == 1
            %%% Calls external subfunction: a nice text display box with a slider if the help is too long.
            CPtextdisplaybox(HelpText,'CellProfiler image analysis module help');
        else
            CPtextdisplaybox('Sorry, there is no help information for this image analysis module.','Image analysis module help');
        end
    catch
        CPtextdisplaybox(NoModuleSelectedHelpMsg,'Help for choosing an analysis module');
    end
end

function ModulesHelp_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
ListOfTools = handles.Current.ModulesFilenames;
ToolsHelpSubfunction(handles, 'Modules', ListOfTools)

function ImageToolsHelp_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
ListOfTools = handles.Current.ImageToolsFilenames;
ToolsHelpSubfunction(handles, 'Image Tools', ListOfTools)

function DataToolsHelp_Callback(hObject, eventdata, handles) %#ok We want to ignore MLint error checking for this line.
ListOfTools = handles.Current.DataToolsFilenames;
ToolsHelpSubfunction(handles, 'Data Tools', ListOfTools)

function HelpFiles_Callback(hObject,eventdata, handles) %#ok Ignore MLint
if strcmp(eventdata,'GS');
    ListOfHelp = handles.Current.GSFilenames;
    for i=1:length(ListOfHelp)
        if strncmpi(ListOfHelp{i},'help',4)
            ListOfHelp{i} = ListOfHelp{i}(5:end);
        end
    end
    ToolsHelpSubfunction(handles,'Getting Started',ListOfHelp)
elseif strcmp(eventdata,'Help');
    ListOfHelp = handles.Current.HelpFilenames;
    for i=1:length(ListOfHelp)
        if strncmpi(ListOfHelp{i},'help',4)
            ListOfHelp{i} = ListOfHelp{i}(5:end);
        end
    end
    ToolsHelpSubfunction(handles,'Help',ListOfHelp)
else
    CPerrordlg('Something is wrong.');
end

%%% SUBFUNCTION %%%
function ToolsHelpSubfunction(handles, ImageDataOrHelp, ToolsCellArray)
global toolsChoice;
ToolsCellArray(1) = [];
okbuttoncallback = 'ToolsHelpWindowHandle = findobj(''tag'',''ToolsHelpWindow''); toolsbox = findobj(''tag'',''toolsbox''); global toolsChoice; toolsChoice = get(toolsbox,''value''); close(ToolsHelpWindowHandle), clear ToolsHelpWindowHandle toolsbox toolsChoice';
cancelbuttoncallback = 'ToolsHelpWindowHandle = findobj(''tag'',''ToolsHelpWindow''); global toolsChoice; toolsChoice = 0; close(ToolsHelpWindowHandle), clear ToolsHelpWindowHandle toolsbox toolsChoice';

MainWinPos = get(handles.figure1,'Position');
Color = [0.7 0.7 0.9];

%%% If there is a (are) ToolsHelpWindow(s) open, close it (them);
%%% otherwise, ok/cancel callbacks can get confused
ToolsHelpWindowHandles = findobj('tag','ToolsHelpWindow');
if ~isempty(ToolsHelpWindowHandles)
    try
        close(ToolsHelpWindowHandles);
    end
end

%%% Label we attach to figures (as UserData) so we know they are ours
userData.Application = 'CellProfiler';
userData.MyHandles=handles;
ToolsHelpWindowHandle = figure(...
    'Units','pixels',...
    'CloseRequestFcn','delete(gcf)',...
    'Color',Color,...
    'DockControls','off',...
    'MenuBar','none',...
    'Name','ToolsHelpWindow',...
    'NumberTitle','off',...
    'Position',[MainWinPos(1)+MainWinPos(3)/4 MainWinPos(2)+MainWinPos(4)/5 MainWinPos(3)/2 MainWinPos(4)*2/3],...
    'Resize','off',...
    'HandleVisibility','on',...
    'Tag','ToolsHelpWindow',...
    'UserData',userData);

if strcmp(ImageDataOrHelp,'Modules')
    set(ToolsHelpWindowHandle,'name','Modules Help');
    TextString = sprintf(['To view help for individual ' ImageDataOrHelp ', choose one below.\nYou can add your own tools by writing Matlab m-files, placing them in the ', ImageDataOrHelp, ' folder, and restarting CellProfiler.']);
elseif strcmp(ImageDataOrHelp,'Image Tools')
    set(ToolsHelpWindowHandle,'name','Image Tools Help');
    TextString = sprintf(['To view help for individual ' ImageDataOrHelp ', choose one below.\nYou can add your own tools by writing Matlab m-files, placing them in the ', ImageDataOrHelp, ' folder, and restarting CellProfiler.']);
elseif strcmp(ImageDataOrHelp,'Data Tools')
    set(ToolsHelpWindowHandle,'name','Data Tools Help');
    TextString = sprintf(['To view help for individual ' ImageDataOrHelp ', choose one below.\nYou can add your own tools by writing Matlab m-files, placing them in the ', ImageDataOrHelp, ' folder, and restarting CellProfiler.']);
elseif strcmp(ImageDataOrHelp,'Help')
    set(ToolsHelpWindowHandle,'name','General Help');
    TextString = sprintf('CellProfiler version 1.0.4684\n\nPlease choose specific help below:');
elseif strcmp(ImageDataOrHelp,'Getting Started')
    set(ToolsHelpWindowHandle,'name','Getting Started');
    TextString = sprintf('CellProfiler version 1.0.4684\n\nPlease choose specific help below:');
end

FontSize = handles.Preferences.FontSize;

choosetext = uicontrol(...
    'Parent',ToolsHelpWindowHandle,...
    'BackGroundColor', Color,...
    'Units','normalized',...
    'Position',[0.10 0.6 0.80 0.31],...
    'String',TextString,...
    'Style','text',...
    'FontSize',FontSize,...
    'Tag','informtext'); %#ok Ignore MLint

listboxcallback = 'ToolsHelpWindowHandle = findobj(''tag'',''ToolsHelpWindow''); if (strcmpi(get(ToolsHelpWindowHandle,''SelectionType''),''open'')==1) toolsbox = findobj(''tag'',''toolsbox''); global toolsChoice; toolsChoice = get(toolsbox,''value''); close(ToolsHelpWindowHandle); end; clear ToolsHelpWindowHandle toolsChoice toolsbox';
toolsbox = uicontrol(...
    'Parent',ToolsHelpWindowHandle,...
    'Units','normalized',...
    'backgroundColor',Color,...
    'Position',[0.20 0.18 0.65 0.464],...
    'String',ToolsCellArray,...
    'Style','listbox',...
    'Callback',listboxcallback,...
    'Value',1,...
    'Tag','toolsbox',...
    'FontSize',FontSize,...
    'Behavior',get(0,'defaultuicontrolBehavior')); %#ok Ignore MLint

okbutton = uicontrol(...
    'Parent',ToolsHelpWindowHandle,...
    'BackGroundColor', Color,...
    'Units','normalized',...
    'Callback',okbuttoncallback,...
    'Position',[0.30 0.077 0.2 0.06],...
    'String','Ok',...
    'Tag','okbutton'); %#ok Ignore MLint

cancelbutton = uicontrol(...
    'Parent',ToolsHelpWindowHandle,...
    'BackGroundColor', Color,...
    'Units','normalized',...
    'Callback',cancelbuttoncallback,...
    'Position',[0.55 0.077 0.2 0.06],...
    'String','Cancel',...
    'Tag','cancelbutton'); %#ok Ignore MLint

toolsChoice = 0; %%% Makes sure toolsChoice indicates no selection
%%% in case user closes window using x icon or Close Windows button
uiwait(ToolsHelpWindowHandle);

if(toolsChoice ~= 0)
    if strcmp(ImageDataOrHelp,'Modules')
        HelpText = handles.Current.ModulesHelp{toolsChoice};
        CPtextdisplaybox(HelpText,'CellProfiler Modules Help');
    elseif strcmp(ImageDataOrHelp,'Image Tools')
        HelpText = handles.Current.ImageToolHelp{toolsChoice};
        CPtextdisplaybox(HelpText,'CellProfiler Image Tools Help');
    elseif strcmp(ImageDataOrHelp,'Data Tools')
        HelpText = handles.Current.DataToolHelp{toolsChoice};
        CPtextdisplaybox(HelpText,'CellProfiler Data Tools Help');
    elseif strcmp(ImageDataOrHelp,'Help')
        HelpText = handles.Current.Help{toolsChoice};
        CPtextdisplaybox(HelpText,'CellProfiler Help');
    elseif strcmp(ImageDataOrHelp,'Getting Started')
        HelpText = handles.Current.GS{toolsChoice};
        CPtextdisplaybox(HelpText,'CellProfiler Help');
    end
end
clear toolsChoice;

%%% END OF HELP HELP HELP HELP HELP HELP BUTTONS %%%



%%% This function is currently never called/used.
function DownloadModules_Callback(hObject, eventdata, handles)

Answer = CPquestdlg('Are you sure you want to over-write all your existing CellProfiler files?','Overwrite Files?','Yes','No','No');
if strcmp(Answer,'Yes')
    CPPath = which('CellProfiler.m');
    if ispc
        CPPath = CPPath(1:max(strfind(CPPath,'\'))-1);
    else
        CPPath = CPPath(1:max(strfind(CPPath,'/'))-1);
    end
    ModulePathName = fullfile(CPPath, 'Modules');
    DataPathName = fullfile(CPPath, 'DataTools');
    ImagePathName = fullfile(CPPath, 'ImageTools');

    try
        Modules = urlread('http://jura.wi.mit.edu/cellprofiler/updates/Modules/ModuleList.txt');
        DataTools = urlread('http://jura.wi.mit.edu/cellprofiler/updates/DataTools/DataList.txt');
        ImageTools = urlread('http://jura.wi.mit.edu/cellprofiler/updates/ImageTools/ImageList.txt');
    catch
        CPwarndlg('The file containing the list of modules could not be downloaded.');
        return;
    end

    p=1;
    while true
        [t,y] = strtok(Modules(p:end));
        try
            urlwrite(['http://jura.wi.mit.edu/cellprofiler/updates/Modules/',t],fullfile(ModulePathName,t));
        catch
            CPwarndlg([t,' could not be downloaded.']);
        end
        if isempty(y)
            break
        end
        p = p + length(t) + 1;
    end

    p=1;
    while true
        [t,y] = strtok(DataTools(p:end));
        try
            urlwrite(['http://jura.wi.mit.edu/cellprofiler/updates/DataTools/',t],fullfile(DataPathName,t));
        catch
            CPwarndlg([t,' could not be downloaded.']);
        end
        if isempty(y)
            break
        end
        p = p + length(t) + 1;
    end

    p=1;
    while true
        [t,y] = strtok(ImageTools(p:end));
        try
            urlwrite(['http://jura.wi.mit.edu/cellprofiler/updates/ImageTools/',t],fullfile(ImagePathName,t));
        catch
            CPwarndlg([t,' could not be downloaded.']);
        end
        if isempty(y)
            break
        end
        p = p + length(t) + 1;
    end

    try
        urlwrite('http://jura.wi.mit.edu/cellprofiler/updates/CellProfiler.m',fullfile(CPPath,'CellProfiler.m'));
    catch
        CPwarndlg('CellProfiler.m could not be downloaded.');
    end

    CPhelpdlg('Update Complete!');
end

function ReportBugs_Callback(hObject, eventdata, handles)

appdata.lastValidTag = 'figure1';
appdata.GUIDELayoutEditor = [];

ReportBugsWindow = figure(...
    'Units','characters',...
    'Color',[0.7 0.7 0.9],...
    'Colormap',[0 0 0.5625;0 0 0.625;0 0 0.6875;0 0 0.75;0 0 0.8125;0 0 0.875;0 0 0.9375;0 0 1;0 0.0625 1;0 0.125 1;0 0.1875 1;0 0.25 1;0 0.3125 1;0 0.375 1;0 0.4375 1;0 0.5 1;0 0.5625 1;0 0.625 1;0 0.6875 1;0 0.75 1;0 0.8125 1;0 0.875 1;0 0.9375 1;0 1 1;0.0625 1 1;0.125 1 0.9375;0.1875 1 0.875;0.25 1 0.8125;0.3125 1 0.75;0.375 1 0.6875;0.4375 1 0.625;0.5 1 0.5625;0.5625 1 0.5;0.625 1 0.4375;0.6875 1 0.375;0.75 1 0.3125;0.8125 1 0.25;0.875 1 0.1875;0.9375 1 0.125;1 1 0.0625;1 1 0;1 0.9375 0;1 0.875 0;1 0.8125 0;1 0.75 0;1 0.6875 0;1 0.625 0;1 0.5625 0;1 0.5 0;1 0.4375 0;1 0.375 0;1 0.3125 0;1 0.25 0;1 0.1875 0;1 0.125 0;1 0.0625 0;1 0 0;0.9375 0 0;0.875 0 0;0.8125 0 0;0.75 0 0;0.6875 0 0;0.625 0 0;0.5625 0 0],...
    'IntegerHandle','off',...
    'InvertHardcopy',get(0,'defaultfigureInvertHardcopy'),...
    'MenuBar','none',...
    'Name','Report Bugs',...
    'NumberTitle','off',...
    'PaperPosition',get(0,'defaultfigurePaperPosition'),...
    'Position',[103.8 32.2115384615385 81.8333333333333 29.25],...
    'Resize','off',...
    'HandleVisibility','callback',...
    'Tag','Report Bugs Window',...
    'UserData',[],...
    'Behavior',get(0,'defaultfigureBehavior'),...
    'Visible','on',...
    'CreateFcn', {@local_CreateFcn, '', appdata} );

appdata = [];
appdata.lastValidTag = 'edit1';

ReportBugsWindowHandles.MailBody = uicontrol(...
    'Parent',ReportBugsWindow,...
    'Units','characters',...
    'BackgroundColor',[1 1 1],...
    'FontSize','fontsize',handles.Preferences.FontSize,...
    'Position',[2.83333333333333 4.75 75.1666666666667 20.0833333333333],...
    'String','',...
    'HorizontalAlignment','left',...
    'Style','edit',...
    'Max',1000,...
    'Min',0,...
    'CreateFcn', {@local_CreateFcn, '', appdata} ,...
    'Tag','edit1',...
    'Behavior',get(0,'defaultuicontrolBehavior'));

appdata = [];
appdata.lastValidTag = 'edit2';

ReportBugsWindowHandles.MailSubject = uicontrol(...
    'Parent',ReportBugsWindow,...
    'Units','characters',...
    'BackgroundColor',[1 1 1],...
    'FontSize','fontsize',handles.Preferences.FontSize,...
    'Position',[15.5 26.0833333333333 62.5 1.91666666666667],...
    'String','',...
    'HorizontalAlignment','left',...
    'Style','edit',...
    'CreateFcn', {@local_CreateFcn, '', appdata} ,...
    'Tag','edit2',...
    'Behavior',get(0,'defaultuicontrolBehavior'));

appdata = [];
appdata.lastValidTag = 'text1';

uicontrol(...
    'Parent',ReportBugsWindow,...
    'Units','characters',...
    'BackgroundColor',[0.7 0.7 0.9],...
    'FontName','Helvetica',...
    'FontSize','fontsize',handles.Preferences.FontSize,...
    'FontWeight','bold',...
    'Position',[2.5 26.3333333333333 12.3333333333333 1.5],...
    'String','Subject:',...
    'Style','text',...
    'Tag','text1',...
    'Behavior',get(0,'defaultuicontrolBehavior'),...
    'CreateFcn', {@local_CreateFcn, '', appdata} );

appdata = [];
appdata.lastValidTag = 'pushbutton2';

uicontrol(...
    'Parent',ReportBugsWindow,...
    'Units','characters',...
    'BackgroundColor',[0.7 0.7 0.9],...
    'Callback','close ''Report Bugs''',...
    'FontSize','fontsize',handles.Preferences.FontSize,...
    'FontWeight','bold',...
    'Position',[61.8333333333333 1.16666666666667 16.1666666666667 1.83333333333333],...
    'String','Cancel',...
    'Tag','Cancel',...
    'Behavior',get(0,'defaultuicontrolBehavior'),...
    'CreateFcn', {@local_CreateFcn, '', appdata} );

appdata = [];
appdata.lastValidTag = 'pushbutton1';

ReportBugsWindowHandles.Submit = uicontrol(...
    'Parent',ReportBugsWindow,...
    'Units','characters',...
    'BackgroundColor',[0.7 0.7 0.9],...
    'Callback','try;setpref(''Internet'',''SMTP_Server'',''mail'');fig=guidata(gcf);sendmail(''cellprofiler@csail.mit.edu'',get(fig.MailSubject,''String''),get(fig.MailBody,''String''));CPmsgbox(''Bug report successfuly submitted.'');catch;CPwarndlg(''Error while sending mail'');end;',...
    'CData',[],...
    'FontName','Helvetica',...
    'FontSize','fontsize',handles.Preferences.FontSize,...
    'FontWeight','bold',...
    'Position',[44.1666666666667 1.16666666666667 16.1666666666667 1.83333333333333],...
    'String','Submit Report',...
    'Tag','Submit Report',...
    'UserData',[],...
    'Behavior',get(0,'defaultuicontrolBehavior'),...
    'CreateFcn', {@local_CreateFcn, '', appdata} );

guidata(ReportBugsWindow, ReportBugsWindowHandles);

function OpenImage_Callback(hObject, eventdata, handles)
% hObject    handle to OpenImage (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
OpenNewImageFile(handles);

function LoadPreferences_Callback(hObject,eventdata,handles)
%%% This function will load settings which have been saved from the Set
%%% Preferences window

[SettingsFileName, SettingsPathname] = CPuigetfile('*.mat','Choose a preferences file',matlabroot);
if isequal(SettingsFileName,0) || isequal(SettingsPathname,0)
    return
else
    try
        load(fullfile(SettingsPathname,SettingsFileName));
        EnteredPreferences = SavedPreferences;
        clear SavedPreferences
    catch
        CPwarndlg('The file chosen does not exist');
        return
    end
end
LoadPreferences_Helper(hObject,eventdata,handles,EnteredPreferences);

function LoadPreferences_Helper(hObject,eventdata,handles,EnteredPreferences)

if exist('EnteredPreferences','var')
    if ~isempty(EnteredPreferences)
        %%% Retrieves the data that the user entered and saves it to the
        %%% handles structure.
        handles.Preferences.PixelSize = EnteredPreferences.PixelSize;
        handles.Preferences.FontSize  = str2double(EnteredPreferences.FontSize);
        handles.Preferences.DefaultImageDirectory = EnteredPreferences.DefaultImageDirectory;
        handles.Preferences.DefaultOutputDirectory = EnteredPreferences.DefaultOutputDirectory;
        handles.Preferences.DefaultModuleDirectory = EnteredPreferences.DefaultModuleDirectory;
        handles.Preferences.IntensityColorMap = EnteredPreferences.IntensityColorMap;
        handles.Preferences.LabelColorMap = EnteredPreferences.LabelColorMap;
        handles.Preferences.StripPipeline = EnteredPreferences.StripPipeline;
        handles.Preferences.SkipErrors = EnteredPreferences.SkipErrors;
        try
            handles.Preferences.DisplayModeValue = EnteredPreferences.DisplayModeValue;
        catch
            handles.Preferences.DisplayModeValue = 1;
        end
        clear global EnteredPreferences

        %%% Now that handles.Preferences.(5 different variables) has been filled
        %%% in, the handles.Current values and edit box displays are set.
        handles.Current.DefaultOutputDirectory = handles.Preferences.DefaultOutputDirectory;
        handles.Current.DefaultImageDirectory = handles.Preferences.DefaultImageDirectory;
        %        handles.Current.PixelSize = handles.Preferences.PixelSize;
        % handles.Current.FontSize  = str2num(handles.Preferences.FontSize);
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
        if ~isdeployed
            addpath(handles.Preferences.DefaultModuleDirectory)
        end

        %%% Set new fontsize...
        names = fieldnames(handles);
        for k = 1:length(names)
            if ishandle(handles.(names{k}))
                set(findobj(handles.(names{k}),'-property','FontSize'),'FontSize',handles.Preferences.FontSize,'FontName','helvetica');
            end
        end
        %%% ... and make it the new default.
        set(0, 'defaultuicontrolfontsize', handles.Preferences.FontSize);
        set(0, 'defaultuicontrolfontname', 'helvetica');

        %%% Updates the handles structure to incorporate all the changes.
        guidata(gcbo, handles);
    end
end

function ZipFiles_Callback(hObject, eventdata, handles)
ListOfThingsToSave = {'CPsubfunctions/CPsplash.jpg' ...
    'CPsubfunctions/*.m' 'CPsubfunctions/graphAnalysisToolbox-1.0/*.m' ...
    'CPsubfunctions/CPljosaprobseg.*' ...
    'DataTools/*.m' 'ImageTools/*.m' 'Modules/*.m' ...
    'Modules/IdentifySecPropagateSubfunction.*' 'Help/*.m' 'CellProfiler.m'};
if ispc
    for i=1:numel(ListOfThingsToSave)
        ListOfThingsToSave{i} = strrep(ListOfThingsToSave{i}, '/', '\');
    end
end
try
    ZipFileName = [handles.Current.DefaultOutputDirectory '/CellProfilerCode_',date,'.zip'];
    zip(ZipFileName,ListOfThingsToSave,handles.Current.CellProfilerPathname);
    CPmsgbox(['The files have been saved to ', ZipFileName, '.']);
catch
    CPhelpdlg(['The files could not be saved for some reason.  This could be because you do not have access to folder ' handles.Current.DefaultOutputDirectory '  Make sure you have access or you can change the default output directory by going to ''set preferences'' on the main menu.']);
end

% --- Executes just before AddModuleWindow_export is made visible.
function AddModuleWindow_OpeningFcn(hObject, eventdata, AddModuleWindowHandles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% AddModuleWindowHandles    structure with AddModuleWindowHandles and user data (see GUIDATA)
% varargin   command line arguments to AddModuleWindow_export (see VARARGIN)

% Choose default command line output for AddModuleWindow_export

% Update AddModuleWindowHandles structure
handles=guidata(hObject);
load_listbox(handles.Preferences.DefaultModuleDirectory,AddModuleWindowHandles);

% UIWAIT makes AddModuleWindow_export wait for user response (see UIRESUME)
% uiwait(AddModuleWindowHandles.figure1);

% --- Executes on selection change in PreProcessingListBox.
function AddModuleListBox_Callback(hObject, eventdata, AddModuleWindowHandles) %#ok
% hObject    handle to PreProcessingListBox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% AddModuleWindowHandles    structure with AddModuleWindowHandles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns PreProcessingListBox contents as cell array
%        contents{get(hObject,'Value')} returns selected item from
%        PreProcessingListBox
if strcmp(get(gcf,'SelectionType'),'open')
    if(~isempty(get(AddModuleWindowHandles.ModulesListBox,'Value')))
        index_selected = get(AddModuleWindowHandles.ModulesListBox,'Value');
        file_list = get(AddModuleWindowHandles.ModulesListBox,'String');
    else
        return;
    end

    handles=guidata(AddModuleWindowHandles.figure1);
    if isdeployed
        filename = [file_list{index_selected} '.txt'];
    else
        filename = [file_list{index_selected} '.m'];
    end
    PutModuleInListBox(filename,handles.Preferences.DefaultModuleDirectory,guidata(AddModuleWindowHandles.figure1),0);
end

% --- Executes on button press in ModuleHelpButton.
function ModuleHelpButton_Callback(hObject, eventdata, AddModuleWindowHandles)
% hObject    handle to ModuleHelpButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% AddModuleWindowHandles    structure with AddModuleWindowHandles and user data (see GUIDATA)

if ~isempty(get(AddModuleWindowHandles.ModulesListBox,'Value'))
    index_selected = get(AddModuleWindowHandles.ModulesListBox,'Value');
    file_list = get(AddModuleWindowHandles.ModulesListBox,'String');
else
    CPwarndlg('You must highlight a module before attempting to get help for it!');
    return;
end
filename = file_list{index_selected};

if isdeployed
    handles = guidata(findobj('tag','figure1'));
    for i = 1:length(handles.Current.ModulesFilenames)
        if strmatch(filename,handles.Current.ModulesFilenames{i},'exact')
            Option = i;
            break
        end
    end
    if ~isempty(Option)
        CPtextdisplaybox(handles.Current.ModulesHelp{Option},'CellProfiler image analysis module help');
    end
else
    CPtextdisplaybox(help(filename),'CellProfiler image analysis module help');
end

function BrowseButton_Callback(hObject, eventdata, AddModuleWindowHandles) %#ok Ignore MLint
% hObject    handle to PreProcessingListBox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% AddModuleWindowHandles    empty - AddModuleWindowHandles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
handles = guidata(AddModuleWindowHandles.figure1);
Path = handles.Preferences.DefaultModuleDirectory;
if ~exist(Path,'dir')
    Path = cd;
end
[FileName PathName] = CPuigetfile('*.m','Choose an image analysis module',Path);
pause(.1);
figure(handles.figure1);
try
    figure(AddModuleWindowHandles.AddModuleWindow);
end
PutModuleInListBox(FileName,PathName,handles,0);

% --- Creates and returns a handle to the GUI figure.
function AddModuleWindowHandles = AddModuleWindow_LayoutFcn(figure1)
% policy - create a new figure or use a singleton. 'new' or 'reuse'.

AddModuleWindowHandles.figure1=figure1;

handles=guidata(figure1);
font=handles.Preferences.FontSize;

appdata = [];
appdata.GUIDEOptions = struct(...
    'active_h', [], ...
    'taginfo', struct(...
    'figure', 2, ...
    'listbox', 6, ...
    'text', 12, ...
    'pushbutton', 6), ...
    'override', 0, ...
    'release', 13, ...
    'resize', 'none', ...
    'accessibility', 'callback', ...
    'mfile', 1, ...
    'callbacks', 1, ...
    'singleton', 1, ...
    'syscolorfig', 1, ...
    'blocking', 0);
appdata.lastValidTag = 'AddModuleWindow';
appdata.GUIDELayoutEditor = [];

set(figure1,'Units','characters');
set(findobj('Tag','ModulePipelineListBox'),'Units','characters');
pos1=get(figure1,'Position');
pos2=get(findobj('Tag','ModulePipelineListBox'),'Position');
set(figure1,'Units','pixels');
set(findobj('Tag','ModulePipelineListBox'),'Units','pixels');

userData.Application = 'CellProfiler';

AddModuleWindow = figure(...
    'Units','characters',...
    'Color',[0.7 0.7 0.9],...
    'Colormap',[0 0 0.5625;0 0 0.625;0 0 0.6875;0 0 0.75;0 0 0.8125;0 0 0.875;0 0 0.9375;0 0 1;0 0.0625 1;0 0.125 1;0 0.1875 1;0 0.25 1;0 0.3125 1;0 0.375 1;0 0.4375 1;0 0.5 1;0 0.5625 1;0 0.625 1;0 0.6875 1;0 0.75 1;0 0.8125 1;0 0.875 1;0 0.9375 1;0 1 1;0.0625 1 1;0.125 1 0.9375;0.1875 1 0.875;0.25 1 0.8125;0.3125 1 0.75;0.375 1 0.6875;0.4375 1 0.625;0.5 1 0.5625;0.5625 1 0.5;0.625 1 0.4375;0.6875 1 0.375;0.75 1 0.3125;0.8125 1 0.25;0.875 1 0.1875;0.9375 1 0.125;1 1 0.0625;1 1 0;1 0.9375 0;1 0.875 0;1 0.8125 0;1 0.75 0;1 0.6875 0;1 0.625 0;1 0.5625 0;1 0.5 0;1 0.4375 0;1 0.375 0;1 0.3125 0;1 0.25 0;1 0.1875 0;1 0.125 0;1 0.0625 0;1 0 0;0.9375 0 0;0.875 0 0;0.8125 0 0;0.75 0 0;0.6875 0 0;0.625 0 0;0.5625 0 0],...
    'DockControls','off',...
    'IntegerHandle','off',...
    'InvertHardcopy',get(0,'defaultfigureInvertHardcopy'),...
    'KeyPressFcn','if strcmp(get(gcf,''CurrentCharacter''),''''), close(gcf), end;',...
    'MenuBar','none',...
    'Name','AddModule',...
    'NumberTitle','off',...
    'PaperPosition',get(0,'defaultfigurePaperPosition'),...
    'Position',[pos1(1)+pos2(1)+pos2(3) pos1(2)+pos2(2) 88 29],...
    'Resize','off',...
    'HandleVisibility','callback',...
    'Tag','AddModuleWindow',...
    'UserData',userData,...
    'Behavior',get(0,'defaultfigureBehavior'),...
    'Visible','on',...
    'CreateFcn', {@local_CreateFcn, '', appdata} );

AddModuleWindowHandles.AddModuleWindow=AddModuleWindow;

LoadListCallback = [...
    'fig=guidata(gcf);'...
    'val=get(fig.ModuleCategoryListBox,''Value'');'...
    'if size(val,2) > 1;'...
    'val = val(1);'...
    'end,'...
    'set(fig.ModuleCategoryListBox,''Value'',val);'...
    'set(fig.ModulesListBox,''value'',1);'...
    'set(fig.ModulesListBox,''string'',fig.ModuleStrings{val});'...
    'clear fig val'];

appdata = [];
appdata.lastValidTag = 'ModuleCategoryListBox';

AddModuleWindowHandles.ModuleCategoryListBox = uicontrol(...
    'Parent',AddModuleWindow,...
    'Units','characters',...
    'BackgroundColor',[1 1 1],...
    'Callback',LoadListCallback,...
    'FontSize',font,...
    'FontName','Helvetica',...
    'Interruptible','off',...
    'Max',2,...
    'Position',[1 18 29 8],...
    'String',{  'Listbox' },...
    'Style','listbox',...
    'Value',1,...
    'Tag','ModuleCategoryListBox',...
    'Behavior',get(0,'defaultuicontrolBehavior'));

appdata = [];
appdata.lastValidTag = 'ModuleCategoryListBoxText';

AddModuleWindowHandles.ModuleCategoryListBoxText = uicontrol(...
    'Parent',AddModuleWindow,...
    'Units','characters',...
    'BackgroundColor',[.7 .7 .9],...
    'FontSize',font,...
    'FontName','Helvetica',...
    'Interruptible','off',...
    'Position',[1 26 29 2],...
    'String','Module Categories',...
    'Style','text',...
    'Tag','ModuleCategoryListBoxText',...
    'Behavior',get(0,'defaultuicontrolBehavior'));

appdata = [];
appdata.lastValidTag = 'ModulesListBox';

AddModuleWindowHandles.ModulesListBox = uicontrol(...
    'Parent',AddModuleWindow,...
    'Units','characters',...
    'BackgroundColor',[1 1 1],...
    'Callback','fig=guidata(gcf);val=get(fig.ModulesListBox,''Value'');if (~isempty(val)); set(fig.ModulesListBox,''Value'',val(1));end;CellProfiler(''AddModuleListBox_Callback'',gcbo,[],guidata(gcf));clear val fig',...
    'FontSize',font,...
    'FontName','Helvetica',...
    'Interruptible','off',...
    'Max',2,...
    'Position',[34 1 50 27],...
    'String',{  'Listbox' },...
    'Style','listbox',...
    'Value',1,...
    'Tag','ModulesListBox',...
    'Behavior',get(0,'defaultuicontrolBehavior'));

appdata = [];
appdata.lastValidTag = 'HelpButton';

AddModuleWindowHandles.HelpButton = uicontrol(...
    'Parent',AddModuleWindow,...
    'Units','characters',...
    'BackgroundColor',[0.701960784313725 0.701960784313725 0.901960784313726],...
    'Callback','handles = guidata(findobj(''tag'',''figure1''));for i = 1:length(handles.Current.GSFilenames), if strfind(handles.Current.GSFilenames{i},''GSGettingStarted''), Option = i;end,end,if ~isempty(Option),CPtextdisplaybox(handles.Current.GS{Option-1},''Getting Started in CellProfiler'');end;clear ans handles Option i;',...
    'FontSize',font,...
    'FontName','Helvetica',...
    'Position',[1 6 29 2],...
    'String','? Getting Started',...
    'Tag','HelpButton',...
    'Behavior',get(0,'defaultuicontrolBehavior'),...
    'CreateFcn', {@local_CreateFcn, '', appdata} );

appdata = [];
appdata.lastValidTag = 'ModuleHelpButton';

AddModuleWindowHandles.ModuleFunctionPanel = uipanel(...
    'Parent',AddModuleWindow,...
    'Title','For Selected Module',...
    'Units','characters',...
    'FontSize',font,...
    'BackgroundColor',[.7 .7 .9],...
    'Position',[.5 10 31.5 6.5],...
    'Tag','ModuleFunctionPanel',...
    'CreateFcn', {@local_CreateFcn, '', appdata} );

appdata = [];
appdata.lastValidTag = 'ModuleFunctionPanel';

AddModuleWindowHandles.ModuleHelpButton = uicontrol(...
    'Parent',AddModuleWindow,...
    'Units','characters',...
    'BackgroundColor',[0.701960784313725 0.701960784313725 0.901960784313726],...
    'Callback','CellProfiler(''ModuleHelpButton_Callback'',gcbo,[],guidata(gcf))',...
    'FontSize',font,...
    'FontName','Helvetica',...
    'Position',[1.5 10.5 29 2],...
    'String','? Module Help',...
    'Tag','ModuleHelpButton',...
    'Behavior',get(0,'defaultuicontrolBehavior'),...
    'CreateFcn', {@local_CreateFcn, '', appdata} );

appdata = [];
appdata.lastValidTag = 'BrowseButton';

AddModuleWindowHandles.BrowseButton = uicontrol(...
    'Parent',AddModuleWindow,...
    'Units','characters',...
    'BackgroundColor',[0.701960784313725 0.701960784313725 0.901960784313726],...
    'Callback','CellProfiler(''BrowseButton_Callback'',gcbo,[],guidata(gcf))',...
    'FontSize',font,...
    'FontName','Helvetica',...
    'Position',[1 3.5 29 2],...
    'String','Browse...',...
    'Tag','BrowseButton',...
    'Behavior',get(0,'defaultuicontrolBehavior'),...
    'CreateFcn', {@local_CreateFcn, '', appdata} );

appdata = [];
appdata.lastValidTag = 'AddModuleButton';

AddModuleWindowHandles.AddModuleButton = uicontrol(...
    'Parent',AddModuleWindow,...
    'Units','characters',...
    'BackgroundColor',[0.701960784313725 0.701960784313725 0.901960784313726],...
    'Callback','set(gcf,''SelectionType'',''open'');CellProfiler(''AddModuleListBox_Callback'',gcbo,[],guidata(gcf))',...
    'FontSize',font,...
    'FontName','Helvetica',...
    'Position',[1.5 13 29 2],...
    'String','+ Add To Pipeline',...
    'Tag','AddModuleButton',...
    'Behavior',get(0,'defaultuicontrolBehavior'),...
    'CreateFcn', {@local_CreateFcn, '', appdata} );

appdata = [];
appdata.lastValidTag = 'DoneButton';

AddModuleWindowHandles.DoneButton = uicontrol(...
    'Parent',AddModuleWindow,...
    'Units','characters',...
    'BackgroundColor',[0.701960784313725 0.701960784313725 0.901960784313726],...
    'Callback','close(gcf)',...
    'FontSize',font,...
    'FontName','Helvetica',...
    'Position',[1 1 29 2],...
    'String','Done',...
    'Tag','DoneButton',...
    'Behavior',get(0,'defaultuicontrolBehavior'),...
    'CreateFcn', {@local_CreateFcn, '', appdata} );

guidata(AddModuleWindow,AddModuleWindowHandles);

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


function load_listbox(dir_path,AddModuleWindowHandles)
%%% Do not remove the BEGIN or END lines, below.  They are used by CompileWizard.m.
%%% Compiler: BEGIN load_listbox
dir_struct = dir([dir_path '/*.m']);
FileProcessingFiles ={};
PreProcessingFiles={};
ObjectProcessingFiles={};
MeasurementFiles={};
OtherFiles={};
for i=1:length(dir_struct)
    name=dir_struct(i).name;
    name=name(1:end-2);
    if file_in_category(dir_struct(i).name, 'File Processing')
        FileProcessingFiles(length(FileProcessingFiles)+1)=cellstr(name);
    elseif file_in_category(dir_struct(i).name, 'Image Processing')
        PreProcessingFiles(length(PreProcessingFiles)+1)=cellstr(name);
    elseif file_in_category(dir_struct(i).name, 'Object Processing')
        ObjectProcessingFiles(length(ObjectProcessingFiles)+1)=cellstr(name);
    elseif file_in_category(dir_struct(i).name, 'Measurement')
        MeasurementFiles(length(MeasurementFiles)+1)=cellstr(name);
    else
        OtherFiles(length(OtherFiles)+1)=cellstr(name);
    end
end
CategoryList = {'File Processing' 'Image Processing' 'Object Processing' 'Measurement' 'Other'};
set(AddModuleWindowHandles.ModuleCategoryListBox,'String',CategoryList,...
    'Value',[])
set(AddModuleWindowHandles.ModulesListBox,'String',FileProcessingFiles,...
    'Value',[])
AddModuleWindowHandles.ModuleStrings{1} = FileProcessingFiles;
AddModuleWindowHandles.ModuleStrings{2} = PreProcessingFiles;
AddModuleWindowHandles.ModuleStrings{3} = ObjectProcessingFiles;
AddModuleWindowHandles.ModuleStrings{4} = MeasurementFiles;
AddModuleWindowHandles.ModuleStrings{5} = OtherFiles;

guidata(AddModuleWindowHandles.AddModuleWindow,AddModuleWindowHandles);
%%% Compiler: END load_listbox
%%% Do not remove the BEGIN or END lines, above.  They are used by CompileWizard.m.

function c = file_in_category(filename, category)
h = help(filename);
c = strfind(h, ['Category: ' category]);


% --- Creates and returns a handle to the GUI figure.
function h1 = CellProfiler_LayoutFcn(policy)
% policy - create a new figure or use a singleton. 'new' or 'reuse'.

persistent hsingleton;
if strcmpi(policy, 'reuse') & ishandle(hsingleton) %#ok Ignore MLint
    h1 = hsingleton;
    return;
end

appdata = [];
appdata.GUIDEOptions = struct(...
    'active_h', 187.003662109375, ...
    'taginfo', struct(...
    'figure', 2, ...
    'listbox', 13, ...
    'popupmenu', 12, ...
    'frame', 38, ...
    'edit', 89, ...
    'pushbutton', 113, ...
    'text', 186, ...
    'axes', 2, ...
    'uipanel', 5, ...
    'slider', 4), ...
    'override', 1, ...
    'release', 13, ...
    'resize', 'none', ...
    'accessibility', 'callback', ...
    'mfile', 1, ...
    'callbacks', 1, ...
    'singleton', 1, ...
    'syscolorfig', 0, ...
    'blocking', 0);
appdata.lastValidTag = 'figure1';
appdata.UsedByGUIData_m = struct(...
    'AlgorithmHighlighted', 'No Algorithms Loaded');
appdata.GUIDELayoutEditor = [];

h1 = figure(...
    'Color',[0.701960784313725 0.701960784313725 0.701960784313725],...
    'Colormap',[0 0 0.5625;0 0 0.625;0 0 0.6875;0 0 0.75;0 0 0.8125;0 0 0.875;0 0 0.9375;0 0 1;0 0.0625 1;0 0.125 1;0 0.1875 1;0 0.25 1;0 0.3125 1;0 0.375 1;0 0.4375 1;0 0.5 1;0 0.5625 1;0 0.625 1;0 0.6875 1;0 0.75 1;0 0.8125 1;0 0.875 1;0 0.9375 1;0 1 1;0.0625 1 1;0.125 1 0.9375;0.1875 1 0.875;0.25 1 0.8125;0.3125 1 0.75;0.375 1 0.6875;0.4375 1 0.625;0.5 1 0.5625;0.5625 1 0.5;0.625 1 0.4375;0.6875 1 0.375;0.75 1 0.3125;0.8125 1 0.25;0.875 1 0.1875;0.9375 1 0.125;1 1 0.0625;1 1 0;1 0.9375 0;1 0.875 0;1 0.8125 0;1 0.75 0;1 0.6875 0;1 0.625 0;1 0.5625 0;1 0.5 0;1 0.4375 0;1 0.375 0;1 0.3125 0;1 0.25 0;1 0.1875 0;1 0.125 0;1 0.0625 0;1 0 0;0.9375 0 0;0.875 0 0;0.8125 0 0;0.75 0 0;0.6875 0 0;0.625 0 0;0.5625 0 0],...
    'DockControls','off',...
    'IntegerHandle','off',...
    'InvertHardcopy',get(0,'defaultfigureInvertHardcopy'),...
    'MenuBar','none',...
    'Name','CellProfiler',...
    'NumberTitle','off',...
    'PaperPosition',get(0,'defaultfigurePaperPosition'),...
    'Position',[428   295   824   460],...
    'Renderer',get(0,'defaultfigureRenderer'),...
    'RendererMode','manual',...
    'Resize','off',...
    'HandleVisibility','callback',...
    'Tag','figure1',...
    'UserData',[],...
    'Behavior',get(0,'defaultfigureBehavior'),...
    'Visible','off',...
    'CreateFcn', {@local_CreateFcn, '', appdata} );

appdata = [];
appdata.lastValidTag = 'toppanel';

h2 = uipanel(...
    'Parent',h1,...
    'Units','pixels',...
    'BorderType','none',...
    'ForegroundColor',[0.698039215686274 0.698039215686274 0.898039215686275],...
    'Tag','toppanel',...
    'UserData',[],...
    'Behavior',get(0,'defaultuipanelBehavior'),...
    'Clipping','on',...
    'BackgroundColor',[.7 .7 .9],...
    'Position',[0 108 824 354],...
    'CreateFcn', {@local_CreateFcn, '', appdata} );

appdata = [];
appdata.lastValidTag = 'CloseFigureButton';

h3 = uicontrol(...
    'Parent',h2,...
    'BackgroundColor',[.7 .7 .9],...
    'Callback','CellProfiler(''CloseFigureButton_Callback'',gcbo,[],guidata(gcbo))',...
    'CData',[],...
    'FontWeight','bold',...
    'Position',[125 18 90 20],...
    'String','Close Figure',...
    'Tag','CloseFigureButton',...
    'UserData',[],...
    'Behavior',get(0,'defaultuicontrolBehavior'),...
    'Visible','off',...
    'CreateFcn', {@local_CreateFcn, '', appdata} ); %#ok Ignore MLint

appdata = [];
appdata.lastValidTag = 'OpenFigureButton';

h4 = uicontrol(...
    'Parent',h2,...
    'BackgroundColor',[.7 .7 .9],...
    'Callback','CellProfiler(''OpenFigureButton_Callback'',gcbo,[],guidata(gcbo))',...
    'CData',[],...
    'FontWeight','bold',...
    'Position',[25 18 90 20],...
    'String','Open Figure',...
    'Tag','OpenFigureButton',...
    'UserData',[],...
    'Behavior',get(0,'defaultuicontrolBehavior'),...
    'Visible','off',...
    'CreateFcn', {@local_CreateFcn, '', appdata} ); %#ok Ignore MLint

appdata = [];
appdata.lastValidTag = 'IndividualModulesText';

h5 = uicontrol(...
    'Parent',h2,...
    'BackgroundColor',[.7 .7 .9],...
    'CData',[],...
    'FontWeight','bold',...
    'HorizontalAlignment','right',...
    'Position',[16 19 118 14],...
    'String','Adjust modules:',...
    'Style','text',...
    'Tag','IndividualModulesText',...
    'UserData',[],...
    'Behavior',get(0,'defaultuicontrolBehavior'),...
    'CreateFcn', {@local_CreateFcn, '', appdata} ); %#ok Ignore MLint

appdata = [];
appdata.lastValidTag = 'AddModule';

h6 = uicontrol(...
    'Parent',h2,...
    'BackgroundColor',[.7 .7 .9],...
    'Callback','CellProfiler(''AddModule_Callback'',gcbo,[],guidata(gcbo))',...
    'CData',[],...
    'FontWeight','bold',...
    'Position',[137 16 18 20],...
    'String','+',...
    'Tag','AddModule',...
    'UserData',[],...
    'Behavior',get(0,'defaultuicontrolBehavior'),...
    'CreateFcn', {@local_CreateFcn, '', appdata} ); %#ok Ignore MLint
appdata = [];
appdata.lastValidTag = 'RemoveModule';

h7 = uicontrol(...
    'Parent',h2,...
    'BackgroundColor',[.7 .7 .9],...
    'Callback','CellProfiler(''RemoveModule_Callback'',gcbo,[],guidata(gcbo))',...
    'FontWeight','bold',...
    'Position',[158 16 18 20],...
    'String','-',...
    'Tag','RemoveModule',...
    'Behavior',get(0,'defaultuicontrolBehavior'),...
    'CreateFcn', {@local_CreateFcn, '', appdata} ); %#ok Ignore MLint

appdata = [];
appdata.lastValidTag = 'MoveUpButton';

h8 = uicontrol(...
    'Parent',h2,...
    'BackgroundColor',[.7 .7 .9],...
    'Callback','CellProfiler(''MoveUpButton_Callback'',gcbo,[],guidata(gcbo))',...
    'CData',[],...
    'Position',[179 16 18 20],...
    'String','^',...
    'Tag','MoveUpButton',...
    'UserData',[],...
    'Behavior',get(0,'defaultuicontrolBehavior'),...
    'CreateFcn', {@local_CreateFcn, '', appdata} ); %#ok Ignore MLint

appdata = [];
appdata.lastValidTag = 'MoveDownButton';

h9 = uicontrol(...
    'Parent',h2,...
    'BackgroundColor',[.7 .7 .9],...
    'Callback','CellProfiler(''MoveDownButton_Callback'',gcbo,[],guidata(gcbo))',...
    'CData',[],...
    'Position',[200 16 18 20],...
    'String','v',...
    'Tag','MoveDownButton',...
    'UserData',[],...
    'Behavior',get(0,'defaultuicontrolBehavior'),...
    'CreateFcn', {@local_CreateFcn, '', appdata} ); %#ok Ignore MLint

appdata = [];
appdata.lastValidTag = 'ModulePipelineListBox';

h10 = uicontrol(...
    'Parent',h2,...
    'BackgroundColor',[1 1 1],...
    'Callback','CellProfiler(''ModulePipelineListBox_Callback'',gcbo,[],guidata(gcbo))',...
    'CData',[],...
    'Max',2,...
    'Position',[12 40 206 235],...
    'String',{  'No Modules Loaded' },...
    'Style','listbox',...
    'Value',1,...
    'Tag','ModulePipelineListBox',...
    'KeyPressFcn',@RemoveModuleByKeyPressFcn,...
    'UserData',[],...
    'Behavior',get(0,'defaultuicontrolBehavior')); %#ok Ignore MLint

appdata = [];
appdata.lastValidTag = 'variablepanel';

h11 = uipanel(...
    'Parent',h2,...
    'Units','pixels',...
    'BorderType','none',...
    'ForegroundColor',[0.698039215686274 0.698039215686274 0.898039215686275],...
    'Tag','variablepanel',...
    'UserData',[],...
    'Behavior',get(0,'defaultuipanelBehavior'),...
    'Clipping','on',...
    'BackgroundColor',[.7 .7 .9],...
    'Position',[240 0 563 584],...
    'CreateFcn', {@local_CreateFcn, '', appdata} ); %#ok Ignore MLint

appdata = [];
appdata.lastValidTag = 'slider1';

h12 = uicontrol(...
    'Parent',h2,...
    'BackgroundColor',[0.9 0.9 0.9],...
    'Callback','CellProfiler(''slider1_Callback'',gcbo,[],guidata(gcbo))',...
    'Position',[800 0 20 348],...
    'String',{  'Slider' },...
    'Style','slider',...
    'SliderStep',[0.02 0.1],...
    'Value',1,...
    'Tag','slider1',...
    'Behavior',get(0,'defaultuicontrolBehavior'),...
    'Visible','off'); %#ok Ignore MLint

appdata = [];
appdata.lastValidTag = 'IndividualModuleHelp';

h13 = uicontrol(...
    'Parent',h2,...
    'BackgroundColor',[.7 .7 .9],...
    'Callback','CellProfiler(''IndividualModuleHelp_Callback'',gcbo,[],guidata(gcbo))',...
    'CData',[],...
    'FontWeight','bold',...
    'Position',[12 17 12 22],...
    'String','?',...
    'Tag','IndividualModuleHelp',...
    'UserData',[],...
    'Behavior',get(0,'defaultuicontrolBehavior'),...
    'CreateFcn', {@local_CreateFcn, '', appdata} ); %#ok Ignore MLint

appdata = [];
appdata.lastValidTag = 'PipelineText';

if ispc
    h14height = 21.5;
else
    h14height = 23.5;
end

h14 = uicontrol(...
    'Parent',h2,...
    'Units','characters',...
    'BackgroundColor',[.7 .7 .9],...
    'FontWeight','bold',...
    'Position',[18 h14height 22 4],...
    'String',{'CellProfiler';'image analysis';'pipeline:'},...
    'Style','text',...
    'Tag','PipelineText',...
    'HorizontalAlignment','center',...
    'Behavior',get(0,'defaultuicontrolBehavior'),...
    'CreateFcn', {@local_CreateFcn, '', appdata} ); %#ok Ignore MLint

appdata = [];
appdata.lastValidTag = 'bottompanel';

h15 = uipanel(...
    'Parent',h1,...
    'Units','pixels',...
    'ShadowColor',[.7 .7 .9],...
    'BorderType','none',...
    'Tag','bottompanel',...
    'UserData',[],...
    'Behavior',get(0,'defaultuipanelBehavior'),...
    'Clipping','on',...
    'BackgroundColor',[.7 .7 .9],...
    'Position',[0 0 824 108],...
    'CreateFcn', {@local_CreateFcn, '', appdata} );

appdata = [];
appdata.lastValidTag = 'AnalyzeImagesButton';

h16 = uicontrol(...
    'Parent',h15,...
    'BackgroundColor',[.7 .7 .9],...
    'Callback','CellProfiler(''AnalyzeImagesButton_Callback'',gcbo,[],guidata(gcbo))',...
    'FontWeight','bold',...
    'Position',[715 8 108 22],...
    'String','Analyze images',...
    'Tag','AnalyzeImagesButton',...
    'Behavior',get(0,'defaultuicontrolBehavior'),...
    'CreateFcn', {@local_CreateFcn, '', appdata} ); %#ok Ignore MLint

appdata = [];
appdata.lastValidTag = 'OutputFilenameHelp';

h17 = uicontrol(...
    'Parent',h15,...
    'BackgroundColor',[.7 .7 .9],...
    'Callback','handles = guidata(findobj(''tag'',''figure1''));for i = 1:length(handles.Current.HelpFilenames), if strfind(handles.Current.HelpFilenames{i},''HelpOutputFilename''), Option = i;end,end,if ~isempty(Option),CPtextdisplaybox(handles.Current.Help{Option-1},''CellProfiler Help'');end;clear ans handles Option i;',...
    'FontWeight','bold',...
    'Position',[683 8 12 22],...
    'String','?',...
    'Tag','OutputFilenameHelp',...
    'Behavior',get(0,'defaultuicontrolBehavior'),...
    'CreateFcn', {@local_CreateFcn, '', appdata} ); %#ok Ignore MLint

appdata = [];
appdata.lastValidTag = 'DefaultImageDirectoryText';

h18 = uicontrol(...
    'Parent',h15,...
    'BackgroundColor',[.7 .7 .9],...
    'CData',[],...
    'FontWeight','bold',...
    'HorizontalAlignment','left',...
    'Position',[235 58 100 35],...
    'String',{  'Default image'; 'folder:' },...
    'Style','text',...
    'Tag','DefaultImageDirectoryText',...
    'UserData',[],...
    'Behavior',get(0,'defaultuicontrolBehavior'),...
    'CreateFcn', {@local_CreateFcn, '', appdata} ); %#ok Ignore MLint

appdata = [];
appdata.lastValidTag = 'BrowseImageDirectoryButton';

h19 = uicontrol(...
    'Parent',h15,...
    'HorizontalAlignment','left',...
    'BackgroundColor',[.7 .7 .9],...
    'Callback','CellProfiler(''BrowseImageDirectoryButton_Callback'',gcbo,[],guidata(gcbo))',...
    'FontWeight','bold',...
    'Position',[760 71 63 20],...
    'String','Browse...',...
    'Tag','BrowseImageDirectoryButton',...
    'Behavior',get(0,'defaultuicontrolBehavior'),...
    'CreateFcn', {@local_CreateFcn, '', appdata} ); %#ok Ignore MLint
appdata = [];
appdata.lastValidTag = 'DefaultImageDirectoryEditBox';

h20 = uicontrol(...
    'Parent',h15,...
    'BackgroundColor',[1 1 1],...
    'Callback','CellProfiler(''DefaultImageDirectoryEditBox_Callback'',gcbo,[],guidata(gcbo))',...
    'Position',[335 70 420 24],...
    'String','',...
    'Style','edit',...
    'Tag','DefaultImageDirectoryEditBox',...
    'Behavior',get(0,'defaultuicontrolBehavior')); %#ok Ignore MLint

appdata = [];
appdata.lastValidTag = 'DefaultImageDirectoryHelp';

h21 = uicontrol(...
    'Parent',h15,...
    'BackgroundColor',[.7 .7 .9],...
    'Callback','handles = guidata(findobj(''tag'',''figure1''));for i = 1:length(handles.Current.HelpFilenames), if strfind(handles.Current.HelpFilenames{i},''HelpDefaultImageFolder''), Option = i;end,end,if ~isempty(Option),CPtextdisplaybox(handles.Current.Help{Option-1},''Default Image Folder Help'');end;clear ans handles Option i;',...
    'FontWeight','bold',...
    'Position',[220 71 12 22],...
    'String','?',...
    'Tag','DefaultImageFolderHelp',...
    'Behavior',get(0,'defaultuicontrolBehavior'),...
    'CreateFcn', {@local_CreateFcn, '', appdata} ); %#ok Ignore MLint

appdata = [];
appdata.lastValidTag = 'NameOutputFileText';

h22 = uicontrol(...
    'Parent',h15,...
    'BackgroundColor',[.7 .7 .9],...
    'CData',[],...
    'FontWeight','bold',...
    'HorizontalAlignment','right',...
    'Position',[435 2 68 30],...
    'String',{  'Output'; 'filename:' },...
    'Style','text',...
    'Tag','NameOutputFileText',...
    'UserData',[],...
    'Behavior',get(0,'defaultuicontrolBehavior'),...
    'CreateFcn', {@local_CreateFcn, '', appdata} ); %#ok Ignore MLint

appdata = [];
appdata.lastValidTag = 'OutputFileNameEditBox';

h23 = uicontrol(...
    'Parent',h15,...
    'BackgroundColor',[1 1 1],...
    'Callback','CellProfiler(''OutputFileNameEditBox_Callback'',gcbo,[],guidata(gcbo))',...
    'Position',[510 8 170 22],...
    'String','',...
    'Style','edit',...
    'Tag','OutputFileNameEditBox',...
    'Behavior',get(0,'defaultuicontrolBehavior')); %#ok Ignore MLint

appdata = [];
appdata.lastValidTag = 'FilenamesListBox';

h24 = uicontrol(...
    'Parent',h15,...
    'BackgroundColor',[1 1 1],...
    'Callback','CellProfiler(''FilenamesListBox_Callback'',gcbo,[],guidata(gcbo))',...
    'Interruptible','off',...
    'Position',[12 8 206 105],...
    'String',{  'Listbox' },...
    'Style','listbox',...
    'Value',1,...
    'Tag','FilenamesListBox',...
    'Behavior',get(0,'defaultuicontrolBehavior')); %#ok Ignore MLint

appdata = [];
appdata.lastValidTag = 'DefaultOutputDirectoryText';

h25 = uicontrol(...
    'Parent',h15,...
    'BackgroundColor',[.7 .7 .9],...
    'CData',[],...
    'FontWeight','bold',...
    'HorizontalAlignment','left',...
    'Position',[235 30 100 35],...
    'String',{  'Default output'; 'folder:' },...
    'Style','text',...
    'Tag','DefaultOutputDirectoryText',...
    'UserData',[],...
    'Behavior',get(0,'defaultuicontrolBehavior'),...
    'CreateFcn', {@local_CreateFcn, '', appdata} ); %#ok Ignore MLint

appdata = [];
appdata.lastValidTag = 'BrowseOutputDirectoryButton';

h26 = uicontrol(...
    'Parent',h15,...
    'HorizontalAlignment','left',...
    'BackgroundColor',[.7 .7 .9],...
    'Callback','CellProfiler(''BrowseOutputDirectoryButton_Callback'',gcbo,[],guidata(gcbo))',...
    'FontWeight','bold',...
    'Position',[760 43 63 20],...
    'String','Browse...',...
    'Tag','BrowseOutputDirectoryButton',...
    'Behavior',get(0,'defaultuicontrolBehavior'),...
    'CreateFcn', {@local_CreateFcn, '', appdata} ); %#ok Ignore MLint

appdata = [];
appdata.lastValidTag = 'DefaultOutputDirectoryEditBox';

h27 = uicontrol(...
    'Parent',h15,...
    'BackgroundColor',[1 1 1],...
    'Callback','CellProfiler(''DefaultOutputDirectoryEditBox_Callback'',gcbo,[],guidata(gcbo))',...
    'Position',[335 42 420 24],...
    'String','',...
    'Style','edit',...
    'Tag','DefaultOutputDirectoryEditBox',...
    'Behavior',get(0,'defaultuicontrolBehavior')); %#ok Ignore MLint

appdata = [];
appdata.lastValidTag = 'DefaultOutputDirectoryHelp';

h28 = uicontrol(...
    'Parent',h15,...
    'BackgroundColor',[.7 .7 .9],...
    'Callback','handles = guidata(findobj(''tag'',''figure1''));for i = 1:length(handles.Current.HelpFilenames), if strfind(handles.Current.HelpFilenames{i},''HelpDefaultOutputFolder''), Option = i;end,end,if ~isempty(Option),CPtextdisplaybox(handles.Current.Help{Option-1},''Default Output Folder Help'');end;clear ans handles Option i;',...
    'FontWeight','bold',...
    'Position',[220 45 12 22],...
    'String','?',...
    'Tag','DefaultOutputFolderHelp',...
    'Behavior',get(0,'defaultuicontrolBehavior'),...
    'CreateFcn', {@local_CreateFcn, '', appdata} ); %#ok Ignore MLint

appdata = [];
appdata.lastValidTag = 'PixelSizeText';

h29 = uicontrol(...
    'Parent',h15,...
    'BackgroundColor',[.7 .7 .9],...
    'CData',[],...
    'FontWeight','bold',...
    'HorizontalAlignment','right',...
    'Position',[342 4 70 22],...
    'String','Pixel size:',...
    'Style','text',...
    'Tag','PixelSizeText',...
    'UserData',[],...
    'Behavior',get(0,'defaultuicontrolBehavior'),...
    'CreateFcn', {@local_CreateFcn, '', appdata} ); %#ok Ignore MLint

appdata = [];
appdata.lastValidTag = 'PixelSizeEditBox';

h30 = uicontrol(...
    'Parent',h15,...
    'BackgroundColor',[1 1 1],...
    'Callback','CellProfiler(''PixelSizeEditBox_Callback'',gcbo,[],guidata(gcbo))',...
    'Position',[417 8 25 22],...
    'String','1',...
    'Style','edit',...
    'Tag','PixelSizeEditBox',...
    'Behavior',get(0,'defaultuicontrolBehavior')); %#ok Ignore MLint

appdata = [];
appdata.lastValidTag = 'PixelSizeHelp';

h31 = uicontrol(...
    'Parent',h15,...
    'BackgroundColor',[.7 .7 .9],...
    'Callback','handles = guidata(findobj(''tag'',''figure1''));for i = 1:length(handles.Current.HelpFilenames), if strfind(handles.Current.HelpFilenames{i},''HelpPixelSize''), Option = i;end,end,if ~isempty(Option),CPtextdisplaybox(handles.Current.Help{Option-1},''Help for Pixel Size edit box'');end;clear ans handles Option i;',...
    'FontWeight','bold',...
    'Position',[335 8 12 22],...
    'String','?',...
    'Tag','PixelSizeHelp',...
    'Behavior',get(0,'defaultuicontrolBehavior'),...
    'CreateFcn', {@local_CreateFcn, '', appdata} ); %#ok Ignore MLint

hsingleton = h1;

function RemoveModuleByKeyPressFcn(Listbox_Handle,eventdata)

if strcmp(eventdata.Key,'delete') || strcmp(eventdata.Key,'backspace')
    handles = guidata(get(get(Listbox_Handle,'parent'),'parent'));
    RemoveModule_Callback(handles.figure1,[],handles)
end

function varargout = gui_mainfcn(gui_State, varargin)

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
    % CellProfiler
    % create the GUI
    gui_Create = 1;
elseif isequal(ishandle(varargin{1}), 1) && ispc && iscom(varargin{1}) && isequal(varargin{1},gcbo)
    % CellProfiler(ACTIVEX,...)
    vin{1} = gui_State.gui_Name;
    vin{2} = [get(varargin{1}.Peer, 'Tag'), '_', varargin{end}];
    vin{3} = varargin{1};
    vin{4} = varargin{end-1};
    vin{5} = guidata(varargin{1}.Peer);
    feval(vin{:});
    return;
elseif ischar(varargin{1}) && numargin>1 && isequal(ishandle(varargin{2}), 1)
    % CellProfiler('CALLBACK',hObject,eventData,handles,...)
    gui_Create = 0;
else
    % CellProfiler(...)
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
    persistent gui_hFigure %#ok Ignore MLint
    if ~isempty(gui_State.gui_LayoutFcn)
        if ishandle(gui_hFigure)
            display('CellProfiler is already running!!');
            figure(gui_hFigure);
            SplashHandle = findobj('tag','SplashScreenTag');
            if ishandle(SplashHandle)
                close(SplashHandle)
            end
            return;
        end
        gui_hFigure = feval(gui_State.gui_LayoutFcn, gui_SingletonOpt);
        % openfig (called by local_openfig below) does this for guis without
        % the LayoutFcn. Be sure to do it here so guis show up on screen.
        movegui(gui_hFigure,'onscreen')
    else
        if ishandle(gui_hFigure)
            display('CellProfiler is already running!!');
            SplashHandle = findobj('tag','SplashScreenTag');
            if ishandle(SplashHandle)
                close(SplashHandle)
            end
            return;
        end
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

        % Generate HANDLES structure and store with GUIDATA. If there is
        % user set GUI data already, keep that also.
        data = guidata(gui_hFigure);
        handles = guihandles(gui_hFigure);
        if ~isempty(handles)
            if isempty(data)
                data = handles;
            else
                names = fieldnames(handles);
                for k=1:length(names)
                    data.(char(names(k)))=handles.(char(names(k)));
                end
            end
        end
        guidata(gui_hFigure, data);
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
        if length(varargin) == index || ~ischar(varargin{index})
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
            if any(findobj('tag','SplashScreenTag'))
                if toc < 4
                    SplashTime = 4 - toc;
                    pause(SplashTime);
                    close(findobj('tag','SplashScreenTag'));
                else
                    close(findobj('tag','SplashScreenTag'));
                end
            end
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

% this application data is used to indicate the running mode of a GUIDE
% GUI to distinguish it from the design mode of the GUI in GUIDE.
setappdata(0,'OpenGuiWhenRunning',1);

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
rmappdata(0,'OpenGuiWhenRunning');

function SplashScreenHandle = SplashScreen
SplashScreenHandle = figure('MenuBar','None','NumberTitle','off','color',[1 1 1],'tag','SplashScreenTag','name','CellProfiler is loading...','color',[0.7,0.7,0.9]);
axis off;
if isdeployed
    logo = imread('CPsplash.jpg','jpg');
else
    ImageFile = fullfile(fileparts(which('CellProfiler.m')),'CPsubfunctions','CPsplash.jpg');
    logo = imread(ImageFile,'jpg');
end
iptsetpref('ImshowBorder','tight')
imshow(logo);
