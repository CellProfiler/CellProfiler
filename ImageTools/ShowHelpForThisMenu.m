function ShowHelpForThisMenu(vaargin)

% Help for the Show Toolbox Help function:
% Category: Image Tools
%
% 
% SHORT DESCRIPTION:
% Shows Help menu for various Image Toolboxes.
% **********************************************************
%
% Shows Help menu for various Image Toolboxes.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% $Revision$

global toolsChoice;
global ImageToolHelp;

try
    addpath(genpath(fileparts(which('CellProfiler.m'))))
    savepath
catch
    CPerrordlg('You changed the name of CellProfiler.m file. Consequences of this are unknown.');
end
if ~isdeployed
    CellProfilerPathname = fileparts(which('CellProfiler'));
    Pathname = fullfile(CellProfilerPathname,'ImageTools');
else
    uigetdir(cd,'Choose the folder where the image tools are located');
    pause(.1);
    figure(gcf);
end
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
        %%% Looks for .m files and add them to the menu list.
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

ImageToolsFilenames = ListOfTools;
ToolsCellArray = ImageToolsFilenames;
ImageToolHelp = ToolHelp;
ToolsCellArray(1) = [];


ToolsHelpWindowHandles = findobj('tag','ToolsHelpWindow'); % handle for main window
if ~isempty(ToolsHelpWindowHandles)
    try
        close(ToolsHelpWindowHandles);
    end
end



% Get screen parameters and set window size, font and color
ScreenResolution = get(0,'ScreenPixelsPerInch');
FontSize = (220 - ScreenResolution)/13;
Color = [0.7 .7 .9];

MainWinPos = get(gcf,'Position');

ToolsHelpWindowHandle = figure(...
    'Units','pixels',...
    'CloseRequestFcn','delete(gcf)',...
    'Color',Color,...
    'DockControls','off',...
    'MenuBar','none',...
    'Name','ToolsHelpWindow',...
    'NumberTitle','off',...
    'Position',[MainWinPos(1)+MainWinPos(3)/6 MainWinPos(2)+MainWinPos(4)/6 MainWinPos(3)*0.75 MainWinPos(4)*0.75],...
    'Resize','off',...
    'HandleVisibility','on',...
    'Tag','ToolsHelpWindow');


set(ToolsHelpWindowHandle,'name','Image Tools Help');
TextString = sprintf(['To view help for individual Image tool help, choose one below.\nYou can add your own tools by writing Matlab m-files, placing them in the Image tools folder, and restarting CellProfiler.']);

choosetext = uicontrol(...
    'Parent',ToolsHelpWindowHandle,...
    'BackGroundColor', Color,...
    'Units','normalized',...
    'Position',[0.10 0.6 0.80 0.31],...
    'String',TextString,...
    'Style','text',...
    'FontSize',FontSize,...
    'Tag','informtext');

listboxcallback = 'ToolsHelpWindowHandle = findobj(''tag'',''ToolsHelpWindow''); if (strcmpi(get(ToolsHelpWindowHandle,''SelectionType''),''open'')==1) toolsbox = findobj(''tag'',''toolsbox''); global toolsChoice; global ImageToolHelp; toolsChoice = get(toolsbox,''value''); CPtextdisplaybox(ImageToolHelp{toolsChoice},[''Image Tools Help'']); close(ToolsHelpWindowHandle); clear ToolsHelpWindowHandle toolsbox toolsChoice ImageToolHelp; end;';
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
    'Behavior',get(0,'defaultuicontrolBehavior'));


okbuttoncallback = 'ToolsHelpWindowHandle = findobj(''tag'',''ToolsHelpWindow'');  toolsbox = findobj(''tag'',''toolsbox''); global toolsChoice; global ImageToolHelp; toolsChoice = get(toolsbox,''value''); CPtextdisplaybox(ImageToolHelp{toolsChoice},[''Image Tools Help'']); close(ToolsHelpWindowHandle); clear ToolsHelpWindowHandle toolsbox toolsChoice ImageToolHelp;';
okbutton = uicontrol(...
    'Parent',ToolsHelpWindowHandle,...
    'BackGroundColor', Color,...
    'Units','normalized',...
    'Callback',okbuttoncallback,...
    'Position',[0.30 0.077 0.2 0.06],...
    'String','Ok',...
    'Tag','okbutton');


cancelbuttoncallback = 'ToolsHelpWindowHandle = findobj(''tag'',''ToolsHelpWindow''); global toolsChoice; toolsChoice = 0; close(ToolsHelpWindowHandle), clear ToolsHelpWindowHandle toolsChoice';
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

