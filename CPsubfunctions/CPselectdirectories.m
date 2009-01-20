function handles = CPselectdirectories(handles)
% Allows the user to interactively select which directories below the image
% directory to process.

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Determines which cycle is being analyzed.
if isempty(handles)
    return
end
if iscell(handles)
    % The "handles" is just a list of directories
    return_directories = 1;
    pathnames = handles;
    RootDirectoryName = handles{1};
else
    return_directories = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% FIRST CYCLE FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    drawnow

% % Trimming the list of files to be analyzed occurs only the first time
% % through this module.
% if SetBeingAnalyzed ~= 1, return; end
% 
% % There MUST be a LoadImages module immediately before this one
% idxLoadImages = strmatch('LoadImages',handles.Settings.ModuleNames);
% if isempty(idxLoadImages) || max(idxLoadImages)+1 ~= CurrentModuleNum,
%     error([ModuleName, ' must be placed immediately after the last LoadImages module in the pipeline.']);
% end

% Pull the FileList created by LoadImages
    fn = fieldnames(handles.Pipeline);
    prefix = 'filelist';
    fn = fn{strncmpi(fn,prefix,length(prefix))};
    if iscell(fn), fn = fn{1}; end; % There may be several FileLists, but since they are the same, we just need the first one
    pathnames = cellfun(@fileparts,handles.Pipeline.(fn),'UniformOutput',false);
    RootDirectoryName = handles.Pipeline.(['Pathname',fn(length(prefix)+1:end)]);
end
%
% It's necessary to sort the pathnames by directory so that foo\bar occurs
% before foo.bar\baz in order for everything further on to work. So the
% path delimiter should have a lower place in the sort order than any other
% character. This replacement makes it so, in a wierd way - hopefully
%
if ~ all(cellfun(@(x) isempty(x), strfind(pathnames,char(1))))
    error('Some pathname has a character with ASCII value 1 in it');
end

UniqueDirectories = unique(strrep(pathnames,filesep,char(1)));
UniqueDirectories = strrep(UniqueDirectories,char(1),filesep);

% Separate out the unique directory names for listing
DirectoryNames = cell(0,0);     % The separated directories from the path
DirectoryPaths = cell(0,0);     % The full pathname, not just the directory
DirectoryLevel = ones(0,0);     % The level (1..n) from the root
ListingTag = cell(0,0);
UniquePathNumber = zeros(0,0);
if ispc, fileseparator = ['\',filesep]; else fileseparator = filesep; end
if length(UniqueDirectories) == 1 && isempty(UniqueDirectories{1})
    return
end

for i = 1:length(UniqueDirectories),
    p = textscan(UniqueDirectories{i},'%s','delimiter',fileseparator);
    prefix = [];
    while (~isempty(p)) && isempty(p{:}{1})
        p{1} = p{1}(2:end);
        prefix = [prefix,filesep];
    end
    p{1}{1} = [prefix,p{1}{1}];
    DirectoryNames = [DirectoryNames; p{:}];
    DirectoryLevel = [DirectoryLevel; (1:length(p{:}))'];
    DirectoryPath = [];
    for j = 1:length(p{:})
        if isempty(DirectoryPath)
            DirectoryPath = p{1}{j};
        else
            DirectoryPath = fullfile(DirectoryPath,p{1}{j});
        end
        DirectoryPaths = [DirectoryPaths; DirectoryPath];
    end
    str = cell(length(p{1}),1); for j = 1:length(p{1}), str{j} = fullfile('',p{1}{1:j}); end
    ListingTag = [ListingTag; str];
    UniquePathNumber = [UniquePathNumber; i*ones(length(p{:}),1)];
end

% Organize the listing so that directories with a common root that follow 
% each other in the list don't get repeated when displayed
% Find the directory names that (1) share a level (2) under the same
% root...
[ignore,i,j] = unique(DirectoryPaths,'first');
DirectoryPathIdx = i;
UniqueDirectoryLabel = (1:length(i))';  % Each directory gets a numeric label since 'unique' doesn't work row-wise on cell arrays
UniqueDirectoryLabel = UniqueDirectoryLabel(j);
[ignore,idx] = unique(UniqueDirectoryLabel,'first');
EntriesToRemove = true(size(DirectoryPaths));
EntriesToRemove(idx) = false;
% ... and remove them from the listing
DirectoryNames(EntriesToRemove) = [];
DirectoryLevel(EntriesToRemove) = [];
ListingTag(EntriesToRemove) = [];
UniquePathNumber(EntriesToRemove) = [];

NumberOfDirectoryEntries = length(DirectoryNames);

% Create the SelectDirectories window
SelectDirectoryFig = CreateSelectDirectoryWindow(DirectoryNames,DirectoryLevel,ListingTag,NumberOfDirectoryEntries,RootDirectoryName);

uiwait(SelectDirectoryFig);

if ishandle(SelectDirectoryFig)
    appdata = guidata(SelectDirectoryFig);
    Choice = get(appdata.directoryhandles,'Value');
    if iscell(Choice)
        Selection = logical(cat(1,Choice{:}));
    else
        Selection = true(NumberOfDirectoryEntries,1);
    end
    delete(SelectDirectoryFig);
else
    uiwait(CPwarndlg('You have clicked Cancel or closed the window. All directories will be selected.','Warning'));
    Selection = true(NumberOfDirectoryEntries,1);
end

if return_directories
    handles = DirectoryPaths(sort(DirectoryPathIdx(Selection)));
    return
else
% Remove the de-selected directories from the FileLists (since there are
% likely to be less de-selected than selected directories)

% Get the highest level of the de-selected directories by sorting an
% ordered list
[paths,idx] = unique([UniquePathNumber ~Selection],'rows','first');
idx = idx(paths(:,2) == 1);
DirectoriesToRemove = ListingTag(idx);
LevelToRemove = DirectoryLevel(idx);
% Generate the index list to directories to remove
idxToRemove = false(size(handles.Pipeline.(fn)));
for i = 1:length(DirectoriesToRemove)
    idxOfFilesep = regexp(DirectoriesToRemove{i},fileseparator);
    if LevelToRemove(i) <= length(idxOfFilesep),
        idxOfFilesep = idxOfFilesep(LevelToRemove(i));
    else
        idxOfFilesep = length(DirectoriesToRemove{i});
    end
    SearchString = DirectoriesToRemove{i}(1:idxOfFilesep);
    
    idxToRemove = idxToRemove | strncmp(SearchString,handles.Pipeline.(fn),length(SearchString));
end
% Remove the direcrtories from all FileLists (which assumes that the paths
% are the same)
fn = fieldnames(handles.Pipeline);
prefix = 'filelist';
fn = fn(strncmpi(fn,prefix,length(prefix)));
if ischar(fn), fn = {fn}; end
for i = 1:length(fn),
    handles.Pipeline.(fn{i})(idxToRemove) = [];
end
if isfield(handles.Current,'NumberOfImageSets')
    handles.Current.NumberOfImageSets = length(idxToRemove(:)) - sum(idxToRemove);
end
end
%%%%%%%%%%%% Subfunctions %%%%%%%%%%%%%
%%
%%% SUBFUNCTION - SelectDirectories_Callback
function SelectDirectories_Callback(hObject, eventdata)
Checkbox_Handles = findobj(gcbf,'style','checkbox');
Dir = get(Checkbox_Handles,'tag');
Selection = get(hObject,'tag');
idx = find(strncmp(Selection,Dir,length(Selection)));
String = get(Checkbox_Handles(idx),'string');
% This takes advantage of the fact that the deeper directories are always
% listed first
idx = idx(1:find(strcmp(get(hObject,'string'),String))-1);
Value = get(hObject,'value');
set(Checkbox_Handles(idx),{'value'},num2cell(Value));

%%
%%% SUBFUNCTION - doFigureKeyPress
function doFigureKeyPress(obj, evd)
switch(evd.Key)
    case {'return','space'}
        [foo,fig] = gcbo;
        set(fig,'UserData',1);
        uiresume(fig);
    case {'escape'}
        delete(gcf);
end
%%
%%% SUBFUNCTION - Slider_Callback
function Slider_Callback(SliderHandle,PanelHandle)
% Get new position for the panel
PanelPos = get(PanelHandle,'Position');
NewPos = PanelPos(4)-get(SliderHandle,'Value')+0.8;
% Hide children, if needed
Children = get(PanelHandle,'Children');
for i = 1:length(Children)
    CurrentPos = get(Children(i),'Position');
    if CurrentPos(2)+NewPos<0.8 || CurrentPos(2)+CurrentPos(4)+NewPos>0.8+PanelPos(4)
        set(Children(i),'Visible','off');
    else
        set(Children(i),'Visible','on');
    end
end
% Set the new position
set(PanelHandle,'Position',[PanelPos(1) NewPos PanelPos(3) PanelPos(4)]);
%%
%%% SUBFUNCTION - SelectModules_SelectAllNone
function SelectModules_SelectAllNone(hObject,eventdata)

appdata = guidata(hObject);
hdl = appdata.directoryhandles;
appdata.selectallnone = ~appdata.selectallnone;
set(hdl,'value',appdata.selectallnone);
guidata(hObject,appdata);

%%
%%% SUBFUNCTION - SelectModules_InvertSelection
function SelectModules_InvertSelection(hObject,eventdata)

appdata = guidata(hObject);
hdl = appdata.directoryhandles;
if iscell(get(hdl,'value')),
    set(hdl,{'value'},num2cell(cellfun(@not,get(hdl,'value'))));
else
    set(hdl,'value',~get(hdl,'value'));
end

%%
%%% SUBFUNCTION - CreateSelectDirectoryWindow
function SelectDirectoryFig = CreateSelectDirectoryWindow(DirectoryNames,DirectoryLevel,ListingTag,NumberOfDirectoryEntries,RootDirectoryName)

% Create Select Directory window
SelectDirectoryFig = CPfigure('Units','Inches','Resize','Off','Menubar','None','Toolbar','None','NumberTitle','Off','Name','Select Directories','Color',[.7 .7 .9],'UserData',0);

% Set window location and size
% Get current position
Pos = get(SelectDirectoryFig,'Position');
% Get screen height
set(0,'Units','inches')
[ScreenWidth,ScreenHeight] = CPscreensize;
set(0,'Units','pixels')
% Estimate window Height
uiheight = 0.25;
Height = (NumberOfDirectoryEntries+5)*uiheight;
% Determine if a slider is needed
if Height > .75*ScreenHeight
    Height = .75*ScreenHeight;
    ReqSlid = 1;
else
    ReqSlid = 0;
end
% Center its height in the screen
YDist = (ScreenHeight-Height)/2;
% Select width
Width = 4.2;
% Set position
set(SelectDirectoryFig,'Position',[Pos(1)+1 YDist Width Height]);

% Create text and special checkboxes
uicontrol(SelectDirectoryFig,'Style','Text','String',['Select which directories in ',RootDirectoryName,' to process: '],...
    'HorizontalAlignment','Left','Units','Inches','Position',[0.1 Height-0.5 3.5 0.4],'BackgroundColor',[.7 .7 .9]);

% Create panel and slider, if needed
if ReqSlid
    % Panel Stuff
    SelectDirectoryPanel = uipanel(SelectDirectoryFig,'units','inches','position',[0 .8 4 Height-5*uiheight],'bordertype','none','BackgroundColor',[.7 .7 .9]);
    PanelPosition = get(SelectDirectoryPanel,'Position');
    PanelHeight = PanelPosition(4);
    Fits = floor(PanelHeight/uiheight);
    % Slider Stuff
    SliderData.Callback = @Slider_Callback;
    SliderData.Panel = SelectDirectoryPanel;
    SliderHandle = uicontrol(SelectDirectoryFig,'style','slider','units','inches','position',[4 .8 .2 Height-4*uiheight],'userdata',SliderData,...
        'Callback','SliderData = get(gco,''UserData''); feval(SliderData.Callback,gco,SliderData.Panel); clear SliderData',...
        'Max',Height-4*uiheight,'Min',Height-4*uiheight-((NumberOfDirectoryEntries-Fits)*uiheight),'Value',Height-4*uiheight,...
        'SliderStep',[1/(NumberOfDirectoryEntries-Fits) 3/(NumberOfDirectoryEntries-Fits)]);
    % Height to be used when creating other uicontrols
    ypos = Height - 4*uiheight-0.2;
else
    SelectDirectoryPanel = SelectDirectoryFig;
    ypos = Height - 3*uiheight;
end

% Create module names and checkboxes
h = zeros(NumberOfDirectoryEntries,1);
for i = 1:NumberOfDirectoryEntries
    space_offset = 0.2*(DirectoryLevel(i)-1);   % Indent by directory level
    h(i) = uicontrol('parent',SelectDirectoryPanel,'style','checkbox','string',DirectoryNames{i},'units','inches','position',[0.2+space_offset ypos 3.2 0.18],...
        'backgroundcolor',[.7 .7 .9],'Value',1,'tag',ListingTag{i},'callback', @SelectDirectories_Callback);
    ypos = ypos-uiheight;
end

% Hide excess names/checkboxes
if ReqSlid
    Slider_Callback(SliderHandle,SelectDirectoryPanel)
end

% Create special features
uicontrol(SelectDirectoryFig,'Style','pushbutton',   'Value',0,'String','Select All/None',   'Units','Inches','BackgroundColor',[.7 .7 .9], 'Position',[0.2 0.5 1.7 .2],'Callback',@SelectModules_SelectAllNone);
uicontrol(SelectDirectoryFig,'Style','pushbutton',   'Value',0,'String','Invert Selection',  'Units','Inches','BackgroundColor',[.7 .7 .9], 'Position',[2.2 0.5 1.7 .2],'Callback',@SelectModules_InvertSelection);
appdata.directoryhandles = h;
appdata.selectallnone = 1;
guidata(SelectDirectoryFig,appdata);

% Create OK and Cancel buttons
posx = (Width - 1.7)/2;               % Centers buttons horizontally
ButtonWidth = .75;
uicontrol(SelectDirectoryFig,...
    'style','pushbutton',...
    'String','OK',...
    'units','inches',...
    'KeyPressFcn', @doFigureKeyPress,...
    'position',[posx 0.1 ButtonWidth 0.25],...
    'Callback','[foo,fig] = gcbo;set(fig,''UserData'',1);uiresume(fig);clear fig foo',...
    'BackgroundColor',[.7 .7 .9],...
    'tag',[mfilename,'_OKButton']);
uicontrol(SelectDirectoryFig,...
    'style','pushbutton',...
    'String','Cancel',...
    'units','inches',...
    'position',[Width-posx-ButtonWidth 0.1 ButtonWidth 0.25],...
    'Callback','delete(gcbf)',...
    'BackgroundColor',[.7 .7 .9],...
    'tag',[mfilename,'_CancelButton']); %#ok Ignore MLint