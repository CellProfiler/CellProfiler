function Selection = CPselectmodules(ModuleNames)

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

NumberOfModules = length(ModuleNames);

%%% Create Select Display window
SelectDisplay = CPfigure('Units','Inches','Resize','Off','Menubar','None','Toolbar','None','NumberTitle','Off','Name','Select Display Window','Color',[.7 .7 .9],'UserData',0);

%%% Set window location and size
% Get current position
Pos = get(SelectDisplay,'Position');
% Get screen height
set(0,'Units','inches')
[ScreenWidth,ScreenHeight] = CPscreensize;
set(0,'Units','pixels')
% Estimate window Height
uiheight = 0.3;
Height = (NumberOfModules+4)*uiheight;
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
set(SelectDisplay,'Position',[Pos(1)+1 YDist Width Height]);

%%% Create text and special checkboxes
uicontrol(SelectDisplay,'Style','Text','String','Select which module windows to display: ','HorizontalAlignment','Left','Units','Inches','Position',[0.2 Height-0.25 3.5 0.2],'BackgroundColor',[.7 .7 .9]);

%%% Create panel and slider, if needed
if ReqSlid
    % Panel Stuff
    SelectDisplayPanel = uipanel(SelectDisplay,'units','inches','position',[0 .8 4 Height-4*uiheight],'bordertype','none','BackgroundColor',[.7 .7 .9]);
    PanelPosition = get(SelectDisplayPanel,'Position');
    PanelHeight = PanelPosition(4);
    Fits = floor(PanelHeight/uiheight);
    % Slider Stuff
    SliderData.Callback = @Slider_Callback;
    SliderData.Panel = SelectDisplayPanel;
    SliderHandle = uicontrol(SelectDisplay,'style','slider','units','inches','position',[4 .8 .2 Height-4*uiheight],'userdata',SliderData,'Callback','SliderData = get(gco,''UserData''); feval(SliderData.Callback,gco,SliderData.Panel); clear SliderData','Max',Height-4*uiheight,'Min',Height-4*uiheight-((NumberOfModules-Fits)*uiheight),'Value',Height-4*uiheight,'SliderStep',[1/(NumberOfModules-Fits) 3/(NumberOfModules-Fits)]);
    % Height to be used when creating other uicontrols
    ypos = Height - 4*uiheight-0.2;
else
    SelectDisplayPanel = SelectDisplay;
    ypos = Height - 2*uiheight;
end

%%% Create module names and checkboxes
h = [];
for k = 1:NumberOfModules
    h(k) = uicontrol(SelectDisplayPanel,'Style','checkbox','String',ModuleNames{k},'units','inches','position',[0.2 ypos 3.2 .18],...
        'BackgroundColor',[.7 .7 .9],'Value',1);
    ypos=ypos-uiheight;
end

%%% Hide excess names/checkboxes
if ReqSlid
    Slider_Callback(SliderHandle,SelectDisplayPanel)
end

%%% Create special features
uicontrol(SelectDisplay,'Style','pushbutton',   'Value',0,'String','Select All/None',   'Units','Inches','BackgroundColor',[.7 .7 .9], 'Position',[0.2 0.5 1.7 .2],'Callback',@SelectModules_SelectAllNone);
uicontrol(SelectDisplay,'Style','pushbutton',   'Value',0,'String','Invert Selection',  'Units','Inches','BackgroundColor',[.7 .7 .9], 'Position',[2.2 0.5 1.7 .2],'Callback',@SelectModules_InvertSelection);
appdata.modulehandles = h;
appdata.selectallnone = 1;
guidata(SelectDisplay,appdata);

%%% Create OK and Cancel buttons
posx = (Width - 1.7)/2;               % Centers buttons horizontally
ButtonWidth = .75;
okbutton = uicontrol(SelectDisplay,...
    'style','pushbutton',...
    'String','OK',...
    'units','inches',...
    'KeyPressFcn', @doFigureKeyPress,...
    'position',[posx 0.1 ButtonWidth 0.3],...
    'Callback','[foo,fig] = gcbo;set(fig,''UserData'',1);uiresume(fig);clear fig foo',...
    'BackgroundColor',[.7 .7 .9]);
cancelbutton = uicontrol(SelectDisplay,...
    'style','pushbutton',...
    'String','Cancel',...
    'units','inches',...
    'position',[Width-posx-ButtonWidth 0.1 ButtonWidth 0.3],...
    'Callback','delete(gcf)',...
    'BackgroundColor',[.7 .7 .9]); %#ok Ignore MLint

uicontrol(okbutton)
uiwait(SelectDisplay)

if ishandle(SelectDisplay)
    Choice = get(h,'Value');
    if iscell(Choice)
        Selection = cat(1,Choice{:});
    elseif isscalar(Choice)
        Selection = Choice;
    else
        Selection = ones(NumberOfModules,1);
    end
    delete(SelectDisplay);
else
    return
end


%%%%%%%%%%%% Subfunctions %%%%%%%%%%%%%
%%
%%% SUBFUNCTION - doFigureKeyPress
function doFigureKeyPress(obj, evd)
switch(evd.Key)
    case {'return','space'}
        [foo,fig] = gcbo;set(fig,'UserData',1);uiresume(fig);clear fig foo;
    case {'escape'}
        delete(gcf);
end
%%
%%% SUBFUNCTION - Slider_Callback
function Slider_Callback(SliderHandle,PanelHandle)
%%% Get new position for the panel
PanelPos = get(PanelHandle,'Position');
NewPos = PanelPos(4)-get(SliderHandle,'Value')+0.8;
%%% Hide children, if needed
Children = get(PanelHandle,'Children');
for i = 1:length(Children)
    CurrentPos = get(Children(i),'Position');
    if CurrentPos(2)+NewPos<0.8 || CurrentPos(2)+CurrentPos(4)+NewPos>0.8+PanelPos(4)
        set(Children(i),'Visible','off');
    else
        set(Children(i),'Visible','on');
    end
end
%%% Set the new position
set(PanelHandle,'Position',[PanelPos(1) NewPos PanelPos(3) PanelPos(4)]);
%%
%%% SUBFUNCTION - SelectModules_SelectAllNone
function SelectModules_SelectAllNone(hObject,eventdata)

appdata = guidata(hObject);
hdl = appdata.modulehandles;
appdata.selectallnone = ~appdata.selectallnone;
set(hdl,'value',appdata.selectallnone);
guidata(hObject,appdata);

%%
%%% SUBFUNCTION - SelectModules_InvertSelection
function SelectModules_InvertSelection(hObject,eventdata)

appdata = guidata(hObject);
hdl = appdata.modulehandles;
if iscell(get(hdl,'value')),
    set(hdl,{'value'},num2cell(cellfun(@not,get(hdl,'value'))));
else
    set(hdl,'value',~get(hdl,'value'));
end
