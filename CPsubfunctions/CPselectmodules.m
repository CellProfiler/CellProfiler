function Selection = CPselectmodules(ModuleNames,TestNum)

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
%
% Website: http://www.cellprofiler.org
%
% $Revision: 4010 $

%%% Create Select Display window
SelectDisplay = CPfigure('Units','Inches','Resize','Off','Menubar','None','Toolbar','None','NumberTitle','Off','Name','Select Display Window','Color',[.7 .7 .9],'UserData',0);

%%% Set window location and size
% Get current position
Pos = get(SelectDisplay,'Position');
% Get screen height
set(0,'Units','inches')
ScreenSize = get(0,'ScreenSize');
set(0,'Units','pixels')
ScreenHeight = ScreenSize(4);
% Estimate window Height
uiheight = 0.3;
Height = (TestNum+4)*uiheight;
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
FontSize = 11;
uicontrol(SelectDisplay,'Style','Text','String','Select which module windows to display: ','FontName','Times','FontSize',FontSize, 'HorizontalAlignment','Left','Units','Inches','Position',[0.2 Height-0.25 3.5 0.2],'BackgroundColor',[.7 .7 .9]);

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
    SliderHandle = uicontrol(SelectDisplay,'style','slider','units','inches','position',[4 .8 .2 Height-4*uiheight],'userdata',SliderData,'Callback','SliderData = get(gco,''UserData''); feval(SliderData.Callback,gco,SliderData.Panel); clear SliderData','Max',Height-4*uiheight,'Min',Height-4*uiheight-((TestNum-Fits)*uiheight),'Value',Height-4*uiheight,'SliderStep',[1/(TestNum-Fits) 3/(TestNum-Fits)]);
    % Height to be used when creating other uicontrols
    ypos = Height - 4*uiheight-0.2;
else
    SelectDisplayPanel = SelectDisplay;
    ypos = Height - 2*uiheight;
end

%%% Create module names and checkboxes
h = [];
for k = 1:TestNum
    uicontrol(SelectDisplayPanel,'style','text','String',ModuleNames{k},'FontName','Times','FontSize',FontSize,'HorizontalAlignment','left',...
        'units','inches','position',[0.6 ypos 3 .18],'BackgroundColor',[.7 .7 .9])
    h(k) = uicontrol(SelectDisplayPanel,'Style','checkbox','units','inches','position',[0.2 ypos .2 .18],...
        'BackgroundColor',[.7 .7 .9],'Value',1);
    ypos=ypos-uiheight;
end

%%% Hide excess names/checkboxes
if ReqSlid
    Slider_Callback(SliderHandle,SelectDisplayPanel)
end

%%% Create special features
uicontrol(SelectDisplay,'Style','Checkbox','Units','Inches','BackgroundColor',[.7 .7 .9],'Position',[0.2 0.5 .2 .2],'Value',1,'UserData',h,'Callback','if get(gcbo,''Value''), set(get(gcbo,''UserData''),''Value'',1); else, set(get(gcbo,''UserData''),''Value'',0); end;');
uicontrol(SelectDisplay,'Style','Text','String','Select All/None','FontName','Times','FontSize',FontSize,'HorizontalAlignment','Left','Units','Inches','Position',[0.6 0.5 1 .2],'BackgroundColor',[.7 .7 .9]);
uicontrol(SelectDisplay,'Style','Checkbox','Units','Inches','BackgroundColor',[.7 .7 .9],'Position',[2.2 0.5 .2 .2],'UserData',h,'Callback','Checkboxes = get(gcbo,''UserData''); for i = 1:length(Checkboxes), if get(Checkboxes(i),''Value'')==1, set(Checkboxes(i),''Value'',0); else, set(Checkboxes(i),''Value'',1); end; end; clear Checkboxes;');
uicontrol(SelectDisplay,'Style','Text','String','Invert Selection','FontName','Times','FontSize',FontSize,'HorizontalAlignment','Left','Units','Inches','Position',[2.4 0.5 1 .2],'BackgroundColor',[.7 .7 .9]);

%%% Create OK and Cancel buttons
posx = (Width - 1.7)/2;               % Centers buttons horizontally
ButtonWidth = .75;
okbutton = uicontrol(SelectDisplay,...
    'style','pushbutton',...
    'String','OK',...
    'FontName','Times',...
    'FontSize',FontSize,...
    'units','inches',...
    'KeyPressFcn', @doFigureKeyPress,...
    'position',[posx 0.1 ButtonWidth 0.3],...
    'Callback','[foo,fig] = gcbo;set(fig,''UserData'',1);uiresume(fig);clear fig foo',...
    'BackgroundColor',[.7 .7 .9]);
cancelbutton = uicontrol(SelectDisplay,...
    'style','pushbutton',...
    'String','Cancel',...
    'FontName','Times',...
    'FontSize',FontSize,...
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
    else
        Selection = ones(TestNum, 1);
    end
    delete(SelectDisplay);
else
    Selection = [];
end


%%%%%%%%%%%% Subfunctions %%%%%%%%%%%%%

function doFigureKeyPress(obj, evd)
switch(evd.Key)
    case {'return','space'}
        [foo,fig] = gcbo;set(fig,'UserData',1);uiresume(fig);clear fig foo;
    case {'escape'}
        delete(gcf);
end

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