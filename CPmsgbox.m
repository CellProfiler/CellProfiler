function varargout = CPmsgbox(varargin)

%  This msgbox function is specially modified to set the correct
%  fontsize for the CellProfiler software.


error(nargchk(1,6,nargin));
error(nargoutchk(0,1,nargout));

BodyTextString=varargin{1};
if ~iscell(BodyTextString),BodyTextString=cellstr(BodyTextString);end

%%%%%%%%%%%%%%%%%%%%
%%% Nargin Check %%%
%%%%%%%%%%%%%%%%%%%%

Interpreter='none';
switch nargin,
  case 1,
    TitleString=' ';
    IconString='none';IconData=[];IconCMap=[];
    CreateMode='non-modal';
  case 2,
    [Flag,CreateMode,Interpreter]=InternalCreateFlag(varargin{2});
    if Flag, % CreateMode specified
      TitleString=' ';
      IconString='none';IconData=[];IconCMap=[];
    else,
      TitleString=varargin{2};
      IconString='none';IconData=[];IconCMap=[];
    end      
  case 3,
    [Flag,CreateMode,Interpreter]=InternalCreateFlag(varargin{3});
    if Flag, % CreateMode specified
      TitleString=varargin{2};
      IconString='none';IconData=[];IconCMap=[];
    else,
      TitleString=varargin{2};
      IconString=varargin{3};IconData=[];IconCMap=[];
    end      
  case 4,
    TitleString=varargin{2};
    IconString=varargin{3};IconData=[];IconCMap=[];
    [Flag,CreateMode,Interpreter]=InternalCreateFlag(varargin{4});
    
  case 5,
    [Flag,CreateMode,Interpreter]=InternalCreateFlag(varargin{5});
    if Flag, % CreateMode specified
      error(['A Colormap must be specified when calling MSGBOX with 5 ', ...
             'input arguments.']);
    else,
      TitleString=varargin{2};
      IconString=varargin{3};IconData=varargin{4};
      IconCMap=varargin{5};
      if ~strcmp(lower(IconString),'custom'),
        warning(['Icon must be ''custom'' when specifying icon data ' ...
                 'in MSGBOX']);
        IconString='custom';
      end        
    end      
  case 6,
    [Flag,CreateMode,Interpreter]=InternalCreateFlag(varargin{6});
    TitleString=varargin{2};
    IconString=varargin{3};IconData=varargin{4};
    IconCMap=varargin{5};
    
end  

CreateMode=lower(CreateMode);
if ~strcmp(CreateMode,'non-modal')& ~strcmp(CreateMode,'modal') & ...
   ~strcmp(CreateMode,'replace'),
    warning('Invalid string for CreateMode in MSGBOX.');
    CreateMode='non-modal';
end    

IconString=lower(IconString);
if ~(strcmp(IconString,'none')|strcmp(IconString,'help')  ...
    |strcmp(IconString,'warn')|strcmp(IconString,'error') ...
    |strcmp(IconString,'custom')),
    warning('Invalid string for Icon in MSGBOX.');
    IconString='none';
end

%%%%%%%%%%%%%%%%%%%%%
%%% General Info. %%%
%%%%%%%%%%%%%%%%%%%%%
Black      =[0       0        0      ]/255;
LightGray  =[192     192      192    ]/255;
LightGray2 =[160     160      164    ]/255;
MediumGray =[128     128      128    ]/255;
White      =[255     255      255    ]/255;

%%%%%%%%%%%%%%%%%%%%%
%%% Set Positions %%%
%%%%%%%%%%%%%%%%%%%%%
DefFigPos=get(0,'DefaultFigurePosition');

MsgOff=7;
%IconWidth=32;
IconWidth=38;

if strcmp(IconString,'none'),
  FigWidth=325;
  MsgTxtWidth=FigWidth-2*MsgOff;
else,
  FigWidth=390;
  MsgTxtWidth=FigWidth-2*MsgOff-IconWidth;
end
FigHeight=50;
DefFigPos(3:4)=[FigWidth FigHeight];

OKWidth=40;
OKHeight=17;
OKXOffset=(FigWidth-OKWidth)/2;
OKYOffset=MsgOff;


MsgTxtXOffset=MsgOff;
MsgTxtYOffset=MsgOff+OKYOffset+OKHeight;
MsgTxtHeight=FigHeight-MsgOff-MsgTxtYOffset;
MsgTxtForeClr=Black;

%IconHeight=32;
IconHeight=38;
IconXOffset=MsgTxtXOffset;
IconYOffset=FigHeight-MsgOff-IconHeight;

%%%%%%%%%%%%%%%%%%%%%
%%% Create MsgBox %%%
%%%%%%%%%%%%%%%%%%%%%

CreateModeFlag=1;

% See if a modal or replace dialog already exists and delete all of its 
% children
MsgboxTag = ['Msgbox_', TitleString];
if ~strcmp(CreateMode,'non-modal'),
  TempHide=get(0,'ShowHiddenHandles');
  set(0,'ShowHiddenHandles','on');
  OldFig=findobj(0,'Type','figure','Tag',MsgboxTag,'Name',TitleString);
  if ~isempty(OldFig),
    CreateModeFlag=0;
    if length(OldFig)>1,
      BoxHandle=OldFig(1);
      close(OldFig(2:end));
      OldFig(2:end)=[];
    end % if length
    BoxHandle=OldFig;
    CurPos=get(BoxHandle,'Position');
    CurPos(3:4)=[FigWidth FigHeight];
    set(BoxHandle,'Position',CurPos);
    BoxChildren=get(BoxHandle,'Children');
    delete(BoxChildren);
    figure(BoxHandle);
  end
  set(0,'ShowHiddenHandles',TempHide);
end

if strcmp(CreateMode,'modal'),
  WindowStyle='modal';
else,
  WindowStyle='normal';
end

if CreateModeFlag,
  BoxHandle=dialog(                                            ...
                  'Name'            ,TitleString             , ...
                  'Pointer'         ,'arrow'                 , ...
                  'Units'           ,'points'                , ...
                  'Visible'         ,'off'                   , ...
                  'KeyPressFcn'     ,@doKeyPress             , ...
                  'WindowStyle'     ,WindowStyle             , ...
                  'HandleVisibility','callback'              , ...
                  'Toolbar'         ,'none'                  , ...
                  'Tag'             ,MsgboxTag                 ...
                  );
else,
  set(BoxHandle,   ...
     'WindowStyle'     ,WindowStyle, ...
     'HandleVisibility','on'         ...
     );
end %if strcmp

%%% Label we attach to figures (as UserData) so we know they are ours
userData.Application = 'CellProfiler';
set(BoxHandle,'UserData',userData);

FigColor=get(BoxHandle,'Color');

MsgTxtBackClr=FigColor;

Font.FontUnits = 'points';
Font.FontSize = get(0,'UserData');
if isempty(Font.FontSize)
    Font.FontSize = 9;
end
Font.FontName = 'Times';

OKHandle=uicontrol(BoxHandle           , Font                             , ...
                  'Style'              ,'pushbutton'                      , ...
                  'Units'              ,'points'                          , ...
                  'Position'           ,[ OKXOffset OKYOffset OKWidth OKHeight ] , ...
                  'CallBack'           ,'delete(gcbf)'                    , ...
                  'KeyPressFcn'        ,@doKeyPress                       , ...
                  'String'             ,'OK'                              , ...
                  'HorizontalAlignment','center'                          , ...
                  'Tag'                ,'OKButton'                          ...
                  );
                                   
MsgHandle=uicontrol(BoxHandle           , Font          , ...
                   'Style'              ,'text'         , ...
                   'Units'              ,'points'       , ...
                   'Position'           ,[MsgTxtXOffset ...
                                          MsgTxtYOffset ...
                                          MsgTxtWidth   ...
                                          MsgTxtHeight  ...
                                         ]              , ...
                   'String'             ,' '            , ...
                   'Tag'                ,'MessageBox'   , ...
                   'HorizontalAlignment','left'         , ...    
                   'BackgroundColor'    ,MsgTxtBackClr  , ...
                   'ForegroundColor'    ,MsgTxtForeClr    ...
                   );

               
[WrapString,NewMsgTxtPos]=textwrap(MsgHandle,BodyTextString,75);
NumLines=size(WrapString,1);

MsgTxtWidth=max(MsgTxtWidth,NewMsgTxtPos(3));
MsgTxtHeight=max(MsgTxtHeight,NewMsgTxtPos(4));

if ~strcmp(IconString,'none'),
  MsgTxtXOffset=IconXOffset+IconWidth+MsgOff;
  FigWidth=MsgTxtXOffset+MsgTxtWidth+MsgOff;  
  % Center Vertically around icon  
  if IconHeight>MsgTxtHeight,
    IconYOffset=OKYOffset+OKHeight+MsgOff;
    MsgTxtYOffset=IconYOffset+(IconHeight-MsgTxtHeight)/2;
    FigHeight=IconYOffset+IconHeight+MsgOff;    
  % center around text    
  else,
    MsgTxtYOffset=OKYOffset+OKHeight+MsgOff;
    IconYOffset=MsgTxtYOffset+(MsgTxtHeight-IconHeight)/2;
    FigHeight=MsgTxtYOffset+MsgTxtHeight+MsgOff;    
  end    
  
else,
  FigWidth=MsgTxtWidth+2*MsgOff;  
  MsgTxtYOffset=OKYOffset+OKHeight+MsgOff;
  FigHeight=MsgTxtYOffset+MsgTxtHeight+MsgOff;
end % if ~strcmp

OKXOffset=(FigWidth-OKWidth)/2; 
DefFigPos(3:4)=[FigWidth FigHeight];
DefFigPos = getnicedialoglocation(DefFigPos, get(BoxHandle,'Units'));

% if there is a figure out there and it's modal, we need to be modal too
if ~isempty(gcbf) && strcmp(get(gcbf,'WindowStyle'),'modal')
    set(BoxHandle,'WindowStyle','modal');
end

set(BoxHandle,'Position',DefFigPos);
set(OKHandle,'Position',[OKXOffset OKYOffset OKWidth OKHeight]);  
  
set(OKHandle,'Units','pixels');
BtnPos = get(OKHandle,'Position');
set(OKHandle,'Units','points');
h = uicontrol(BoxHandle,'BackgroundColor', 'k', ...
              'Style','frame','Position',[ BtnPos(1)-1 BtnPos(2)-1 BtnPos(3)+2 BtnPos(4)+2 ]);
uistack(h,'bottom')

delete(MsgHandle);
AxesHandle=axes('Parent',BoxHandle,'Position',[0 0 1 1],'Visible','off');

MsgHandle=text( ...
    'Parent'              ,AxesHandle                        , ...
    'Units'               ,'points'                          , ...
    'Color'               ,get(OKHandle,'ForegroundColor')   , ...
    Font                  , ...
    'HorizontalAlignment' ,'left'                            , ...
    'VerticalAlignment'   ,'bottom'                          , ...
    'Position'            ,[ MsgTxtXOffset MsgTxtYOffset+5 0], ...
    'String'              ,WrapString                        , ...
    'Interpreter'         ,Interpreter                       , ...
    'Tag'                 ,'MessageBox'                        ...
    );


if ~strcmp(IconString,'none'),
  IconAxes=axes(                                            ...
               'Parent'          ,BoxHandle               , ...
               'Units'           ,'points'                , ...
               'Position'        ,[IconXOffset IconYOffset ...
                                   IconWidth IconHeight]  , ...
               'Tag'             ,'IconAxes'                ...
               );         
 

  if ~strcmp(IconString,'custom'),IconCMap=[Black;FigColor];end
 
 load dialogicons.mat
 if strcmp('warn',IconString),
            IconData=warnIconData;
            warnIconMap(256,:)=get(BoxHandle,'color');
            IconCMap=warnIconMap;
    
  elseif strcmp('help',IconString),
            IconData=helpIconData;
            helpIconMap(256,:)=get(BoxHandle,'color');
            IconCMap=helpIconMap;

  elseif strcmp('error',IconString),
            IconData=errorIconData;
            errorIconMap(146,:)=get(BoxHandle,'color');
            IconCMap=errorIconMap;
  end
  
  Img=image('CData',IconData,'Parent',IconAxes);  
  set(BoxHandle, 'Colormap', IconCMap);
  set(IconAxes          , ...
      'XLim'            ,get(Img,'XData')+[-0.5 0.5], ...
      'YLim'            ,get(Img,'YData')+[-0.5 0.5], ...
      'Visible'         ,'off'                      , ...
      'YDir'            ,'reverse'                    ...      
      );
  
end % if ~strcmp

% make sure we are on screen
movegui(BoxHandle)

set(BoxHandle,'HandleVisibility','callback','Visible','on');

% make sure the window gets drawn even if we are in a pause
drawnow


if nargout==1,varargout{1}=BoxHandle;end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% InternalCreateFlag %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Flag,CreateMode,Interpreter]=InternalCreateFlag(String)

Flag=0;
CreateMode='non-modal';
Interpreter='none';

if iscell(String),String=String{:};end
if isstr(String),
  if strcmp(String,'non-modal') | strcmp(String,'modal') | ...
     strcmp(String,'replace'),
    Flag=1;
    CreateMode=String;
  end
elseif isstruct(String),
  Flag=1;
  CreateMode=String.WindowStyle;
  Interpreter=String.Interpreter;
end  

function doKeyPress(obj, evd)
switch(evd.Key)
 case {'return','space','escape'}
  delete(gcbf);
end


function figure_size = getnicedialoglocation(figure_size, figure_units)
parentHandle = gcbf;
propName = 'Position';
if isempty(parentHandle)
    parentHandle = 0;
    propName = 'ScreenSize';
end

old_u = get(parentHandle,'Units');
set(parentHandle,'Units',figure_units);
container_size=get(parentHandle,propName);
set(parentHandle,'Units',old_u);

figure_size(1) = container_size(1)  + 1/2*(container_size(3) - figure_size(3));
figure_size(2) = container_size(2)  + 2/3*(container_size(4) - figure_size(4));
