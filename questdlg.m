function ButtonName=questdlg(Question,Title,Btn1,Btn2,Btn3,Default)
%
% This function should only be used within CellProfiler
%
%QUESTDLG Question dialog box.
%  ButtonName=QUESTDLG(Question) creates a modal dialog box that 
%  automatically wraps the cell array or string (vector or matrix) 
%  Question to fit an appropriately sized window.  The name of the 
%  button that is pressed is returned in ButtonName.  The Title of 
%  the figure may be specified by adding a second string argument.  
%  Question will be interpreted as a normal string.  
%
%  QUESTDLG uses UIWAIT to suspend execution until the user responds.
%
%  The default set of buttons names for QUESTDLG are 'Yes','No' and 
%  'Cancel'.  The default answer for the above calling syntax is 'Yes'.  
%  This can be changed by adding a third argument which specifies the 
%  default Button.  i.e. ButtonName=questdlg(Question,Title,'No').
%
%  Up to 3 custom button names may be specified by entering
%  the button string name(s) as additional arguments to the function 
%  call.  If custom ButtonName's are entered, the default ButtonName
%  must be specified by adding an extra argument DEFAULT, i.e.
%
%    ButtonName=questdlg(Question,Title,Btn1,Btn2,DEFAULT);
%
%  where DEFAULT=Btn1.  This makes Btn1 the default answer. If the 
%  DEFAULT string does not match any of the button string names, a
%  warning message is displayed.
%
%  To use TeX interpretation for the Question string, a data
%  structure must be used for the last argument, i.e.
%
%    ButtonName=questdlg(Question,Title,Btn1,Btn2,OPTIONS);
%
%  The OPTIONS structure must include the fields Default and Interpreter.  
%  Interpreter may be 'none' or 'tex' and Default is the default button
%  name to be used.
%
%  If the dialog is closed without a valid selection, the return value
%  is empty.
%
%  Example:
%
%  ButtonName=questdlg('What is your wish?', ...
%                      'Genie Question', ...
%                      'Food','Clothing','Money','Money');
%
%  
%  switch ButtonName,
%    case 'Food', 
%     disp('Food is delivered');
%    case 'Clothing',
%     disp('The Emperor''s  new clothes have arrived.')
%     case 'Money',
%      disp('A ton of money falls out the sky.');
%  end % switch
%
%  See also TEXTWRAP, INPUTDLG, INPUTDLG.

%  Copyright 1984-2003 The MathWorks, Inc.
%  $Revision$

if nargin<1,error('Too few arguments for QUESTDLG');end

Interpreter='none';
if ~iscell(Question),Question=cellstr(Question);end

%%%%%%%%%%%%%%%%%%%%%
%%% General Info. %%%
%%%%%%%%%%%%%%%%%%%%%
Black      =[0       0        0      ]/255;
LightGray  =[192     192      192    ]/255;
LightGray2 =[160     160      164    ]/255;
MediumGray =[128     128      128    ]/255;
White      =[255     255      255    ]/255;

%%%%%%%%%%%%%%%%%%%%
%%% Nargin Check %%%
%%%%%%%%%%%%%%%%%%%%
if nargout>1,error('Wrong number of output arguments for QUESTDLG');end
if nargin==1,Title=' ';end
if nargin<=2, Default='Yes';end
if nargin==3, Default=Btn1 ;end
if nargin<=3, Btn1='Yes'; Btn2='No'; Btn3='Cancel';NumButtons=3;end
if nargin==4, Default=Btn2;Btn2=[];Btn3=[];NumButtons=1;end
if nargin==5, Default=Btn3;Btn3=[];NumButtons=2;end
if nargin==6, NumButtons=3;end
if nargin>6, error('Too many input arguments');NumButtons=3;end

if isstruct(Default),
    Interpreter=Default.Interpreter;
    Default=Default.Default;
end


%%%%%%%%%%%%%%%%%%%%%%%
%%% Create QuestFig %%%
%%%%%%%%%%%%%%%%%%%%%%%
FigPos    = get(0,'DefaultFigurePosition');
FigPos(3) = 267;
FigPos(4) =  70;
FigPos    = getnicedialoglocation(FigPos, get(0,'DefaultFigureUnits'));

QuestFig=dialog(                                    ...
    'Visible'         ,'off'                      , ...
    'Name'            ,Title                      , ...
    'Pointer'         ,'arrow'                    , ...
    'Position'        ,FigPos                     , ...
    'KeyPressFcn'     ,@doFigureKeyPress          , ...
    'IntegerHandle'   ,'off'                      , ...
    'WindowStyle'     ,'normal'                   , ... 
    'HandleVisibility','callback'                 , ...
    'Tag'             ,Title                        ...
    );

%%%%%%%%%%%%%%%%%%%%%
%%% Set Positions %%%
%%%%%%%%%%%%%%%%%%%%%
DefOffset  =10;

IconWidth  =54;
IconHeight =54;
IconXOffset=DefOffset;
IconYOffset=FigPos(4)-DefOffset-IconHeight;
IconCMap=[Black;get(QuestFig,'Color')];

DefBtnWidth =56;
BtnHeight   =22;

BtnYOffset=DefOffset;

BtnWidth=DefBtnWidth;

ExtControl=uicontrol(QuestFig   , ...
                     'Style'    ,'pushbutton', ...
                     'String'   ,' '          ...
                     );

btnMargin=1.4;
set(ExtControl,'String',Btn1);
BtnExtent=get(ExtControl,'Extent');
BtnWidth=max(BtnWidth,BtnExtent(3)+8);
if NumButtons > 1
    set(ExtControl,'String',Btn2);
    BtnExtent=get(ExtControl,'Extent');
    BtnWidth=max(BtnWidth,BtnExtent(3)+8);
    if NumButtons > 2
        set(ExtControl,'String',Btn3);
        BtnExtent=get(ExtControl,'Extent');
        BtnWidth=max(BtnWidth,BtnExtent(3)*btnMargin);
    end 
end
BtnHeight = max(BtnHeight,BtnExtent(4)*btnMargin);

delete(ExtControl);

MsgTxtXOffset=IconXOffset+IconWidth;

FigPos(3)=max(FigPos(3),MsgTxtXOffset+NumButtons*(BtnWidth+2*DefOffset));
set(QuestFig,'Position',FigPos);

BtnXOffset=zeros(NumButtons,1);

if NumButtons==1,
    BtnXOffset=(FigPos(3)-BtnWidth)/2;
elseif NumButtons==2,
    BtnXOffset=[MsgTxtXOffset
                FigPos(3)-DefOffset-BtnWidth];
elseif NumButtons==3,
    BtnXOffset=[MsgTxtXOffset
                0
                FigPos(3)-DefOffset-BtnWidth];
    BtnXOffset(2)=(BtnXOffset(1)+BtnXOffset(3))/2;
end

MsgTxtYOffset=DefOffset+BtnYOffset+BtnHeight;
MsgTxtWidth=FigPos(3)-DefOffset-MsgTxtXOffset-IconWidth;
MsgTxtHeight=FigPos(4)-DefOffset-MsgTxtYOffset;
MsgTxtForeClr=Black;
MsgTxtBackClr=get(QuestFig,'Color');

CBString='uiresume(gcbf)';

% Check to see if the Default string passed does match one of the
% strings on the buttons in the dialog. If not, throw a warning.
setappdata(QuestFig, 'DefaultValid', false);
ButtonString=Btn1;
ButtonTag='Btn1';
BtnHandle = [];
DefaultButton = 0;
if strcmp(ButtonString, Default)
    setappdata(QuestFig, 'DefaultValid', true);
    DefaultButton = 1;
end
BtnHandle(end+1)=uicontrol(QuestFig            , ...
                       'Style'              ,'pushbutton', ...
                       'Position'           ,[ BtnXOffset(1) BtnYOffset BtnWidth BtnHeight ]           , ...
                       'KeyPressFcn'        ,@doControlKeyPress , ...
                       'CallBack'           ,CBString    , ...
                       'String'             ,ButtonString, ...
                       'FontSize'           ,get(0,'UserData'), ...
                       'Fontname'          ,'Times', ...
                       'HorizontalAlignment','center'    , ...
                       'Tag'                ,ButtonTag     ...
                       );

if NumButtons > 1
    ButtonString=Btn2;
    ButtonTag='Btn2';
    if strcmp(ButtonString, Default)
        setappdata(QuestFig, 'DefaultValid', true);
        DefaultButton = 2;
    end
    BtnHandle(end+1)=uicontrol(QuestFig            , ...
                           'Style'              ,'pushbutton', ...
                           'Position'           ,[ BtnXOffset(2) BtnYOffset BtnWidth BtnHeight ]        , ...
                           'KeyPressFcn'        ,@doControlKeyPress , ...
                           'CallBack'           ,CBString    , ...
                           'String'             ,ButtonString, ...
                           'FontSize'           ,get(0,'UserData'), ...
                           'Fontname'          ,'Times',...
                           'HorizontalAlignment','center'    , ...
                           'Tag'                ,ButtonTag     ...
                           );
    if NumButtons > 2
        ButtonString=Btn3;
        ButtonTag='Btn3';
        if strcmp(ButtonString, Default)
            setappdata(QuestFig, 'DefaultValid', true);
            DefaultButton = 3;
        end
        BtnHandle(end+1)=uicontrol(QuestFig            , ...
                               'Style'              ,'pushbutton', ...
                               'Position'           ,[ BtnXOffset(3) BtnYOffset BtnWidth BtnHeight ]           , ...
                               'KeyPressFcn'        ,@doControlKeyPress , ...
                               'CallBack'           ,CBString    , ...
                               'String'             ,ButtonString, ...
                               'FontSize'           ,get(0,'UserData'), ...
                               'Fontname'          ,'Times',...
                               'HorizontalAlignment','center'    , ...
                               'Tag'                ,ButtonTag     ...
                               );
    end
end

if (getappdata(QuestFig, 'DefaultValid') ~= true)
    warnstate = warning('backtrace','off');
    warning('MATLAB:QUESTDLG:stringMismatch','Default string does not match any button string name.');
    warning(warnstate);
end

MsgHandle=uicontrol(QuestFig            , ...
                    'Style'              ,'text'         , ...
                    'Position'           ,[MsgTxtXOffset MsgTxtYOffset 0.95*MsgTxtWidth MsgTxtHeight ]              , ...
                    'String'             ,{' '}          , ...
                    'Tag'                ,'Question'     , ...
                    'HorizontalAlignment','left'         , ...    
                    'FontWeight'         ,'bold'         , ...
                    'FontSize'           ,get(0,'UserData'), ...
                    'Fontname'          ,'Times',...
                    'BackgroundColor'    ,MsgTxtBackClr  , ...
                    'ForegroundColor'    ,MsgTxtForeClr    ...
                    );

[WrapString,NewMsgTxtPos]=textwrap(MsgHandle,Question,75);

NumLines=size(WrapString,1);

% The +2 is to add some slop for the border of the control.
MsgTxtWidth=max(MsgTxtWidth,NewMsgTxtPos(3)+2);
MsgTxtHeight=NewMsgTxtPos(4)+2;

MsgTxtXOffset=IconXOffset+IconWidth+DefOffset;
FigPos(3)=max(NumButtons*(BtnWidth+DefOffset)+DefOffset, ...
             MsgTxtXOffset+MsgTxtWidth+DefOffset);


% Center Vertically around icon  
if IconHeight>MsgTxtHeight,
    IconYOffset=BtnYOffset+BtnHeight+DefOffset;
    MsgTxtYOffset=IconYOffset+(IconHeight-MsgTxtHeight)/2;
    FigPos(4)=IconYOffset+IconHeight+DefOffset;    
    % center around text    
else,
    MsgTxtYOffset=BtnYOffset+BtnHeight+DefOffset;
    IconYOffset=MsgTxtYOffset+(MsgTxtHeight-IconHeight)/2;
    FigPos(4)=MsgTxtYOffset+MsgTxtHeight+DefOffset;    
end    

if NumButtons==1,
    BtnXOffset=(FigPos(3)-BtnWidth)/2;
elseif NumButtons==2,
    BtnXOffset=[(FigPos(3)-DefOffset)/2-BtnWidth
                (FigPos(3)+DefOffset)/2      
               ];
    
elseif NumButtons==3,
    BtnXOffset(2)=(FigPos(3)-BtnWidth)/2;
    BtnXOffset=[BtnXOffset(2)-DefOffset-BtnWidth
                BtnXOffset(2)
                BtnXOffset(2)+BtnWidth+DefOffset
               ];              
end

set(QuestFig ,'Position',getnicedialoglocation(FigPos, get(QuestFig,'Units')));

BtnPos=get(BtnHandle,{'Position'});
BtnPos=cat(1,BtnPos{:});
BtnPos(:,1)=BtnXOffset;
BtnPos=num2cell(BtnPos,2);  
set(BtnHandle,{'Position'},BtnPos);  

if (getappdata(QuestFig, 'DefaultValid') == true)
    h = uicontrol(QuestFig,'BackgroundColor', 'k', ...
              'Style','frame','Position',[ BtnXOffset(DefaultButton)-1 BtnYOffset-1 BtnWidth+2 BtnHeight+2 ]);
    uistack(h,'bottom')
end
delete(MsgHandle);
AxesHandle=axes('Parent',QuestFig,'Position',[0 0 1 1],'Visible','off');

MsgHandle=text( ...
    'Parent'              ,AxesHandle                      , ...
    'Units'               ,'pixels'                        , ...
    'Color'               ,get(BtnHandle(1),'ForegroundColor')   , ...
    'HorizontalAlignment' ,'left'                          , ...
    'FontName'            ,get(BtnHandle(1),'FontName')    , ...
    'FontSize'            ,get(BtnHandle(1),'FontSize')    , ...
    'VerticalAlignment'   ,'bottom'                        , ...
    'Position'            ,[MsgTxtXOffset MsgTxtYOffset 0] , ...
    'String'              ,WrapString                      , ...
    'Interpreter'         ,Interpreter                     , ...
    'Tag'                 ,'Question'                        ...
    );

IconAxes=axes(                                      ...
    'Parent'      ,QuestFig              , ...  
    'Units'       ,'Pixels'              , ...
    'Position'    ,[IconXOffset IconYOffset IconWidth IconHeight], ...
    'NextPlot'    ,'replace'             , ...
    'Tag'         ,'IconAxes'              ...
    );         

set(QuestFig ,'NextPlot','add');

load dialogicons.mat
IconData=questIconData;
questIconMap(256,:)=get(QuestFig,'color');
IconCMap=questIconMap;

Img=image('CData',IconData,'Parent',IconAxes);
set(QuestFig, 'Colormap', IconCMap);
set(IconAxes, ...
    'Visible','off'           , ...
    'YDir'   ,'reverse'       , ...
    'XLim'   ,get(Img,'XData'), ...
    'YLim'   ,get(Img,'YData')  ...
    );

% make sure we are on screen
movegui(QuestFig)

set(QuestFig ,'WindowStyle','modal','Visible','on');
drawnow;

uiwait(QuestFig);

if ishandle(QuestFig)
    if isappdata(QuestFig,'DefaultWasPressed') && getappdata(QuestFig,'DefaultWasPressed') == true
        ButtonName=Default;
    else
        ButtonName=get(get(QuestFig,'CurrentObject'),'String');
    end
    delete(QuestFig);
else
    ButtonName='';
end

function doFigureKeyPress(obj, evd)
switch(evd.Key)
 case {'return','space'}
  if getappdata(gcbf,'DefaultValid')
      setappdata(gcbf,'DefaultWasPressed',true)
      uiresume(gcbf);
  end
 case 'escape'
  delete(gcbf)
end

function doControlKeyPress(obj, evd)
switch(evd.Key)
 case {'return'}
  if getappdata(gcbf,'DefaultValid')
      setappdata(gcbf,'DefaultWasPressed',true)
      uiresume(gcbf);
  end
 case 'escape'
  delete(gcbf)
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

