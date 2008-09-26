function [text, ok]=CPeditbox(text, varargin)

% Multiline edit box to let users examine and edit text.
%
% Arguments: text - the text to edit
%            varargin - properties passed into the figure
% Returns:   text as edited (or same text if canceled)
%            ok - true if OK pressed, false if user exited box in
%            some other way.
%
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

hFig = figure('WindowStyle','modal',...
              'ResizeFcn',@Resize_Callback,...
              varargin{:});
hData = struct('hFig',hFig);
hData.hEdit = uicontrol(hFig,...
		        'Style','edit',...
                'HorizontalAlignment','left',...
    			'Max',1000,...
        		'String',text);
hData.hOK = uicontrol(hFig,...
		      'Style','pushbutton',...
		      'Callback', {@Exit_Callback, true},...
		      'String','OK');
hData.hCancel = uicontrol(hFig,...
			  'Style','pushbutton',...
			  'Callback', {@Exit_Callback, false},...
			  'String','Cancel');
ok = false;
hData.text = text;
Resize_Callback(hFig,[]);
uiwait(hFig);


function Exit_Callback(hObject, eventdata, result)
    ok = result;
    if result
        text=get(hData.hEdit,'String');
    end
    close(hFig);
end

function Resize_Callback(hObject,eventdata)
    FigPosition = get(hData.hFig,'Position');
    FigWidth = FigPosition(3);
    FigHeight = FigPosition(4);
    ButtonHeight = 20;
    ButtonWidth  = 60;
    Padding = 4;
    EditBottom = Padding*2+ButtonHeight;
    EditHeight = max([FigHeight - EditBottom-Padding,20]);
    OKLeft = max([(FigWidth-ButtonWidth)/4,Padding]);
    CancelLeft = min([(3*FigWidth-ButtonWidth)/4,FigWidth-ButtonWidth-Padding]);
    set(hData.hEdit,'Position',[...
        Padding,Padding*3+ButtonHeight,...
        FigWidth-Padding*2,EditHeight]);
    set(hData.hOK,'Position',[OKLeft,Padding,ButtonWidth, ButtonHeight]);
    set(hData.hCancel,'Position',[CancelLeft,Padding,ButtonWidth, ButtonHeight]);

    end
end