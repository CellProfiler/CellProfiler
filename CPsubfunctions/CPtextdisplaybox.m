function helpFig = CPtextdisplaybox(Text,title)

% Custom text display box for CellProfiler, with a slider if the text
% becomes too long to fit into a small window.
%
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
% $Revision$

helpFig = CPfigure;
set(helpFig,'Resize','off');
set(helpFig,'NumberTitle','off');
set(helpFig,'name', title);
set(helpFig,'units','characters','color',[0.7 0.7 0.9]);
helpFigPos = get(helpFig,'position');
%set(helpFig,'position',[helpFigPos(1),helpFigPos(2),87,helpFigPos(4)]);

WidthOfWindow = 105;
set(helpFig,'position',[helpFigPos(1),helpFigPos(2),WidthOfWindow,helpFigPos(4)]);

try
    handles = guidata(findobj('Tag','figure1'));
    FontSize = handles.Preferences.FontSize;
catch
    FontSize = 11;
end

helpUI = uicontrol(...
    'Parent',helpFig,...
    'Enable','inactive',...
    'Units','characters',...
    'HorizontalAlignment','left',...
    'Max',2,...
    'Min',0,...
    'Position',[1 1 WidthOfWindow helpFigPos(4)],...
    'String',Text,...
    'BackgroundColor',[0.7 0.7 0.9],...
    'Style','text', ...
    'FontSize',FontSize);
if FontSize <= 8
    WrapNum = 80;
elseif FontSize <= 11 && FontSize > 8
    WrapNum = 50;
elseif FontSize <= 13 && FontSize > 11
    WrapNum = 25;
else
    WrapNum = 10;
end
[outstring,position] = textwrap(helpUI,{Text},WrapNum);
set(helpUI,'position',[1 1.5+30-position(4) WidthOfWindow-7 position(4)]);
if(length(outstring) > 27),
    helpUIPosition = get(helpUI,'position');
    helpScrollCallback = ['set(',num2str(helpUI,'%.13f'),',''position'',[', ...
        num2str(helpUIPosition(1)),' ',num2str(helpUIPosition(2)),'+get(gcbo,''max'')-get(gcbo,''value'') ', num2str(WidthOfWindow), ...
        ' ', num2str(helpUIPosition(4)),'])'];

    helpScrollUI = uicontrol(...
        'Parent',helpFig,...
        'Callback',helpScrollCallback,...
        'Units','characters',...
        'Visible', 'on',...
        'BackgroundColor',[0.7 0.7 0.9],...
        'Style', 'slider',...
        'SliderStep', [.1 .6],...
        'Position',[WidthOfWindow-6 1 4 30],...
        'FontSize',FontSize);
    set(helpScrollUI,'max',(length(outstring)-27)*1.09);
    set(helpScrollUI,'value',(length(outstring)-27)*1.09);
end