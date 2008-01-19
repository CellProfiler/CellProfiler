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
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% $Revision$

helpFig = CPfigure;
set(helpFig,'Resize','off');
set(helpFig,'NumberTitle','off');
set(helpFig,'name', title);
set(helpFig,'units','characters','color',[0.7 0.7 0.9], 'menubar', 'none');
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
    'Enable','on',...
    'Units','normalized',...
    'HorizontalAlignment','left',...
    'Max',100,...
    'Position',[0.05, 0.05, 0.95, 0.95],...
    'String',Text,...
    'BackgroundColor',[0.7 0.7 0.9],...
    'Style','edit', ...
    'FontSize',FontSize);
