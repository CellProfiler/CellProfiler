function HelpColormaps

% Default colormaps can be set in File > Set preferences.
%
% Label colormap - affects how objects are colored. Colorcube (and possibly
% other colormaps) is not recommended because some intensity values are
% displayed as black. Jet is the default.
%
% Intensity colormap - affects how grayscale images are displayed.
% Colorcube (and possibly other colormaps) is not recommended because some
% intensity values are displayed as black. Gray is recommended.
%
% Choose from these colormaps:
% autumn bone colorcube cool copper flag gray hot hsv jet lines pink
% prism spring summer white winter
CPtextandfiguredisplaybox(help('HelpColormaps'),'Help Colormaps');

% We have one line of actual code in these files so that the help is
% visible. We are not using CPhelpdlg because using helpdlg instead allows
% the help to be accessed from the command line of MATLAB. The one line of
% code in each help file (helpdlg) is never run from inside CP anyway.

function helpFig = CPtextandfiguredisplaybox(Text,Title)

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
%   Peter Swire
%   Rodrigo Ipince
%   Vicky Lay
%   Jun Liu
%   Chris Gan
%
% Website: http://www.cellprofiler.org
%
% $Revision: 4076 $
helpFig = CPfigure;
set(helpFig,'Resize','off');
set(helpFig,'NumberTitle','off');
set(helpFig,'name', Title);
set(helpFig,'units','characters','color',[0.7 0.7 0.9]);
helpFigPos = get(helpFig,'position');
%set(helpFig,'position',[helpFigPos(1),helpFigPos(2),87,helpFigPos(4)]);

WidthOfWindow = 105;
HeightOfWindow = helpFigPos(4)*1.5;
TextBoxFraction = 0.4;
% make window 150% taller and move into window
set(helpFig,'position',[helpFigPos(1),helpFigPos(2)-HeightOfWindow*TextBoxFraction,WidthOfWindow,HeightOfWindow]);
try
    handles = guidata(findobj('Tag','figure1'));
    FontSize = handles.Preferences.FontSize;
catch
    FontSize = 11;
end
% move text box to top and make 33% the size of the entire window
helpUI = uicontrol(...
    'Parent',helpFig,...
    'Enable','inactive',...
    'Units','characters',...
    'HorizontalAlignment','left',...
    'Max',2,...
    'Min',0,...
    'Position',[1 (HeightOfWindow-HeightOfWindow*TextBoxFraction) (WidthOfWindow) (HeightOfWindow*TextBoxFraction)],...
    'String',Text,...
    'BackgroundColor',[0.7 0.7 0.9],...
    'Style','text', ...
    'FontSize',FontSize);
if FontSize <= 8
    WrapNum = 60;
elseif FontSize <= 11 && FontSize > 8
    WrapNum = 50;
elseif FontSize <= 13 && FontSize > 11
    WrapNum = 25;
else
    WrapNum = 10;
end
[outstring,position] = textwrap(helpUI,{Text},WrapNum);
% change this so that the top of the text is lined up with the top of the text box
%set(helpUI,'position',[1 1.5+30-position(4) WidthOfWindow-7 position(4)]);
%set(helpUI,'position',[1 (HeightOfWindow-position(4)) (WidthOfWindow-7) (position(4))]);
LinesToShow = 27;
if(length(outstring) > LinesToShow),
    helpUIPosition = get(helpUI,'position');
    helpScrollCallback = ['set(',num2str(helpUI,'%.13f'),',''position'',[', ...
        num2str(helpUIPosition(1)),' ',num2str(helpUIPosition(2)),'+get(gcbo,''max'')-get(gcbo,''value'') ', num2str(WidthOfWindow-7), ...
        ' ', num2str(helpUIPosition(4)),'])'];

    helpScrollUI = uicontrol(...
        'Parent',helpFig,...
        'Callback',helpScrollCallback,...
        'Units','characters',...
        'Visible', 'on',...
        'BackgroundColor',[0.7 0.7 0.9],...
        'Style', 'slider',...
        'SliderStep', [.1 .6],...
        'Position',[WidthOfWindow-6 HeightOfWindow-HeightOfWindow*TextBoxFraction  4 HeightOfWindow*TextBoxFraction],...
        'FontSize',FontSize);
    set(helpScrollUI,'max',(length(outstring)-LinesToShow)*1.09);
    set(helpScrollUI,'value',(length(outstring)-LinesToShow)*1.09);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
hp = uipanel(...
        'Title','',...
        'ShadowColor',[.5 .5 .5],...
        'FontSize',12,...
        'BackgroundColor',[0.7 0.7 0.9], ...
        'Position',[0.01 0.01 0.98 1-TextBoxFraction]);
a = axes('Parent', hp);
axis off
I = imread('rice.png'); 
BW = im2bw(I, graythresh(I)); 
Image = bwlabel(BW);
GenerateColormaps(Image);

function GenerateColormaps(Image)
ColorImg = label2rgb(Image,'winter','k','shuffle'); subplot(3,3,1), imagesc(ColorImg), title('winter')
ColorImg = label2rgb(Image,'summer','k','shuffle'); subplot(3,3,2), imagesc(ColorImg), title('summer')
ColorImg = label2rgb(Image,'spring','k','shuffle'); subplot(3,3,3), imagesc(ColorImg), title('spring')
ColorImg = label2rgb(Image,'autumn','k','shuffle'); subplot(3,3,4), imagesc(ColorImg), title('autumn')
ColorImg = label2rgb(Image,'colorcube','k','shuffle'); subplot(3,3,5), imagesc(ColorImg), title('colorcube')
ColorImg = label2rgb(Image,'cool','k','shuffle'); subplot(3,3,6), imagesc(ColorImg), title('cool')
ColorImg = label2rgb(Image,'hsv','k','shuffle'); subplot(3,3,7), imagesc(ColorImg), title('hsv')
ColorImg = label2rgb(Image,'hot','k','shuffle'); subplot(3,3,8), imagesc(ColorImg), title('hot')
ColorImg = label2rgb(Image,'jet','k','shuffle'); subplot(3,3,9), imagesc(ColorImg), title('jet')
