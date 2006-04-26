function CPresizefigure(OrigImage,Layout,FigHandle)

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
% $Revision: 2807 $

FigPosOrig = get(FigHandle,'Position');
FigPos = FigPosOrig;

%%% Handles Text display windows.
if strcmp(Layout,'NarrowText') && (FigPos(3) ~= 280)
    FigPos(3) = 280;
    %%% Sets the figure position and size.
    set(FigHandle,'Position',FigPos);
    return
end

ImagePos = size(OrigImage);
ScSize = get(0,'ScreenSize');
FigureWidth = FigPosOrig(3);
FigureHeight = FigPosOrig(4);
ScreenWidth = ScSize(3);
ScreenHeight = ScSize(4);
ImageWidth = ImagePos(2);
ImageHeight = ImagePos(1);

%%% If the window is 1 row x 1 column (half width and half height):
if strcmp(Layout,'OneByOne')
    %%% Makes the figure half width and half height.
    FigureWidth = .5*FigureWidth;
    FigureHeight = .5*FigureHeight;
    if (FigureWidth/FigureHeight) < (ImageWidth/ImageHeight)
        %%% 40 is added to allow for axes labels.
        FigureHeight = 40 + FigureWidth * ImageHeight / ImageWidth;
    else
        %%% 40 is added to allow for axes labels.
        FigureWidth = 40 + FigureHeight * ImageWidth / ImageHeight;
    end
%%% If the window is 2 rows x 2 columns (standard):
%%% 2*ImageHeight is because there are two rows.
%%% 2*ImageWidth is because there are two columns.
elseif strcmp(Layout,'TwoByTwo')
    if (FigureWidth/FigureHeight) < ((2*ImageWidth)/(2*ImageHeight))
        %%% 40 is added to allow for axes labels.
        FigureHeight = 40 + FigureWidth * (2*ImageHeight) / (2*ImageWidth);
    else
        %%% 40 is added to allow for axes labels.
        FigureWidth = 40 + FigureHeight * (2*ImageWidth) / (2*ImageHeight);
    end
%%% If the window is 2 rows x 1 column (narrow):
elseif strcmp(Layout,'TwoByOne')
    %%% Makes the figure half width.    
    FigureWidth = .5*FigureWidth;
    %%% 2*ImageHeight is because there are two rows.
    if (FigureWidth/FigureHeight) < (ImageWidth/(2*ImageHeight))
        %%% 40 is added to allow for axes labels.
        FigureHeight = 40 + FigureWidth * (2*ImageHeight) / ImageWidth;
    else
        %%% 40 is added to allow for axes labels.
        FigureWidth = 40 + FigureHeight * ImageWidth / (2*ImageHeight);
    end
end

%%% Checks whether the resulting figure would take up more than 40% of
%%% the screen height or width and adjusts accordingly.
HorRatio = FigureWidth / ScreenWidth;
VerRatio = FigureHeight / ScreenHeight;
if(HorRatio > .8) || (VerRatio > .8)
    if(HorRatio > VerRatio)
        FigureWidth = ScreenWidth * 0.8;
        FigureHeight = FigureHeight / FigureWidth * ScreenWidth * 0.8;
    else
        FigureHeight = ScreenHeight * 0.8;
        FigureWidth = FigureWidth / FigureHeight * ScreenHeight * 0.8;
    end
end

%%% If the height has changed as a result of the above, adjusts the
%%% position to be sure it's still at the top of the screen.
if FigureHeight ~= FigPosOrig(4)
    FigPos(2) = ScreenHeight - (FigureHeight+80);
end

%%% Sets the newly calculated values.
FigPos(3) = FigureWidth;
FigPos(4) = FigureHeight;

%%% Sets the figure position and size.
set(FigHandle,'Position',FigPos);