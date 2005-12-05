function handles = StretchAlignCrop(handles)

% Help for the Stretch, Align, and Crop module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Stretches, aligns, and crops one image relative to another. Useful for
% valdiation of hand outlined images.
% *************************************************************************
%
% This module was written for a specific purpose: to take a larger
% image of the outlines of cells, with some space around the outside,
% and align it with a real image of cells (or of nuclei).
% Fortunately, it can work for almost any set of images for which one
% set is larger than the other and rescaling must be done (and, of
% course, there are features shared which can be used to align the
% images).  Just enter the larger images as the traced ones, and the
% smaller ones as the real images.
%
% This module determines the best alignment of two images.  It expects
% the first to be a scanned, traced transparency of 8 1/2 by 11
% inches, and the second should be a real image from a microscope.
% The addition of colored registration marks is optional, although
% very helpful in alignment.  The module stretches the real image to
% be at the same resolution as the traced image, then creates a black
% background for it of the same size, then aligns the images, and
% finally crops them all to be the same size and taking up the same
% area as on the original real image. In the future, this may be
% developed to include a way of finding the optimal scaling of the
% images to each other; however, no guarantees.
%
% Note that as long as the input real images of nuclei and of cells
% are the same dimensions, their output files will have the same
% dimensions as well.
%
% See also ALIGN.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne Carpenter
%   Thouis Jones
%   In Han Kang
%   Ola Friman
%   Steve Lowe
%   Joo Han Chang
%   Colin Clarke
%   Mike Lamprecht
%   Susan Ma
%   Wyman Li
%
% Website: http://www.cellprofiler.org
%
% $Revision: 2782 $

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow


[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the traced images?
%infotypeVAR01 = imagegroup
TracedImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What did you call the real images?
%defaultVAR02 = OrigReal
%infotypeVAR02 = imagegroup
RealImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 = What do you want to call the aligned, cropped traced images?
%defaultVAR03 = ACTraced
%infotypeVAR03 = imagegroup indep
FinishedTracedImageName = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = What do you want to call the aligned, cropped real images?
%defaultVAR04 = ACReal
%infotypeVAR04 = imagegroup indep
FinishedRealImageName = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = Enter the printed size of the real image in inches as "height,width" (no quotes).
%choiceVAR05 = 11,8.5
PrintedImageSize = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu custom

%textVAR06 = Enter the page orientation of the traced images (portrait or landscape)
%choiceVAR06 = portrait
%choiceVAR06 = landscape
Orientation = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%inputtypeVAR06 = popupmenu

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Separates the entered dimensions into two variables
PrintedImageSizeNumerical = str2double(PrintedImageSize);
PrintedHeight = PrintedImageSizeNumerical(1);
PrintedWidth = PrintedImageSizeNumerical(2);

%%% Checks that the page orientation is a valid entry
if strcmp(upper(Orientation),'PORTRAIT') ~= 1
    if strcmp(upper(Orientation),'LANDSCAPE') ~= 1
        error(['Image processing was canceled in the ', ModuleName, ' module because you have not entered a valid response for the page orientaion in the AlignAndCrop module.  Type either landscape or portrait and try again.'])
    end
end

%%% Checks whether the image to be analyzed exists in the handles structure.
if isfield(handles.Pipeline, TracedImageName) == 0
    %%% If the image is not there, an error message is produced.  The error
    %%% is not displayed: The error function halts the current function and
    %%% returns control to the calling function (the analyze all images
    %%% button callback.)  That callback recognizes that an error was
    %%% produced because of its try/catch loop and breaks out of the image
    %%% analysis loop without attempting further modules.
    error(['Image processing was canceled in the ', ModuleName, ' module because it could not find the input image.  It was supposed to be named ', TracedImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
%%% Reads the image.
TracedImage = handles.Pipeline.(TracedImageName);

%%% Checks whether the image to be analyzed exists in the handles structure.
if isfield(handles, RealImageName) == 0
    %%% If the image is not there, an error message is produced.  The error
    %%% is not displayed: The error function halts the current function and
    %%% returns control to the calling function (the analyze all images
    %%% button callback.)  That callback recognizes that an error was
    %%% produced because of its try/catch loop and breaks out of the image
    %%% analysis loop without attempting further modules.
    error(['Image processing was canceled in the ', ModuleName, ' module because it could not find the input image.  It was supposed to be named ', RealImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
%%% Reads the image.
RealImage = handles.Pipeline.(RealImageName);

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% creates a two-dimensional, b&w Traced Image to work with during this section
ExTracedImage = TracedImage(:,:,1);
PreTracedImage = im2bw(ExTracedImage,.5);
%%% creates a 2D real image to work with for the size
ExRealImage = RealImage(:,:,1);
PreRealImage = im2bw(ExRealImage,.5);
%%% finds the size of each of the images
[TracedY,TracedX] = size(PreTracedImage);
[RealY,RealX] = size(PreRealImage);
%CPwarndlg(['The dimensions are as follows: TY=',num2str(TracedY),' TX=',num2str(TracedX),' RY=',num2str(RealY),' RX=',num2str(RealX)],'Alert!')
%%% checks that for both dimensions, the traced image is bigger than the
%%% real one
drawnow
if TracedX < RealX
    error(['Image processing was canceled in the ', ModuleName, ' module because the real image is wider than the traced image. Make sure that the resolution is the same for both images.'])
elseif TracedY < RealY
    error(['Image processing was canceled in the ', ModuleName, ' module because the real image is taller than the traced image.  Make sure that the resolution is the same for both images.'])
end

%%% Finds the resolution along both axes for both sets of images
RealResolution = RealX / PrintedWidth;
RealResolution2 = RealY / PrintedHeight;
if strcmp(upper(Orientation),'PORTRAIT') == 1
    TracedResolution = TracedX /8.5;
    TracedResolution2 = TracedY /11;
    %CPwarndlg('I think that you wrote portrait')
else TracedResolution = TracedX / (11);
    TracedResolution2 = TracedY / (8.5);
    %CPwarndlg('I think that you wrote landscape')
end
%%% activate this line to pop up a dialog box with the resolution values in it
%CPwarndlg(['RR is ',num2str(RealResolution),',RR2 is ',num2str(RealResolution2),', TR is ',num2str(TracedResolution),', TR2 is ',num2str(TracedResolution2)],'Alert!')
drawnow

if RealResolution ~= RealResolution2
    error(['Image processing was canceled in the ', ModuleName, ' module because of some fundamental problem in coding or possibly the resolution is different depending upon which axis you calculate it.  Re-measure your printed image: you may be trying to be too accurate.'])
end
if TracedResolution ~= TracedResolution2
    error(['Image processing was canceled in the ', ModuleName, ' module because of some fundamental problem in coding or possibly the resolution is different depending upon which axis you calculate it.  Most likely, you scanned the image not perfectly at 8.5 by 11 and caused major problems.'])
end
drawnow
%%% Resizes the real image to be at the same resolution as the traced image
ResizingCoefficient = TracedResolution / RealResolution;
%CPwarndlg(['Resizing Coefficient = ',num2str(ResizingCoefficient)],'NFO 4 U')
ResizedRealImage = imresize(RealImage,ResizingCoefficient,'bicubic');
%%% finds the dimensions of the resized real image
[RRealX,RRealY] = size(ResizedRealImage);
%CPwarndlg(['RRealX = ',num2str(RRealX),', RRealY = ',num2str(RRealY)],'Yeahhhhh, About That...')
%%% finds the difference in dimensions to create a margin value
XDifference = TracedX - RRealX;
YDifference = TracedY - RRealY;
XMargin = XDifference / 2;
YMargin = YDifference / 2;

drawnow
%%% creates a matrix of zeros the size of the traced image
RealImageBlankSlate = zeros(TracedY,TracedX);
%%% makes sure images are the right size
NewImageSizeX = TracedX - 2*(XMargin);
NewImageSizeY = TracedY - 2*(YMargin);
if isequal(NewImageSizeX,RRealX) == 0
    error(['Image processing was canceled in the ', ModuleName, ' module because the value (TracedX - 2(XMargin)) is not the same as the value (RRealX).'])
elseif isequal(NewImageSizeY,RRealY) == 0
    error(['Image processing was canceled in the ', ModuleName, ' module because the value (TracedY - 2(YMargin)) is not the same as the value (RRealY).'])
end
drawnow
%%% rounds each number up, then subtracts the original from that and tests
%%% equality with zero to see if the numbers are not whole, which then pops
%%% up messages but does not abort. then it pastes the resized real image
%%% over the blank image
RXM = round(XMargin);
RYM = round(YMargin);
XMD = RXM - XMargin;
YMD = RYM - YMargin;
if isequal(XMD,0) == 0
    if isequal(YMD,0) == 0
        CPwarndlg(['Warning in the ', ModuleName, ' module: neither margin value is an integer, and therefore problems may appear in the very near future. XM = ',num2str(XMargin),', YM = ',num2str(YMargin)],'Feeling Unwhole')
        YMargin1 = ceil(YMargin);
        YMargin2 = floor(YMargin);
    else CPwarndlg(['Warning in the ', ModuleName, ' module: the XMargin number is not an integer, and therefore may cause problems in the very near future. Fortunately, the YMargin value is an integer'],'Feeling Unwhole')
    end
    XMargin1 = ceil(XMargin);
    XMargin2 = floor(XMargin);
elseif isequal(YMD,0) == 0
    CPwarndlg(['Warning in the ', ModuleName, ' module: the YMargin number is not an integer, and therefore may cause problems in the very near future. Fortunately, the XMargin value is an integer'],'Feeling Unwhole')
    YMargin1 = ceil(YMargin);
    YMargin2 = floor(YMargin);
else XMargin1 = XMargin;
    XMargin2 = XMargin;
    YMargin1 = YMargin;
    YMargin2 = YMargin;
end
YBegin = YMargin2 + 1;
YEnd = TracedY - YMargin1;
XBegin = XMargin2 + 1;
XEnd = TracedX - XMargin1;
RealImageBlankSlate(YBegin:YEnd,XBegin:XEnd) = ResizedRealImage;
ExpandedRealImage = RealImageBlankSlate;

drawnow
%%% makes a 2D traced image to work with
TwoDTracedImage = rgb2gray(TracedImage);
%%% runs the alignment subfunctions on the two images, working with just
%%% one layer but applying movements to the whole image
InvertedTraced = imcomplement(TwoDTracedImage);
[sx,sy] = autoalign(InvertedTraced,ExpandedRealImage);
%CPwarndlg('The autoalign step has completed','Notice:')
AlignedTracedImage = subim(TwoDTracedImage, sx, sy);
AlignedRealImage = subim(ExpandedRealImage, -sx, -sy);
%CPwarndlg('The subim steps have completed','Notice')
% Results = ['(Traced vs. Real: X ',num2str(sx),', Y ',num2str(sy),')'];
%CPwarndlg(['All image processing has completed. Results are ',Results],'Notice:')
%%% Checks that the size of aligned images is the same
if isequal(size(AlignedTracedImage),size(AlignedRealImage)) == 0
    error(['Image processing was canceled in the ', ModuleName, ' module because after the alignment step was completed, the two images were different sizes for some reason.'])
end
%%% finds the end and begin points for all dimensions of the new images and
%%% creates the cropping rectangle matrix
YBeginARI = YBegin + sy;
YEndARI = YEnd + sy;
XBeginARI = XBegin + sx;
XEndARI = XEnd + sx;
CropRectARI = [XBeginARI YBeginARI (XEndARI - XBeginARI) (YEndARI - YBeginARI)];
%%% crops both aligned images
CroppedAlignedTracedImage = imcrop(AlignedTracedImage,CropRectARI);
CroppedAlignedRealImage = imcrop(AlignedRealImage,CropRectARI);

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines the figure number to display in.
ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber) == 1;
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    subplot(2,2,1); 
    CPimagesc(TracedImage,handles);
    title(['Traced Input, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    subplot(2,2,2); 
    CPimagesc(RealImage,handles); 
    title('Real Input Image');
    subplot(2,2,3); 
    CPimagesc(CroppedAlignedTracedImage,handles);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% The adjusted image is saved to the handles structure so it can be used
%%% by subsequent modules.
handles.Pipeline.(FinishedTracedImageName) = CroppedAlignedTracedImage;
handles.Pipeline.(FinishedRealImageName) = CroppedAlignedRealImage;

%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTIONS %%%
%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Written by Thouis R. Jones in the Align module

function [shiftx, shifty] = autoalign(in1, in2)
%%% Aligns two images using mutual-information and hill-climbing.
best = mutualinf(in1, in2);
bestx = 0;
besty = 0;
%%% Checks which one-pixel move is best.
for dx=-1:1,
    for dy=-1:1,
        cur = mutualinf(subim(in1, dx, dy), subim(in2, -dx, -dy));
        if (cur > best),
            best = cur;
            bestx = dx;
            besty = dy;
        end
    end
end
if (bestx == 0) && (besty == 0),
    shiftx = 0;
    shifty = 0;
    return;
end
%%% Remembers the lastd direction we moved.
lastdx = bestx;
lastdy = besty;
%%% Loops until things stop improving.
while true,
    [nextx, nexty, newbest] = one_step(in1, in2, bestx, besty, lastdx, lastdy, best);
    if (nextx == 0) && (nexty == 0),
        shiftx = bestx;
        shifty = besty;
        return;
    else
        bestx = bestx + nextx;
        besty = besty + nexty;
        best = newbest;
    end
end

function [nx, ny, nb] = one_step(in1, in2, bx, by, ldx, ldy, best)
%%% Finds the best one pixel move, but only in the same direction(s) we
%%% moved last time (no sense repeating evaluations)
nb = best;
for dx=-1:1,
    for dy=-1:1,
        if (dx == ldx) || (dy == ldy),
            cur = mutualinf(subim(in1, bx+dx, by+dy), subim(in2, -(bx+dx), -(by+dy)));
            if (cur > nb),
                nb = cur;
                nx = dx;
                ny = dy;
            end
        end
    end
end
if (best == nb),
    %%% no change, so quit searching
    nx = 0;
    ny = 0;
end

function sub = subim(im, dx, dy)
%%% Subimage with positive or negative offsets
if (dx > 0),
    sub = im(:,dx+1:end);
else
    sub = im(:,1:end+dx);
end
if (dy > 0),
    sub = sub(dy+1:end,:);
else
    sub = sub(1:end+dy,:);
end

function H = entropy(X)
%%% Entropy of samples X
S = imhist(X,256);
%%% if S is probability distribution function N is 1
N=sum(sum(S));
if ((N>0) && (min(S(:))>=0))
    Snz=nonzeros(S);
    H=log2(N)-sum(Snz.*log2(Snz))/N;
else
    H=0;
end

function H = entropy2(X,Y)
%%% joint entropy of paired samples X and Y
%%% Makes sure images are binned to 256 graylevels
X = double(im2uint8(X));
Y = double(im2uint8(Y));
%%% Creates a combination image of X and Y
XY = 256*X + Y;
S = histc(XY(:),0:(256*256-1));
%%% If S is probability distribution function N is 1
N=sum(sum(S));
if ((N>0) && (min(S(:))>=0))
    Snz=nonzeros(S);
    H=log2(N)-sum(Snz.*log2(Snz))/N;
else
    H=0;
end

function I = mutualinf(X, Y)
%%% Mutual information of images X and Y
I = entropy(X) + entropy(Y) - entropy2(X,Y);