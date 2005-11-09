function handles = Align(handles)

% Help for the Align module:
% Category: Image Processing
%
% SHORT DESCRIPTION: Aligns two or three images relative to each other.
% Particularly useful to align microscopy images acquired from different
% color channels.
% *************************************************************************
%
% For two or three input images, this module determines the optimal
% alignment among them.  This works whether the images are correlated or
% anti-correlated (bright in one = bright in the other, or bright in one =
% dim in the other).  This is useful when the microscope is not perfectly
% calibrated, because, for example, proper alignment is necessary for
% primary objects to be helpful to identify secondary objects. The images
% are cropped appropriately according to this alignment, so the final
% images will be smaller than the originals by a few pixels if alignment is
% necessary.
%
% SAVING IMAGES: Any of the three aligned images produced by this module
% can be easily saved using the Save Images module, using the names you
% assign. If you want to save other intermediate images, alter the code for
% this module to save those images to the handles structure (see the
% SaveImages module help) and then use the Save Images module.
%
% See also <nothing>.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne Carpenter <carpenter@wi.mit.edu>
%   Thouis Jones   <thouis@csail.mit.edu>
%   In Han Kang    <inthek@mit.edu>
%   Ola Friman     <friman@bwh.harvard.edu>
%   Steve Lowe     <stevelowe@alum.mit.edu>
%   Joo Han Chang  <joohan.chang@gmail.com>
%   Colin Clarke   <colinc@mit.edu>
%   Mike Lamprecht <mrl@wi.mit.edu>
%   Susan Ma       <xuefang_ma@wi.mit.edu>
%
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);
ModuleName = char(handles.Settings.ModuleNames(CurrentModuleNum));

%textVAR01 = What did you call the first image to be aligned? (will be displayed as blue)
%infotypeVAR01 = imagegroup
Image1Name = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the aligned first image?
%defaultVAR02 = AlignedBlue
%infotypeVAR02 = imagegroup indep
AlignedImage1Name = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = What did you call the second image to be aligned? (will be displayed as green)
%infotypeVAR03 = imagegroup
Image2Name = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 = What do you want to call the aligned second image?
%defaultVAR04 = AlignedGreen
%infotypeVAR04 = imagegroup indep
AlignedImage2Name = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = What did you call the third image to be aligned? (will be displayed as red)
%choiceVAR05 = Do not use
%infotypeVAR05 = imagegroup
Image3Name = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu

%textVAR06 = What do you want to call the aligned third image?
%defaultVAR06 = Do not use
%infotypeVAR06 = imagegroup indep
AlignedImage3Name = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = This module calculates the alignment shift and stores it as a measurement. Do you want to actually shift the images and crop them to produce the aligned images?
%choiceVAR07 = Yes
%choiceVAR07 = No
AdjustImage = char(handles.Settings.VariableValues{CurrentModuleNum,7});
%inputtypeVAR07 = popupmenu

%%%VariableRevisionNumber = 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Checks whether the image to be analyzed exists in the handles structure.
if ~isfield(handles.Pipeline, Image1Name)
    %%% If the image is not there, an error message is produced.  The error
    %%% is not displayed: The error function halts the current function and
    %%% returns control to the calling function (the analyze all images
    %%% button callback.)  That callback recognizes that an error was
    %%% produced because of its try/catch loop and breaks out of the image
    %%% analysis loop without attempting further modules.
    error(['Image processing was canceled in the ', ModuleName, ' module because the input image could not be found.  It was supposed to be named ', Image1Name, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
%%% Reads the image.
Image1 = handles.Pipeline.(Image1Name);

if max(Image1(:)) > 1 || min(Image1(:)) < 0
    CPwarndlg('The first image that you loaded is outside the 0-1 range, and you may be losing data.','Outside 0-1 Range','replace');
end

%%% Same for Image 2.
if ~isfield(handles.Pipeline, Image2Name)
    error(['Image processing was canceled in the ', ModuleName, ' module because the input image could not be found.  It was supposed to be named ', Image2Name, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
Image2 = handles.Pipeline.(Image2Name);

if max(Image2(:)) > 1 || min(Image2(:)) < 0
    CPwarndlg('The second image that you loaded is outside the 0-1 range, and you may be losing data.','Outside 0-1 Range','replace');
end

%%% Same for Image 3.
if ~strcmp(Image3Name,'Do not use')
    if ~isfield(handles.Pipeline, Image3Name)
        error(['Image processing was canceled in the ', ModuleName, ' module because the input image could not be found.  It was supposed to be named ', Image3Name, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
    end
    Image3 = handles.Pipeline.(Image3Name);
    if max(Image3(:)) > 1 || min(Image3(:)) < 0
        CPwarndlg('The third image that you loaded is outside the 0-1 range, and you may be losing data.','Outside 0-1 Range','replace');
    end

    if ndims(Image3) ~= 2
        error(['Image processing was canceled in the ', ModuleName, ' module because it requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.']);
    end
end

if ndims(Image1) ~= 2 || ndims(Image2) ~= 2
    error(['Image processing was canceled in the ', ModuleName, ' module because it requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.']);
end

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Aligns three input images.
if ~strcmp(Image3Name,'Do not use')
    %%% Aligns 1 and 2 (see subfunctions at the end of the module).
    [sx, sy] = autoalign(Image1, Image2);
    Temp1 = subim(Image1, sx, sy);
    Temp2 = subim(Image2, -sx, -sy);
    %%% Assumes 3 is stuck to 2.
    Temp3 = subim(Image3, -sx, -sy);
    %%% Aligns 2 and 3.
    [sx2, sy2] = autoalign(Temp2, Temp3);
    Results = ['(1 vs 2: X ', num2str(sx), ', Y ', num2str(sy), ...
        ') (2 vs 3: X ', num2str(sx2), ', Y ', num2str(sy2),')'];
    if strcmp(AdjustImage,'Yes') == 1
        AlignedImage2 = subim(Temp2, sx2, sy2);
        AlignedImage3 = subim(Temp3, -sx2, -sy2);
        %%% 1 was already aligned with 2.
        AlignedImage1 = subim(Temp1, sx2, sy2);
    end
else %%% Aligns two input images.
    [sx, sy] = autoalign(Image1, Image2);
    Results = ['(1 vs 2: X ', num2str(sx), ', Y ', num2str(sy),')'];
    if strcmp(AdjustImage,'Yes') == 1
        AlignedImage1 = subim(Image1, sx, sy);
        AlignedImage2 = subim(Image2, -sx, -sy);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines the figure number to display in.
fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
if any(findobj == ThisModuleFigureNumber) == 1;
    if strcmp(AdjustImage,'Yes') == 1
        %%% For three input images.
        if strcmp(Image3Name,'Do not use') ~= 1
            OriginalRGB(:,:,1) = Image3;
            OriginalRGB(:,:,2) = Image2;
            OriginalRGB(:,:,3) = Image1;
            AlignedRGB(:,:,1) = AlignedImage3;
            AlignedRGB(:,:,2) = AlignedImage2;
            AlignedRGB(:,:,3) = AlignedImage1;
        else %%% For two input images.
            OriginalRGB(:,:,1) = zeros(size(Image1));
            OriginalRGB(:,:,2) = Image2;
            OriginalRGB(:,:,3) = Image1;
            AlignedRGB(:,:,1) = zeros(size(AlignedImage1));
            AlignedRGB(:,:,2) = AlignedImage2;
            AlignedRGB(:,:,3) = AlignedImage1;
        end
    end
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        %%% Sets the window to be only 250 pixels wide.
        originalsize = get(ThisModuleFigureNumber, 'position');
        newsize = originalsize;
        newsize(3) = 250;
        set(ThisModuleFigureNumber, 'position', newsize);
    end

    drawnow
    %%% Activates the appropriate figure window.
    CPfigure(handles,ThisModuleFigureNumber);
    if strcmp(AdjustImage,'Yes') == 1
        %%% A subplot of the figure window is set to display the original image.
        subplot(2,1,1);
        ImageHandle = imagesc(OriginalRGB);
        set(ImageHandle,'ButtonDownFcn','CPImageTool(gco)');
        title(['Input Images, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
        %%% A subplot of the figure window is set to display the adjusted
        %%%  image.
        subplot(2,1,2);
        ImageHandle = imagesc(AlignedRGB);
        set(ImageHandle,'ButtonDownFcn','CPImageTool(gco)');
        title('Aligned Images');
    end
    displaytexthandle = uicontrol(ThisModuleFigureNumber,'style','text', 'position', [0 0 235 30],'fontname','helvetica','backgroundcolor',[0.7,0.7,0.9],'FontSize',handles.Current.FontSize);
    set(displaytexthandle,'string',['Offset: ',Results])
    set(ThisModuleFigureNumber,'toolbar','figure')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

if strcmp(AdjustImage,'Yes') == 1
    %%% Saves the adjusted image to the
    %%% handles structure so it can be used by subsequent modules.
    handles.Pipeline.(AlignedImage1Name) = AlignedImage1;
    handles.Pipeline.(AlignedImage2Name) = AlignedImage2;
    if strcmp(Image3Name,'Do not use') ~= 1
        handles.Pipeline.(AlignedImage3Name) = AlignedImage3;
    end
end

%%% Stores the shift in alignment as a measurement for quality control
%%% purposes.
fieldname = ['ImageXAlign', AlignedImage1Name,AlignedImage2Name];
handles.Measurements.(fieldname)(handles.Current.SetBeingAnalyzed) = {sx};
fieldname = ['ImageYAlign', AlignedImage1Name,AlignedImage2Name];
handles.Measurements.(fieldname)(handles.Current.SetBeingAnalyzed) = {sy};

%%% If three images were aligned:
if strcmp(Image3Name,'Do not use') ~= 1
    fieldname = ['ImageXAlignFirstTwoImages',AlignedImage3Name];
    handles.Measurements.(fieldname)(handles.Current.SetBeingAnalyzed) = {sx2};
    fieldname = ['ImageYAlignFirstTwoImages',AlignedImage3Name];
    handles.Measurements.(fieldname)(handles.Current.SetBeingAnalyzed) = {sy2};
end

%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTIONS %%%
%%%%%%%%%%%%%%%%%%%%

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