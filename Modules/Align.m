function handles = AlgAlign(handles)

% Help for the Align module: 
% 
% The optimal alignment between 2 or 3 incoming images is determined.
% The images are cropped appropriately according to this alignment, so
% the final images will be smaller than the originals by a few pixels
% if alignment is necessary.
% 
% Which image is displayed as which color can be changed by going into
% the module's ".m" file and changing the lines after "FOR DISPLAY
% PURPOSES ONLY".  The first line in each set is red, then green, then
% blue.
 
% The contents of this file are subject to the Mozilla Public License
% Version 1.1 (the "License"); you may not use this file except in
% compliance with the License. You may obtain a copy of the License at 
% http://www.mozilla.org/MPL/
% 
% Software distributed under the License is distributed on an "AS IS"
% basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
% License for the specific language governing rights and limitations under
% the License.
% 
% 
% The Original Code is the Align module.
% 
% The Initial Developer of the Original Code is
% Whitehead Institute for Biomedical Research
% Portions created by the Initial Developer are Copyright (C) 2003,2004
% the Initial Developer. All Rights Reserved.
% 
% Contributor(s):
%   Anne Carpenter <carpenter@wi.mit.edu>
%   Thouis Jones   <thouis@csail.mit.edu>
%   In Han Kang    <inthek@mit.edu>
%
% $Revision$

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow

%%% Reads the current algorithm number, since this is needed to find 
%%% the variable values that the user entered.
CurrentAlgorithm = handles.currentalgorithm;
CurrentAlgorithmNum = str2double(handles.currentalgorithm);

%textVAR01 = What did you call the first image to be aligned? (will be displayed as blue)
%defaultVAR01 = OrigBlue
Image1Name = char(handles.Settings.Vvariable{CurrentAlgorithmNum,1});

%textVAR02 = What do you want to call the aligned first image?
%defaultVAR02 = AlignedBlue
AlignedImage1Name = char(handles.Settings.Vvariable{CurrentAlgorithmNum,2});

%textVAR03 = What did you call the second image to be aligned? (will be displayed as green)
%defaultVAR03 = OrigGreen
Image2Name = char(handles.Settings.Vvariable{CurrentAlgorithmNum,3});

%textVAR04 = What do you want to call the aligned second image?
%defaultVAR04 = AlignedGreen
AlignedImage2Name = char(handles.Settings.Vvariable{CurrentAlgorithmNum,4});

%textVAR05 = What did you call the third image to be aligned? (will be displayed as red)
%defaultVAR05 = /
Image3Name = char(handles.Settings.Vvariable{CurrentAlgorithmNum,5});

%textVAR06 = What do you want to call the aligned third image?
%defaultVAR06 = /
AlignedImage3Name = char(handles.Settings.Vvariable{CurrentAlgorithmNum,6});

%textVAR07 = This module calculates the alignment shift. Do you want to actually adjust the images?
%defaultVAR07 = N
AdjustImage = upper(char(handles.Settings.Vvariable{CurrentAlgorithmNum,7}));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

if strcmp(Image1Name,'/') == 1
    error('Image processing was canceled because no image was loaded in the Align module''s first image slot')
end
%%% Checks whether the image to be analyzed exists in the handles structure.
fieldname = ['dOT', Image1Name];
if isfield(handles, fieldname) == 0
    %%% If the image is not there, an error message is produced.  The error
    %%% is not displayed: The error function halts the current function and
    %%% returns control to the calling function (the analyze all images
    %%% button callback.)  That callback recognizes that an error was
    %%% produced because of its try/catch loop and breaks out of the image
    %%% analysis loop without attempting further modules.
    error(['Image processing was canceled because the Align module could not find the input image.  It was supposed to be named ', Image1Name, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
%%% Reads the image.
Image1 = handles.(fieldname);

%%% Same for Image 2.
if strcmp(Image2Name,'/') == 1
    error('Image processing was canceled because no image was loaded in the Align module''s second image slot')
end
fieldname = ['dOT', Image2Name];
if isfield(handles, fieldname) == 0
    error(['Image processing was canceled because the Align module could not find the input image.  It was supposed to be named ', Image2Name, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
Image2 = handles.(fieldname);

%%% Same for Image 3.
if strcmp(Image3Name,'/') ~= 1
    fieldname = ['dOT', Image3Name];
    if isfield(handles, fieldname) == 0
        error(['Image processing was canceled because the Align module could not find the input image.  It was supposed to be named ', Image3Name, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
    end
    Image3 = handles.(fieldname);
end

%%% Determine the filenames of the images to be analyzed.
fieldname = ['dOTFilename', Image1Name];
FileName1 = handles.(fieldname)(handles.setbeinganalyzed);
fieldname = ['dOTFilename', Image2Name];
FileName2 = handles.(fieldname)(handles.setbeinganalyzed);
if strcmp(upper(Image3Name),'N') ~= 1
    fieldname = ['dOTFilename', Image3Name];
    FileName3 = handles.(fieldname)(handles.setbeinganalyzed);
end

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Aligns three input images.
if strcmp(Image3Name,'/') ~= 1
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
    if strcmp(AdjustImage,'Y') == 1
        AlignedImage2 = subim(Temp2, sx2, sy2);
        AlignedImage3 = subim(Temp3, -sx2, -sy2);
        %%% 1 was already aligned with 2.
        AlignedImage1 = subim(Temp1, sx2, sy2);
    end
else %%% Aligns two input images.
    [sx, sy] = autoalign(Image1, Image2);
    Results = ['(1 vs 2: X ', num2str(sx), ', Y ', num2str(sy),')'];
    if strcmp(AdjustImage,'Y') == 1
        AlignedImage1 = subim(Image1, sx, sy);
        AlignedImage2 = subim(Image2, -sx, -sy);
    end
end

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines the figure number to display in.
fieldname = ['figurealgorithm',CurrentAlgorithm];
ThisAlgFigureNumber = handles.(fieldname);
%%% Note: Everything between the "if" and "end" is not carried out if the 
%%% user has closed
%%% the figure window, so do not do any important calculations here.
%%% Otherwise an error message will be produced if the user has closed the
%%% window but you have attempted to access data that was supposed to be
%%% produced by this part of the code.

%%% Check whether that figure is open. This checks all the figure handles
%%% for one whose handle is equal to the figure number for this algorithm.
%%% Also, check whether any of these images 
if any(findobj == ThisAlgFigureNumber) == 1;

    %%% START CALCULATE IMAGES FOR DISPLAY ONLY %%%
    if strcmp(AdjustImage,'Y') == 1
        %%% For three input images.
        if strcmp(Image3Name,'/') ~= 1
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
    %%% END CALCULATE IMAGES FOR DISPLAY ONLY %%%

    if handles.setbeinganalyzed == 1
        %%% Sets the window to be only 250 pixels wide.
        originalsize = get(ThisAlgFigureNumber, 'position');
        newsize = originalsize;
        newsize(3) = 250;
        set(ThisAlgFigureNumber, 'position', newsize);
    end
    %%% The "drawnow" function executes any pending figure window-related
    %%% commands.  In general, Matlab does not update figure windows
    %%% until breaks between image analysis modules, or when a few select
    %%% commands are used. "figure" and "drawnow" are two of the commands
    %%% that allow Matlab to pause and carry out any pending figure window-
    %%% related commands (like zooming, or pressing timer pause or cancel
    %%% buttons or pressing a help button.)  If the drawnow command is not
    %%% used immediately prior to the figure(ThisAlgFigureNumber) line,
    %%% then immediately after the figure line executes, the other commands
    %%% that have been waiting are executed in the other windows.  Then,
    %%% when Matlab returns to this module and goes to the subplot line,
    %%% the figure which is active is not necessarily the correct one.
    %%% This results in strange things like the subplots appearing in the
    %%% timer window or in the wrong figure window, or in help dialog boxes.
    drawnow
    %%% Activates the appropriate figure window.
    figure(ThisAlgFigureNumber);
    if strcmp(AdjustImage,'Y') == 1
        %%% A subplot of the figure window is set to display the original image.
        subplot(2,1,1); imagesc(OriginalRGB);
        title(['Input Images, Image Set # ',num2str(handles.setbeinganalyzed)]);
        %%% A subplot of the figure window is set to display the adjusted
        %%%  image.
        subplot(2,1,2); imagesc(AlignedRGB); title('Aligned Images');
    end
    displaytexthandle = uicontrol(ThisAlgFigureNumber,'style','text', 'position', [0 0 235 30],'fontname','fixedwidth','backgroundcolor',[0.7,0.7,0.7]);
    set(displaytexthandle,'string',Results)
    set(ThisAlgFigureNumber,'toolbar','figure')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

if strcmp(AdjustImage,'Y') == 1
    %%% Saves the adjusted image to the
    %%% handles structure so it can be used by subsequent algorithms.
    fieldname = ['dOT', AlignedImage1Name];
    handles.(fieldname) = AlignedImage1;
    fieldname = ['dOT', AlignedImage2Name];
    handles.(fieldname) = AlignedImage2;
    if strcmp(Image3Name,'/') ~= 1
        fieldname = ['dOT', AlignedImage3Name];
        handles.(fieldname) = AlignedImage3;
    end
end
%%% Saves the original file name ito the handles structure in a
%%% field named after the adjusted image name.
fieldname = ['dOTFilename', AlignedImage1Name];
handles.(fieldname)(handles.setbeinganalyzed) = FileName1;
fieldname = ['dOTFilename', AlignedImage2Name];
handles.(fieldname)(handles.setbeinganalyzed) = FileName2;
if strcmp(Image3Name,'/') ~= 1
fieldname = ['dOTFilename', AlignedImage3Name];
handles.(fieldname)(handles.setbeinganalyzed) = FileName3;
end

%%% Stores the shift in alignment as a measurement for quality control
%%% purposes.
fieldname = ['dMTXAlign', AlignedImage1Name,AlignedImage2Name];
handles.(fieldname)(handles.setbeinganalyzed) = {sx};
fieldname = ['dMTYAlign', AlignedImage1Name,AlignedImage2Name];
handles.(fieldname)(handles.setbeinganalyzed) = {sy};

%%% If three images were aligned:
if strcmp(Image3Name,'/') ~= 1
fieldname = ['dMTXAlignFirstTwoImages',AlignedImage3Name];
handles.(fieldname)(handles.setbeinganalyzed) = {sx2};
fieldname = ['dMTYAlignFirstTwoImages',AlignedImage3Name];
handles.(fieldname)(handles.setbeinganalyzed) = {sy2};
end

%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTIONS %%%
%%%%%%%%%%%%%%%%%%%

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