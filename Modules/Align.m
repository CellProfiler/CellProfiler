function handles = AlgAlign2(handles)
%%% Reads the current algorithm number, since this is needed to find 
%%% the variable values that the user entered.
CurrentAlgorithm = handles.currentalgorithm;

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow

%textVAR01 = What did you call the first image to be aligned?
%defaultVAR01 = OrigBlue
fieldname = ['Vvariable',CurrentAlgorithm,'_01'];
Image1Name = handles.(fieldname);
%textVAR02 = What do you want to call the aligned first image?
%defaultVAR02 = AlignedBlue
fieldname = ['Vvariable',CurrentAlgorithm,'_02'];
AlignedImage1Name = handles.(fieldname);
%textVAR03 = To save the adjusted first image, enter text to append to the image name 
%defaultVAR03 = N
fieldname = ['Vvariable',CurrentAlgorithm,'_03'];
SaveImage1 = handles.(fieldname);
%textVAR04 = What did you call the second image to be aligned?
%defaultVAR04 = OrigGreen
fieldname = ['Vvariable',CurrentAlgorithm,'_04'];
Image2Name = handles.(fieldname);
%textVAR05 = What do you want to call the aligned second image?
%defaultVAR05 = AlignedGreen
fieldname = ['Vvariable',CurrentAlgorithm,'_05'];
AlignedImage2Name = handles.(fieldname);
%textVAR06 = To save the adjusted second image, enter text to append to the image name 
%defaultVAR06 = N
fieldname = ['Vvariable',CurrentAlgorithm,'_06'];
SaveImage2 = handles.(fieldname);
%textVAR07 = What did you call the third image to be aligned?
%defaultVAR07 = /
fieldname = ['Vvariable',CurrentAlgorithm,'_07'];
Image3Name = handles.(fieldname);
%textVAR08 = What do you want to call the aligned third image?
%defaultVAR08 = /
fieldname = ['Vvariable',CurrentAlgorithm,'_08'];
AlignedImage3Name = handles.(fieldname);
%textVAR09 = To save the adjusted third image, enter text to append to the image name 
%defaultVAR09 = N
fieldname = ['Vvariable',CurrentAlgorithm,'_09'];
SaveImage3 = handles.(fieldname);
%textVAR10 =  Otherwise, leave as "N". To save or display other images, press Help button
%textVAR11 = In what file format do you want to save images? Do not include a period
%defaultVAR11 = tif
fieldname = ['Vvariable',CurrentAlgorithm,'_11'];
FileFormat = handles.(fieldname);

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
% figure, imshow(Image1), title('Image1')

%%% Same for Image 2.
if strcmp(Image2Name,'/') == 1
    error('Image processing was canceled because no image was loaded in the Align module''s second image slot')
end
fieldname = ['dOT', Image2Name];
if isfield(handles, fieldname) == 0
    error(['Image processing was canceled because the Align module could not find the input image.  It was supposed to be named ', Image2Name, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
Image2 = handles.(fieldname);
% figure, imshow(Image2), title('Image2')

%%% Same for Image 3.
if strcmp(Image3Name,'/') ~= 1
    fieldname = ['dOT', Image3Name];
    if isfield(handles, fieldname) == 0
        error(['Image processing was canceled because the Align module could not find the input image.  It was supposed to be named ', Image3Name, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
    end
    Image3 = handles.(fieldname);
    % figure, imshow(Image1), title('Image3')
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

%%% Checks whether the file format the user entered is readable by Matlab.
IsFormat = imformats(FileFormat);
if isempty(IsFormat) == 1
    error('The image file type entered in the Align module is not recognized by Matlab. Or, you may have entered a period in the box. For a list of recognizable image file formats, type "imformats" (no quotes) at the command line in Matlab.','Error')
end

%%% Checks whether the appendages to be added to the file names of images
%%% will result in overwriting the original file, or in a file name that
%%% contains spaces.
if strcmp(upper(SaveImage1),'N') ~= 1
    %%% Finds and removes the file format extension within the original file
    %%% name, but only if it is at the end. Strips the original file format extension 
    %%% off of the file name, if it is present, otherwise, leaves the original
    %%% name intact.
    CharFileName1 = char(FileName1);
    PotentialDot1 = CharFileName1(end-3:end-3);
    if strcmp(PotentialDot1,'.') == 1
        BareFileName1 = CharFileName1(1:end-4);
    else BareFileName1 = CharFileName1;
    end
    %%% Assembles the new image name.
    NewImageName1 = [BareFileName1,SaveImage1,'.',FileFormat];
    %%% Checks whether the new image name is going to result in a name with
    %%% spaces.
    A = isspace(SaveImage1);
    if any(A) == 1
        error('Image processing was canceled because you have entered one or more spaces in the box of text to append to the aligned image name in the Align module.  If you do not want to save the image to the hard drive, type "N" into the appropriate box.')
        return
    end
    %%% Checks whether the new image name is going to result in overwriting the
    %%% original file.
    B = strcmp(upper(CharFileName1), upper(NewImageName1));
    if B == 1
        error('Image processing was canceled because you have not entered text to append to the aligned image name in the Align module.  If you do not want to save the aligned image to the hard drive, type "N" into the appropriate box.')
        return
    end
end

%%% Same for Image 2.
if strcmp(upper(SaveImage2),'N') ~= 1
    CharFileName2 = char(FileName2);
    PotentialDot2 = CharFileName2(end-3:end-3);
    if strcmp(PotentialDot2,'.') == 1
        BareFileName2 = CharFileName2(1:end-4);
    else BareFileName2 = CharFileName2;
    end
    NewImageName2 = [BareFileName2,SaveImage2,'.',FileFormat];
    A = isspace(SaveImage2);
    if any(A) == 1
        error('Image processing was canceled because you have entered one or more spaces in the box of text to append to the aligned image name in the Align module.  If you do not want to save the image to the hard drive, type "N" into the appropriate box.')
        return
    end
    B = strcmp(upper(CharFileName2), upper(NewImageName2));
    if B == 1
        error('Image processing was canceled because you have not entered text to append to the aligned image name in the Align module.  If you do not want to save the aligned image to the hard drive, type "N" into the appropriate box.')
        return
    end
end
if strcmp(upper(SaveImage3),'N') ~= 1
    CharFileName3 = char(FileName3);
    PotentialDot3 = CharFileName3(end-3:end-3);
    if strcmp(PotentialDot3,'.') == 1
        BareFileName3 = CharFileName3(1:end-4);
    else BareFileName3 = CharFileName3;
    end
    NewImageName3 = [BareFileName3,SaveImage3,'.',FileFormat];
    A = isspace(SaveImage3);
    if any(A) == 1
        error('Image processing was canceled because you have entered one or more spaces in the box of text to append to the aligned image name in the Align module.  If you do not want to save the image to the hard drive, type "N" into the appropriate box.')
        return
    end
    B = strcmp(upper(CharFileName3), upper(NewImageName3));
    if B == 1
        error('Image processing was canceled because you have not entered text to append to the aligned image name in the Align module.  If you do not want to save the aligned image to the hard drive, type "N" into the appropriate box.')
        return
    end
end 
drawnow

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%

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
    AlignedImage2 = subim(Temp2, sx2, sy2);
    AlignedImage3 = subim(Temp3, -sx2, -sy2);
    %%% 1 was already aligned with 2.
    AlignedImage1 = subim(Temp1, sx2, sy2);
    Results = ['(1 vs 2: X ', num2str(sx), ', Y ', num2str(sy), ...
            ') (2 vs 3: X ', num2str(sx2), ', Y ', num2str(sy2),')'];
else %%% Aligns two input images.
    %%% Aligns 1 and 2.
    [sx, sy] = autoalign(Image1, Image2);
    AlignedImage1 = subim(Image1, sx, sy);
    AlignedImage2 = subim(Image2, -sx, -sy);
    Results = ['(1 vs 2: X ', num2str(sx), ', Y ', num2str(sy),')'];
end

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Note: Everything between the "if" and "end" is not carried out if the 
%%% user has closed
%%% the figure window, so do not do any important calculations here.
%%% Otherwise an error message will be produced if the user has closed the
%%% window but you have attempted to access data that was supposed to be
%%% produced by this part of the code.

%%% Determines the figure number to display in.
fieldname = ['figurealgorithm',CurrentAlgorithm];
ThisAlgFigureNumber = handles.(fieldname);
%%% Check whether that figure is open. This checks all the figure handles
%%% for one whose handle is equal to the figure number for this algorithm.
%%% Also, check whether any of these images 
if any(findobj == ThisAlgFigureNumber) == 1;
    
    %%% START CALCULATE IMAGES FOR DISPLAY ONLY %%%
    
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
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,1,1); imagesc(OriginalRGB);
    title(['Input Images, Image Set # ',num2str(handles.setbeinganalyzed)]);
    %%% A subplot of the figure window is set to display the adjusted
    %%%  image.
    subplot(2,1,2); imagesc(AlignedRGB); title('Aligned Images');
    displaytexthandle = uicontrol(ThisAlgFigureNumber,'style','text', 'position', [0 0 235 30],'fontname','fixedwidth','backgroundcolor',[0.7,0.7,0.7]);
    set(displaytexthandle,'string',Results)
    set(ThisAlgFigureNumber,'toolbar','figure')
end
%%% Executes pending figure-related commands so that the results are
%%% displayed.
drawnow

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE PROCESSED IMAGE TO HARD DRIVE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Determine whether the user wanted to save the adjusted image
%%% by comparing their entry "SaveImage" with "N" (after
%%% converting SaveImage to uppercase).
if strcmp(upper(SaveImage1),'N') ~= 1
%%% Save the image to the hard drive.    
imwrite(AlignedImage1, NewImageName1, FileFormat);
end
if strcmp(upper(SaveImage2),'N') ~= 1
imwrite(AlignedImage2, NewImageName2, FileFormat);
end
if strcmp(upper(SaveImage3),'N') ~= 1
imwrite(AlignedImage3, NewImageName3, FileFormat);
end
drawnow

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% The adjusted image is saved to the
%%% handles structure so it can be used by subsequent algorithms.
fieldname = ['dOT', AlignedImage1Name];
handles.(fieldname) = AlignedImage1;
fieldname = ['dOT', AlignedImage2Name];
handles.(fieldname) = AlignedImage2;
if strcmp(Image3Name,'/') ~= 1
fieldname = ['dOT', AlignedImage3Name];
handles.(fieldname) = AlignedImage3;
end

%%% The original file name is saved to the handles structure in a
%%% field named after the adjusted image name.
fieldname = ['dOTFilename', AlignedImage1Name];
handles.(fieldname)(handles.setbeinganalyzed) = FileName1;
fieldname = ['dOTFilename', AlignedImage2Name];
handles.(fieldname)(handles.setbeinganalyzed) = FileName2;
if strcmp(Image3Name,'/') ~= 1
fieldname = ['dOTFilename', AlignedImage3Name];
handles.(fieldname)(handles.setbeinganalyzed) = FileName3;
end
%%% Removed for parallel: guidata(gcbo, handles);

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
if (bestx == 0) & (besty == 0),
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
  if (nextx == 0) & (nexty == 0),
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
    if (dx == ldx) | (dy == ldy),
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
if ((N>0) & (min(S(:))>=0))
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
if ((N>0) & (min(S(:))>=0))
   Snz=nonzeros(S);
   H=log2(N)-sum(Snz.*log2(Snz))/N;
else
   H=0;
end

function I = mutualinf(X, Y)
%%% Mutual information of images X and Y
I = entropy(X) + entropy(Y) - entropy2(X,Y);

%%%%%%%%%%%
%%% HELP %%%
%%%%%%%%%%%

%%%%% Help for the Align module: 
%%%%% .
%%%%% The optimal alignment between 2 or 3 incoming images is determined.
%%%%% The images are cropped appropriately according to this alignment, so
%%%%% the final images will be smaller than the originals by a few pixels
%%%%% if alignment is necessary.
%%%%% .
%%%%% Which image is displayed as which color can be changed by going into the module's ".m" file
%%%%% and changing the lines after "FOR DISPLAY PURPOSES ONLY".  The first
%%%%% line in each set is red, then green, then blue.