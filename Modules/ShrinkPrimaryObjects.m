function handles = AlgShrinkPrimaryObjects(handles)

% The contents of this file are subject to the Mozilla Public License Version 
% 1.1 (the "License"); you may not use this file except in compliance with 
% the License. You may obtain a copy of the License at 
% http://www.mozilla.org/MPL/
% 
% Software distributed under the License is distributed on an "AS IS" basis,
% WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
% for the specific language governing rights and limitations under the
% License.
% 
% 
% The Original Code is the Shrink Primary Objects Module.
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

%%% Reads the current algorithm number, since this is needed to find 
%%% the variable values that the user entered.
CurrentAlgorithm = handles.currentalgorithm;

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow

%textVAR01 = What did you call the objects that you want to shrink?
%defaultVAR01 = Nuclei
fieldname = ['Vvariable',CurrentAlgorithm,'_01'];
ObjectName = handles.(fieldname);
%textVAR02 = What do you want to call the shrunken objects?
%defaultVAR02 = ShrunkenNuclei
fieldname = ['Vvariable',CurrentAlgorithm,'_02'];
ShrunkenObjectName = handles.(fieldname);
%textVAR04 = How much do you want to shrink the objects? (Positive number, or "Inf" to shrink to a point)
%defaultVAR04 = 1
fieldname = ['Vvariable',CurrentAlgorithm,'_04'];
ShrinkingNumber = handles.(fieldname);
%textVAR08 = To save the shrunken image as colored blocks, enter text to append to the image name 
%defaultVAR08 = N
fieldname = ['Vvariable',CurrentAlgorithm,'_08'];
SaveImage = handles.(fieldname);
%textVAR09 =  Otherwise, leave as "N". To save or display other images, press Help button
%textVAR10 = In what file format do you want to save images? Do not include a period
%defaultVAR10 = tif
fieldname = ['Vvariable',CurrentAlgorithm,'_10'];
FileFormat = handles.(fieldname);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow
%%% Checks whether the file format the user entered is readable by Matlab.
IsFormat = imformats(FileFormat);
if isempty(IsFormat) == 1
    error('The image file type entered in the Shrink Primary Objects module is not recognized by Matlab. Or, you may have entered a period in the box. For a list of recognizable image file formats, type "imformats" (no quotes) at the command line in Matlab.','Error')
end

%%% Retrieves the segmented image, not edited for objects along the edges or
%%% for size.
fieldname = ['dOTPrelimSegmented',ObjectName];
%%% Check whether the image to be analyzed exists in the handles structure.
if isfield(handles, fieldname) == 0
    error(['Image processing was canceled because the Shrink Primary Objects module could not find the input image.  It was supposed to be produced by an Identify Primary module in which the objects were named ', ObjectName, '.  Perhaps there is a typo in the name.'])
end
PrelimSegmentedImage = handles.(fieldname);

%%% Retrieves the segmented image, only edited for small objects.
fieldname = ['dOTPrelimSmallSegmented',ObjectName];
%%% Check whether the image to be analyzed exists in the handles structure.
if isfield(handles, fieldname) == 0
    error(['Image processing was canceled because the Shrink Primary Objects module could not find the input image.  It was supposed to be produced by an Identify Primary module in which the objects were named ', ObjectName, '.  Perhaps there is a typo in the name.'])
end
PrelimSmallSegmentedImage = handles.(fieldname);

%%% Retrieves the final segmented label matrix image.
fieldname = ['dOTSegmented',ObjectName];
%%% Check whether the image to be analyzed exists in the handles structure.
if isfield(handles, fieldname) == 0
    error(['Image processing was canceled because the Shrink Primary Objects module could not find the input image.  It was supposed to be produced by an Identify Primary module in which the objects were named ', ObjectName, '.  Perhaps there is a typo in the name.'])
end
SegmentedImage = handles.(fieldname);

%%% Assemble the new image name.
NewImageName = [ObjectName,num2str(handles.setbeinganalyzed),SaveImage,'.',FileFormat];
%%% Check whether the new image name is going to result in a name with
%%% spaces.
A = isspace(SaveImage);
if any(A) == 1
    error('Image processing was canceled because you have entered one or more spaces in the box of text to append to the object outlines image name in the Shrink Primary Objects module.  If you do not want to save the object outlines image to the hard drive, type "N" into the appropriate box.')
    return
end

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%

%%% Shrinks the three incoming images.  The "thin" option nicely removes
%%% one pixel border from objects with each iteration.  When carried out
%%% for an infinite number of iterations, however, it produces one-pixel
%%% width objects (points, lines, or branched lines) rather than a single
%%% pixel.  The "shrink" option uses a peculiar algorithm to remove border
%%% pixels that does not result in nice uniform shrinking of objects, but
%%% it does have the capability, when used with an infinite number of
%%% iterations, to reduce objects to a single point (one pixel).
%%% Therefore, if the user wants a single pixel for each object, the
%%% "shrink" option is used; otherwise, the "thin" option is used.
if strcmp(upper(ShrinkingNumber),'INF') == 1
    ShrunkenPrelimSegmentedImage = bwmorph(PrelimSegmentedImage, 'shrink', Inf);
    ShrunkenPrelimSmallSegmentedImage = bwmorph(PrelimSmallSegmentedImage, 'shrink', Inf);
    ShrunkenSegmentedImage = bwmorph(SegmentedImage, 'shrink', Inf);
else 
    try ShrinkingNumber = str2num(ShrinkingNumber);
        ShrunkenPrelimSegmentedImage = bwmorph(PrelimSegmentedImage, 'thin', ShrinkingNumber);
        ShrunkenPrelimSmallSegmentedImage = bwmorph(PrelimSmallSegmentedImage, 'thin', ShrinkingNumber);
        ShrunkenSegmentedImage = bwmorph(SegmentedImage, 'thin', ShrinkingNumber);
    catch error('Image processing was canceled because the value entered in the Shrink Primary Objects module must either be a number or the text "Inf" (no quotes).')
    end
end

%%% For the ShrunkenSegmentedImage, the objects are relabeled so that their
%%% numbers correspond to the numbers used for nuclei.  This is important
%%% so that if the user has made measurements on the non-shrunk objects,
%%% the order of these objects will be exactly the same as the shrunk
%%% objects, which may go on to be used to identify secondary objects.
FinalShrunkenPrelimSegmentedImage = ShrunkenSegmentedImage.*PrelimSegmentedImage;
FinalShrunkenPrelimSmallSegmentedImage = ShrunkenSegmentedImage.*PrelimSmallSegmentedImage;
FinalShrunkenSegmentedImage = ShrunkenSegmentedImage.*SegmentedImage;

%%% THE FOLLOWING CALCULATIONS ARE FOR DISPLAY PURPOSES ONLY: The resulting
%%% images are shown in the figure window (if open), or saved to the hard
%%% drive (if desired).  To speed execution, these lines can be removed (or
%%% have a % sign placed in front of them) as long as all the lines which
%%% depend on the resulting images are also removed (e.g. in the figure
%%% window display section).  Alternately, all of this code can be moved to
%%% within the if loop in the figure window display section and then after
%%% starting image analysis the figure window can be closed.  Just remember
%%% that when the figure window is closed, nothing within the if loop is
%%% carried out, so you would not be able to use the imwrite lines below to
%%% save images to the hard drive, for example.

%%% Calculates the OriginalColoredLabelMatrixImage for displaying in the figure
%%% window in subplot(2,1,1).
%%% Note that the label2rgb function doesn't work when there are no objects
%%% in the label matrix image, so there is an "if".
if sum(sum(SegmentedImage)) >= 1
    OriginalColoredLabelMatrixImage = label2rgb(SegmentedImage,'jet', 'k', 'shuffle');
        % figure, imshow(SegmentedImage, []), title('SegmentedImage')
        % figure, imshow(OriginalColoredLabelMatrixImage, []), title('OriginalColoredLabelMatrixImage')
        % imwrite(OriginalColoredLabelMatrixImage, [BareFileName,'OCLMI','.',FileFormat], FileFormat);
else  OriginalColoredLabelMatrixImage = SegmentedImage;
end

%%% Calculates the ShrunkenColoredLabelMatrixImage for displaying in the figure
%%% window in subplot(2,1,2).
%%% Note that the label2rgb function doesn't work when there are no objects
%%% in the label matrix image, so there is an "if".
if sum(sum(SegmentedImage)) >= 1
    ShrunkenColoredLabelMatrixImage = label2rgb(FinalShrunkenSegmentedImage,'jet', 'k', 'shuffle');
        % figure, imshow(FinalShrunkenSegmentedImage, []), title('FinalShrunkenSegmentedImage')
        % figure, imshow(ShrunkenColoredLabelMatrixImage, []), title('ShrunkenColoredLabelMatrixImage')
        % imwrite(ShrunkenColoredLabelMatrixImage, [BareFileName,'SCLMI','.',FileFormat], FileFormat);
else  ShrunkenColoredLabelMatrixImage = FinalShrunkenSegmentedImage;
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
if any(findobj == ThisAlgFigureNumber) == 1;
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
    %%% Sets the width of the figure window to be appropriate (half width).
    if handles.setbeinganalyzed == 1
        originalsize = get(ThisAlgFigureNumber, 'position');
        newsize = originalsize;
        newsize(3) = 0.5*originalsize(3);
        set(ThisAlgFigureNumber, 'position', newsize);
    end
    %%% Activates the appropriate figure window.
    figure(ThisAlgFigureNumber);
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,1,1); imagesc(OriginalColoredLabelMatrixImage);colormap(gray);
    title([ObjectName, ' Image Set # ',num2str(handles.setbeinganalyzed)]);
    subplot(2,1,2); imagesc(ShrunkenColoredLabelMatrixImage); title(ShrunkenObjectName);colormap(gray);
%     subplot(2,2,2); imagesc(ShrunkenPrelimSegmentedImage); title('ShrunkenPrelimSegmentedImage');colormap(gray);
%     subplot(2,2,3); imagesc(ShrunkenPrelimSmallSegmentedImage); title('ShrunkenPrelimSmallSegmentedImage');colormap(gray);
end
%%% Executes pending figure-related commands so that the results are
%%% displayed.
drawnow

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Saves the segmented image, not edited for objects along the edges or
%%% for size, to the handles structure.
fieldname = ['dOTPrelimSegmented',ShrunkenObjectName];
handles.(fieldname) = FinalShrunkenPrelimSegmentedImage;

%%% Saves the segmented image, only edited for small objects, to the
%%% handles structure.
fieldname = ['dOTPrelimSmallSegmented',ShrunkenObjectName];
handles.(fieldname) = FinalShrunkenPrelimSmallSegmentedImage;

%%% Saves the final segmented label matrix image to the handles structure.
fieldname = ['dOTSegmented',ShrunkenObjectName];
handles.(fieldname) = FinalShrunkenSegmentedImage;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE PROCESSED IMAGE TO HARD DRIVE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Determine whether the user wanted to save the Thresholded image
%%% by comparing their entry "SaveImage" with "N" (after
%%% converting SaveImage to uppercase).
if strcmp(upper(SaveImage),'N') ~= 1
%%% Save the image to the hard drive.    
imwrite(ShrunkenColoredLabelMatrixImage, NewImageName, FileFormat);
end

drawnow

%%%%%%%%%%%
%%% HELP %%%
%%%%%%%%%%%

%%%%% Help for the Shrink Primary Objects module: 
%%%%% .
%%%%% .
%%%%% DISPLAYING AND SAVING PROCESSED IMAGES 
%%%%% PRODUCED BY THIS IMAGE ANALYSIS MODULE:
%%%%% Note: Images saved using the boxes in the main CellProfiler window
%%%%% will be saved in the default directory specified in STEP 1.
%%%%% .
%%%%% If you want to save other processed images, open the m-file for this 
%%%%% image analysis module, go to the line in the
%%%%% m-file where the image is generated, and there should be 2 lines
%%%%% which have been inactivated.  These are green comment lines that are
%%%%% indented. To display an image, remove the percent sign before
%%%%% the line that says "figure, imshow...". This will cause the image to
%%%%% appear in a fresh display window for every image set. To save an
%%%%% image to the hard drive, remove the percent sign before the line
%%%%% that says "imwrite..." and adjust the file type and appendage to the
%%%%% file name as desired.  When you have finished removing the percent
%%%%% signs, go to File > Save As and save the m file with a new name.
%%%%% Then load the new image analysis module into the CellProfiler as
%%%%% usual.
%%%%% Please note that not all of these imwrite lines have been checked for
%%%%% functionality: it may be that you will have to alter the format of
%%%%% the image before saving.  Try, for example, adding the uint8 command:
%%%%% uint8(Image) surrounding the image prior to using the imwrite command
%%%%% if the image is not saved correctly.
