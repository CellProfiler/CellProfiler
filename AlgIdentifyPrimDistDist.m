function handles = AlgIdentifyPrimDistDist8(handles)
%%% This image analysis module identifies objects by finding peaks in
%%% the distance transform of a thresholded image.  Once a marker
%%% for each object has been identified in this way, a watershed function
%%% identifies the lines between objects that are touching each other by
%%% looking for the dimmest points between them.  To identify the edges of
%%% non-clumped objects, a simple threshold is applied.
%%% The algorithm works best for objects that are round-shaped.  The cells
%%% need not be brighter towards the interior as is required for the
%%% intensity-based algorithm.

%%% Reads the current algorithm number, since this is needed to find 
%%% the variable values that the user entered.
CurrentAlgorithm = handles.currentalgorithm;

%%% The "drawnow" function allows figure windows to be updated and buttons
%%% to be pushed (like the pause, cancel, help, and view buttons).  The
%%% "drawnow" function is sprinkled throughout the algorithm so there are
%%% plenty of breaks where the figure windows/buttons can be interacted
%%% with.
drawnow 

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%

%textVAR01 = What did you call the images you want to process? 
%defaultVAR01 = OrigBlue
fieldname = ['Vvariable',CurrentAlgorithm,'_01'];
ImageName = handles.(fieldname);
%textVAR02 = What do you want to call the objects segmented by this algorithm?
%defaultVAR02 = Nuclei
fieldname = ['Vvariable',CurrentAlgorithm,'_02'];
ObjectName = handles.(fieldname);
%textVAR03 = Size range (in pixels) of objects to include (1,99999 = do not discard any)
%defaultVAR03 = 1,99999
fieldname = ['Vvariable',CurrentAlgorithm,'_03'];
SizeRange = handles.(fieldname);
%textVAR04 = Enter the threshold [0 = automatically calculate] (Positive number, Max = 1):
%defaultVAR04 = 0
fieldname = ['Vvariable',CurrentAlgorithm,'_04'];
Threshold = str2num(handles.(fieldname));
%textVAR05 = If auto threshold, enter an adjustment factor (Positive number, 1 = no adjustment):
%defaultVAR05 = 1
fieldname = ['Vvariable',CurrentAlgorithm,'_05'];
ThresholdAdjustmentFactor = str2num(handles.(fieldname));
%textVAR06 = Enter the Max Suppress N'hood (Non-negative integer ~ the radius of objects)
%defaultVAR06 = 10
fieldname = ['Vvariable',CurrentAlgorithm,'_06'];
MaximaSuppressionNeighborhood = str2num(handles.(fieldname));

%textVAR08 = To save object outlines as an image, enter text to append to the name 
%defaultVAR08 = N
fieldname = ['Vvariable',CurrentAlgorithm,'_08'];
SaveObjectOutlines = handles.(fieldname);
%textVAR09 = To save colored object blocks as an image, enter text to append to the name 
%defaultVAR09 = N
fieldname = ['Vvariable',CurrentAlgorithm,'_09'];
SaveColoredObjects = handles.(fieldname);
%textVAR10 = Otherwise, leave as "N". To save or display other images, press Help button
%textVAR11 = If saving images, what file format do you want to use? Do not include a period.
%defaultVAR11 = tif
fieldname = ['Vvariable',CurrentAlgorithm,'_11'];
FileFormat = handles.(fieldname);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Determines what the user entered for the size range.
SizeRangeNumerical = str2num(SizeRange);
MinSize = SizeRangeNumerical(1);
MaxSize = SizeRangeNumerical(2);

%%% Checks whether the file format the user entered is readable by Matlab.
IsFormat = imformats(FileFormat);
if isempty(IsFormat) == 1
    error('The image file type entered in the Segment Intensity module is not recognized by Matlab. Or, you may have entered a period in the box. For a list of recognizable image file formats, type "imformats" (no quotes) at the command line in Matlab.','Error')
end
%%% Read (open) the image you want to analyze and assign it to a variable,
%%% "OrigImageToBeAnalyzed".
fieldname = ['dOT', ImageName];
%%% Checks whether the image exists in the handles structure.
    if isfield(handles, fieldname) == 0
    error(['Image processing has been canceled. Prior to running the Segment Intensity module, you must have previously run an algorithm to load an image. You specified in the Segment Intensity module that this image was called ', ImageName, ' which should have produced a field in the handles structure called ', fieldname, '. The Segment Intensity module cannot find this image.']);
    end
OrigImageToBeAnalyzed = handles.(fieldname);
%%% Update the handles structure.
%%% Removed for parallel: guidata(gcbo, handles);

%%% Check whether the appendages to be added to the file names of images
%%% will result in overwriting the original file, or in a file name that
%%% contains spaces.
        
%%% Determine the filename of the image to be analyzed.
fieldname = ['dOTFilename', ImageName];
FileName = handles.(fieldname)(handles.setbeinganalyzed);
%%% Find and remove the file format extension within the original file
%%% name, but only if it is at the end. Strip the original file format extension 
%%% off of the file name, if it is present, otherwise, leave the original
%%% name intact.
CharFileName = char(FileName);
PotentialDot = CharFileName(end-3:end-3);
if strcmp(PotentialDot,'.') == 1
    BareFileName = CharFileName(1:end-4);
else BareFileName = CharFileName;
end

%%% Assemble the new image name.
NewImageNameSaveObjectOutlines = [BareFileName,SaveObjectOutlines,'.',FileFormat];
%%% Check whether the new image name is going to result in a name with
%%% spaces.
A = isspace(SaveObjectOutlines);
if any(A) == 1
    error('Image processing was canceled because you have entered one or more spaces in the box of text to append to the object outlines image name in the Segment Intensity module.  If you do not want to save the object outlines image to the hard drive, type "N" into the appropriate box.')
end
%%% Check whether the new image name is going to result in overwriting the
%%% original file.
B = strcmp(upper(CharFileName), upper(NewImageNameSaveObjectOutlines));
if B == 1
    error('Image processing was canceled because you have not entered text to append to the object outlines image name in the Segment Intensity module.  If you do not want to save the object outlines image to the hard drive, type "N" into the appropriate box.')
end

%%% Repeat the above for the other image to be saved: 
NewImageNameSaveColoredObjects = [BareFileName,SaveColoredObjects,'.',FileFormat];
A = isspace(SaveColoredObjects);
if any(A) == 1
    error('Image processing was canceled because you have entered one or more spaces in the box of text to append to the colored objects image name in the Segment Intensity module.  If you do not want to save the colored objects image to the hard drive, type "N" into the appropriate box.')
end
B = strcmp(upper(CharFileName), upper(NewImageNameSaveColoredObjects));
if B == 1
    error('Image processing was canceled because you have not entered text to append to the colored objects image name in the Segment Intensity module.  If you do not want to save the colored objects image to the hard drive, type "N" into the appropriate box.')
end

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%

%%% Checks that the original image is two-dimensional (i.e. not a color
%%% image), which would disrupt several of the image functions.
if ndims(OrigImageToBeAnalyzed) ~= 2
    error('Image processing was canceled because the Segment Distance module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
end

%%% STEP 1: Finds markers for each object based on maxima in the distance
%%% transform.

%%% Determines the threshold to be used, if the user has left the Threshold
%%% variable set to 0.
if Threshold == 0
    Threshold = graythresh(OrigImageToBeAnalyzed);
    Threshold = Threshold*ThresholdAdjustmentFactor;
end
%%% Thresholds the image.
ThresholdedOrigImage = im2bw(OrigImageToBeAnalyzed, Threshold);
    % figure, imshow(ThresholdedOrigImage, []), title('ThresholdedOrigImage')
    % imwrite(ThresholdedOrigImage, [BareFileName,'TOI','.',FileFormat], FileFormat);
%%% Fills holes in the thresholded image so that stray dim pixels within the
%%% objects do not count as edges of the objects.
FilledThresholdedImage = imfill(ThresholdedOrigImage, 'holes');
    % figure, imshow(FilledThresholdedImage, []), title('FilledThresholdedImage')
    % imwrite(FilledThresholdedImage, [BareFileName,'FTI','.',FileFormat], FileFormat);
%%% Computes distance transform.
DistanceTransformedImage = bwdist(~FilledThresholdedImage);
    % figure, imagesc(DistanceTransformedImage), title('DistanceTransformedImage')
%%% Essentially thresholds again to get rid of background pixels.
% DistanceTransformedImage(~FilledThresholdedImage) = sum(size(OrigImageToBeAnalyzed)); 
    % figure, imagesc(DistanceTransformedImage), title('DistanceTransformedImage')
    % imwrite(DistanceTransformedImage, [BareFileName,'DTI','.',FileFormat], FileFormat);

%%% Perturb the distance image so that local maxima near each other with
%%% identical values will now have slightly different values.

%%% Save off the random number generator's state, and set the state to
%%% a particular value (for repeatability)
oldstate = rand('state');
rand('state',0);
%%% Add a random value between 0 and 0.05 to each pixel in the DistanceTransformedImage
DistanceTransformedImage = DistanceTransformedImage + 0.05*rand(size(DistanceTransformedImage));
%%% Restore the random number generator's state
rand('state',oldstate);

%%% Extracts local maxima and filters them by eliminating maxima that are
%%% within a certain distance of each other.
MaximaImage = OrigImageToBeAnalyzed;
MaximaImage(~FilledThresholdedImage) = 0;
     % figure, imshow(MaximaImage, []), title('MaximaImage')
MaximaMask = strel('disk', MaximaSuppressionNeighborhood);
MaximaImage(DistanceTransformedImage < ordfilt2(DistanceTransformedImage,sum(sum(getnhood(MaximaMask))),getnhood(MaximaMask))) = 0;
     % figure, imshow(MaximaImage, []), title('MaximaImage2')
     % imwrite(MaximaImage, [BareFileName,'MI','.',FileFormat], FileFormat);
%%% Converts the maxima image to black and white.
    MaximaImage(MaximaImage ~= 0) = 1;
     % figure, imshow(MaximaImage, []), title('MaximaImage3')
     
%%% STEP 2: Performs watershed function on the DistanceTransformedImage.
drawnow 
%%% Inverts image.
InvertedDistanceTransformedImage = -DistanceTransformedImage;
        % figure, imshow(InvertedOriginal, []), title('InvertedOriginal')
        % imwrite(InvertedOriginal, [BareFileName,'IO','.',FileFormat], FileFormat);
%%% Overlays the nuclear markers (maxima) on the inverted DistanceTransformedImage so
%%% there are black dots on top of each dark nucleus on a white background.
Overlaid = imimposemin(InvertedDistanceTransformedImage,MaximaImage);
        % figure, imagesc(Overlaid), title('Overlaid')
        % imwrite(Overlaid, [BareFileName,'O','.',FileFormat], FileFormat);
%%% Identifies watershed lines.
BlackWatershedLinesPre = watershed(Overlaid);
        % figure, imshow(BlackWatershedLinesPre, []), title('BlackWatershedLinesPre')
        % imwrite(BlackWatershedLinesPre, [BareFileName,'BWLP','.',FileFormat], FileFormat);
%%% Superimposes watershed lines as white (255) onto the inverted original
%%% image.
InvertedOrigImage = imcomplement(OrigImageToBeAnalyzed);
WhiteWatershedOnInvertedOrig = InvertedOrigImage;
WhiteWatershedOnInvertedOrig(BlackWatershedLinesPre == 0) = 255;
        % figure, imshow(WhiteWatershedOnInvertedOrig, []), title('WhiteWatershedOnInvertedOrig')
        % imwrite(WhiteWatershedOnInvertedOrig, [BareFileName,'WWOIO','.',FileFormat], FileFormat);

%%% STEP 3: Identifies and extracts the objects, using the watershed lines.
drawnow 

%%% Thresholds the WhiteWatershedOnInvertedOrig image, using the same
%%% threshold as used for the maxima detection, except the number is inverted
%%% since we are working with an inverted image now.
InvertedThreshold = 1 - Threshold;
BinaryObjectsImage = im2bw(WhiteWatershedOnInvertedOrig,InvertedThreshold);
        % figure, imshow(BinaryObjectsImage, []), title('BinaryObjectsImage')
        % imwrite(BinaryObjectsImage, [BareFileName,'BOI','.',FileFormat], FileFormat);
%%% Inverts the BinaryObjectsImage.
InvertedBinaryImage = imcomplement(BinaryObjectsImage);
        % figure, imshow(InvertedBinaryImage, []), title('InvertedBinaryImage')
        % imwrite(InvertedBinaryImage, [BareFileName,'IBI','.',FileFormat], FileFormat);
%%% Fills holes, then identifies objects in the binary image.
PrelimLabelMatrixImage1 = bwlabel(imfill(InvertedBinaryImage,'holes'));
    % figure, imshow(PrelimLabelMatrixImage1, []), title('PrelimLabelMatrixImage1')
    % imwrite(PrelimLabelMatrixImage1, [BareFileName,'PLMI1','.',FileFormat], FileFormat);
%%% Finds objects larger and smaller than the user-specified size.
%%% Finds the locations and labels for the pixels that are part of an object.
AreaLocations = find(PrelimLabelMatrixImage1);
AreaLabels = PrelimLabelMatrixImage1(AreaLocations);
%%% Creates a sparse matrix with column as label and row as location,
%%% with a 1 at (A,B) if location A has label B.  Summing the columns
%%% gives the count of area pixels with a given label.  E.g. Areas(L) is the
%%% number of pixels with label L.
Areas = full(sum(sparse(AreaLocations, AreaLabels, 1)));
Map = [0,Areas];
AreasImage = Map(PrelimLabelMatrixImage1 + 1);
    % figure, imshow(AreasImage, []), title('AreasImage')
    % imwrite(AreasImage, [BareFileName,'AI','.',FileFormat], FileFormat);
%%% The small objects are overwritten with zeros.
PrelimLabelMatrixImage2 = PrelimLabelMatrixImage1;
PrelimLabelMatrixImage2(AreasImage < MinSize) = 0;
%%% Relabels so that labels are consecutive. This is important for
%%% downstream modules (IdentifySec).
PrelimLabelMatrixImage2 = bwlabel(im2bw(PrelimLabelMatrixImage2,.1));
%%% The large objects are overwritten with zeros.
PrelimLabelMatrixImage3 = PrelimLabelMatrixImage2;
if MaxSize ~= 99999
    PrelimLabelMatrixImage3(AreasImage > MaxSize) = 0;
        % figure, imshow(PrelimLabelMatrixImage3, []), title('PrelimLabelMatrixImage3')
        % imwrite(PrelimLabelMatrixImage3, [BareFileName,'PLMI3','.',FileFormat], FileFormat);
end
%%% Removes objects that are touching the edge of the image, since they
%%% won't be measured properly.
PrelimLabelMatrixImage4 = imclearborder(PrelimLabelMatrixImage3,8);
    % figure, imshow(PrelimLabelMatrixImage4, []), title('PrelimLabelMatrixImage4')
    % imwrite(PrelimLabelMatrixImage4, [BareFileName,'PLMI4','.',FileFormat], FileFormat);
%%% The PrelimLabelMatrixImage4 is converted to binary.
FinalBinaryPre = im2bw(PrelimLabelMatrixImage4,1);
% figure, imshow(FinalBinaryPre, []), title('FinalBinaryPre')
% imwrite(FinalBinaryPre, [BareFileName,'FBP','.',FileFormat], FileFormat);
%%% Holes in the FinalBinaryPre image are filled in.
FinalBinary = imfill(FinalBinaryPre, 'holes');
%%% The image is converted to label matrix format. Even if the above step
%%% is excluded (filling holes), it is still necessary to do this in order
%%% to "compact" the label matrix: this way, each number corresponds to an
%%% object, with no numbers skipped.
FinalLabelMatrixImage = bwlabel(FinalBinary);
% figure, imshow(FinalLabelMatrixImage, []), title('FinalLabelMatrixImage')
% imwrite(FinalLabelMatrixImage, [BareFileName,'FLMInuc','.',FileFormat], FileFormat);
drawnow 

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

%%% Calculates the ColoredLabelMatrixImage for displaying in the figure
%%% window in subplot(2,2,2).  
%%% Note that the label2rgb function doesn't work when there are no objects
%%% in the label matrix image, so there is an "if".
if sum(sum(FinalLabelMatrixImage)) >= 1
    ColoredLabelMatrixImage = label2rgb(FinalLabelMatrixImage, 'jet', 'k', 'shuffle');
    % figure, imshow(ColoredLabelMatrixImage, []), title('ColoredLabelMatrixImage')
    % imwrite(ColoredLabelMatrixImage, [BareFileName,'CLMI','.',FileFormat], FileFormat);
else  ColoredLabelMatrixImage = FinalLabelMatrixImage;
end

%%% Calculates the object outlines, which are overlaid on the original
%%% image and displayed in figure subplot (2,2,4).
%%% Creates the structuring element that will be used for dilation.
StructuringElement = strel('square',3);
%%% Converts the FinalLabelMatrixImage to binary.
FinalBinaryImage = im2bw(FinalLabelMatrixImage,1);
%%% Dilates the FinalBinaryImage by one pixel (8 neighborhood).
DilatedBinaryImage = imdilate(FinalBinaryImage, StructuringElement);
        % figure, imshow(DilatedBinaryImage, []), title('DilatedBinaryImage')
        % imwrite(DilatedBinaryImage, [BareFileName,'DBI','.',FileFormat], FileFormat);
%%% Subtracts the FinalBinaryImage from the DilatedBinaryImage,
%%% which leaves the PrimaryObjectOutlines.
PrimaryObjectOutlines = DilatedBinaryImage - FinalBinaryImage;
        % figure, imshow(PrimaryObjectOutlines, []), title('PrimaryObjectOutlines')
        % imwrite(PrimaryObjectOutlines, [BareFileName,'POO','.',FileFormat], FileFormat);
%%% Overlays the object outlines on the original image.
ObjectOutlinesOnOriginalImage = OrigImageToBeAnalyzed;
    %%% Determines the grayscale intensity to use for the cell outlines.
    LineIntensity = max(OrigImageToBeAnalyzed(:));
ObjectOutlinesOnOriginalImage(PrimaryObjectOutlines == 1) = LineIntensity;
        % figure, imshow(ObjectOutlinesOnOriginalImage, []), title('ObjectOutlinesOnOriginalImage')
        % imwrite(ObjectOutlinesOnOriginalImage, [BareFileName,'OOOOI','.',FileFormat], FileFormat);

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
    figure(ThisAlgFigureNumber);
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,2,1); imagesc(OrigImageToBeAnalyzed);colormap(gray);
    title(['Input Image, Image Set # ',num2str(handles.setbeinganalyzed)]);
    %%% A subplot of the figure window is set to display the colored label
    %%% matrix image.
    subplot(2,2,2); imagesc(ColoredLabelMatrixImage); title(['Segmented ',ObjectName]);
    %%% A subplot of the figure window is set to display the Overlaid image,
    %%% where the maxima are imposed on the inverted original image
    subplot(2,2,3); imagesc(Overlaid); colormap(gray); title([ObjectName, ' markers']);
    %%% A subplot of the figure window is set to display the inverted original
    %%% image with watershed lines drawn to divide up clusters of objects.
    subplot(2,2,4); imagesc(ObjectOutlinesOnOriginalImage);colormap(gray); title([ObjectName, ' Outlines on Input Image']);
end
%%% Executes pending figure-related commands so that the results are
%%% displayed.
drawnow

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Saves the segmented image, not edited for objects along the edges or
%%% for size, to the handles structure.
fieldname = ['dOTPrelimSegmented',ObjectName];
handles.(fieldname) = PrelimLabelMatrixImage1;

%%% Saves the segmented image, only edited for small objects, to the
%%% handles structure.
fieldname = ['dOTPrelimSmallSegmented',ObjectName];
handles.(fieldname) = PrelimLabelMatrixImage2;

%%% Saves the final segmented label matrix image to the handles structure.
fieldname = ['dOTSegmented',ObjectName];
handles.(fieldname) = FinalLabelMatrixImage;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE PROCESSED IMAGE TO HARD DRIVE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Determine whether the user wanted to save the image of object outlines
%%% by comparing their entry "SaveObjectOutlines" with "N" (after
%%% converting SaveObjectOutlines to uppercase).  The appropriate names
%%% were determined towards the beginning of the module during error
%%% checking.
if strcmp(upper(SaveObjectOutlines),'N') ~= 1
%%% Save the image to the hard drive.    
imwrite(PrimaryObjectOutlines, NewImageNameSaveObjectOutlines, FileFormat);
end
%%% Same for the SaveColoredObjects image.
if strcmp(upper(SaveColoredObjects),'N') ~= 1
%%% Save the image to the hard drive.    
imwrite(ColoredLabelMatrixImage, NewImageNameSaveColoredObjects, FileFormat);
end

drawnow 

%%%%%%%%%%%%%%
%%%%%% HELP %%%%%
%%%%%%%%%%%%%%

%%%%% Help for Segment Distance module: 
%%%%% .
%%%%% 
%%%%% .
%%%%% DISPLAYING AND SAVING PROCESSED IMAGES 
%%%%% PRODUCED BY THIS IMAGE ANALYSIS MODULE:
%%%%% Note: Images saved using the boxes in the main CellProfiler window
%%%%% will be saved in the default directory specified in STEP 1.
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
% The Original Code is the ______________________.
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
