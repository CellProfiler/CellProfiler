function handles = AlgIdentifySecWatershed(handles)

% Help for the Identify Secondary Watershed module: 
% 
% This module assumes that you have a set of primary objects contained
% within a group of secondary objects, and you want to identify the
% outlines of the secondary objects.  The primary objects are assumed to be
% completely within secondary objects (e.g. nuclei within cells).  The
% module is especially good at determining the dividing lines between
% clustered cells.
%
% SETTINGS:
% Threshold adjustment factor: this setting affects the stringency of
% object outlines that border the background.  It does not affect the
% dividing lines between clumped objects. A higher number will result
% in smaller objects (more stringent). A lower number will result in
% large objects (less stringent), but at a certain point, depending on
% the particular image, the objects will become huge and the processing
% will take a really long time.  To determine whether the number is too
% low, you can just test it (of course), but a shortcut would be to go
% into the m-file for this module, take the percent sign off the
% beginning of the line that says "figure,
% imshow(InvertedThresholdedOrigImage,...", save the file and load the
% algorithm.  The resulting image that pops up during processing should
% not have lots of speckles - this adds to the processing time.
% Rather, there should be rather large regions of black where the cells
% are located.
% 
% Note: Primary segmenters produce two output images that are used by
% this module.  The dOTSegmented image contains the final, edited
% primary objects (i.e. objects at the border and those that are too
% small or large have been excluded).  The dOTPrelimSmallSegmented
% image is the same except that the objects at the border and the large
% objects have been included.  These extra objects are used to perform
% the identification of secondary object outlines, since they are
% probably real objects (even if we don't want to measure them).  Small
% objects are not used at this stage because they are more likely to be
% artifactual, and so they therefore should not "claim" any secondary
% object pixels.

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
% The Original Code is the Identify Secondary Watershed module.
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

%textVAR01 = What did you call the images you want to process? 
%defaultVAR01 = OrigGreen
ImageName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,1});

%textVAR02 = What did you call the objects that will be used to mark the centers of these objects?
%defaultVAR02 = Nuclei
PrimaryObjectName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,2});

%textVAR03 = What do you want to call the objects identified by this algorithm?
%defaultVAR03 = Cells
SecondaryObjectName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,3});
%textVAR04 = (Note: Data will be produced based on this name, e.g. dMCTotalAreaCells)

%textVAR05 = Enter the threshold [0 = automatically calculate] (Positive number, Max = 1):
%defaultVAR05 = 0
Threshold = str2double(char(handles.Settings.Vvariable{CurrentAlgorithmNum,5}));

%textVAR06 = If auto threshold, enter an adjustment factor (Positive number, 1 = no adjustment):
%defaultVAR06 = 0.75
ThresholdAdjustmentFactor = str2double(char(handles.Settings.Vvariable{CurrentAlgorithmNum,6}));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a variable,
%%% "OrigImage".
fieldname = ['dOT', ImageName];
%%% Checks whether the image to be analyzed exists in the handles structure.
if isfield(handles, fieldname) == 0
    %%% If the image is not there, an error message is produced.  The error
    %%% is not displayed: The error function halts the current function and
    %%% returns control to the calling function (the analyze all images
    %%% button callback.)  That callback recognizes that an error was
    %%% produced because of its try/catch loop and breaks out of the image
    %%% analysis loop without attempting further modules.
    error(['Image processing was canceled because the Identify Secondary Watershed module could not find the input image.  It was supposed to be named ', ImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
OrigImageToBeAnalyzed = handles.(fieldname);

%%% Checks that the original image is two-dimensional (i.e. not a color
%%% image), which would disrupt several of the image functions.
if ndims(OrigImageToBeAnalyzed) ~= 2
    error('Image processing was canceled because the Identify Secondary Watershed module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
end

%%% Retrieves the preliminary label matrix image that contains the primary
%%% segmented objects which have only been edited to discard objects 
%%% that are smaller than a certain size.  This image
%%% will be used as markers to segment the secondary objects with this
%%% algorithm.  Checks first to see whether the appropriate image exists.
fieldname = ['dOTPrelimSmallSegmented',PrimaryObjectName];
%%% Checks whether the image exists in the handles structure.
    if isfield(handles, fieldname) == 0
    error(['Image processing has been canceled. Prior to running the Identify Secondary Watershed module, you must have previously run an algorithm that generates an image with the preliminary primary objects identified.  You specified in the Identify Secondary Watershed module that the primary objects were named ', PrimaryObjectName, ' as a result of the previous algorithm, which should have produced an image called ', fieldname, ' in the handles structure.  The Identify Secondary Watershed module cannot locate this image.']);
    end
PrelimPrimaryLabelMatrixImage = handles.(fieldname);

%%% Retrieves the label matrix image that contains the edited primary
%%% segmented objects which will be used to weed out which objects are
%%% real - not on the edges and not below or above the specified size
%%% limits. Checks first to see whether the appropriate image exists.
fieldname = ['dOTSegmented',PrimaryObjectName];
%%% Checks whether the image exists in the handles structure.
    if isfield(handles, fieldname) == 0
    error(['Image processing has been canceled. Prior to running the Identify Secondary Watershed module, you must have previously run an algorithm that generates an image with the preliminary primary objects identified.  You specified in the Identify Secondary Watershed module that the primary objects were named ', PrimaryObjectName, ' as a result of the previous algorithm, which should have produced an image called ', fieldname, ' in the handles structure.  The Identify Secondary Watershed module cannot locate this image.']);
    end
EditedPrimaryLabelMatrixImage = handles.(fieldname);

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% In order to use the watershed transform to find dividing lines between
%%% the secondary objects, it is necessary to identify the foreground
%%% objects and to identify a portion of the background.  The foreground
%%% objects are retrieved as the binary image of primary objects from the
%%% previously run image analysis module.   This forces the secondary
%%% object's outline to extend at least as far as the edge of the primary
%%% objects.

%%% STEP 1: Marks at least some of the background by applying a 
%%% weak threshold to the original image of the secondary objects.
drawnow
%%% Determines the threshold to use. 
if Threshold == 0
    Threshold = graythresh(OrigImageToBeAnalyzed);
    %%% Adjusts the threshold by a correction factor.  
    Threshold = Threshold*ThresholdAdjustmentFactor;
end

%%% Thresholds the original image.
ThresholdedOrigImage = im2bw(OrigImageToBeAnalyzed, Threshold);
%%% Inverts the image.
InvertedThresholdedOrigImage = imcomplement(ThresholdedOrigImage);

%%% NOTE: There are two other ways to mark the background prior to
%%% watershedding; I think the method used above is best, but I have
%%% included the ideas for two alternate methods. 
%%% METHOD (2): Threshold the original image (or a smoothed image)
%%% so that background pixels are black.  This is overly strong, so instead
%%% of weakly thresholding the image as is done in METHOD (1),  you can then "thin"
%%% the background pixels by computing the SKIZ 
%%% (skeleton of influence zones), which is done by watershedding the
%%% distance transform of the thresholded image.  These watershed lines are
%%% then superimposed on the marked image that will be watershedded to
%%% segment the objects.  I think this would not produce results different
%%% from METHOD 1 (the one used above), since METHOD 1 overlays the
%%% outlines of the primary objects anyway.
%%% This method is based on the Mathworks Image Processing Toolbox demo
%%% "Marker-Controlled Watershed Segmentation".  I found it online; I don't
%%% think it is in the Matlab Demos that are found through help.  It uses
%%% an image of a box of oranges.
%%%
%%% METHOD (3):  (I think this method does not work well for clustered
%%% objects.)  The distance transformed image containing the marked objects
%%% is watershedded, which produces lines midway between the marked
%%% objects.  These lines are superimposed on the marked image that will be
%%% watershedded to segment the objects. But if marked objects are
%%% clustered and not a uniform distance from each other, this will produce
%%% background lines on top of actual objects. 
%%% This method is based on Gonzalez, et al. Digital Image Processing using
%%% Matlab, page 422-425.

%%% STEP 2: Identify the outlines of each primary object, so that each
%%% primary object can be definitely separated from the background.  This
%%% solves the problem of some primary objects running
%%% right up against the background pixels and therefore getting skipped.
%%% Note: it is less accurate and less fast to use edge detection (sobel)
%%% to identify the edges of the primary objects.
drawnow
%%% Converts the PrelimPrimaryLabelMatrixImage to binary.
PrelimPrimaryBinaryImage = im2bw(PrelimPrimaryLabelMatrixImage, 1);
%%% Creates the structuring element that will be used for dilation.
StructuringElement = strel('square',3);
%%% Dilates the Primary Binary Image by one pixel (8 neighborhood).
DilatedPrimaryBinaryImage = imdilate(PrelimPrimaryBinaryImage, StructuringElement);
%%% Subtracts the PrelimPrimaryBinaryImage from the DilatedPrimaryBinaryImage,
%%% which leaves the PrimaryObjectOutlines.
PrimaryObjectOutlines = DilatedPrimaryBinaryImage - PrelimPrimaryBinaryImage;

%%% STEP 3: Produce the marker image which will be used for the first
%%% watershed.
drawnow
%%% Combines the foreground markers and the background markers.
BinaryMarkerImagePre = PrelimPrimaryBinaryImage | InvertedThresholdedOrigImage;
%%% Overlays the PrimaryObjectOutlines to maintain distinctions between each
%%% primary object and the background.
BinaryMarkerImage = BinaryMarkerImagePre;
BinaryMarkerImage(PrimaryObjectOutlines == 1) = 0;
        
%%% STEP 4: Calculate the Sobel image, which reflects gradients, which will
%%% be used for the watershedding function.  
drawnow
%%% Calculates the 2 sobel filters.  The sobel filter is directional, so it
%%% is used in both the horizontal & vertical directions and then the
%%% results are combined.
filter1 = fspecial('sobel');
filter2 = filter1';
%%% Applies each of the sobel filters to the original image.
I1 = imfilter(OrigImageToBeAnalyzed, filter1);
I2 = imfilter(OrigImageToBeAnalyzed, filter2);
%%% Adds the two images.  
%%% The Sobel operator results in negative values, so the absolute values
%%% are calculated to prevent errors in future steps.
AbsSobeledImage = abs(I1) + abs(I2);

%%% STEP 5: Perform the first watershed.
drawnow

%%% Overlays the foreground and background markers onto the
%%% absolute value of the Sobel Image, so there are black nuclei on top of
%%% each dark object, with black background.
Overlaid = imimposemin(AbsSobeledImage, BinaryMarkerImage);
%%% Perform the watershed on the marked absolute-value Sobel Image. 
BlackWatershedLinesPre = watershed(Overlaid);
%%% Bug workaround (see step 9).
BlackWatershedLinesPre2 = im2bw(BlackWatershedLinesPre,1);
BlackWatershedLines = bwlabel(BlackWatershedLinesPre2);

%%% STEP 6: Identify and extract the secondary objects, using the watershed
%%% lines.
drawnow
%%% The BlackWatershedLines image is a label matrix where the watershed
%%% lines = 0 and each distinct object is assigned a number starting at 1.
%%% This image is converted to a binary image where all the objects = 1.
SecondaryObjects1 = im2bw(BlackWatershedLines,1);
%%% Identifies objects in the binary image using bwlabel.
%%% Note: Matlab suggests that in some circumstances bwlabeln is faster
%%% than bwlabel, even for 2D images.  I found that in this case it is
%%% about 10 times slower.
LabelMatrixImage1 = bwlabel(SecondaryObjects1,4);
drawnow

%%% STEP 7: Discarding background "objects".  The first watershed function
%%% simply divides up the image into regions.  Most of these regions
%%% correspond to actual objects, but there are big blocks of background
%%% that are recognized as objects. These can be distinguished from actual
%%% objects because they do not overlap a primary object.

%%% The following changes all the labels in LabelMatrixImage1 to match the
%%% centers they enclose (from PrelimPrimaryBinaryImage), and marks as background 
%%% any labeled regions that don't overlap a center. This function assumes
%%% that every center is entirely contained in one labeled area.  The
%%% results if otherwise may not be well-defined. The non-background labels
%%% will be renumbered according to the center they enclose.

%%% Finds the locations and labels for different regions.
area_locations = find(LabelMatrixImage1);
area_labels = LabelMatrixImage1(area_locations);
%%% Creates a sparse matrix with column as label and row as location,
%%% with the value of the center at (I,J) if location I has label J.
%%% Taking the maximum of this matrix gives the largest valued center
%%% overlapping a particular label.  Tacking on a zero and pushing
%%% labels through the resulting map removes any background regions.
map = [0 full(max(sparse(area_locations, area_labels, PrelimPrimaryBinaryImage(area_locations))))];
ActualObjectsBinaryImage = map(LabelMatrixImage1 + 1);

%%% STEP 8: Produce the marker image which will be used for the second
%%% watershed.
drawnow
%%% The algorithm has now produced a binary image of actual secondary
%%% objects.  The gradient (Sobel) image was used for watershedding, which
%%% produces very nice divisions between objects that are clumped, but it
%%% is too stringent at the edges of objects that are isolated, and at the
%%% edges of clumps of objects. Therefore, the stringently identified
%%% secondary objects are used as markers for a second round of
%%% watershedding, this time based on the original (intensity) image rather
%%% than the gradient image.

%%% Creates the structuring element that will be used for dilation.
StructuringElement = strel('square',3);
%%% Dilates the Primary Binary Image by one pixel (8 neighborhood).
DilatedActualObjectsBinaryImage = imdilate(ActualObjectsBinaryImage, StructuringElement);
%%% Subtracts the PrelimPrimaryBinaryImage from the DilatedPrimaryBinaryImage,
%%% which leaves the PrimaryObjectOutlines.
ActualObjectOutlines = DilatedActualObjectsBinaryImage - ActualObjectsBinaryImage;
%%% Produces the marker image which will be used for the watershed. The
%%% foreground markers are taken from the ActualObjectsBinaryImage; the
%%% background markers are taken from the same image as used in the first
%%% round of watershedding: InvertedThresholdedOrigImage.
BinaryMarkerImagePre2 = ActualObjectsBinaryImage | InvertedThresholdedOrigImage;
%%% Overlays the ActualObjectOutlines to maintain distinctions between each
%%% secondary object and the background.
BinaryMarkerImage2 = BinaryMarkerImagePre2;
BinaryMarkerImage2(ActualObjectOutlines == 1) = 0;

%%% STEP 9: Perform the second watershed.
%%% As described above, the second watershed is performed on the original
%%% intensity image rather than on a gradient (Sobel) image.
drawnow
%%% Inverts the original image.
InvertedOrigImage = imcomplement(OrigImageToBeAnalyzed);
%%% Overlays the foreground and background markers onto the
%%% InvertedOrigImage, so there are black secondary object markers on top
%%% of each dark secondary object, with black background. 
MarkedInvertedOrigImage = imimposemin(InvertedOrigImage, BinaryMarkerImage2);
%%% Performs the watershed on the MarkedInvertedOrigImage. 
SecondWatershedPre = watershed(MarkedInvertedOrigImage);
%%% BUG WORKAROUND: 
%%% There is a bug in the watershed function of Matlab that often results in 
%%% the label matrix result having two objects labeled with the same label.
%%% I am not sure whether it is a bug in how the watershed image is
%%% produced (it seems so: the resulting objects often are nowhere near the
%%% regional minima) or whether it is simply a problem in the final label
%%% matrix calculation. Matlab has been informed of this issue and has
%%% confirmed that it is a bug (February 2004). I think that it is a
%%% reasonable fix to convert the result of the watershed to binary and
%%% remake the label matrix so that each label is used only once. In later
%%% steps, inappropriate regions are weeded out anyway.
SecondWatershedPre2 = im2bw(SecondWatershedPre,1);
SecondWatershed = bwlabel(SecondWatershedPre2);
drawnow

%%% STEP 10: As in step 7, remove objects that are actually background
%%% objects.  See step 7 for description. This time, the edited primary object image is
%%% used rather than the preliminary one, so that objects whose nuclei are
%%% on the edge of the image and who are larger or smaller than the
%%% specified size are discarded.

%%% Converts the EditedPrimaryBinaryImage to binary.
EditedPrimaryBinaryImage = im2bw(EditedPrimaryLabelMatrixImage);
%%% Finds the locations and labels for different regions.
area_locations2 = find(SecondWatershed);
area_labels2 = SecondWatershed(area_locations2);
%%% Creates a sparse matrix with column as label and row as location,
%%% with the value of the center at (I,J) if location I has label J.
%%% Taking the maximum of this matrix gives the largest valued center
%%% overlapping a particular label.  Tacking on a zero and pushing
%%% labels through the resulting map removes any background regions.
map2 = [0 full(max(sparse(area_locations2, area_labels2, EditedPrimaryBinaryImage(area_locations2))))];
FinalBinaryImagePre = map2(SecondWatershed + 1);
%%% Fills holes in the FinalBinaryPre image.
FinalBinaryImage = imfill(FinalBinaryImagePre, 'holes');
%%% Converts the image to label matrix format. Even if the above step
%%% is excluded (filling holes), it is still necessary to do this in order
%%% to "compact" the label matrix: this way, each number corresponds to an
%%% object, with no numbers skipped.
ActualObjectsLabelMatrixImage3 = bwlabel(FinalBinaryImage);
%%% The final objects are relabeled so that their numbers
%%% correspond to the numbers used for nuclei.
%%% For each object, one label and one label location is acquired and
%%% stored.
[LabelsUsed,LabelLocations] = unique(EditedPrimaryLabelMatrixImage);
%%% The +1 increment accounts for the fact that there are zeros in the
%%% image, while the LabelsUsed starts at 1.
LabelsUsed(ActualObjectsLabelMatrixImage3(LabelLocations(2:end))+1) = EditedPrimaryLabelMatrixImage(LabelLocations(2:end));
FinalLabelMatrixImagePre = LabelsUsed(ActualObjectsLabelMatrixImage3+1);
%%% The following is a workaround for what seems to be a bug in the
%%% watershed function: very very rarely two nuclei end up sharing one
%%% "cell" object, so that one of the nuclei ends up without a
%%% corresponding cell.  I am trying to determine why this happens exactly.
%%% When the cell is measured, the area (and other
%%% measurements) are recorded as [], which causes problems when dependent
%%% measurements (e.g. perimeter/area) are attempted.  It results in divide
%%% by zero errors and the mean area = NaN and so on.  So, the Primary
%%% label matrix image (where it is nonzero) is written onto the Final cell
%%% label matrix image pre so that every primary object has at least some
%%% pixels of secondary object.
FinalLabelMatrixImage = FinalLabelMatrixImagePre;
FinalLabelMatrixImage(EditedPrimaryLabelMatrixImage ~= 0) = EditedPrimaryLabelMatrixImage(EditedPrimaryLabelMatrixImage ~= 0);
        
%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines the figure number to display in.
fieldname = ['figurealgorithm',CurrentAlgorithm];
ThisAlgFigureNumber = handles.(fieldname);
%%% Checks whether that figure is open. This checks all the figure handles
%%% for one whose handle is equal to the figure number for this algorithm.
%%% Note: Everything between the "if" and "end" is not carried out if the
%%% user has closed the figure window, so do not do any important
%%% calculations here. Otherwise an error message will be produced if the
%%% user has closed the window but you have attempted to access data that
%%% was supposed to be produced by this part of the code.
if any(findobj == ThisAlgFigureNumber) == 1;
    %%% THE FOLLOWING CALCULATIONS ARE FOR DISPLAY PURPOSES ONLY: The
    %%% resulting images are shown in the figure window (if open), or saved
    %%% to the hard drive (if desired).  To speed execution, all of this
    %%% code has been moved to within the if statement in the figure window
    %%% display section and then after starting image analysis, the figure
    %%% window can be closed.  Just remember that when the figure window is
    %%% closed, nothing within the if loop is carried out, so you would not
    %%% be able to save images depending on these lines to the hard drive,
    %%% for example.  If you plan to save images, these lines should be
    %%% moved outside this if statement.

    %%% Calculates the ColoredLabelMatrixImage for displaying in the figure
    %%% window in subplot(2,2,2).
    %%% Note that the label2rgb function doesn't work when there are no objects
    %%% in the label matrix image, so there is an "if".
    if sum(sum(FinalLabelMatrixImage)) >= 1
        ColoredLabelMatrixImage2 = label2rgb(FinalLabelMatrixImage,'jet', 'k', 'shuffle');
    else  ColoredLabelMatrixImage2 = FinalLabelMatrixImage;
    end

    %%% Calculates WatershedOnOriginalImage for displaying in the figure
    %%% window in subplot(2,2,3).
    %%% Calculates the final watershed lines for the secondary objects.
    %%% Creates the structuring element that will be used for dilation.
    StructuringElement = strel('square',3);
    %%% Dilates the FinalBinaryImage by one pixel (8 neighborhood).
    DilatedFinalSecondaryImage = imdilate(FinalBinaryImage, StructuringElement);
    %%% Subtracts the FinalBinaryImage from the DilatedPrimaryBinaryImage,
    %%% which leaves the SecondaryObjectOutlines.
    SecondaryObjectOutlines = DilatedFinalSecondaryImage - FinalBinaryImage;
    %%% Overlays the watershed lines on the original image.
    WatershedOnOriginalImage = OrigImageToBeAnalyzed;
    %%% Determines the grayscale intensity to use for the cell outlines.
    LineIntensity = max(OrigImageToBeAnalyzed(:));
    WatershedOnOriginalImage(SecondaryObjectOutlines == 1) = LineIntensity;

    %%% Calculates BothOutlinesOnOriginalImage for displaying in the figure
    %%% window in subplot(2,2,4).
    BothOutlinesOnOriginalImage = WatershedOnOriginalImage;
    BothOutlinesOnOriginalImage(PrimaryObjectOutlines == 1) = LineIntensity;

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
    subplot(2,2,1); imagesc(OrigImageToBeAnalyzed);colormap(gray);
    title(['Input Image, Image Set # ',num2str(handles.setbeinganalyzed)]);
    %%% A subplot of the figure window is set to display the colored label
    %%% matrix image.
    subplot(2,2,2); imagesc(ColoredLabelMatrixImage2); title(['Segmented ',SecondaryObjectName]);
    %%% A subplot of the figure window is set to display the original image
    %%% with watershed lines drawn on top.
    subplot(2,2,3); imagesc(WatershedOnOriginalImage); colormap(gray); title([SecondaryObjectName, ' Outlines on Input Image']);
    %%% A subplot of the figure window is set to display the original
    %%% image with watershed lines drawn for both the primary and secondary
    %%% objects.
    subplot(2,2,4); imagesc(BothOutlinesOnOriginalImage); colormap(gray); title(['Outlines of ', PrimaryObjectName, ' and ', SecondaryObjectName, ' on Input Image']);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the final, segmented label matrix image of secondary objects to
%%% the handles structure so it can be used by subsequent algorithms.
fieldname = ['dOTSegmented',SecondaryObjectName];
handles.(fieldname) = FinalLabelMatrixImage;

%%%% Saves the Threshold value to the handles structure.
fieldname = ['dMTThreshold', SecondaryObjectName];
handles.(fieldname)(handles.setbeinganalyzed) = {Threshold};

%% Determines the filename of the image to be analyzed.
fieldname = ['dOTFilename', ImageName];
FileName = handles.(fieldname)(handles.setbeinganalyzed);
%%% Saves the filename of the image to be analyzed.
fieldname = ['dOTFilename', SecondaryObjectName];
handles.(fieldname)(handles.setbeinganalyzed) = FileName;