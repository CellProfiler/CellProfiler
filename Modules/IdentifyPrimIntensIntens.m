function handles = AlgIdentifyPrimIntensIntens(handles)

% Help for the Identify Primary Intensity Intensity module: 
% 
% This image analysis module identifies objects by finding peaks in
% intensity, after the image has been blurred to remove texture.  Once a
% marker for each object has been identified in this way, a watershed
% function identifies the lines between objects that are touching each
% other by looking for the dimmest points between them.  To identify the
% edges of non-clumped objects, a simple threshold is applied. The
% algorithm works best for objects that are brighter towards the interior;
% the objects can be any shape, so they need not be round and uniform in
% size as would be required for a distance-based algorithm.  The dividing
% lines between clumped objects should be dim.
%
% SPEED OPTIMIZATION: Note that increasing the blur radius increases
% the processing time exponentially.

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
% The Original Code is the Identify Primary Intensity Intensity module.
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
%defaultVAR01 = OrigBlue
ImageName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,1});

%textVAR02 = What do you want to call the objects identified by this algorithm?
%defaultVAR02 = Nuclei
ObjectName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,2});

%textVAR03 = Size range (in pixels) of objects to include (1,99999 = do not discard any)
%defaultVAR03 = 1,99999
SizeRange = char(handles.Settings.Vvariable{CurrentAlgorithmNum,3});

%textVAR04 = Enter the threshold [0 = automatically calculate] (Positive number, Max = 1):
%defaultVAR04 = 0
Threshold = str2double(char(handles.Settings.Vvariable{CurrentAlgorithmNum,4}));

%textVAR05 = If auto threshold, enter an adjustment factor (Positive number, 1 = no adjustment):
%defaultVAR05 = 1
ThresholdAdjustmentFactor = str2double(char(handles.Settings.Vvariable{CurrentAlgorithmNum,5}));

%textVAR06 = Set the Maxima Suppression Neighborhood (Non-negative integer, Default = 6):
%defaultVAR06 = 6
MaximaSuppressionNeighborhood = str2double(char(handles.Settings.Vvariable{CurrentAlgorithmNum,6}));

%textVAR07 = Set the blur radius (Lower is faster; Non-negative number; Default = 3):
%defaultVAR07 = 3
BlurRadius = str2double(char(handles.Settings.Vvariable{CurrentAlgorithmNum,7}));


%%% Determines what the user entered for the size range.
SizeRangeNumerical = str2num(SizeRange); %#ok We want to ignore MLint error checking for this line.
MinSize = SizeRangeNumerical(1);
MaxSize = SizeRangeNumerical(2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Note to programmers: *Any* data that you write to the handles structure
%%% must be preceded by 'dMT', 'dMC', or 'dOT' so that the data is deleted
%%% at the end of the processing batch.  dMT stands for deletable
%%% Measurements - Total image (i.e. any case where there is one
%%% measurement for the entire image).  dMC stands for deletable
%%% Measurements - Cell by cell (i.e. any case where there is more than one
%%% measurement for the entire image). dOT stands for deletable - OTher,
%%% which would typically include images that are stored in the handles
%%% structure for use by other algorithms.  If the data is not deleted, it
%%% is still available for use if the user runs a completely separate
%%% analysis.  This could be disastrous: For example, a user might process
%%% 12 image sets of nuclei which results in a set of 12 measurements
%%% ("TotalNucArea") stored in the handles structure.  In addition, a
%%% processed image of nuclei from the last image set is left in the
%%% handles structure ("SegmNucImg").  Now, if the user uses a different
%%% algorithm which happens to have the same measurement output name
%%% "TotalNucArea" to analyze 4 image sets, the 4 measurements will
%%% overwrite the first 4 measurements of the previous analysis, but the
%%% remaining 8 measurements will still be present.  So, the user will end
%%% up with 12 measurements from the 4 sets.  Another potential problem is
%%% that if, in the second analysis run, the user runs only an algorithm
%%% which depends on the output "SegmNucImg" but does not run an algorithm
%%% that produces an image by that name, the algorithm will run just fine: it will  
%%% just repeatedly use the processed image of nuclei leftover from the last
%%% image set, which was left in the handles structure ("SegmNucImg").

%%% As a policy, I think it is probably wise to disallow more than one
%%% "column" of data per object, to allow for uniform extraction of data
%%% later. So, for example, instead of creating a field of XY locations
%%% stored in pairs, it is better to store a field of X locations and a
%%% field of Y locations.

%%% Reads (opens) the image you want to analyze and assigns it to a variable,
%%% "OrigImageToBeAnalyzed".
fieldname = ['dOT', ImageName];

%%% Checks whether the image exists in the handles structure.
if isfield(handles, fieldname) == 0
    error(['Image processing has been canceled. Prior to running the Identify Primary Intensity module, you must have previously run an algorithm to load an image. You specified in the Identify Primary Intensity module that this image was called ', ImageName, ' which should have produced a field in the handles structure called ', fieldname, '. The Identify Primary Intensity module cannot find this image.']);
end
OrigImageToBeAnalyzed = handles.(fieldname);

%%% Checks that the original image is two-dimensional (i.e. not a color
%%% image), which would disrupt several of the image functions.
if ndims(OrigImageToBeAnalyzed) ~= 2
    error('Image processing was canceled because the Identify Primary Intensity module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
end

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%

%%% STEP 1: Finds markers for each nucleus based on local maxima in the
%%% intensity image.
drawnow 
if BlurRadius == 0
    BlurredImage = OrigImageToBeAnalyzed;
else
%%% Blurs the image.
%%% Note: using filter2 is much faster than imfilter (e.g. 14.5 sec vs. 99.1 sec).
FiltSize = max(3,ceil(4*BlurRadius));
BlurredImage = filter2(fspecial('gaussian',FiltSize, BlurRadius), OrigImageToBeAnalyzed);
end

%%% Perturbs the blurred image so that local maxima near each other with
%%% identical values will now have slightly different values.
%%% Saves off the random number generator's state, and set the state to
%%% a particular value (for repeatability)
oldstate = rand('state');
rand('state',0);
%%% Adds a random value between 0 and 0.002 to each pixel in the
%%% BlurredImage. We chose .002
BlurredImage = BlurredImage + 0.002*rand(size(BlurredImage));
%%% Restores the random number generator's state.
rand('state',oldstate);

%%% Extracts local maxima and filters them by eliminating maxima that are
%%% within a certain distance of each other.
MaximaImage = BlurredImage;
MaximaMask = strel('disk', MaximaSuppressionNeighborhood);
MaximaImage(BlurredImage < ordfilt2(BlurredImage,sum(sum(getnhood(MaximaMask))),getnhood(MaximaMask))) = 0;
%%% Determines the threshold to be used, if the user has left the Threshold
%%% variable set to 0.
if Threshold == 0
    Threshold = graythresh(OrigImageToBeAnalyzed);
    Threshold = Threshold*ThresholdAdjustmentFactor;
end
%%% Thresholds the image to eliminate dim maxima.
MaximaImage(~im2bw(OrigImageToBeAnalyzed, Threshold))=0;

%%% STEP 2: Performs watershed function on the original intensity
%%% (grayscale) image.
drawnow 
%%% Inverts original image.
InvertedOriginal = imcomplement(OrigImageToBeAnalyzed);
%%% Overlays the nuclear markers (maxima) on the inverted original image so
%%% there are black dots on top of each dark nucleus on a white background.
Overlaid = imimposemin(InvertedOriginal,MaximaImage);
%%% Identifies watershed lines.
BlackWatershedLinesPre = watershed(Overlaid);
%%% Superimposes watershed lines as white (255) onto the inverted original
%%% image.
WhiteWatershedOnInvertedOrig = InvertedOriginal;
WhiteWatershedOnInvertedOrig(BlackWatershedLinesPre == 0) = 255;

%%% STEP 3: Identifies and extracts the objects, using the watershed lines.
drawnow
%%% Thresholds the WhiteWatershedOnInvertedOrig image, using the same
%%% threshold as used for the maxima detection, except the number is inverted
%%% since we are working with an inverted image now.
InvertedThreshold = 1 - Threshold;
BinaryObjectsImage = im2bw(WhiteWatershedOnInvertedOrig,InvertedThreshold);
%%% Inverts the BinaryObjectsImage.
InvertedBinaryImage = imcomplement(BinaryObjectsImage);
%%% Fills holes, then identifies objects in the binary image.
PrelimLabelMatrixImage1 = bwlabel(imfill(InvertedBinaryImage,'holes'));
%%% Finds objects larger and smaller than the user-specified size.
%%% Finds the locations and labels for the pixels that are part of an object.
AreaLocations = find(PrelimLabelMatrixImage1);
AreaLabels = PrelimLabelMatrixImage1(AreaLocations);
drawnow
%%% Creates a sparse matrix with column as label and row as location,
%%% with a 1 at (A,B) if location A has label B.  Summing the columns
%%% gives the count of area pixels with a given label.  E.g. Areas(L) is the
%%% number of pixels with label L.
Areas = full(sum(sparse(AreaLocations, AreaLabels, 1)));
Map = [0,Areas];
AreasImage = Map(PrelimLabelMatrixImage1 + 1);
%%% The small objects are overwritten with zeros.
PrelimLabelMatrixImage2 = PrelimLabelMatrixImage1;
PrelimLabelMatrixImage2(AreasImage < MinSize) = 0;
drawnow
%%% Relabels so that labels are consecutive. This is important for
%%% downstream modules (IdentifySec).
PrelimLabelMatrixImage2 = bwlabel(im2bw(PrelimLabelMatrixImage2,.1));
%%% The large objects are overwritten with zeros.
PrelimLabelMatrixImage3 = PrelimLabelMatrixImage2;
if MaxSize ~= 99999
    PrelimLabelMatrixImage3(AreasImage > MaxSize) = 0;
end
%%% Removes objects that are touching the edge of the image, since they
%%% won't be measured properly.
PrelimLabelMatrixImage4 = imclearborder(PrelimLabelMatrixImage3,8);
%%% The PrelimLabelMatrixImage4 is converted to binary.
FinalBinaryPre = im2bw(PrelimLabelMatrixImage4,1);
drawnow
%%% Holes in the FinalBinaryPre image are filled in.
FinalBinary = imfill(FinalBinaryPre, 'holes');
%%% The image is converted to label matrix format. Even if the above step
%%% is excluded (filling holes), it is still necessary to do this in order
%%% to "compact" the label matrix: this way, each number corresponds to an
%%% object, with no numbers skipped.
FinalLabelMatrixImage = bwlabel(FinalBinary);
   
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
        ColoredLabelMatrixImage = label2rgb(FinalLabelMatrixImage, 'jet', 'k', 'shuffle');
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
    %%% Subtracts the FinalBinaryImage from the DilatedBinaryImage,
    %%% which leaves the PrimaryObjectOutlines.
    PrimaryObjectOutlines = DilatedBinaryImage - FinalBinaryImage;
    %%% Overlays the object outlines on the original image.
    ObjectOutlinesOnOriginalImage = OrigImageToBeAnalyzed;
    %%% Determines the grayscale intensity to use for the cell outlines.
    LineIntensity = max(OrigImageToBeAnalyzed(:));
    ObjectOutlinesOnOriginalImage(PrimaryObjectOutlines == 1) = LineIntensity;
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

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

%%% Saves the Threshold value to the handles structure.
fieldname = ['dMTThreshold', ObjectName];
handles.(fieldname)(handles.setbeinganalyzed) = {Threshold};

%%% Determines the filename of the image to be analyzed.
fieldname = ['dOTFilename', ImageName];
FileName = handles.(fieldname)(handles.setbeinganalyzed);
%%% Saves the filename of the image to be analyzed.
fieldname = ['dOTFilename', ObjectName];
handles.(fieldname)(handles.setbeinganalyzed) = FileName;