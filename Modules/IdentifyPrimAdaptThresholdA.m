function handles = AlgIdentifyPrimAdaptThresholdA(handles)

% Help for Identify Primary Adaptive Threshold A module: 
% This image analysis module identifies objects by applying an adaptive
% threshold to a grayscale image.
%
% BLOCK SIZE: should be set large enough that every square block of
% pixels is likely to contain some background and some foreground.
% Smaller block sizes take more processing time.

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
% The Original Code is the Identify Primary Adaptive Threshold A module.
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
CurrentAlgorithmNum = str2num(handles.currentalgorithm);

%textVAR01 = What did you call the images you want to process? 
%defaultVAR01 = OrigBlue
ImageName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,1});

%textVAR02 = What do you want to call the objects identified by this algorithm?
%defaultVAR02 = Nuclei
ObjectName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,2});

%textVAR03 = Size range (in pixels) of objects to include (1,99999 = do not discard any)
%defaultVAR03 = 1,99999
SizeRange = char(handles.Settings.Vvariable{CurrentAlgorithmNum,3});

%textVAR04 = Enter the desired minimum threshold (0 to 1), or "A" to calculate automatically
%defaultVAR04 = A
MinimumThreshold = char(handles.Settings.Vvariable{CurrentAlgorithmNum,4});

%textVAR05 = Enter the threshold adjustment factor (>1 = more stringent, <1 = less stringent)
%defaultVAR05 = 1
ThresholdAdjustmentFactor = str2num(char(handles.Settings.Vvariable{CurrentAlgorithmNum,5}));

%textVAR06 = Block size, in pixels
%defaultVAR06 = 100
BlockSize = str2num(char(handles.Settings.Vvariable{CurrentAlgorithmNum,6}));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines what the user entered for the size range.
SizeRangeNumerical = str2num(SizeRange);
MinSize = SizeRangeNumerical(1);
MaxSize = SizeRangeNumerical(2);

%%% Reads (opesn) the image you want to analyze and assigns it to a variable,
%%% "OrigImageToBeAnalyzed".
fieldname = ['dOT', ImageName];
%%% Checks whether the image exists in the handles structure.
if isfield(handles, fieldname) == 0
    error(['Image processing has been canceled. Prior to running the Identify Primary Adaptive Threshold module, you must have previously run an algorithm to load an image. You specified in the Identify Primary Adaptive Threshold module that this image was called ', ImageName, ' which should have produced a field in the handles structure called ', fieldname, '. The Identify Primary Adaptive Threshold module cannot find this image.']);
end
OrigImageToBeAnalyzed = handles.(fieldname);

%%% Checks that the original image is two-dimensional (i.e. not a color
%%% image), which would disrupt several of the image functions.
if ndims(OrigImageToBeAnalyzed) ~= 2
    error('Image processing was canceled because the Identify Primary Adaptive Threshold module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
end

%%% Check whether the chosen block size is larger than the image itself.
[m,n] = size(OrigImageToBeAnalyzed);
MinLengthWidth = min(m,n);
if BlockSize >= MinLengthWidth
    error('Image processing was canceled because in the Identify Primary Adaptive Threshold module the selected block size is greater than or equal to the image size itself.')
end

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%

drawnow
%%% Calculates the MinimumThreshold automatically, if requested.
if strncmp(upper(MinimumThreshold),'A',1) == 1
    GlobalThreshold = graythresh(OrigImageToBeAnalyzed);
    %%% 0.7 seemed to produce good results; there is no theoretical basis
    %%% for choosing that exact number.
    MinimumThreshold = GlobalThreshold*0.7;
else
    try MinimumThreshold = str2num(MinimumThreshold);
    catch error('The value entered for the minimum threshold in the Identify Primary Adaptive Threshold module was not correct.')
    end
end
%%% Calculates the best block size that prevents padding with zeros, which
%%% would produce edge artifacts. This is based on Matlab's bestblk
%%% function, but changing the minimum of the range searched to be 75% of
%%% the suggested block size rather than 50%. Defines acceptable block
%%% sizes.  m and n were calculated above as the size of the original
%%% image.
MM = floor(BlockSize):-1:floor(min(ceil(m/10),ceil(BlockSize*3/4)));
NN = floor(BlockSize):-1:floor(min(ceil(n/10),ceil(BlockSize*3/4)));
%%% Chooses the acceptable block that has the minimum padding.
[dum,ndx] = min(ceil(m./MM).*MM-m);
BestBlockSize(1) = MM(ndx);
[dum,ndx] = min(ceil(n./NN).*NN-n);
drawnow
BestBlockSize(2) = NN(ndx);
BestRows = BestBlockSize(1)*ceil(m/BestBlockSize(1));
BestColumns = BestBlockSize(2)*ceil(n/BestBlockSize(2));
RowsToAdd = BestRows - m;
ColumnsToAdd = BestColumns - n;
%%% Pads the image so that the blocks fit properly.
PaddedImage = padarray(OrigImageToBeAnalyzed,[RowsToAdd ColumnsToAdd],'replicate','post');
%%% Calculates the threshold for each block in the image.
drawnow
SmallImageOfThresholds = blkproc(PaddedImage,[BestBlockSize(1) BestBlockSize(2)],'graythresh(x)');
%%% Resizes the block-produced image to be the size of the padded image.
%%% Bilinear prevents dipping below zero.
PaddedImageOfThresholds = imresize(SmallImageOfThresholds, size(PaddedImage), 'bilinear');
%%% "Crops" the image to get rid of the padding, to make the result the same
%%% size as the original image.
drawnow
ImageOfThresholds = PaddedImageOfThresholds(1:m,1:n);
%%% Multiplies an adjustment factor against the thresholds to reduce or
%%% increase them all proportionally.
CorrectedImageOfThresholds = ThresholdAdjustmentFactor*ImageOfThresholds;
%%% FOR DISPLAY ONLY
PreThresholdedImage = OrigImageToBeAnalyzed;
PreThresholdedImage(PreThresholdedImage <= CorrectedImageOfThresholds) = 0;
drawnow
PreThresholdedImage(PreThresholdedImage > CorrectedImageOfThresholds) = 1;
%%% For any of the threshold values that is lower than the user-specified
%%% minimum threshold, set to equal the minimum threshold.  Thus, if there
%%% are no objects within a block (e.g. if cells are very sparse), an
%%% unreasonable threshold will be overridden by the minimum threshold.
drawnow
MinImageOfThresholds = CorrectedImageOfThresholds;
MinImageOfThresholds(MinImageOfThresholds <= MinimumThreshold) = MinimumThreshold;
%%% Applies the thresholds to the image.
ThresholdedImage = OrigImageToBeAnalyzed;
ThresholdedImage(ThresholdedImage <= MinImageOfThresholds) = 0;
drawnow
ThresholdedImage(ThresholdedImage > MinImageOfThresholds) = 1;
ThresholdedImage = logical(ThresholdedImage);
%%% Holes in the ThresholdedImage image are filled in.
ThresholdedImage = imfill(ThresholdedImage, 'holes');

%%% Identifies objects in the binary image.
PrelimLabelMatrixImage1 = bwlabel(ThresholdedImage);
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
drawnow
AreasImage = Map(PrelimLabelMatrixImage1 + 1);
%%% The small objects are overwritten with zeros.
PrelimLabelMatrixImage2 = PrelimLabelMatrixImage1;
PrelimLabelMatrixImage2(AreasImage < MinSize) = 0;
%%% Relabels so that labels are consecutive. This is important for
%%% downstream modules (IdentifySec).
PrelimLabelMatrixImage2 = bwlabel(im2bw(PrelimLabelMatrixImage2,.1));
%%% The large objects are overwritten with zeros.
drawnow
PrelimLabelMatrixImage3 = PrelimLabelMatrixImage2;
if MaxSize ~= 99999
    PrelimLabelMatrixImage3(AreasImage > MaxSize) = 0;
end
%%% Removes objects that are touching the edge of the image, since they
%%% won't be measured properly.
PrelimLabelMatrixImage4 = imclearborder(PrelimLabelMatrixImage3,8);
%%% The PrelimLabelMatrixImage4 is converted to binary.
drawnow
FinalBinaryPre = im2bw(PrelimLabelMatrixImage4,1);
%%% Holes in the FinalBinaryPre image are filled in.
FinalBinary = imfill(FinalBinaryPre, 'holes');
%%% The image is converted to label matrix format. Even if the above step
%%% is excluded (filling holes), it is still necessary to do this in order
%%% to "compact" the label matrix: this way, each number corresponds to an
%%% object, with no numbers skipped.
FinalLabelMatrixImage = bwlabel(FinalBinary);

%%% POTENTIAL IMPROVEMENT TO MAKE:  May want to blur the Image of Thresholds.

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
    drawnow
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
    drawnow
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
    subplot(2,2,2); imagesc(ColoredLabelMatrixImage); title(['Identified ',ObjectName]);
    %%% A subplot of the figure window is set to display the prethresholded
    %%% image.
    subplot(2,2,3); imagesc(PreThresholdedImage);colormap(gray); title('Without applying minimum threshold');
    %%% A subplot of the figure window is set to display the inverted original
    %%% image with outlines drawn on top.
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