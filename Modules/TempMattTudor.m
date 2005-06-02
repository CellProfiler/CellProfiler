function handles = TempMattTudor(handles)

% Help for the Identify Primary Adaptive Threshold C module: 
% Category: Testing
%
% This image analysis module identifies objects by applying an adaptive
% threshold to the image.
% .
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
% The Original Code is the the Identify Primary Adaptive Threshold C module.
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

%%% Reads the current module number, because this is needed to find 
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

%textVAR01 = What did you call the images you want to process? 
%defaultVAR01 = OrigBlue
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = What do you want to call the objects identified by this algorithm?
%defaultVAR02 = Nuclei
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Size range (in pixels) of objects to include (1,99999 = do not discard any)
%defaultVAR03 = 1,99999
SizeRange = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = Enter the desired minimum threshold (0 to 1), or "A" to calculate automatically
%defaultVAR04 = A
MinimumThreshold = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = Enter the threshold adjustment factor (>1 = more stringent, <1 = less stringent)
%defaultVAR05 = 1
ThresholdAdjustmentFactor = str2num(char(handles.Settings.VariableValues{CurrentModuleNum,5}));

%textVAR06 = Block size, in pixels
%defaultVAR06 = 100
BlockSize = str2num(char(handles.Settings.VariableValues{CurrentModuleNum,6}));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines what the user entered for the size range.
SizeRangeNumerical = str2num(SizeRange);
MinSize = SizeRangeNumerical(1);
MaxSize = SizeRangeNumerical(2);

%%% Reads (opens) the image you want to analyze and assigns it to a variable,
%%% "OrigImage".
fieldname = ['dOT', ImageName];
%%% Checks whether the image exists in the handles structure.
    if isfield(handles, fieldname) == 0
    error(['Image processing has been canceled. Prior to running the Identify Primary Adaptive Threshold module, you must have previously run a module to load an image. You specified in the Identify Primary Adaptive Threshold module that this image was called ', ImageName, ' which should have produced a field in the handles structure called ', fieldname, '. The Identify Primary Adaptive Threshold module cannot find this image.']);
    end
OrigImage = handles.(fieldname);

%%% Checks that the original image is two-dimensional (i.e. not a color
%%% image), which would disrupt several of the image functions.
if ndims(OrigImage) ~= 2
    error('Image processing was canceled because the Identify Primary Adaptive Threshold module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
end

%%% Checks whether the chosen block size is larger than the image itself.
[m,n] = size(OrigImage);
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
    GlobalThreshold = CPgraythresh(OrigImage);
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
PaddedImage = padarray(OrigImage,[RowsToAdd ColumnsToAdd],'replicate','post');
drawnow
%%% Calculates the threshold for each block in the image.
SmallImageOfThresholds = blkproc(PaddedImage,[BestBlockSize(1) BestBlockSize(2)],@isodata);
%%% Resizes the block-produced image to be the size of the padded image.
%%% Bilinear prevents dipping below zero.
PaddedImageOfThresholds = imresize(SmallImageOfThresholds, size(PaddedImage), 'bilinear');
drawnow
%%% "Crops" the image to get rid of the padding, to make the result the same
%%% size as the original image.
ImageOfThresholds = PaddedImageOfThresholds(1:m,1:n);
%%% Multiplies an adjustment factor against the thresholds to reduce or
%%% increase them all proportionally.
CorrectedImageOfThresholds = ThresholdAdjustmentFactor*ImageOfThresholds;
drawnow
%%% For any of the threshold values that is lower than the user-specified
%%% minimum threshold, set to equal the minimum threshold.  Thus, if there
%%% are no objects within a block (e.g. if cells are very sparse), an
%%% unreasonable threshold will be overridden by the minimum threshold.
MinImageOfThresholds = CorrectedImageOfThresholds;
MinImageOfThresholds(MinImageOfThresholds <= MinimumThreshold) = MinimumThreshold;
%%% Applies the thresholds to the image.
ThresholdedImage = OrigImage;
ThresholdedImage(ThresholdedImage <= MinImageOfThresholds) = 0;
ThresholdedImage(ThresholdedImage > MinImageOfThresholds) = 1;
drawnow
ThresholdedImage = logical(ThresholdedImage);
%%% Holes in the ThresholdedImage image are filled in.
ThresholdedImage = imfill(ThresholdedImage, 'holes');

%%% POTENTIAL IMPROVEMENT TO MAKE:  May want to blur the Image of Thresholds.

%%% Identifies objects in the binary image.
drawnow
PrelimLabelMatrixImage1 = bwlabel(ThresholdedImage);
%%% Finds objects larger and smaller than the user-specified size.
%%% Finds the locations and labels for the pixels that are part of an object.
AreaLocations = find(PrelimLabelMatrixImage1);
drawnow
AreaLabels = PrelimLabelMatrixImage1(AreaLocations);
%%% Creates a sparse matrix with column as label and row as location,
%%% with a 1 at (A,B) if location A has label B.  Summing the columns
%%% gives the count of area pixels with a given label.  E.g. Areas(L) is the
%%% number of pixels with label L.
Areas = full(sum(sparse(AreaLocations, AreaLabels, 1)));
Map = [0,Areas];
AreasImage = Map(PrelimLabelMatrixImage1 + 1);
drawnow
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
FinalBinaryPre = im2bw(PrelimLabelMatrixImage4,1);
%%% Holes in the FinalBinaryPre image are filled in.
FinalBinary = imfill(FinalBinaryPre, 'holes');
drawnow
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
fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
%%% Checks whether that figure is open. This checks all the figure handles
%%% for one whose handle is equal to the figure number for this algorithm.
%%% Note: Everything between the "if" and "end" is not carried out if the
%%% user has closed the figure window, so do not do any important
%%% calculations here. Otherwise an error message will be produced if the
%%% user has closed the window but you have attempted to access data that
%%% was supposed to be produced by this part of the code.
if any(findobj == ThisModuleFigureNumber) == 1;
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
    %%% Calculates the PreThresholdedImage for displaying in the figure
    %%% window in subplot(2,2,3).
    PreThresholdedImage = OrigImage;
    PreThresholdedImage(PreThresholdedImage <= CorrectedImageOfThresholds) = 0;
    PreThresholdedImage(PreThresholdedImage > CorrectedImageOfThresholds) = 1;
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
    ObjectOutlinesOnOrigImage = OrigImage;
    %%% Determines the grayscale intensity to use for the cell outlines.
    LineIntensity = max(OrigImage(:));
    ObjectOutlinesOnOrigImage(PrimaryObjectOutlines == 1) = LineIntensity;
    %%% The "drawnow" function executes any pending figure window-related
    %%% commands.  In general, Matlab does not update figure windows
    %%% until breaks between image analysis modules, or when a few select
    %%% commands are used. "figure" and "drawnow" are two of the commands
    %%% that allow Matlab to pause and carry out any pending figure window-
    %%% related commands (like zooming, or pressing timer pause or cancel
    %%% buttons or pressing a help button.)  If the drawnow command is not
    %%% used immediately prior to the figure(ThisModuleFigureNumber) line,
    %%% then immediately after the figure line executes, the other commands
    %%% that have been waiting are executed in the other windows.  Then,
    %%% when Matlab returns to this module and goes to the subplot line,
    %%% the figure which is active is not necessarily the correct one.
    %%% This results in strange things like the subplots appearing in the
    %%% timer window or in the wrong figure window, or in help dialog boxes.
    drawnow
    figure(ThisModuleFigureNumber);
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,2,1); imagesc(OrigImage);colormap(gray);
    title(['Input Image, Image Set # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the colored label
    %%% matrix image.
    subplot(2,2,2); imagesc(ColoredLabelMatrixImage); title(['Segmented ',ObjectName]);
    %%% A subplot of the figure window is set to display the prethresholded
    %%% image.
    subplot(2,2,3); imagesc(PreThresholdedImage);colormap(gray); title('Without applying minimum threshold');
    %%% A subplot of the figure window is set to display the inverted original
    %%% image with outlines drawn on top.
    subplot(2,2,4); imagesc(ObjectOutlinesOnOrigImage);colormap(gray); title([ObjectName, ' Outlines on Input Image']);
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

%%% Determines the filename of the image to be analyzed.
fieldname = ['dOTFilename', ImageName];
FileName = handles.(fieldname)(handles.Current.SetBeingAnalyzed);
%%% Saves the filename of the image to be analyzed.
fieldname = ['dOTFilename', ObjectName];
handles.(fieldname)(handles.Current.SetBeingAnalyzed) = FileName;

%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% ISODATA SUBFUNCTION %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%

function level=isodata(I)
%   ISODATA Compute global image threshold using iterative isodata method.
%   LEVEL = ISODATA(I) computes a global threshold (LEVEL) that can be
%   used to convert an intensity image to a binary image with IM2BW. LEVEL
%   is a normalized intensity value that lies in the range [0, 1].
%   This iterative technique for choosing a threshold was developed by Ridler and Calvard .
%   The histogram is initially segmented into two parts using a starting threshold value such as 0 = 2B-1,
%   half the maximum dynamic range.
%   The sample mean (mf,0) of the gray values associated with the foreground pixels and the sample mean (mb,0)
%   of the gray values associated with the background pixels are computed. A new threshold value 1 is now computed
%   as the average of these two sample means. The process is repeated, based upon the new threshold,
%   until the threshold value does not change any more.
% Reference :T.W. Ridler, S. Calvard, Picture thresholding using an iterative selection method,
%            IEEE Trans. System, Man and Cybernetics, SMC-8 (1978) 630-632.

% Convert all N-D arrays into a single column.  Convert to uint8 for
% fastest histogram computation.
%MT:  ran into trouble with low intensity images where the dynamic range
%is 4-7 bins. so will stretch data before binning & restore scale at end.
Irange=prctile(I(:),[1,99]);
I = imadjust(I);
I = im2uint8(I(:));

% STEP 1: Compute mean intensity of image from histogram, set T=mean(I)
[counts,N]=imhist(I);
i=1;
mu=cumsum(counts);
T(i)=(sum(N.*counts))/mu(end);
T(i)=round(T(i));

%%% Errors were resulting in the mu2(end) line below if the mean intensity
%%% is zero, so I added the following if statement.
if T(i) == 0
    level = 0;
    return
end

% STEP 2: compute Mean above T (MAT) and Mean below T (MBT) using T from
% step 1
mu2=cumsum(counts(1:T(i)));
MBT=sum(N(1:T(i)).*counts(1:T(i)))/mu2(end);

mu3=cumsum(counts(T(i):end));
MAT=sum(N(T(i):end).*counts(T(i):end))/mu3(end);
i=i+1;
T(i)=round((MAT+MBT)/2);
%%% I added the following line because for some images,
%%% Threshold ends up as an undefined variable if the while function below
%%% does not even get started.
Threshold = T(i);

%%% Errors were resulting in the mu2(end) line below if the mean intensity
%%% is zero, so I added the following if statement.
if T(i) == 0
    level = 0;
    return
end

% STEP 3 to n: repeat step 2 if T(i)~=T(i-1)
while abs(T(i)-T(i-1))>=1
    mu2=cumsum(counts(1:T(i)));
    if mu2(end) == 0
        Threshold=T(i);
        break
    end
    MBT=sum(N(1:T(i)).*counts(1:T(i)))/mu2(end);

    mu3=cumsum(counts(T(i):end));
    if mu3(end) == 0
        Threshold=T(i);
        break
    end
    MAT=sum(N(T(i):end).*counts(T(i):end))/mu3(end);

    i=i+1;
    T(i)=round((MAT+MBT)/2); 
    Threshold=T(i);
end

% Normalize the threshold to the range [i, 1].
%MT:  on the _original_ scale
level = (Irange(2)-Irange(1))*(Threshold - 1) / (N(end) - 1);function handles = AlgIdentifyPrimAdaptThresholdB(handles)

% Help for the Identify Primary Adaptive Threshold B module: 
% 
% This image analysis module identifies objects by applying an adaptive
% threshold to the image.
%
% NEIGHBORHOOD SIZE: should be set large enough that every square block of
% pixels is likely to contain some background and some foreground.
% Smaller neighborhood sizes take less processing time.

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
% The Original Code is the Identify Primary Adaptive Threshold B module.
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

%%% Reads the current module number, because this is needed to find 
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

%textVAR01 = What did you call the images you want to process? 
%defaultVAR01 = OrigBlue
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = What do you want to call the objects identified by this algorithm?
%defaultVAR02 = Nuclei
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Size range (in pixels) of objects to include (1,99999 = do not discard any)
%defaultVAR03 = 1,99999
SizeRange = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = Enter the desired minimum threshold (0 to 1), or "A" to calculate automatically
%defaultVAR04 = A
MinimumThreshold = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = Enter the threshold adjustment factor (>1 = more stringent, <1 = less stringent)
%defaultVAR05 = 1
ThresholdAdjustmentFactor = str2num(char(handles.Settings.VariableValues{CurrentModuleNum,5}));

%textVAR06 = Neighborhood size, in pixels (Odd number)
%defaultVAR06 = 51
NeighborhoodSize = str2num(char(handles.Settings.VariableValues{CurrentModuleNum,6}));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines what the user entered for the size range.
SizeRangeNumerical = str2num(SizeRange);
MinSize = SizeRangeNumerical(1);
MaxSize = SizeRangeNumerical(2);

%%% Reads (opens) the image you want to analyze and assigns it to a variable,
%%% "OrigImage".
fieldname = ['dOT', ImageName];
%%% Checks whether the image exists in the handles structure.
    if isfield(handles, fieldname) == 0
    error(['Image processing has been canceled. Prior to running the Identify Primary Adaptive Threshold module, you must have previously run an algorithm to load an image. You specified in the Identify Primary Adaptive Threshold module that this image was called ', ImageName, ' which should have produced a field in the handles structure called ', fieldname, '. The Identify Primary Adaptive Threshold module cannot find this image.']);
    end
OrigImage = handles.(fieldname);

%%% Checks that the original image is two-dimensional (i.e. not a color
%%% image), which would disrupt several of the image functions.
if ndims(OrigImage) ~= 2
    error('Image processing was canceled because the Identify Primary Adaptive Threshold module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
end

%%% Checks whether the chosen block size is larger than the image itself.
[m,n] = size(OrigImage);
MinLengthWidth = min(m,n);
if NeighborhoodSize >= MinLengthWidth
    error('Image processing was canceled because in the Identify Primary Adaptive Threshold module the selected block size is greater than or equal to the image size itself.')
end

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Calculates the MinimumThreshold automatically, if requested.
if strncmp(upper(MinimumThreshold),'A',1) == 1
    GlobalThreshold = CPgraythresh(OrigImage);
    %%% 0.2 seemed to produce good results; there is no theoretical basis
    %%% for choosing that exact number.
    MinimumThreshold = GlobalThreshold*0.2;
else
    try MinimumThreshold = str2num(MinimumThreshold);
    catch error('The value entered for the minimum threshold in the Identify Primary Adaptive Threshold module was not correct.')
    end
end
%%% Neighborhood must be an odd number.
if rem(NeighborhoodSize,2) == 0
    NeighborhoodSize = NeighborhoodSize - 1;
    if handles.Current.SetBeingAnalyzed == 1
        warndlg(['The neighborhood size in the Identify Primary Adaptive Threshold module must be an odd number. The value that will be used is ', num2str(NeighborhoodSize), '.'])
        drawnow
    end
end
%%% Converts the image before processing.  This assumes that it's coming in
%%% as a double image from 0 to 1.
%%% Invert the image to make dark objects on a white background.
Image1 = imcomplement(OrigImage);
%%% Stretch the image to use the full dynamic range from 0 to 1. 
Image = imadjust(Image1,[min(min(Image1)) max(max(Image1))],[0 1]);
%%% Performs adaptive thresholding.
%%% This code was adapted from ñImage Segmentation by adaptive
%%% thresholdingî by Nir Milstein of Technion - Israel Institute of
%%% Technology, The Faculty for Computer Sciences.  The theory is that the
%%% average value within a neighborhood is likely to be a good threshold
%%% (this assumes a rather sparse distribution of objects so that the
%%% background predominates in any given neighborhood.)
AverageFilter = ones(NeighborhoodSize, NeighborhoodSize) / (NeighborhoodSize^2);
FirstNumber = ceil(NeighborhoodSize/2);
SecondNumber = NeighborhoodSize - FirstNumber;
%MT:  need to pad array to get around convolution-derived edge effects
%which cause entire image to be filled in in the imfill('holes') step
Threshold = conv2(padarray(OrigImage,[FirstNumber FirstNumber],'symmetric'), AverageFilter,'same');
drawnow
ThresholdMask = (OrigImage - Threshold(FirstNumber:m+SecondNumber, FirstNumber:n+SecondNumber));
drawnow
% AdjustedThresholdMask = ThresholdMask*ThresholdAdjustmentFactor;
% PreThresholdedImage = AdjustedThresholdMask > 0;
%MT:  original line makes no sense:  multiplying by positive number will
%not change which points are above zero... so adjustment factor makes
%no adjustment.  changed to test for greater than
%MinimumThreshold*ThresholdAdjustmentFactor
PreThresholdedImage = ThresholdMask > MinimumThreshold*ThresholdAdjustmentFactor;
drawnow
ThresholdedImage = PreThresholdedImage;
ThresholdedImage(OrigImage <= MinimumThreshold) = 0;
%%% Holes in the ThresholdedImage image are filled in.
drawnow
ThresholdedImage = imfill(ThresholdedImage, 'holes');
%%% Identifies objects in the binary image.
PrelimLabelMatrixImage1 = bwlabel(ThresholdedImage);
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
drawnow
%%% The small objects are overwritten with zeros.
PrelimLabelMatrixImage2 = PrelimLabelMatrixImage1;
PrelimLabelMatrixImage2(AreasImage < MinSize) = 0;
%%% Relabels so that labels are consecutive. This is important for
%%% downstream modules (IdentifySec).
PrelimLabelMatrixImage2 = bwlabel(im2bw(PrelimLabelMatrixImage2,.1));
%%% The large objects are overwritten with zeros.
PrelimLabelMatrixImage3 = PrelimLabelMatrixImage2;
drawnow
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
fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
%%% Check whether that figure is open. This checks all the figure handles
%%% for one whose handle is equal to the figure number for this algorithm.
%%% Note: Everything between the "if" and "end" is not carried out if the
%%% user has closed the figure window, so do not do any important
%%% calculations here. Otherwise an error message will be produced if the
%%% user has closed the window but you have attempted to access data that
%%% was supposed to be produced by this part of the code.
if any(findobj == ThisModuleFigureNumber) == 1;
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
    ObjectOutlinesOnOrigImage = OrigImage;
    %%% Determines the grayscale intensity to use for the cell outlines.
    LineIntensity = max(OrigImage(:));
    ObjectOutlinesOnOrigImage(PrimaryObjectOutlines == 1) = LineIntensity;
    %%% The "drawnow" function executes any pending figure window-related
    %%% commands.  In general, Matlab does not update figure windows
    %%% until breaks between image analysis modules, or when a few select
    %%% commands are used. "figure" and "drawnow" are two of the commands
    %%% that allow Matlab to pause and carry out any pending figure window-
    %%% related commands (like zooming, or pressing timer pause or cancel
    %%% buttons or pressing a help button.)  If the drawnow command is not
    %%% used immediately prior to the figure(ThisModuleFigureNumber) line,
    %%% then immediately after the figure line executes, the other commands
    %%% that have been waiting are executed in the other windows.  Then,
    %%% when Matlab returns to this module and goes to the subplot line,
    %%% the figure which is active is not necessarily the correct one.
    %%% This results in strange things like the subplots appearing in the
    %%% timer window or in the wrong figure window, or in help dialog boxes.
    drawnow
    figure(ThisModuleFigureNumber);
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,2,1); imagesc(OrigImage);colormap(pink);
    title(['Input Image, Image Set # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the colored label
    %%% matrix image.
    subplot(2,2,2); imagesc(ColoredLabelMatrixImage); title(['Segmented ',ObjectName]);
    %%% A subplot of the figure window is set to display the prethresholded
    %%% image.
    subplot(2,2,3); imagesc(PreThresholdedImage);colormap(pink); title('Without applying minimum threshold');
    %%% A subplot of the figure window is set to display the inverted original
    %%% image with outlines drawn on top.
    subplot(2,2,4); imagesc(ObjectOutlinesOnOrigImage);colormap(pink); title([ObjectName, ' Outlines on Input Image']);
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

%%% Determines the filename of the image to be analyzed.
fieldname = ['dOTFilename', ImageName];
FileName = handles.(fieldname)(handles.Current.SetBeingAnalyzed);
%%% Saves the filename of the image to be analyzed.
fieldname = ['dOTFilename', ObjectName];
handles.(fieldname)(handles.Current.SetBeingAnalyzed) = FileName;