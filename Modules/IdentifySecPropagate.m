function handles = AlgIdentifySecPropagate9(handles)

%%% Reads the current algorithm number, since this is needed to find 
%%% the variable values that the user entered.
CurrentAlgorithm = handles.currentalgorithm;
CurrentAlgorithmNum = str2num(handles.currentalgorithm);

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow

%textVAR01 = What did you call the images you want to process? 
%defaultVAR01 = OrigGreen
fieldname = ['Vvariable',CurrentAlgorithm,'_01'];
ImageName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,1});
%textVAR02 = What did you call the objects that will be used to mark the centers of these objects?
%defaultVAR02 = Nuclei
fieldname = ['Vvariable',CurrentAlgorithm,'_02'];
PrimaryObjectName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,2});
%textVAR03 = What do you want to call the objects identified by this algorithm?
%defaultVAR03 = Cells
fieldname = ['Vvariable',CurrentAlgorithm,'_03'];
SecondaryObjectName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,3});
%%% The following is just an extra line of text describing variable 3.
%textVAR04 = (Note: Data will be produced based on this name, e.g. dMCTotalAreaCells)

%textVAR05 = Enter the threshold [0 = automatically calculate] (Positive number, Max = 1):
%defaultVAR05 = 0
fieldname = ['Vvariable',CurrentAlgorithm,'_05'];
Threshold = str2num(char(handles.Settings.Vvariable{CurrentAlgorithmNum,5}));
%textVAR06 = If auto threshold, enter an adjustment factor (Positive number, 1 = no adjustment):
%defaultVAR06 = 1
fieldname = ['Vvariable',CurrentAlgorithm,'_06'];
ThresholdAdjustmentFactor = str2num(char(handles.Settings.Vvariable{CurrentAlgorithmNum,6}));

%textVAR07 = Regularization factor (0 to infinity). Larger=distance,0=intensity
%defaultVAR07 = 0.05
fieldname = ['Vvariable',CurrentAlgorithm,'_07'];
RegularizationFactor = str2num(char(handles.Settings.Vvariable{CurrentAlgorithmNum,7}));

%textVAR08 = To save object outlines as an image, enter text to append to the name 
%defaultVAR08 = N
fieldname = ['Vvariable',CurrentAlgorithm,'_08'];
SaveObjectOutlines = char(handles.Settings.Vvariable{CurrentAlgorithmNum,8});
%textVAR09 = To save colored object blocks as an image, enter text to append to the name 
%defaultVAR09 = N
fieldname = ['Vvariable',CurrentAlgorithm,'_09'];
SaveColoredObjects = char(handles.Settings.Vvariable{CurrentAlgorithmNum,9});
%textVAR10 = Otherwise, leave as "N". To save or display other images, press Help button
%textVAR11 = If saving images, what file format do you want to use? Do not include a period.
%defaultVAR11 = tif
fieldname = ['Vvariable',CurrentAlgorithm,'_11'];
FileFormat = char(handles.Settings.Vvariable{CurrentAlgorithmNum,11});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Checks whether the file format the user entered is readable by Matlab.
IsFormat = imformats(FileFormat);
if isempty(IsFormat) == 1
    error('The image file type entered in the Identify Secondary Propagate module is not recognized by Matlab. Or, you may have entered a period in the box. For a list of recognizable image file formats, type "imformats" (no quotes) at the command line in Matlab.','Error')
end
%%% Read (open) the image you want to analyze and assign it to a variable,
%%% "OrigImageToBeAnalyzed".
fieldname = ['dOT', ImageName];
%%% Check whether the image to be analyzed exists in the handles structure.
if isfield(handles, fieldname) == 0
    %%% If the image is not there, an error message is produced.  The error
    %%% is not displayed: The error function halts the current function and
    %%% returns control to the calling function (the analyze all images
    %%% button callback.)  That callback recognizes that an error was
    %%% produced because of its try/catch loop and breaks out of the image
    %%% analysis loop without attempting further modules.
    error(['Image processing was canceled because the Identify Secondary Propagate module could not find the input image.  It was supposed to be named ', ImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
OrigImageToBeAnalyzed = handles.(fieldname);
        % figure, imshow(OrigImageToBeAnalyzed), title('OrigImageToBeAnalyzed')
%%% Update the handles structure.
%%% Removed for parallel: guidata(gcbo, handles);

%%% Checks that the original image is two-dimensional (i.e. not a color
%%% image), which would disrupt several of the image functions.
if ndims(OrigImageToBeAnalyzed) ~= 2
    error('Image processing was canceled because the Identify Secondary Propagate module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
end

%%% Retrieves the preliminary label matrix image that contains the primary
%%% segmented objects which have only been edited to discard objects 
%%% that are smaller than a certain size.  This image
%%% will be used as markers to segment the secondary objects with this
%%% algorithm.  Checks first to see whether the appropriate image exists.
fieldname = ['dOTPrelimSmallSegmented',PrimaryObjectName];
%%% Checks whether the image exists in the handles structure.
    if isfield(handles, fieldname) == 0
    error(['Image processing has been canceled. Prior to running the Identify Secondary Propagate module, you must have previously run an algorithm that generates an image with the preliminary primary objects identified.  You specified in the Identify Secondary Propagate module that the primary objects were named ', PrimaryObjectName, ' as a result of the previous algorithm, which should have produced an image called ', fieldname, ' in the handles structure.  The Identify Secondary Propagate module cannot locate this image.']);
    end
PrelimPrimaryLabelMatrixImage = handles.(fieldname);
     % figure, imshow(PrelimPrimaryLabelMatrixImage), title('PrelimPrimaryLabelMatrixImage')
        
%%% Retrieve the label matrix image that contains the edited primary
%%% segmented objects which will be used to weed out which objects are
%%% real - not on the edges and not below or above the specified size
%%% limits. Checks first to see whether the appropriate image exists.
fieldname = ['dOTSegmented',PrimaryObjectName];
%%% Checks whether the image exists in the handles structure.
    if isfield(handles, fieldname) == 0
    error(['Image processing has been canceled. Prior to running the Identify Secondary Propagate module, you must have previously run an algorithm that generates an image with the preliminary primary objects identified.  You specified in the Identify Secondary Propagate module that the primary objects were named ', PrimaryObjectName, ' as a result of the previous algorithm, which should have produced an image called ', fieldname, ' in the handles structure.  The Identify Secondary Propagate module cannot locate this image.']);
    end
EditedPrimaryLabelMatrixImage = handles.(fieldname);
       % figure, imshow(EditedPrimaryLabelMatrixImage), title('EditedPrimaryLabelMatrixImage')
       % save temp2 EditedPrimaryLabelMatrixImage
       
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
    error('Image processing was canceled because you have entered one or more spaces in the box of text to append to the object outlines image name in the Identify Secondary Propagate module.  If you do not want to save the object outlines image to the hard drive, type "N" into the appropriate box.')
end
%%% Check whether the new image name is going to result in overwriting the
%%% original file.
B = strcmp(upper(CharFileName), upper(NewImageNameSaveObjectOutlines));
if B == 1
    error('Image processing was canceled because you have not entered text to append to the object outlines image name in the Identify Secondary Propagate module.  If you do not want to save the object outlines image to the hard drive, type "N" into the appropriate box.')
end

%%% Repeat the above for the other image to be saved: 
NewImageNameSaveColoredObjects = [BareFileName,SaveColoredObjects,'.',FileFormat];
A = isspace(SaveColoredObjects);
if any(A) == 1
    error('Image processing was canceled because you have entered one or more spaces in the box of text to append to the colored objects image name in the Identify Secondary Propagate module.  If you do not want to save the colored objects image to the hard drive, type "N" into the appropriate box.')
end
B = strcmp(upper(CharFileName), upper(NewImageNameSaveColoredObjects));
if B == 1
    error('Image processing was canceled because you have not entered text to append to the colored objects image name in the Identify Secondary Propagate module.  If you do not want to save the colored objects image to the hard drive, type "N" into the appropriate box.')
end

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% This algorithm assumes that you have a set of primary objects contained
%%% within a group of secondary objects, and you want to identify the
%%% outlines of the secondary objects.  The primary objects
%%% are assumed to be completely within secondary objects (e.g. nuclei
%%% within cells).  The algorithm is especially good at determining the
%%% dividing lines between clustered cells.

%%% STEP 1: The distinction between objects and background is determined
%%% using the user-specified threshold.
%%% Determine the threshold to use. 
if Threshold == 0
    Threshold = graythresh(OrigImageToBeAnalyzed);
    %%% Adjust the threshold by a correction factor.  
    Threshold = Threshold*ThresholdAdjustmentFactor;
end
%%% Threshold the original image.
ThresholdedOrigImage = im2bw(OrigImageToBeAnalyzed, Threshold);
    % figure, imshow(ThresholdedOrigImage, []), title('ThresholdedOrigImage')
    % imwrite(ThresholdedOrigImage, [BareFileName,'TOI','.',FileFormat], FileFormat);

%%% STEP 2: Starting from the identified primary objects, the secondary
%%% objects are identified using the propagate function, written by Thouis R.
%%% Jones.
%%% Calls the function "AlgIdentifySecPropagateSubfunction.mexmac", which consists
%%% of C code that has been compiled to run quickly within Matlab.
PropagatedImage = AlgIdentifySecPropagateSubfunction(PrelimPrimaryLabelMatrixImage,OrigImageToBeAnalyzed,ThresholdedOrigImage,RegularizationFactor);
    % figure, imshow(PropagatedImage, []), title('PropagatedImage')
    % imwrite(PropagatedImage, [BareFileName,'PI','.',FileFormat], FileFormat);

%%% STEP 3: Remove objects that are not desired, edited objects.  The
%%% edited primary object image is used rather than the preliminary one, so
%%% that objects whose nuclei are on the edge of the image and who are
%%% larger or smaller than the specified size are discarded.
%%% Converts the EditedPrimaryBinaryImage to binary.
EditedPrimaryBinaryImage = im2bw(EditedPrimaryLabelMatrixImage);
%%% Finds the locations and labels for different regions.
area_locations2 = find(PropagatedImage);
area_labels2 = PropagatedImage(area_locations2);
%%% Creates a sparse matrix with column as label and row as location,
%%% with the value of the center at (I,J) if location I has label J.
%%% Taking the maximum of this matrix gives the largest valued center
%%% overlapping a particular label.  Tacking on a zero and pushing
%%% labels through the resulting map removes any background regions.
map2 = [0 full(max(sparse(area_locations2, area_labels2, EditedPrimaryLabelMatrixImage(area_locations2))))];
HoleyPrelimLabelMatrixImage = map2(PropagatedImage + 1);
    % figure, imshow(HoleyPrelimLabelMatrixImage, []), title('HoleyPrelimLabelMatrixImage')
    % imwrite(HoleyPrelimLabelMatrixImage, [BareFileName,'HPLMI','.',FileFormat], FileFormat);
%%% Fills in holes in the HoleyPrelimLabelMatrixImage image.
%%% Filters the image for maxima (Plus sign neighborhood, ignoring zeros).
MaximaImage = ordfilt2(HoleyPrelimLabelMatrixImage, 5, [0 1 0; 1 1 1 ; 0 1 0]);
%%% This is a pain.  On sparse matrices, min returns zero almost always
%%% (because the matrices are mostly zero, of course).  So we need to invert
%%% the labels so we can use max to find the minimum adjacent label as well,
%%% below.  This also takes care of boundaries, which otherwise return zero
%%% in the min filter.
LargestLabelImage = max(HoleyPrelimLabelMatrixImage(:));
TempImage = HoleyPrelimLabelMatrixImage;
TempImage(HoleyPrelimLabelMatrixImage > 0) = LargestLabelImage - TempImage(HoleyPrelimLabelMatrixImage > 0) + 1;
%%% Filters the image for minima (Plus sign neighborhood).
MinimaImage = ordfilt2(TempImage, 5, [0 1 0; 1 1 1 ; 0 1 0]);
%%% Marks and labels the zero regions.
ZeroRegionImage = bwlabel(HoleyPrelimLabelMatrixImage==0, 4);
%%% Uses sparse matrices to find the minimum and maximum label adjacent
%%% to each zero-region.
ZeroLocations = find(ZeroRegionImage);
ZeroLabels = ZeroRegionImage(ZeroLocations);
MinByRegion = full(max(sparse(ZeroLocations, ZeroLabels, MinimaImage(ZeroLocations))));
%%% remaps to correct order (see above).
MinByRegion = LargestLabelImage - MinByRegion + 1;
MaxByRegion = full(max(sparse(ZeroLocations, ZeroLabels, MaximaImage(ZeroLocations))));
%%% Anywhere the min and max are the same is a region surrounded by a
%%% single value.
Surrounded = (MinByRegion == MaxByRegion);
%%% Creates a map that turns a labelled zero-region into the surrounding
%%% label if it's surrounded, and into zero if it's not surrounded.
%%% (Pad by a leading zero so 0 maps to 0 when 1 is added.)
Remap = [ 0 (Surrounded .* MinByRegion)];
ZeroRegionImage = Remap(ZeroRegionImage + 1);
%%% Now all surrounded zeroregions should have been remapped to their
%%% new value, or zero if not surrounded.
PrelimLabelMatrixImage = max(HoleyPrelimLabelMatrixImage, ZeroRegionImage);

%%% STEP 4: The final objects are relabeled so that their numbers
%%% correspond to the numbers used for nuclei.
%%% For each object, one label and one label location is acquired and
%%% stored.
[LabelsUsed,LabelLocations,Irrelevant] = unique(EditedPrimaryLabelMatrixImage);
%%% The +1 increment accounts for the fact that there are zeros in the
%%% image, while the LabelsUsed starts at 1.
LabelsUsed(PrelimLabelMatrixImage(LabelLocations(2:end))+1) = EditedPrimaryLabelMatrixImage(LabelLocations(2:end));
FinalLabelMatrixImage = LabelsUsed(PrelimLabelMatrixImage+1);
    % figure, imshow(FinalLabelMatrixImagePre, []), title('FinalLabelMatrixImagePre')
    % imwrite(FinalLabelMatrixImagePre, [BareFileName,'FLMIP','.',FileFormat], FileFormat);
    
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

%%% Calculate the ColoredLabelMatrixImage for displaying in the figure
%%% window in subplot(2,2,2).
%%% Note that the label2rgb function doesn't work when there are no objects
%%% in the label matrix image, so there is an "if".
if sum(sum(FinalLabelMatrixImage)) >= 1
    ColoredLabelMatrixImage2 = label2rgb(FinalLabelMatrixImage,'jet', 'k', 'shuffle');
        % figure, imshow(ColoredLabelMatrixImage2, []), title('ColoredLabelMatrixImage2')
        % imwrite(ColoredLabelMatrixImage2, [BareFileName,'CLMI2','.',FileFormat], FileFormat);
else  ColoredLabelMatrixImage2 = FinalLabelMatrixImage;
end
        
%%% Calculate OutlinesOnOriginalImage for displaying in the figure
%%% window in subplot(2,2,3).        
    %%% Note: these outlines are not perfectly accurate; for some reason it
    %%% produces more objects than in the original image.  But it is OK for
    %%% display purposes.
%%% Maximum filters the image with a 3x3 neighborhood.
MaxFilteredImage = ordfilt2(FinalLabelMatrixImage,9,ones(3,3),'symmetric');
    % figure, imshow(MaxFilteredImage, []), title('MaxFilteredImage')
%%% Determines the outlines.
IntensityOutlines = FinalLabelMatrixImage - MaxFilteredImage;
    % figure, imshow(IntensityOutlines, []), title('IntensityOutlines')
%%% Converts to logical.
warning off MATLAB:conversionToLogical
LogicalOutlines = logical(IntensityOutlines);
warning on MATLAB:conversionToLogical
    % figure, imshow(LogicalOutlines, []), title('LogicalOutlines')
%%% Determines the grayscale intensity to use for the cell outlines.
LineIntensity = max(OrigImageToBeAnalyzed(:));
%%% Overlays the outlines on the original image.
OutlinesOnOriginalImage = OrigImageToBeAnalyzed;
OutlinesOnOriginalImage(LogicalOutlines) = LineIntensity;
    % figure, imshow(OutlinesOnOriginalImage, []), title('OutlinesOnOriginalImage')
    % imwrite(OutlinesOnOriginalImage, [BareFileName,'OOOI','.',FileFormat], FileFormat);

%%% Calculate BothOutlinesOnOriginalImage for displaying in the figure
%%% window in subplot(2,2,4).
%%% Create the structuring element that will be used for dilation.
StructuringElement = strel('square',3);
%%% Dilate the Primary Binary Image by one pixel (8 neighborhood).
DilatedPrimaryBinaryImage = imdilate(EditedPrimaryBinaryImage, StructuringElement);
        % figure, imshow(DilatedActualObjectsBinaryImage, []), title('DilatedActualObjectsBinaryImage')
        % imwrite(DilatedActualObjectsBinaryImage, [BareFileName,'DAOBI','.',FileFormat], FileFormat);
%%% Subtract the PrelimPrimaryBinaryImage from the DilatedPrimaryBinaryImage,
%%% which leaves the PrimaryObjectOutlines.
PrimaryObjectOutlines = DilatedPrimaryBinaryImage - EditedPrimaryBinaryImage;
        % figure, imshow(ActualObjectOutlines, []), title('ActualObjectOutlines')
        % imwrite(ActualObjectOutlines, [BareFileName,'AOO','.',FileFormat], FileFormat);
BothOutlinesOnOriginalImage = OutlinesOnOriginalImage;
BothOutlinesOnOriginalImage(PrimaryObjectOutlines == 1) = LineIntensity;
        % figure, imshow(BothOutlinesOnOriginalImage, []), title('BothOutlinesOnOriginalImage')
        % imwrite(BothOutlinesOnOriginalImage, [BareFileName,'BOOOI','.',FileFormat], FileFormat);

%%% For display only: overlaying the nuclear markers on top of the
%%% color label matrix image to see how it looks: 
% OverlaidColorLabelMatrixImage = ColoredLabelMatrixImage2;
% OverlaidColorLabelMatrixImage(PrelimPrimaryBinaryImage == 1) = 0;
% figure, imshow(OverlaidColorLabelMatrixImage, []), title('OverlaidColorLabelMatrixImage')
% imwrite(OverlaidColorLabelMatrix, [BareFileName,'OCLMI','.',FileFormat], FileFormat);
        
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
    %%% Activates the appropriate figure window.
    figure(ThisAlgFigureNumber);
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,2,1); imagesc(OrigImageToBeAnalyzed);colormap(gray);
    title(['Input Image, Image Set # ',num2str(handles.setbeinganalyzed)]);
    %%% A subplot of the figure window is set to display the colored label
    %%% matrix image.
    subplot(2,2,2); imagesc(ColoredLabelMatrixImage2); title(['Segmented ',SecondaryObjectName]);
    %%% A subplot of the figure window is set to display the original image
    %%% with secondary object outlines drawn on top.
    subplot(2,2,3); imagesc(OutlinesOnOriginalImage); colormap(gray); title([SecondaryObjectName, ' Outlines on Input Image']);
    %%% A subplot of the figure window is set to display the original
    %%% image with outlines drawn for both the primary and secondary
    %%% objects.
    subplot(2,2,4); imagesc(BothOutlinesOnOriginalImage); colormap(gray); title(['Outlines of ', PrimaryObjectName, ' and ', SecondaryObjectName, ' on Input Image']);
end
%%% Executes pending figure-related commands so that the results are
%%% displayed.
drawnow

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% The final, segmented label matrix image of secondary objects is saved to the
%%% handles structure so it can be used by subsequent algorithms.
fieldname = ['dOTSegmented',SecondaryObjectName];
handles.(fieldname) = FinalLabelMatrixImage;
%%% Removed for parallel: guidata(gcbo, handles);

%%% Save the filename of the image to be analyzed.
fieldname = ['dOTFilename', SecondaryObjectName];
handles.(fieldname)(handles.setbeinganalyzed) = FileName;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE PROCESSED IMAGE TO HARD DRIVE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Determine whether the user wanted to save the image of object outlines
%%% by comparing their entry "SaveObjectOutlines" with "N" (after
%%% converting SaveObjectOutlines to uppercase).
if strcmp(upper(SaveObjectOutlines),'N') ~= 1
%%% Save the image to the hard drive.    
imwrite(LogicalOutlines, NewImageNameSaveObjectOutlines, FileFormat);
end
%%% Same for the SaveColoredObjects image.
if strcmp(upper(SaveColoredObjects),'N') ~= 1
%%% Save the image to the hard drive.    
imwrite(ColoredLabelMatrixImage2, NewImageNameSaveColoredObjects, FileFormat);
end

drawnow

%%%%%%%%%%%
%%% HELP %%%
%%%%%%%%%%%

%%%%% Help for the Identify Secondary Propagate module: 
%%%%% .
%%%%% SETTINGS:
%%%%% Threshold: this setting affects the stringency of
%%%%% object outlines that border the background.  It does not affect the
%%%%% dividing lines between clumped objects. A higher number will result
%%%%% in smaller objects (more stringent). A lower number will result in
%%%%% large objects (less stringent). 
%%%%% Regularization factor: This algorithm takes two factors into account
%%%%% when deciding where to draw the dividing line between two touching
%%%%% secondary objects: the distance to the nearest primary object, and
%%%%% the intensity of the secondary object image.  The regularization
%%%%% factor controls the balance between these two considerations: A value
%%%%% of zero means that the distance to the nearest primary object is
%%%%% ignored and the decision is made entirely on the intensity gradient
%%%%% between the two competing primary objects. Larger values weight the
%%%%% distance between the two values more and more heavily.  The
%%%%% regularization factor can be infinitely large, but around 10 or so,
%%%%% the intensity image is almost completely ignored and the dividing
%%%%% line will simply be halfway between the two competing primary
%%%%% objects.
%%%%% 
%%%%% Note: Primary segmenters produce two output images that are used by
%%%%% this module.  The dOTSegmented image contains the final, edited
%%%%% primary objects (i.e. objects at the border and those that are too
%%%%% small or large have been excluded).  The dOTPrelimSmallSegmented
%%%%% image is the same except that the objects at the border and the large
%%%%% objects have been included.  These extra objects are used to perform
%%%%% the identification of secondary object outlines, since they are
%%%%% probably real objects (even if we don't want to measure them).  Small
%%%%% objects are not used at this stage because they are more likely to be
%%%%% artifactual, and so they therefore should not "claim" any secondary
%%%%% object pixels.
%%%%% 
%%%%% TECHNICAL DESCRIPTION OF THE ALGORITHM: 
%%%%% Propagate labels from LABELS_IN to LABELS_OUT, steered by IMAGE and 
%%%%% limited to MASK.  MASK should be a logical array.  LAMBDA is a 
%%%%% regularization parameter, larger being closer to Euclidean distance in
%%%%% the image plane, and zero being entirely controlled by IMAGE. 
%%%%% Propagation of labels is by shortest path to a nonzero label in
%%%%% LABELS_IN.  Distance is the sum of absolute differences in the image
%%%%% in a 3x3 neighborhood, combined with LAMBDA via sqrt(differences^2 +
%%%%% LAMBDA^2).  Note that there is no separation between adjacent areas
%%%%% with different labels (as there would be using, e.g.,
%%%%% watershed).  Such boundaries must be added in a postprocess.
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