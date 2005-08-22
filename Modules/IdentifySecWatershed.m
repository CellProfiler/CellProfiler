function handles = IdentifySecWatershed(handles)

% Help for the Identify Secondary Watershed module:
% Category: Object Identification and Modification
%
% This module identifies secondary objects based on a previous
% module's identification of primary objects.  Each primary object is
% assumed to be completely within a secondary object (e.g. nuclei
% within cells stained for actin). The dividing lines between objects
% are determined by looking for dim lines between objects. It would
% not be difficult to write a module that looks for bright lines
% between objects, based on this one.
%
% Settings:
%
% Threshold: The threshold affects the stringency of the lines between
% the objects and the background. You may enter an absolute number
% between 0 and 1 for the threshold (use 'Show pixel data' to see the
% pixel intensities for your images in the appropriate range of 0 to
% 1), or you may have it calculated for each image individually by
% typing 0.  There are advantages either way.  An absolute number
% treats every image identically, but an automatically calculated
% threshold is more realistic/accurate, though occasionally subject to
% artifacts.  The threshold which is used for each image is recorded
% as a measurement in the output file, so if you find unusual
% measurements from one of your images, you might check whether the
% automatically calculated threshold was unusually high or low
% compared to the remaining images.  When an automatic threshold is
% selected, it may consistently be too stringent or too lenient, so an
% adjustment factor can be entered as well. The number 1 means no
% adjustment, 0 to 1 makes the threshold more lenient and greater than
% 1 (e.g. 1.3) makes the threshold more stringent.
%
% Perhaps outdated note about the threshold adjustment factor: A
% higher number will result in smaller objects (more stringent). A
% lower number will result in large objects (less stringent), but at a
% certain point, depending on the particular image, the objects will
% become huge and the processing will take a really long time.  To
% determine whether the number is too low, you can just test it (of
% course), but a shortcut would be to alter the code (m-file) for this
% module to display the image called InvertedThresholdedOrigImage. The
% resulting image that pops up during processing should not have lots
% of speckles - this adds to the processing time. Rather, there should
% be rather large regions of black where the cells are located.
%
% Note: Primary segmenters produce two output images that are used by
% this module.  The Pipeline.Segmented image contains the final, edited
% primary objects (i.e. objects at the border and those that are too
% small or large have been excluded).  The SmallRemovedSegmented
% image is the same except that the objects at the border and the
% large objects have been included.  These extra objects are used to
% perform the identification of secondary object outlines, since they
% are probably real objects (even if we don't want to measure them).
% Small objects are not used at this stage because they are more
% likely to be artifactual, and so they therefore should not "claim"
% any secondary object pixels.
%
% SAVING IMAGES: In addition to the object outlines and the
% pseudo-colored object images that can be saved using the
% instructions in the main CellProfiler window for this module, this
% module produces a grayscale image where each object is a different
% intensity, which you can save using the Save Images module using the
% name: Segmented + whatever you called the objects (e.g.
% SegmentedCells).
%
%    Additional image(s) are calculated by this module and can be 
% saved by altering the code for the module (see the SaveImages module
% help for instructions).
%
% See also IDENTIFYSECPROPAGATE, IDENTIFYSECDISTANCE.

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
%
% $Revision$

% PROGRAMMING NOTE
% HELP:
% The first unbroken block of lines will be extracted as help by
% CellProfiler's 'Help for this analysis module' button as well as Matlab's
% built in 'help' and 'doc' functions at the command line. It will also be
% used to automatically generate a manual page for the module. An example
% image demonstrating the function of the module can also be saved in tif
% format, using the same name as the module, and it will automatically be
% included in the manual page as well.  Follow the convention of: purpose
% of the module, description of the variables and acceptable range for
% each, how it works (technical description), info on which images can be
% saved, and See also CAPITALLETTEROTHERMODULES. The license/author
% information should be separated from the help lines with a blank line so
% that it does not show up in the help displays.  Do not change the
% programming notes in any modules! These are standard across all modules
% for maintenance purposes, so anything module-specific should be kept
% separate.

% PROGRAMMING NOTE
% DRAWNOW:
% The 'drawnow' function allows figure windows to be updated and
% buttons to be pushed (like the pause, cancel, help, and view
% buttons).  The 'drawnow' function is sprinkled throughout the code
% so there are plenty of breaks where the figure windows/buttons can
% be interacted with.  This does theoretically slow the computation
% somewhat, so it might be reasonable to remove most of these lines
% when running jobs on a cluster where speed is important.
drawnow

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%

% PROGRAMMING NOTE
% VARIABLE BOXES AND TEXT:
% The '%textVAR' lines contain the variable descriptions which are
% displayed in the CellProfiler main window next to each variable box.
% This text will wrap appropriately so it can be as long as desired.
% The '%defaultVAR' lines contain the default values which are
% displayed in the variable boxes when the user loads the module.
% The line of code after the textVAR and defaultVAR extracts the value
% that the user has entered from the handles structure and saves it as
% a variable in the workspace of this module with a descriptive
% name. The syntax is important for the %textVAR and %defaultVAR
% lines: be sure there is a space before and after the equals sign and
% also that the capitalization is as shown.
% CellProfiler uses VariableRevisionNumbers to help programmers notify
% users when something significant has changed about the variables.
% For example, if you have switched the position of two variables,
% loading a pipeline made with the old version of the module will not
% behave as expected when using the new version of the module, because
% the settings (variables) will be mixed up. The line should use this
% syntax, with a two digit number for the VariableRevisionNumber:
% '%%%VariableRevisionNumber = 01'  If the module does not have this
% line, the VariableRevisionNumber is assumed to be 00.  This number
% need only be incremented when a change made to the modules will affect
% a user's previously saved settings. There is a revision number at
% the end of the license info at the top of the m-file for revisions
% that do not affect the user's previously saved settings files.

%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

%textVAR01 = What did you call the images you want to process?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What did you call the objects that will be used to mark the centers of these objects?
%infotypeVAR02 = objectgroup
PrimaryObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 = What do you want to call the objects identified by this module? (Note: Data will be produced based on this name, e.g. ObjectTotalAreaCells)
%infotypeVAR03 = objectgroup indep
%defaultVAR03 = Cells
SecondaryObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = Enter the threshold (Positive number, Max = 1):
%choiceVAR04 = Automatic
Threshold = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu custom

%textVAR05 = If auto threshold, enter an adjustment factor (Positive number, 1 = no adjustment):
%defaultVAR05 = 0.75
ThresholdAdjustmentFactor = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,5}));

%textVAR06 = Enter the minimum allowable threshold (Range = 0 to 1; this prevents an unreasonably low threshold from counting noise as objects when there are no bright objects in the field of view). This is intended for use with automatic thresholding, but will override an absolute threshold entered above:
%defaultVAR06 = 0
MinimumThreshold = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = What do you want to call the image of the outlines of the objects?
%choiceVAR07 = Do not save
%choiceVAR07 = OutlinedNuclei
SaveOutlined = char(handles.Settings.VariableValues{CurrentModuleNum,7}); 
%inputtypeVAR07 = popupmenu custom

%textVAR08 =  What do you want to call the labeled matrix image?
%infotypeVAR08 = imagegroup indep
%choiceVAR08 = Do not save
%choiceVAR08 = LabeledNuclei
SaveColored = char(handles.Settings.VariableValues{CurrentModuleNum,8}); 
%inputtypeVAR08 = popupmenu custom

%textVAR09 = Do you want to save the labeled matrix image in RGB or grayscale?
%choiceVAR09 = RGB
%choiceVAR09 = Grayscale
SaveMode = char(handles.Settings.VariableValues{CurrentModuleNum,9}); 
%inputtypeVAR09 = popupmenu

%%%VariableRevisionNumber = 3

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a
%%% variable.
fieldname = ['', ImageName];
%%% Checks whether the image to be analyzed exists in the handles structure.
if isfield(handles.Pipeline, fieldname)==0,
    %%% If the image is not there, an error message is produced.  The error
    %%% is not displayed: The error function halts the current function and
    %%% returns control to the calling function (the analyze all images
    %%% button callback.)  That callback recognizes that an error was
    %%% produced because of its try/catch loop and breaks out of the image
    %%% analysis loop without attempting further modules.
    error(['Image processing was canceled because the Identify Secondary Watershed module could not find the input image.  It was supposed to be named ', ImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
OrigImage = handles.Pipeline.(fieldname);

%%% Checks that the original image is two-dimensional (i.e. not a color
%%% image), which would disrupt several of the image functions.
if ndims(OrigImage) ~= 2
    error('Image processing was canceled because the Identify Secondary Watershed module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
end

%%% Retrieves the preliminary label matrix image that contains the primary
%%% segmented objects which have only been edited to discard objects
%%% that are smaller than a certain size.  This image
%%% will be used as markers to segment the secondary objects with this
%%% module.  Checks first to see whether the appropriate image exists.
fieldname = ['SmallRemovedSegmented', PrimaryObjectName];
%%% Checks whether the image exists in the handles structure.
if isfield(handles.Pipeline, fieldname)==0,
    error(['Image processing has been canceled. Prior to running the Identify Secondary Watershed module, you must have previously run a module that generates an image with the preliminary primary objects identified.  You specified in the Identify Secondary Watershed module that the primary objects were named ', PrimaryObjectName, ' as a result of the previous module, which should have produced an image called ', fieldname, ' in the handles structure.  The Identify Secondary Watershed module cannot locate this image.']);
    end
PrelimPrimaryLabelMatrixImage = handles.Pipeline.(fieldname);

%%% Retrieves the label matrix image that contains the edited primary
%%% segmented objects which will be used to weed out which objects are
%%% real - not on the edges and not below or above the specified size
%%% limits. Checks first to see whether the appropriate image exists.
fieldname = ['Segmented', PrimaryObjectName];
%%% Checks whether the image exists in the handles structure.
if isfield(handles.Pipeline, fieldname)==0,
    error(['Image processing has been canceled. Prior to running the Identify Secondary Watershed module, you must have previously run a module that generates an image with the preliminary primary objects identified.  You specified in the Identify Secondary Watershed module that the primary objects were named ', PrimaryObjectName, ' as a result of the previous module, which should have produced an image called ', fieldname, ' in the handles structure.  The Identify Secondary Watershed module cannot locate this image.']);
    end
EditedPrimaryLabelMatrixImage = handles.Pipeline.(fieldname);

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow

% PROGRAMMING NOTE
% TO TEMPORARILY SHOW IMAGES DURING DEBUGGING:
% figure, imshow(BlurredImage, []), title('BlurredImage')
% TO TEMPORARILY SAVE IMAGES DURING DEBUGGING:
% imwrite(BlurredImage, FileName, FileFormat);
% Note that you may have to alter the format of the image before
% saving.  If the image is not saved correctly, for example, try
% adding the uint8 command:
% imwrite(uint8(BlurredImage), FileName, FileFormat);
% To routinely save images produced by this module, see the help in
% the SaveImages module.

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
if strcmp(Threshold,'Automatic')
    Threshold = CPgraythresh(OrigImage,handles,ImageName);
    %%% Replaced the following line to accomodate calculating the
    %%% threshold for images that have been masked.
    %    Threshold = CPgraythresh(OrigImage);
    %%% Adjusts the threshold by a correction factor.
    Threshold = Threshold*ThresholdAdjustmentFactor;
else
    Threshold=str2double(Threshold);
end
MinimumThreshold = str2num(MinimumThreshold);
Threshold = max(MinimumThreshold,Threshold);

%%% Thresholds the original image.
ThresholdedOrigImage = im2bw(OrigImage, Threshold);
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
PrelimPrimaryBinaryImage = im2bw(PrelimPrimaryLabelMatrixImage,.5);
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
I1 = imfilter(OrigImage, filter1);
I2 = imfilter(OrigImage, filter2);
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
BlackWatershedLinesPre2 = im2bw(BlackWatershedLinesPre,.5);
BlackWatershedLines = bwlabel(BlackWatershedLinesPre2);

%%% STEP 6: Identify and extract the secondary objects, using the watershed
%%% lines.
drawnow
%%% The BlackWatershedLines image is a label matrix where the watershed
%%% lines = 0 and each distinct object is assigned a number starting at 1.
%%% This image is converted to a binary image where all the objects = 1.
SecondaryObjects1 = im2bw(BlackWatershedLines,.5);
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
%%% The module has now produced a binary image of actual secondary
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
InvertedOrigImage = imcomplement(OrigImage);
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
SecondWatershedPre2 = im2bw(SecondWatershedPre,.5);
SecondWatershed = bwlabel(SecondWatershedPre2);
drawnow

%%% STEP 10: As in step 7, remove objects that are actually background
%%% objects.  See step 7 for description. This time, the edited primary object image is
%%% used rather than the preliminary one, so that objects whose nuclei are
%%% on the edge of the image and who are larger or smaller than the
%%% specified size are discarded.

%%% Converts the EditedPrimaryBinaryImage to binary.
EditedPrimaryBinaryImage = im2bw(EditedPrimaryLabelMatrixImage,.5);
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

% PROGRAMMING NOTE
% DISPLAYING RESULTS:
% Some calculations produce images that are used only for display or
% for saving to the hard drive, and are not used by downstream
% modules. To speed processing, these calculations are omitted if the
% figure window is closed and the user does not want to save the
% images.

fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
if any(findobj == ThisModuleFigureNumber) == 1 | strncmpi(SaveColored,'Y',1) == 1 | strncmpi(SaveOutlined,'Y',1) == 1
    %%% Calculates the ColoredLabelMatrixImage for displaying in the figure
    %%% window in subplot(2,2,2).
    %%% Note that the label2rgb function doesn't work when there are no objects
    %%% in the label matrix image, so there is an "if".
    if sum(sum(FinalLabelMatrixImage)) >= 1
        ColoredLabelMatrixImage = label2rgb(FinalLabelMatrixImage,'jet', 'k', 'shuffle');
    else  ColoredLabelMatrixImage = FinalLabelMatrixImage;
    end
    %%% Calculates ObjectOutlinesOnOrigImage for displaying in the figure
    %%% window in subplot(2,2,3).
    %%% Calculates the final watershed lines for the secondary objects.
    %%% Creates the structuring element that will be used for dilation.
    StructuringElement = strel('square',3);
    %%% Dilates the FinalBinaryImage by one pixel (8 neighborhood).
    DilatedFinalSecondaryImage = imdilate(FinalBinaryImage, StructuringElement);
    %%% Subtracts the FinalBinaryImage from the DilatedPrimaryBinaryImage,
    %%% which leaves the SecondaryObjectOutlines.
    SecondaryObjectOutlines = DilatedFinalSecondaryImage - FinalBinaryImage;
    %%% Determines the grayscale intensity to use for the cell outlines.
    LineIntensity = max(OrigImage(:));
    %%% Overlays the secondary object lines on the original image.
    ObjectOutlinesOnOrigImage = OrigImage;
    ObjectOutlinesOnOrigImage(SecondaryObjectOutlines == 1) = LineIntensity;
    %%% Calculates BothOutlinesOnOrigImage for displaying in the figure
    %%% window in subplot(2,2,4).
    BothOutlinesOnOrigImage = ObjectOutlinesOnOrigImage;
    BothOutlinesOnOrigImage(PrimaryObjectOutlines == 1) = LineIntensity;
% PROGRAMMING NOTE
% DRAWNOW BEFORE FIGURE COMMAND:
% The "drawnow" function executes any pending figure window-related
% commands.  In general, Matlab does not update figure windows until
% breaks between image analysis modules, or when a few select commands
% are used. "figure" and "drawnow" are two of the commands that allow
% Matlab to pause and carry out any pending figure window- related
% commands (like zooming, or pressing timer pause or cancel buttons or
% pressing a help button.)  If the drawnow command is not used
% immediately prior to the figure(ThisModuleFigureNumber) line, then
% immediately after the figure line executes, the other commands that
% have been waiting are executed in the other windows.  Then, when
% Matlab returns to this module and goes to the subplot line, the
% figure which is active is not necessarily the correct one. This
% results in strange things like the subplots appearing in the timer
% window or in the wrong figure window, or in help dialog boxes.
    drawnow
    %%% Activates the appropriate figure window.
    CPfigure(handles,ThisModuleFigureNumber);
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,2,1); imagesc(OrigImage);colormap(gray);
    title(['Input Image, Image Set # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the colored label
    %%% matrix image.
    subplot(2,2,2); imagesc(ColoredLabelMatrixImage); title(['Segmented ',SecondaryObjectName]);
    %%% A subplot of the figure window is set to display the original image
    %%% with watershed lines drawn on top.
    subplot(2,2,3); imagesc(ObjectOutlinesOnOrigImage); colormap(gray); title([SecondaryObjectName, ' Outlines on Input Image']);
    %%% A subplot of the figure window is set to display the original
    %%% image with watershed lines drawn for both the primary and secondary
    %%% objects.
    subplot(2,2,4); imagesc(BothOutlinesOnOrigImage); colormap(gray); title(['Outlines of ', PrimaryObjectName, ' and ', SecondaryObjectName, ' on Input Image']);
    CPFixAspectRatio(OrigImage);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

% PROGRAMMING NOTE
% HANDLES STRUCTURE:
%       In CellProfiler (and Matlab in general), each independent
% function (module) has its own workspace and is not able to 'see'
% variables produced by other modules. For data or images to be shared
% from one module to the next, they must be saved to what is called
% the 'handles structure'. This is a variable, whose class is
% 'structure', and whose name is handles. The contents of the handles
% structure are printed out at the command line of Matlab using the
% Tech Diagnosis button. The only variables present in the main
% handles structure are handles to figures and gui elements.
% Everything else should be saved in one of the following
% substructures:
%
% handles.Settings:
%       Everything in handles.Settings is stored when the user uses
% the Save pipeline button, and these data are loaded into
% CellProfiler when the user uses the Load pipeline button. This
% substructure contains all necessary information to re-create a
% pipeline, including which modules were used (including variable
% revision numbers), their setting (variables), and the pixel size.
%   Fields currently in handles.Settings: PixelSize, ModuleNames,
% VariableValues, NumbersOfVariables, VariableRevisionNumbers.
%
% handles.Pipeline:
%       This substructure is deleted at the beginning of the
% analysis run (see 'Which substructures are deleted prior to an
% analysis run?' below). handles.Pipeline is for storing data which
% must be retrieved by other modules. This data can be overwritten as
% each image set is processed, or it can be generated once and then
% retrieved during every subsequent image set's processing, or it can
% be saved for each image set by saving it according to which image
% set is being analyzed, depending on how it will be used by other
% modules. Any module which produces or passes on an image needs to
% also pass along the original filename of the image, named after the
% new image name, so that if the SaveImages module attempts to save
% the resulting image, it can be named by appending text to the
% original file name.
%   Example fields in handles.Pipeline: FileListOrigBlue,
% PathnameOrigBlue, FilenameOrigBlue, OrigBlue (which contains the actual image).
%
% handles.Current:
%       This substructure contains information needed for the main
% CellProfiler window display and for the various modules to
% function. It does not contain any module-specific data (which is in
% handles.Pipeline).
%   Example fields in handles.Current: NumberOfModules,
% StartupDirectory, DefaultOutputDirectory, DefaultImageDirectory,
% FilenamesInImageDir, CellProfilerPathname, ImageToolHelp,
% DataToolHelp, FigureNumberForModule01, NumberOfImageSets,
% SetBeingAnalyzed, TimeStarted, CurrentModuleNumber.
%
% handles.Preferences:
%       Everything in handles.Preferences is stored in the file
% CellProfilerPreferences.mat when the user uses the Set Preferences
% button. These preferences are loaded upon launching CellProfiler.
% The PixelSize, DefaultImageDirectory, and DefaultOutputDirectory
% fields can be changed for the current session by the user using edit
% boxes in the main CellProfiler window, which changes their values in
% handles.Current. Therefore, handles.Current is most likely where you
% should retrieve this information if needed within a module.
%   Fields currently in handles.Preferences: PixelSize, FontSize,
% DefaultModuleDirectory, DefaultOutputDirectory,
% DefaultImageDirectory.
%
% handles.Measurements
%      Data extracted from input images are stored in the
% handles.Measurements substructure for exporting or further analysis.
% This substructure is deleted at the beginning of the analysis run
% (see 'Which substructures are deleted prior to an analysis run?'
% below). The Measurements structure is organized in two levels. At
% the first level, directly under handles.Measurements, there are
% substructures (fields) containing measurements of different objects.
% The names of the objects are specified by the user in the Identify
% modules (e.g. 'Cells', 'Nuclei', 'Colonies').  In addition to these
% object fields is a field called 'Image' which contains information
% relating to entire images, such as filenames, thresholds and
% measurements derived from an entire image. That is, the Image field
% contains any features where there is one value for the entire image.
% As an example, the first level might contain the fields
% handles.Measurements.Image, handles.Measurements.Cells and
% handles.Measurements.Nuclei.
%      In the second level, the measurements are stored in matrices
% with dimension [#objects x #features]. Each measurement module
% writes its own block; for example, the MeasureAreaShape module
% writes shape measurements of 'Cells' in
% handles.Measurements.Cells.AreaShape. An associated cell array of
% dimension [1 x #features] with suffix 'Features' contains the names
% or descriptions of the measurements. The export data tools, e.g.
% ExportData, triggers on this 'Features' suffix. Measurements or data
% that do not follow the convention described above, or that should
% not be exported via the conventional export tools, can thereby be
% stored in the handles.Measurements structure by leaving out the
% '....Features' field. This data will then be invisible to the
% existing export tools.
%      Following is an example where we have measured the area and
% perimeter of 3 cells in the first image and 4 cells in the second
% image. The first column contains the Area measurements and the
% second column contains the Perimeter measurements.  Each row
% contains measurements for a different cell:
% handles.Measurements.Cells.AreaShapeFeatures = {'Area' 'Perimeter'}
% handles.Measurements.Cells.AreaShape{1} = 	40		20
%                                               100		55
%                                              	200		87
% handles.Measurements.Cells.AreaShape{2} = 	130		100
%                                               90		45
%                                               100		67
%                                               45		22
%
% Which substructures are deleted prior to an analysis run?
%       Anything stored in handles.Measurements or handles.Pipeline
% will be deleted at the beginning of the analysis run, whereas
% anything stored in handles.Settings, handles.Preferences, and
% handles.Current will be retained from one analysis to the next. It
% is important to think about which of these data should be deleted at
% the end of an analysis run because of the way Matlab saves
% variables: For example, a user might process 12 image sets of nuclei
% which results in a set of 12 measurements ("ImageTotalNucArea")
% stored in handles.Measurements. In addition, a processed image of
% nuclei from the last image set is left in the handles structure
% ("SegmNucImg"). Now, if the user uses a different algorithm which
% happens to have the same measurement output name "ImageTotalNucArea"
% to analyze 4 image sets, the 4 measurements will overwrite the first
% 4 measurements of the previous analysis, but the remaining 8
% measurements will still be present. So, the user will end up with 12
% measurements from the 4 sets. Another potential problem is that if,
% in the second analysis run, the user runs only a module which
% depends on the output "SegmNucImg" but does not run a module that
% produces an image by that name, the module will run just fine: it
% will just repeatedly use the processed image of nuclei leftover from
% the last image set, which was left in handles.Pipeline.

%%% Saves the final, segmented label matrix image of secondary objects to
%%% the handles structure so it can be used by subsequent modules.
fieldname = ['Segmented',SecondaryObjectName];
handles.Pipeline.(fieldname) = FinalLabelMatrixImage;

%%% Saves the Threshold value to the handles structure.
%%% Storing the threshold is a little more complicated than storing other measurements
%%% because several different modules will write to the handles.Measurements.Image.Threshold
%%% structure, and we should therefore probably append the current threshold to an existing structure.
% First, if the Threshold fields don't exist, initialize them
if ~isfield(handles.Measurements.Image,'ThresholdFeatures')
    handles.Measurements.Image.ThresholdFeatures = {};
    handles.Measurements.Image.Threshold = {};
end
% Search the ThresholdFeatures to find the column for this object type
column = find(~cellfun('isempty',strfind(handles.Measurements.Image.ThresholdFeatures,SecondaryObjectName)));
% If column is empty it means that this particular object has not been segmented before. This will
% typically happen for the first image set. Append the feature name in the
% handles.Measurements.Image.ThresholdFeatures matrix
if isempty(column)
    handles.Measurements.Image.ThresholdFeatures(end+1) = {['Threshold ' SecondaryObjectName]};
    column = length(handles.Measurements.Image.ThresholdFeatures);
end
handles.Measurements.Image.Threshold{handles.Current.SetBeingAnalyzed}(1,column) = Threshold;


%%% Saves the ObjectCount, i.e. the number of segmented objects.
if ~isfield(handles.Measurements.Image,'ObjectCountFeatures')
    handles.Measurements.Image.ObjectCountFeatures = {};
    handles.Measurements.Image.ObjectCount = {};
end
column = find(~cellfun('isempty',strfind(handles.Measurements.Image.ObjectCountFeatures,SecondaryObjectName)));
if isempty(column)
    handles.Measurements.Image.ObjectCountFeatures(end+1) = {['ObjectCount ' SecondaryObjectName]};
    column = length(handles.Measurements.Image.ObjectCountFeatures);
end
handles.Measurements.Image.ObjectCount{handles.Current.SetBeingAnalyzed}(1,column) = max(FinalLabelMatrixImage(:));


%%% Saves the location of each segmented object
handles.Measurements.(SecondaryObjectName).LocationFeatures = {'CenterX','CenterY'};
tmp = regionprops(FinalLabelMatrixImage,'Centroid');
Centroid = cat(1,tmp.Centroid);
handles.Measurements.(SecondaryObjectName).Location(handles.Current.SetBeingAnalyzed) = {Centroid};

%%% Saves images to the handles structure so they can be saved to the hard
%%% drive, if the user requested.
try
    if ~strcmp(SaveColored,'Do not save')
        if strcmp(SaveMode,'RGB')
            handles.Pipeline.(SaveColored) = ColoredLabelMatrixImage;
        else
            handles.Pipeline.(SaveColored) = FinalLabelMatrixImage;
        end
    end
    if ~strcmp(SaveOutlined,'Do not save')
        handles.Pipeline.(SaveOutlined) = ObjectOutlinesOnOrigImage;
    end
catch errordlg('The object outlines or colored objects were not calculated by an identify module (possibly because the window is closed) so these images were not saved to the handles structure. The Save Images module will therefore not function on these images. This is just for your information - image processing is still in progress, but the Save Images module will fail if you attempted to save these images.')
end
