function handles = IdentifyPrimShapeDist(handles)

% Help for the Identify Primary Shape Distance module: 
% Category: Object Identification
% 
% This image analysis module identifies and separates clumps of
% objects in a grayscale input image. The module works best for
% objects that are very round. Separating objects that are clumped is
% based entirely on shape (roundness) rather than intensity gradients
% of the objects. Dividing lines between clumped objects are halfway
% between the 'center' of each object.  Centers are located as peaks
% in the distance-transformed image calculated by thresholding.
% Dividing lines do not depend on the intensity gradients in the
% image. The cells need not be brighter towards the interior as is
% required for the intensity-based module, nor do they need to be
% dimmer or brighter along the lines between clumped objects, as is
% necessary for the Identify Primary Shape Intensity module.
%
% Settings:
%
% Size range: You may exclude objects that are smaller or bigger than
% the size range you specify. A comma should be placed between the
% lower size limit and the upper size limit. The units here are pixels
% so that it is easy to zoom in on found objects and determine the
% size of what you think should be excluded.
%
% Threshold: The threshold affects the stringency of the lines between
% the objects and the background. It also affects a preliminary
% calculation which is used to determine whether whether objects close
% by each other are considered a single object or multiple objects.
% You may enter an absolute number between 0 and 1 for the threshold
% (use 'Show pixel data' to see the pixel intensities for your images
% in the appropriate range of 0 to 1), or you may have it calculated
% for each image individually by typing 0.  There are advantages
% either way.  An absolute number treats every image identically, but
% an automatically calculated threshold is more realistic/accurate,
% though occasionally subject to artifacts.  The threshold which is
% used for each image is recorded as a measurement in the output file,
% so if you find unusual measurements from one of your images, you
% might check whether the automatically calculated threshold was
% unusually high or low compared to the remaining images.  When an
% automatic threshold is selected, it may consistently be too
% stringent or too lenient, so an adjustment factor can be entered as
% well. The number 1 means no adjustment, 0 to 1 makes the threshold
% more lenient and greater than 1 (e.g. 1.3) makes the threshold more
% stringent.
%
% Maxima suppression neighborhood: This variable (together with the
% threshold) affects whether objects close by each other are
% considered a single object or multiple objects. It does not affect
% the dividing lines between an object and the background.  If you see
% too many objects merged that ought to be separate, the value should
% be lower. If you see too many objects split up that ought to be
% merged, the value should be higher. The value should be set to be
% roughly equivalent to the minimum radius of a real object of
% interest. Basically, any distinct 'objects' which are found but are
% within two times this distance from each other will be assumed to be
% actually two lumpy parts of the same object, and they will be
% merged.
%
% How it works: this module identifies objects by finding peaks in the
% distance transform of a thresholded image.  Once a marker for each
% object has been identified in this way, a watershed function on the
% distance transformed image identifies the lines between objects that
% are touching each other.  To identify the dividing lines between
% object and background, a threshold is applied. Objects on the border
% of the image are ignored, and the user can select a size range,
% outside which objects will be ignored.
%
% What does Primary mean? Identify Primary modules identify objects
% without relying on any information other than a single grayscale
% input image (e.g. nuclei are typically primary objects). Identify
% Secondary modules require a grayscale image plus an image where
% primary objects have already been identified, because the secondary
% objects' locations are determined in part based on the primary
% objects (e.g. cells can be secondary objects). Identify Tertiary
% modules require images where two sets of objects have already been
% identified (e.g. nuclei and cell regions are used to define the
% cytoplasm objects, which are tertiary objects).
%
% SAVING IMAGES: This module produces several images which can be
% easily saved using the Save Images module. These will be grayscale
% images where each object is a different intensity. (1) The
% preliminary segmented image, which includes objects on the edge of
% the image and objects that are outside the size range can be saved
% using the name: PrelimSegmented + whatever you called the objects
% (e.g. PrelimSegmentedNuclei). (2) The preliminary segmented image
% which excludes objects smaller than your selected size range can be
% saved using the name: PrelimSmallSegmented + whatever you called the
% objects (e.g. PrelimSmallSegmented Nuclei) (3) The final segmented
% image which excludes objects on the edge of the image and excludes
% objects outside the size range can be saved using the name:
% Segmented + whatever you called the objects (e.g. SegmentedNuclei)
% 
% Several additional images are normally calculated for display only,
% including the colored label matrix image (the objects displayed as
% arbitrary colors), object outlines, and object outlines overlaid on
% the original image. These images can be saved by altering the code
% for this module to save those images to the handles structure (see
% the SaveImages module help) and then using the Save Images module.
% Important note: The calculations of these display images are only
% performed if the figure window is open, so the figure window must be
% left open or the Save Images module will fail.  If you are running
% the job on a cluster, figure windows are not open, so the Save
% Images module will also fail, unless you go into the code for this
% module and remove the 'if/end' statement surrounding the DISPLAY
% RESULTS section.
%
% See also IDENTIFYPRIMADAPTTHRESHOLDA,
% IDENTIFYPRIMADAPTTHRESHOLDB,
% IDENTIFYPRIMADAPTTHRESHOLDC, 
% IDENTIFYPRIMADAPTTHRESHOLDD,
% IDENTIFYPRIMTHRESHOLD,
% IDENTIFYPRIMSHAPEINTENS, 
% IDENTIFYPRIMINTENSINTENS.

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
%defaultVAR01 = OrigBlue
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = What do you want to call the objects segmented by this module?
%defaultVAR02 = Nuclei
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Size range (in pixels) of objects to include (1,99999 = do not discard any)
%defaultVAR03 = 1,99999
SizeRange = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = Enter the threshold [0 = automatically calculate] (Positive number, Max = 1):
%defaultVAR04 = 0
Threshold = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,4}));

%textVAR05 = If auto threshold, enter an adjustment factor (Positive number, 1 = no adjustment):
%defaultVAR05 = 1
ThresholdAdjustmentFactor = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,5}));

%textVAR06 = Enter the Max Suppress N'hood (Non-negative integer ~ the radius of objects)
%defaultVAR06 = 10
MaximaSuppressionNeighborhood = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,6}));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines what the user entered for the size range.
SizeRangeNumerical = str2num(SizeRange); %#ok We want to ignore MLint error checking for this line.
MinSize = SizeRangeNumerical(1);
MaxSize = SizeRangeNumerical(2);

%%% Reads (opens) the image you want to analyze and assigns it to a variable,
%%% "OrigImageToBeAnalyzed".
%%% Checks whether the image exists in the handles structure.
if isfield(handles.Pipeline, ImageName)==0
    error(['Image processing has been canceled. Prior to running the Identify Primary Shape Distance module, you must have previously run a module to load an image. You specified in the Segment Intensity module that this image was called ', ImageName, ' which should have produced a field in the handles structure called ', ImageName, '. The Segment Intensity module cannot find this image.']);
end
OrigImageToBeAnalyzed = handles.Pipeline.(ImageName);

%%% Checks that the original image is two-dimensional (i.e. not a color
%%% image), which would disrupt several of the image functions.
if ndims(OrigImageToBeAnalyzed) ~= 2
    error('Image processing was canceled because the Identify Primary Shape Distance module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
end

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
%%% Fills holes in the thresholded image so that stray dim pixels within the
%%% objects do not count as edges of the objects.
FilledThresholdedImage = imfill(ThresholdedOrigImage, 'holes');
%%% Computes distance transform.
DistanceTransformedImage = bwdist(~FilledThresholdedImage);
%%% Essentially thresholds again to get rid of background pixels.
% DistanceTransformedImage(~FilledThresholdedImage) = sum(size(OrigImageToBeAnalyzed));

%%% Perturbs the distance image so that local maxima near each other with
%%% identical values will now have slightly different values.

%%% Saves off the random number generator's state, and set the state to
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
MaximaMask = strel('disk', MaximaSuppressionNeighborhood);
MaximaImage(DistanceTransformedImage < ordfilt2(DistanceTransformedImage,sum(sum(getnhood(MaximaMask))),getnhood(MaximaMask))) = 0;
drawnow
%%% Converts the maxima image to black and white.
MaximaImage(MaximaImage ~= 0) = 1;

%%% STEP 2: Performs watershed function on the DistanceTransformedImage.
drawnow
%%% Inverts image.
InvertedDistanceTransformedImage = -DistanceTransformedImage;
%%% Overlays the nuclear markers (maxima) on the inverted DistanceTransformedImage so
%%% there are black dots on top of each dark nucleus on a white background.
Overlaid = imimposemin(InvertedDistanceTransformedImage,MaximaImage);
%%% Identifies watershed lines.
BlackWatershedLinesPre = watershed(Overlaid);
%%% Superimposes watershed lines as white (255) onto the inverted original
%%% image.
InvertedOrigImage = imcomplement(OrigImageToBeAnalyzed);
WhiteWatershedOnInvertedOrig = InvertedOrigImage;
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
%%% Overwrites the small objects with zeros.
PrelimLabelMatrixImage2 = PrelimLabelMatrixImage1;
PrelimLabelMatrixImage2(AreasImage < MinSize) = 0;
drawnow
%%% Relabels so that labels are consecutive. This is important for
%%% downstream modules (IdentifySec).
PrelimLabelMatrixImage2 = bwlabel(im2bw(PrelimLabelMatrixImage2,.1));
%%% Overwrites the large objects with zeros.
PrelimLabelMatrixImage3 = PrelimLabelMatrixImage2;
if MaxSize ~= 99999
    PrelimLabelMatrixImage3(AreasImage > MaxSize) = 0;
end
%%% Removes objects that are touching the edge of the image, since they
%%% won't be measured properly.
PrelimLabelMatrixImage4 = imclearborder(PrelimLabelMatrixImage3,8);
drawnow
%%% Converts PrelimLabelMatrixImage4 to binary.
FinalBinaryPre = im2bw(PrelimLabelMatrixImage4,1);
%%% Fills in holes in the FinalBinaryPre image.
FinalBinary = imfill(FinalBinaryPre, 'holes');
drawnow
%%% Converts the image to label matrix format. Even if the above step
%%% is excluded (filling holes), it is still necessary to do this in order
%%% to "compact" the label matrix: this way, each number corresponds to an
%%% object, with no numbers skipped.
FinalLabelMatrixImage = bwlabel(FinalBinary);

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

% PROGRAMMING NOTE
% DISPLAYING RESULTS:
% Each module checks whether its figure is open before calculating
% images that are for display only. This is done by examining all the
% figure handles for one whose handle is equal to the assigned figure
% number for this module. If the figure is not open, everything
% between the "if" and "end" is ignored (to speed execution), so do
% not do any important calculations here. Otherwise an error message
% will be produced if the user has closed the window but you have
% attempted to access data that was supposed to be produced by this
% part of the code. If you plan to save images which are normally
% produced for display only, the corresponding lines should be moved
% outside this if statement.

fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
if any(findobj == ThisModuleFigureNumber) == 1;
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
    figure(ThisModuleFigureNumber);
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,2,1); imagesc(OrigImageToBeAnalyzed);colormap(gray);
    title(['Input Image, Image Set # ',num2str(handles.Current.SetBeingAnalyzed)]);
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
% handles.Measurements:
%       Everything in handles.Measurements contains data specific to each
% image set analyzed for exporting. It is used by the ExportMeanImage
% and ExportCellByCell data tools. This substructure is deleted at the
% beginning of the analysis run (see 'Which substructures are deleted
% prior to an analysis run?' below).
%    Note that two types of measurements are typically made: Object
% and Image measurements.  Object measurements have one number for
% every object in the image (e.g. ObjectArea) and image measurements
% have one number for the entire image, which could come from one
% measurement from the entire image (e.g. ImageTotalIntensity), or
% which could be an aggregate measurement based on individual object
% measurements (e.g. ImageMeanArea).  Use the appropriate prefix to
% ensure that your data will be extracted properly. It is likely that
% Subobject will become a new prefix, when measurements will be
% collected for objects contained within other objects. 
%       Saving measurements: The data extraction functions of
% CellProfiler are designed to deal with only one "column" of data per
% named measurement field. So, for example, instead of creating a
% field of XY locations stored in pairs, they should be split into a
% field of X locations and a field of Y locations. It is wise to
% include the user's input for 'ObjectName' or 'ImageName' as part of
% the fieldname in the handles structure so that multiple modules can
% be run and their data will not overwrite each other.
%   Example fields in handles.Measurements: ImageCountNuclei,
% ObjectAreaCytoplasm, FilenameOrigBlue, PathnameOrigBlue,
% TimeElapsed.
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
% ("SegmNucImg"). Now, if the user uses a different module which
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

%%% Saves the segmented image, not edited for objects along the edges or
%%% for size, to the handles structure.
fieldname = ['PrelimSegmented',ObjectName];
handles.Pipeline.(fieldname) = PrelimLabelMatrixImage1;

%%% Saves the segmented image, only edited for small objects, to the
%%% handles structure.
fieldname = ['PrelimSmallSegmented',ObjectName];
handles.Pipeline.(fieldname) = PrelimLabelMatrixImage2;

%%% Saves the final segmented label matrix image to the handles structure.
fieldname = ['Segmented',ObjectName];
handles.Pipeline.(fieldname) = FinalLabelMatrixImage;

%%% Saves the Threshold value to the handles structure.
fieldname = ['ImageThreshold', ObjectName];
handles.Measurements.(fieldname)(handles.Current.SetBeingAnalyzed) = {Threshold};

%%% Determines the filename of the image to be analyzed.
fieldname = ['Filename', ImageName];
FileName = handles.Pipeline.(fieldname)(handles.Current.SetBeingAnalyzed);
%%% Saves the filename of the image to be analyzed.
fieldname = ['Filename', ObjectName];
handles.Pipeline.(fieldname)(handles.Current.SetBeingAnalyzed) = FileName;