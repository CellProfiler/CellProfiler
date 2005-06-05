function handles = IdentifySecPropagate(handles)

% Help for the Identify Secondary Propagate module:
% Category: Object Identification
% 
% This module identifies secondary objects based on a previous
% module's identification of primary objects.  Each primary object is
% assumed to be completely within a secondary object (e.g. nuclei
% within cells stained for actin).  The module is especially good at
% determining the dividing lines between clustered secondary objects.
% The dividing lines between objects are determined by a combination
% of the distance to the nearest primary object and intensity
% gradients (dividing lines can be either dim or bright). 
%
% Settings:
%
% Threshold: this setting affects the stringency of object outlines
% that border the background.  It does not affect the dividing lines
% between clumped objects. A higher number will result in smaller
% objects (more stringent). A lower number will result in large
% objects (less stringent).
%
% Regularization factor: This module takes two factors into account
% when deciding where to draw the dividing line between two touching
% secondary objects: the distance to the nearest primary object, and
% the intensity of the secondary object image.  The regularization
% factor controls the balance between these two considerations: A
% value of zero means that the distance to the nearest primary object
% is ignored and the decision is made entirely on the intensity
% gradient between the two competing primary objects. Larger values
% weight the distance between the two values more and more heavily.
% The regularization factor can be infinitely large, but around 10 or
% so, the intensity image is almost completely ignored and the
% dividing line will simply be halfway between the two competing
% primary objects.
% 
% Note: Primary segmenters produce two output images that are used by
% this module.  The Segmented image contains the final, edited
% primary objects (i.e. objects at the border and those that are too
% small or large have been excluded).  The PrelimSmallSegmented
% image is the same except that the objects at the border and the
% large objects have been included.  These extra objects are used to
% perform the identification of secondary object outlines, since they
% are probably real objects (even if we don't want to measure them).
% Small objects are not used at this stage because they are more
% likely to be artifactual, and so they therefore should not "claim"
% any secondary object pixels.
% 
% TECHNICAL DESCRIPTION OF THE MODULE: 
% Propagate labels from LABELS_IN to LABELS_OUT, steered by IMAGE and 
% limited to MASK.  MASK should be a logical array.  LAMBDA is a 
% regularization parameter, larger being closer to Euclidean distance
% in the image plane, and zero being entirely controlled by IMAGE. 
% Propagation of labels is by shortest path to a nonzero label in
% LABELS_IN.  Distance is the sum of absolute differences in the image
% in a 3x3 neighborhood, combined with LAMBDA via sqrt(differences^2 +
% LAMBDA^2).  Note that there is no separation between adjacent areas
% with different labels (as there would be using, e.g., watershed).
% Such boundaries must be added in a postprocess.
%
% What does Secondary mean?
% Identify Primary modules identify objects without relying on any
% information other than a single grayscale input image (e.g. nuclei
% are typically primary objects). Identify Secondary modules require a
% grayscale image plus an image where primary objects have already
% been identified, because the secondary objects' locations are
% determined in part based on the primary objects (e.g. cells can be
% secondary objects). Identify Tertiary modules require images where
% two sets of objects have already been identified (e.g. nuclei and
% cell regions are used to define the cytoplasm objects, which are
% tertiary objects).
%
% SAVING IMAGES: In addition to the object outlines and the
% pseudo-colored object images that can be saved using the
% instructions in the main CellProfiler window for this module, this
% module produces a grayscale image where each object is a different
% intensity, which you can save using the Save Images module using the
% name: Segmented + whatever you called the objects (e.g.
% SegmentedCells).
%
% Additional image(s) are normally calculated for display only,
% including the object outlines alone. These images can be saved by
% altering the code for this module to save those images to the
% handles structure (see the SaveImages module help) and then using
% the Save Images module.
%
% See also IDENTIFYSECPROPAGATESUBFUNCTION, IDENTIFYSECDISTANCE,
% IDENTIFYSECWATERSHED.

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
%defaultVAR01 = OrigGreen
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = What did you call the objects that will be used to mark the centers of these objects?
%defaultVAR02 = Nuclei
PrimaryObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = What do you want to call the objects identified by this module? (Note: Data will be produced based on this name, e.g. ObjectTotalAreaCells)
%defaultVAR03 = Cells
SecondaryObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = Enter the threshold [0 = automatically calculate] (Positive number, Max = 1):
%defaultVAR04 = 0
Threshold = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,4}));

%textVAR05 = If auto threshold, enter an adjustment factor (Positive number, 1 = no adjustment):
%defaultVAR05 = 1
ThresholdAdjustmentFactor = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,5}));

%textVAR06 = Enter the minimum allowable threshold (Range = 0 to 1; this prevents an unreasonably low threshold from counting noise as objects when there are no bright objects in the field of view). This is intended for use with automatic thresholding, but will override an absolute threshold entered above:
%defaultVAR06 = 0
MinimumThreshold = char(handles.Settings.VariableValues{CurrentModuleNum,6}); 

%textVAR07 = Regularization factor (0 to infinity). Larger=distance,0=intensity
%defaultVAR07 = 0.05
RegularizationFactor = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,7}));

%textVAR08 = Will you want to save the outlines of the objects (Yes or No)? If yes, use a Save Images module and type "OutlinedOBJECTNAME" in the first box, where OBJECTNAME is whatever you have called the objects identified by this module.
%defaultVAR08 = No
SaveOutlined = char(handles.Settings.VariableValues{CurrentModuleNum,8}); 

%textVAR09 =  Will you want to save the image of the pseudo-colored objects (Yes or No)? If yes, use a Save Images module and type "ColoredOBJECTNAME" in the first box, where OBJECTNAME is whatever you have called the objects identified by this module.
%defaultVAR09 = No
SaveColored = char(handles.Settings.VariableValues{CurrentModuleNum,9}); 

%%%VariableRevisionNumber = 3

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a variable,
%%% "OrigImage".
fieldname = ['', ImageName];
%%% Checks whether the image to be analyzed exists in the handles structure.
if isfield(handles.Pipeline, fieldname)==0,
    %%% If the image is not there, an error message is produced.  The error
    %%% is not displayed: The error function halts the current function and
    %%% returns control to the calling function (the analyze all images
    %%% button callback.)  That callback recognizes that an error was
    %%% produced because of its try/catch loop and breaks out of the image
    %%% analysis loop without attempting further modules.
    error(['Image processing was canceled because the Identify Secondary Propagate module could not find the input image.  It was supposed to be named ', ImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
OrigImage = handles.Pipeline.(fieldname);


%%% Checks that the original image is two-dimensional (i.e. not a color
%%% image), which would disrupt several of the image functions.
if ndims(OrigImage) ~= 2
    error('Image processing was canceled because the Identify Secondary Propagate module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
end

%%% Retrieves the preliminary label matrix image that contains the primary
%%% segmented objects which have only been edited to discard objects 
%%% that are smaller than a certain size.  This image
%%% will be used as markers to segment the secondary objects with this
%%% module.  Checks first to see whether the appropriate image exists.
fieldname = ['PrelimSmallSegmented', PrimaryObjectName];
%%% Checks whether the image exists in the handles structure.
if isfield(handles.Pipeline, fieldname)==0,
    error(['Image processing has been canceled. Prior to running the Identify Secondary Propagate module, you must have previously run a module that generates an image with the preliminary primary objects identified.  You specified in the Identify Secondary Propagate module that the primary objects were named ', PrimaryObjectName, ' as a result of the previous module, which should have produced an image called ', fieldname, ' in the handles structure.  The Identify Secondary Propagate module cannot locate this image.']);
    end
PrelimPrimaryLabelMatrixImage = handles.Pipeline.(fieldname);

        
%%% Retrieves the label matrix image that contains the edited primary
%%% segmented objects which will be used to weed out which objects are
%%% real - not on the edges and not below or above the specified size
%%% limits. Checks first to see whether the appropriate image exists.
fieldname = ['Segmented', PrimaryObjectName];
%%% Checks whether the image exists in the handles structure.
if isfield(handles.Pipeline, fieldname)==0,
    error(['Image processing has been canceled. Prior to running the Identify Secondary Propagate module, you must have previously run a module that generates an image with the preliminary primary objects identified.  You specified in the Identify Secondary Propagate module that the primary objects were named ', PrimaryObjectName, ' as a result of the previous module, which should have produced an image called ', fieldname, ' in the handles structure.  The Identify Secondary Propagate module cannot locate this image.']);
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

%%% STEP 1: The distinction between objects and background is determined
%%% using the user-specified threshold.
%%% Determines the threshold to use. 
if Threshold == 0
    Threshold = CPgraythresh(OrigImage,handles,ImageName);
    %%% Replaced the following line to accomodate calculating the
    %%% threshold for images that have been masked.
%    Threshold = CPgraythresh(OrigImage);
    %%% Adjusts the threshold by a correction factor.  
    Threshold = Threshold*ThresholdAdjustmentFactor;
end
MinimumThreshold = str2num(MinimumThreshold);
Threshold = max(MinimumThreshold,Threshold);

%%% Thresholds the original image.
ThresholdedOrigImage = im2bw(OrigImage, Threshold);

%%% STEP 2: Starting from the identified primary objects, the secondary
%%% objects are identified using the propagate function, written by Thouis
%%% R. Jones. Calls the function
%%% "IdentifySecPropagateSubfunction.mexmac" (or whichever version is
%%% appropriate for the computer platform being used), which consists of C
%%% code that has been compiled to run quickly within Matlab.
PropagatedImage = IdentifySecPropagateSubfunction(PrelimPrimaryLabelMatrixImage,OrigImage,ThresholdedOrigImage,RegularizationFactor);
drawnow

%%% STEP 3: Remove objects that are not desired, edited objects.  The
%%% edited primary object image is used rather than the preliminary one, so
%%% that objects whose nuclei are on the edge of the image and who are
%%% larger or smaller than the specified size are discarded.
%%% Converts the EditedPrimaryBinaryImage to binary.
EditedPrimaryBinaryImage = im2bw(EditedPrimaryLabelMatrixImage,.5);
%%% Finds the locations and labels for different regions.
area_locations2 = find(PropagatedImage);
area_labels2 = PropagatedImage(area_locations2);
drawnow
%%% Creates a sparse matrix with column as label and row as location,
%%% with the value of the center at (I,J) if location I has label J.
%%% Taking the maximum of this matrix gives the largest valued center
%%% overlapping a particular label.  Tacking on a zero and pushing
%%% labels through the resulting map removes any background regions.
map2 = [0 full(max(sparse(area_locations2, area_labels2, EditedPrimaryLabelMatrixImage(area_locations2))))];
HoleyPrelimLabelMatrixImage = map2(PropagatedImage + 1);
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
drawnow
%%% Uses sparse matrices to find the minimum and maximum label adjacent
%%% to each zero-region.
ZeroLocations = find(ZeroRegionImage);
ZeroLabels = ZeroRegionImage(ZeroLocations);
MinByRegion = full(max(sparse(ZeroLocations, ZeroLabels, MinimaImage(ZeroLocations))));
%%% Remaps to correct order (see above).
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
drawnow

%%% STEP 4: Relabels the final objects so that their numbers
%%% correspond to the numbers used for nuclei.
%%% For each object, one label and one label location is acquired and
%%% stored.
[LabelsUsed,LabelLocations] = unique(EditedPrimaryLabelMatrixImage);
%%% The +1 increment accounts for the fact that there are zeros in the
%%% image, while the LabelsUsed starts at 1.
LabelsUsed(PrelimLabelMatrixImage(LabelLocations(2:end))+1) = EditedPrimaryLabelMatrixImage(LabelLocations(2:end));
FinalLabelMatrixImage = LabelsUsed(PrelimLabelMatrixImage+1);

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
    %%% Calculates OutlinesOnOrigImage for displaying in the figure
    %%% window in subplot(2,2,3).
    %%% Note: these outlines are not perfectly accurate; for some reason it
    %%% produces more objects than in the original image.  But it is OK for
    %%% display purposes.
    %%% Maximum filters the image with a 3x3 neighborhood.
    MaxFilteredImage = ordfilt2(FinalLabelMatrixImage,9,ones(3,3),'symmetric');
    %%% Determines the outlines.
    IntensityOutlines = FinalLabelMatrixImage - MaxFilteredImage;
    %%% Converts to logical.
    warning off MATLAB:conversionToLogical
    LogicalOutlines = logical(IntensityOutlines);
    warning on MATLAB:conversionToLogical
    %%% Determines the grayscale intensity to use for the cell outlines.
    LineIntensity = max(OrigImage(:));
    %%% Overlays the outlines on the original image.
    ObjectOutlinesOnOrigImage = OrigImage;
    ObjectOutlinesOnOrigImage(LogicalOutlines) = LineIntensity;
    %%% Calculates BothOutlinesOnOrigImage for displaying in the figure
    %%% window in subplot(2,2,4).
    %%% Creates the structuring element that will be used for dilation.
    StructuringElement = strel('square',3);
    %%% Dilates the Primary Binary Image by one pixel (8 neighborhood).
    DilatedPrimaryBinaryImage = imdilate(EditedPrimaryBinaryImage, StructuringElement);
    %%% Subtracts the PrelimPrimaryBinaryImage from the DilatedPrimaryBinaryImage,
    %%% which leaves the PrimaryObjectOutlines.
    PrimaryObjectOutlines = DilatedPrimaryBinaryImage - EditedPrimaryBinaryImage;
    BothOutlinesOnOrigImage = ObjectOutlinesOnOrigImage;
    BothOutlinesOnOrigImage(PrimaryObjectOutlines == 1) = LineIntensity;
    drawnow
    %%% Activates the appropriate figure window.
    CPfigure(ThisModuleFigureNumber);
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,2,1); imagesc(OrigImage);colormap(gray);
    title(['Input Image, Image Set # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the colored label
    %%% matrix image.
    subplot(2,2,2); imagesc(ColoredLabelMatrixImage); title(['Segmented ',SecondaryObjectName]);
    %%% A subplot of the figure window is set to display the original image
    %%% with secondary object outlines drawn on top.
    subplot(2,2,3); imagesc(ObjectOutlinesOnOrigImage); colormap(gray); title([SecondaryObjectName, ' Outlines on Input Image']);
    %%% A subplot of the figure window is set to display the original
    %%% image with outlines drawn for both the primary and secondary
    %%% objects.
    subplot(2,2,4); imagesc(BothOutlinesOnOrigImage); colormap(gray); title(['Outlines of ', PrimaryObjectName, ' and ', SecondaryObjectName, ' on Input Image']);
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
    if strncmpi(SaveColored,'Y',1) == 1
        fieldname = ['Colored',SecondaryObjectName];
        handles.Pipeline.(fieldname) = ColoredLabelMatrixImage;
    end
    if strncmpi(SaveOutlined,'Y',1) == 1
        fieldname = ['Outlined',SecondaryObjectName];
        handles.Pipeline.(fieldname) = ObjectOutlinesOnOrigImage;
    end
%%% I am pretty sure this try/catch is no longer necessary, but will
%%% leave in just in case.
catch errordlg('The object outlines or colored objects were not calculated by an identify module (possibly because the window is closed) so these images could not be saved to the handles structure. The Save Images module will therefore not function on these images.')
end