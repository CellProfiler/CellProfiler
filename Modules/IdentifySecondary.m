function handles = IdentifySecondary(handles)

% Help for the Identify Secondary module:
% Category: Object Processing
%
% METHODS OF IDENTIFICATION
%
% Distance:
% Based on another module's identification of primary objects, this
% mode identifies secondary objects when no specific staining is
% available.  The edges of the primary objects are simply expanded a
% particular distance to create the secondary objects. For example, if
% nuclei are labeled but there is no stain to help locate cell edges,
% the nuclei can simply be expanded in order to estimate the cell's
% location.  This is a standard module used in commercial software and
% is known as the 'donut' or 'annulus' approach for identifying the
% cytoplasmic compartment.
%
% Propagation:
% This mode identifies secondary objects based on a previous
% module's identification of primary objects.  Each primary object is
% assumed to be completely within a secondary object (e.g. nuclei
% within cells stained for actin).  The module is especially good at
% determining the dividing lines between clustered secondary objects.
% The dividing lines between objects are determined by a combination
% of the distance to the nearest primary object and intensity
% gradients (dividing lines can be either dim or bright).
%
% Watershed:
% This mode identifies secondary objects based on a previous
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
% small or large have been excluded).  The SmallRemovedSegmented
% image is the same except that the objects at the border and the
% large objects have been included.  These extra objects are used to
% perform the identification of secondary object outlines, since they
% are probably real objects (even if we don't want to measure them).
% Small objects are not used at this stage because they are more
% likely to be artifactual, and so they therefore should not "claim"
% any secondary object pixels.
%
% TECHNICAL DESCRIPTION OF THE PROPAGATION MODE:
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
% Information on IdentifySecPropagateSubfunction:
%
% This is a subfunction implemented in C and MEX to perform the
% propagate algorithm (somewhat similar to watershed).  This help
% documents the arguments and behavior of the propagate algorithm.
%
% Propagate labels from LABELS_IN to LABELS_OUT, steered by IMAGE and
% limited to MASK.  MASK should be a logical array.  LAMBDA is a
% regularization paramter, larger being closer to Euclidean distance
% in the image plane, and zero being entirely controlled by IMAGE.
%
% Propagation of labels is by shortest path to a nonzero label in
% LABELS_IN.  Distance is the sum of absolute differences in the image
% in a 3x3 neighborhood, combined with LAMBDA via sqrt(differences^2 +
% LAMBDA^2).
%
% Note that there is no separation between adjacent areas with
% different labels (as there would be using, e.g., watershed).  Such
% boundaries must be added in a postprocess.
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
% $Revision: 1808 $

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow

%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);
ModuleName = char(handles.Settings.ModuleNames(CurrentModuleNum));

%textVAR01 = What did you call the primary objects you want to create secondary objects around?
%infotypeVAR01 = objectgroup
PrimaryObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the objects identified by this module? (Note: Data will be produced based on this name, e.g. ObjectTotalAreaCells)
%defaultVAR02 = Cells
%infotypeVAR02 = objectgroup indep
SecondaryObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = How do you want to identify the secondary objects (Distance - B uses a background image for identification, Distance - N identifies objects by distance alone)?
%choiceVAR03 = Distance - N
%choiceVAR03 = Distance - B
%choiceVAR03 = Propagation
%choiceVAR03 = Watershed
%inputtypeVAR03 = popupmenu
IdentChoice = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = What did you call the images of the secondary objects? If identifying objects by DISTANCE - N, this will not affect object identification, only the final display.
%infotypeVAR04 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%textVAR05 = Select thresholding method or enter a threshold in the range [0,1] (Choosing 'All' will decide threshold for entire image group).
%choiceVAR05 = MoG Global
%choiceVAR05 = MoG Adaptive
%choiceVAR05 = Otsu Global
%choiceVAR05 = Otsu Adaptive
%choiceVAR05 = All
%choiceVAR05 = Test Mode
Threshold = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu custom

%textVAR06 = Threshold correction factor
%defaultVAR06 = 1
ThresholdCorrection = str2num(char(handles.Settings.VariableValues{CurrentModuleNum,6}));

%textVAR07 = Lower bound on threshold in the range [0,1].
%defaultVAR07 = 0
MinimumThreshold = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = Approximate percentage of image covered by objects (for MoG thresholding only):
%choiceVAR08 = 10%
%choiceVAR08 = 20%
%choiceVAR08 = 30%
%choiceVAR08 = 40%
%choiceVAR08 = 50%
%choiceVAR08 = 60%
%choiceVAR08 = 70%
%choiceVAR08 = 80%
%choiceVAR08 = 90%
pObject = char(handles.Settings.VariableValues{CurrentModuleNum,8});
%inputtypeVAR08 = popupmenu

%textVAR09 = Set the number of pixels by which to expand the primary objects, ONLY if identifying by DISTANCE [Positive number]
%defaultVAR09 = 10
DistanceToDilate = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,9}));

%textVAR10 = Regularization factor, ONLY if identifying by PROPAGATION or DISTANCE - B (0 to infinity). Larger=distance,0=intensity
%defaultVAR10 = 0.05
RegularizationFactor = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,10}));

%textVAR11 = What do you want to call the image of the outlines of the objects?
%choiceVAR11 = Do not save
%choiceVAR11 = OutlinedNuclei
%infotypeVAR11 = imagegroup indep
SaveOutlined = char(handles.Settings.VariableValues{CurrentModuleNum,11});
%inputtypeVAR11 = popupmenu custom

%textVAR12 = What do you want to call the labeled matrix image?
%choiceVAR12 = Do not save
%choiceVAR12 = LabeledNuclei
%infotypeVAR12 = imagegroup indep
SaveColored = char(handles.Settings.VariableValues{CurrentModuleNum,12});
%inputtypeVAR12 = popupmenu custom

%textVAR13 = Do you want to save the labeled matrix image in RGB or grayscale?
%choiceVAR13 = RGB
%choiceVAR13 = Grayscale
SaveMode = char(handles.Settings.VariableValues{CurrentModuleNum,13});
%inputtypeVAR13 = popupmenu

%%%VariableRevisionNumber = 2

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
    error(['Image processing was canceled in the ', ModuleName, ' module because it could not find the input image.  It was supposed to be named ', ImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
OrigImage = handles.Pipeline.(fieldname);

%%% Checks that the original image is two-dimensional (i.e. not a color
%%% image), which would disrupt several of the image functions.
if ndims(OrigImage) ~= 2
    error(['Image processing was canceled in the ', ModuleName, ' module because it requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.'])
end

%%% Retrieves the preliminary label matrix image that contains the primary
%%% segmented objects which have only been edited to discard objects
%%% that are smaller than a certain size.  This image
%%% will be used as markers to segment the secondary objects with this
%%% module.  Checks first to see whether the appropriate image exists.
fieldname = ['SmallRemovedSegmented', PrimaryObjectName];
%%% Checks whether the image exists in the handles structure.
if isfield(handles.Pipeline, fieldname)==0,
    error(['Image processing was canceled in the ', ModuleName, ' module. Prior to running the Identify Secondary Propagate module, you must have previously run a module that generates an image with the preliminary primary objects identified.  You specified in the Identify Secondary Propagate module that the primary objects were named ', PrimaryObjectName, ' as a result of the previous module, which should have produced an image called ', fieldname, ' in the handles structure.  The Identify Secondary Propagate module cannot locate this image.']);
end
PrelimPrimaryLabelMatrixImage = handles.Pipeline.(fieldname);

%%% Retrieves the label matrix image that contains the edited primary
%%% segmented objects which will be used to weed out which objects are
%%% real - not on the edges and not below or above the specified size
%%% limits. Checks first to see whether the appropriate image exists.
fieldname = ['Segmented', PrimaryObjectName];
%%% Checks whether the image exists in the handles structure.
if isfield(handles.Pipeline, fieldname)==0,
    error(['Image processing was canceled in the ', ModuleName, ' module. Prior to running the Identify Secondary Propagate module, you must have previously run a module that generates an image with the preliminary primary objects identified.  You specified in the Identify Secondary Propagate module that the primary objects were named ', PrimaryObjectName, ' as a result of the previous module, which should have produced an image called ', fieldname, ' in the handles structure.  The Identify Secondary Propagate module cannot locate this image.']);
end
EditedPrimaryLabelMatrixImage = handles.Pipeline.(fieldname);

%%% Check that the sizes of the images are equal.
if (size(OrigImage) ~= size(EditedPrimaryLabelMatrixImage)) | (size(OrigImage) ~= size(PrelimPrimaryLabelMatrixImage))
    error(['Image processing was canceled in the ', ModuleName, ' module. The incoming images are not all of equal size.']);
end

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% STEP 1: Marks at least some of the background by applying a
%%% weak threshold to the original image of the secondary objects.
drawnow
%%% Determines the threshold to use.
[handles,Threshold] = CPthreshold(handles,Threshold,pObject,MinimumThreshold,ThresholdCorrection,OrigImage,ImageName,ModuleName);
guidata(handles.figure1,handles);

%%% Thresholds the original image.
ThresholdedOrigImage = im2bw(OrigImage, Threshold);

if strncmp(IdentChoice,'Distance',8)
    if strcmp(IdentChoice(12),'N')
        %%% Creates the structuring element using the user-specified size.
        StructuringElement = strel('disk', DistanceToDilate);
        %%% Dilates the preliminary label matrix image (edited for small only).
        DilatedPrelimSecObjectLabelMatrixImage = imdilate(PrelimPrimaryLabelMatrixImage, StructuringElement);
        %%% Converts to binary.
        DilatedPrelimSecObjectBinaryImage = im2bw(DilatedPrelimSecObjectLabelMatrixImage,.5);
        %%% Computes nearest neighbor image of nuclei centers so that the dividing
        %%% line between secondary objects is halfway between them rather than
        %%% favoring the primary object with the greater label number.
        [ignore, Labels] = bwdist(full(PrelimPrimaryLabelMatrixImage>0)); %#ok We want to ignore MLint error checking for this line.
        drawnow
        %%% Remaps labels in Labels to labels in PrelimPrimaryLabelMatrixImage.
        ExpandedRelabeledDilatedPrelimSecObjectImage = PrelimPrimaryLabelMatrixImage(Labels);
        %%% Removes the background pixels (those not labeled as foreground in the
        %%% DilatedPrelimSecObjectBinaryImage). This is necessary because the
        %%% nearest neighbor function assigns *every* pixel to a nucleus, not just
        %%% the pixels that are part of a secondary object.
        %%% TODO: This is where we would put in thresholding, if we add this as
        %%% an option in the future.
        RelabeledDilatedPrelimSecObjectImage = zeros(size(ExpandedRelabeledDilatedPrelimSecObjectImage));
        RelabeledDilatedPrelimSecObjectImage(DilatedPrelimSecObjectBinaryImage) = ExpandedRelabeledDilatedPrelimSecObjectImage(DilatedPrelimSecObjectBinaryImage);
        drawnow
    elseif strcmp(IdentChoice(12),'B')
        [labels_out,d]=IdentifySecPropagateSubfunction(PrelimPrimaryLabelMatrixImage,OrigImage,ThresholdedOrigImage,1.0);
        labels_out(d>DistanceToDilate) = 0;
        labels_out((PrelimPrimaryLabelMatrixImage > 0)) = PrelimPrimaryLabelMatrixImage((PrelimPrimaryLabelMatrixImage > 0));
        RelabeledDilatedPrelimSecObjectImage = labels_out;
    end

    EditedPrimaryBinaryImage = im2bw(EditedPrimaryLabelMatrixImage,.5);

    %%% Removes objects that are not in the edited EditedPrimaryLabelMatrixImage.
    LookUpTable = sortrows(unique([PrelimPrimaryLabelMatrixImage(:) EditedPrimaryLabelMatrixImage(:)],'rows'),1);
    b=zeros(max(LookUpTable(:,1)+1),2);
    b(LookUpTable(:,1)+1,1)=LookUpTable(:,1);
    b(LookUpTable(:,1)+1,2)=LookUpTable(:,2);
    b(:,1) = 0:length(b)-1;
    LookUpColumn = b(:,2);
    FinalLabelMatrixImage = LookUpColumn(RelabeledDilatedPrelimSecObjectImage+1);

elseif strcmp(IdentChoice,'Propagation')
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
elseif strcmp(IdentChoice,'Watershed')
    %%% In order to use the watershed transform to find dividing lines between
    %%% the secondary objects, it is necessary to identify the foreground
    %%% objects and to identify a portion of the background.  The foreground
    %%% objects are retrieved as the binary image of primary objects from the
    %%% previously run image analysis module.   This forces the secondary
    %%% object's outline to extend at least as far as the edge of the primary
    %%% objects.

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
end

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
if any(findobj == ThisModuleFigureNumber) == 1 | strncmpi(SaveColored,'Y',1) == 1 | strncmpi(SaveOutlined,'Y',1) == 1
    %%% Calculates the ColoredLabelMatrixImage for displaying in the figure
    %%% window in subplot(2,2,2).
    %%% Note that the label2rgb function doesn't work when there are no objects
    %%% in the label matrix image, so there is an "if".
    if sum(sum(FinalLabelMatrixImage)) >= 1
        ColoredLabelMatrixImage = CPlabel2rgb(handles,FinalLabelMatrixImage);
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
    CPfigure(handles,ThisModuleFigureNumber);
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,2,1); imagesc(OrigImage);
    title(['Input Image, Image Set # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the colored label
    %%% matrix image.
    subplot(2,2,2); imagesc(ColoredLabelMatrixImage); title(['Segmented ',SecondaryObjectName]);
    %%% A subplot of the figure window is set to display the original image
    %%% with secondary object outlines drawn on top.
    subplot(2,2,3); imagesc(ObjectOutlinesOnOrigImage);  title([SecondaryObjectName, ' Outlines on Input Image']);
    %%% A subplot of the figure window is set to display the original
    %%% image with outlines drawn for both the primary and secondary
    %%% objects.
    subplot(2,2,4); imagesc(BothOutlinesOnOrigImage);  title(['Outlines of ', PrimaryObjectName, ' and ', SecondaryObjectName, ' on Input Image']);
    CPFixAspectRatio(OrigImage);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the final, segmented label matrix image of secondary objects to
%%% the handles structure so it can be used by subsequent modules.
fieldname = ['Segmented',SecondaryObjectName];
handles.Pipeline.(fieldname) = FinalLabelMatrixImage;

if strcmp(IdentChoice,'Propagation') || strcmp(IdentChoice,'Propagation')
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
end

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