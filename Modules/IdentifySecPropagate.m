function handles = IdentifySecPropagate(handles)

% Help for the Identify Secondary Propagate module:
% Category: Object Processing
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
% small or large have been excluded).  The SmallRemovedSegmented
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




drawnow

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%



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
%defaultVAR05 = 1
ThresholdAdjustmentFactor = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,5}));

%textVAR06 = Enter the minimum allowable threshold (Range = 0 to 1; this prevents an unreasonably low threshold from counting noise as objects when there are no bright objects in the field of view). This is intended for use with automatic thresholding, but will override an absolute threshold entered above:
%defaultVAR06 = 0
MinimumThreshold = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = Regularization factor (0 to infinity). Larger=distance,0=intensity
%defaultVAR07 = 0.05
RegularizationFactor = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,7}));

%textVAR08 = What do you want to call the image of the outlines of the objects?
%choiceVAR08 = Do not save
%choiceVAR08 = OutlinedNuclei
SaveOutlined = char(handles.Settings.VariableValues{CurrentModuleNum,8}); 
%inputtypeVAR08 = popupmenu custom

%textVAR09 =  What do you want to call the labeled matrix image?
%infotypeVAR09 = imagegroup indep
%choiceVAR09 = Do not save
%choiceVAR09 = LabeledNuclei
SaveColored = char(handles.Settings.VariableValues{CurrentModuleNum,9}); 
%inputtypeVAR09 = popupmenu custom

%textVAR10 = Do you want to save the labeled matrix image in RGB or grayscale?
%choiceVAR10 = RGB
%choiceVAR10 = Grayscale
SaveMode = char(handles.Settings.VariableValues{CurrentModuleNum,10}); 
%inputtypeVAR10 = popupmenu

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
fieldname = ['SmallRemovedSegmented', PrimaryObjectName];
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

%%% Check that the sizes of the images are equal.
if (size(OrigImage) ~= size(EditedPrimaryLabelMatrixImage)) | (size(OrigImage) ~= size(PrelimPrimaryLabelMatrixImage))
    error(['Image processing has been canceled. The incoming images are not all of equal size.']);
end


%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow



%%% STEP 1: The distinction between objects and background is determined
%%% using the user-specified threshold.
%%% Determines the threshold to use.
if strcmp(Threshold,'Automatic')
 %   Threshold = CPgraythresh(OrigImage,handles,ImageName);
    %%% Replaced the following line to accomodate calculating the
    %%% threshold for images that have been masked.
%    Threshold = CPgraythresh(OrigImage);
    %%% Adjusts the threshold by a correction factor.
%    Threshold = Threshold*ThresholdAdjustmentFactor;


    Threshold=graythresh(OrigImage);
    ThresholdedOrigImage = im2bw(OrigImage, Threshold);
    while numel(nonzeros(ThresholdedOrigImage & PrelimPrimaryLabelMatrixImage))/numel(nonzeros(PrelimPrimaryLabelMatrixImage))<.95;
        Threshold=Threshold-0.002;
        ThresholdedOrigImage = im2bw(OrigImage, Threshold);
    end
else
    Threshold=str2double(Threshold);
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



fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
if any(findobj == ThisModuleFigureNumber) == 1 | strncmpi(SaveColored,'Y',1) == 1 | strncmpi(SaveOutlined,'Y',1) == 1
    %%% Calculates the ColoredLabelMatrixImage for displaying in the figure
    %%% window in subplot(2,2,2).
    %%% Note that the label2rgb function doesn't work when there are no objects
    %%% in the label matrix image, so there is an "if".
    if sum(sum(FinalLabelMatrixImage)) >= 1
        cmap = jet(max(64,max(FinalLabelMatrixImage(:))));
        ColoredLabelMatrixImage = label2rgb(FinalLabelMatrixImage,cmap, 'k', 'shuffle');
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