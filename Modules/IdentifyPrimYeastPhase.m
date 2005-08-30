function handles = IdentifyPrimYeastPhase(handles)

% Help for the Identify Primary Yeast Phase module:
% Category: Object Processing
%
% This module contains code contributed by Ben Kaufmann of MIT.
%
% This module has been designed to identify yeast cells
% in phase contrast images.
%
% Settings:
%
% Size range: You may exclude objects that are smaller or bigger than
% the size range you specify. A comma should be placed between the
% lower size limit and the upper size limit. The units here are pixels
% so that it is easy to zoom in on found objects and determine the
% size of what you think should be excluded.
%
% Threshold: The threshold affects the identification of the objects
% in a rather complicated way that will not be decribed here (see the
% code itself). You may enter an absolute number (which may be
% negative or positive - use the image tool 'Show pixel data' to see
% the pixel intensities on the relevant image which is labeled
% "Inverted enhanced contrast image"), or you may have it
% automatically calculated for each image individually by typing 0.
% There are advantages either way.  An absolute number treats every
% image identically, but an automatically calculated threshold is more
% realistic/accurate, though occasionally subject to artifacts.  The
% threshold which is used for each image is recorded as a measurement
% in the output file, so if you find unusual measurements from one of
% your images, you might check whether the automatically calculated
% threshold was unusually high or low compared to the remaining
% images.  When an automatic threshold is selected, it may
% consistently be too stringent or too lenient, so an adjustment
% factor can be entered as well. The number 1 means no adjustment, 0
% to 1 makes the threshold more lenient and greater than 1 (e.g. 1.3)
% makes the threshold more stringent. The minimum allowable threshold
% prevents an unreasonably low threshold from counting noise as
% objects when there are no bright objects in the field of view. This
% is intended for use with automatic thresholding; a number entered
% here will override an absolute threshold. The value -Inf will cause
% the threshold specified either absolutely or automatically to always
% be used; this is recommended for this module.
%
% Minimum possible diameter of a real object: This determines how much
% objects will be eroded. Keep in mind that this should not be set
% very stringently (that is, you should set it to a lower value that
% the real minimum acceptable diameter of an object), because during
% the first thresholding step sometimes objects appear a bit smaller
% than their final, actual size.
%
% Several other variables are ill-defined so far; they were
% empirically determined.  They probably have
% some relationship to the typical object's diameter, but we haven't
% characterized them well yet.....
%
% SmallValue1 LargeValue1:
% Used in this line:   disks=[SmallValue1 LargeValue1];
% These are masks passed over the image to select for objects with a
% radius in the range of SmallValue1-LargeValue1 pixels. These can be thought of as the
% smallest and largest feasible radii
%
% Value2:
% Used in this line:   OrigImageMinima = imopen(BWerode,strel('disk', Value2));
%
% Value3:
% Used in this line:   BWsmoothed  = imclose(BW,strel('disk', Value3));
%
% Value4:
% Used in this line:   PrelimLabelMatrixImage1 = imopen(WS,strel('disk', Value4))
%
% How it works:
% This image analysis module identifies objects by finding peaks in
% intensity, after the image has been blurred to remove texture (based
% on blur radius).  Once a marker for each object has been identified
% in this way, a watershed function identifies the lines between
% objects that are touching each other by looking for the dimmest
% points between them. To identify the edges of non-clumped objects, a
% simple threshold is applied. Objects on the border of the image are
% ignored, and the user can select a size range, outside which objects
% will be ignored.
%
% SAVING IMAGES: In addition to the object outlines and the
% pseudo-colored object images that can be saved using the
% instructions in the main CellProfiler window for this module,
% this module produces several additional images which can be
% easily saved using the Save Images module. These will be grayscale
% images where each object is a different intensity. (1) The
% preliminary segmented image, which includes objects on the edge of
% the image and objects that are outside the size range can be saved
% using the name: UneditedSegmented + whatever you called the objects
% (e.g. UneditedSegmentedNuclei). (2) The preliminary segmented image
% which excludes objects smaller than your selected size range can be
% saved using the name: SmallRemovedSegmented + whatever you called the
% objects (e.g. SmallRemovedSegmented Nuclei) (3) The final segmented
% image which excludes objects on the edge of the image and excludes
% objects outside the size range can be saved using the name:
% Segmented + whatever you called the objects (e.g. SegmentedNuclei)
%
%    Additional image(s) are calculated by this module and can be 
% saved by altering the code for the module (see the SaveImages module
% help for instructions).
%
% See also IDENTIFYPRIMADAPTTHRESHOLDA,
% IDENTIFYPRIMADAPTTHRESHOLDB,
% IDENTIFYPRIMADAPTTHRESHOLDC,
% IDENTIFYPRIMADAPTTHRESHOLDD,
% IDENTIFYPRIMTHRESHOLD,
% IDENTIFYPRIMSHAPEDIST,
% IDENTIFYPRIMSHAPEINTENS.

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

%textVAR02 = What do you want to call the objects identified by this module?
%infotypeVAR02 = objectgroup indep
%defaultVAR02 = Yeast
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Size range (in pixels) of objects to include (1,99999 = do not discard any)
%defaultVAR03 = 1,99999
SizeRange = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = Enter the threshold (Positive number, Max = 1):
%choiceVAR04 = Automatic
Threshold = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu custom

%textVAR05 = If auto threshold, enter an adjustment factor (Positive number, >1 = more stringent, <1 = less stringent, 1 = no adjustment):
%defaultVAR05 = 1
ThresholdAdjustmentFactor = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,5}));

%textVAR06 = Enter the minimum allowable threshold (this prevents an unreasonably low threshold from counting noise as objects when there are no bright objects in the field of view. This is intended for use with automatic thresholding; a number entered here will override an absolute threshold entered two boxes above). The value -Inf will cause the threshold specified above to always be used; this is recommended for this module
%defaultVAR06 = -Inf
MinimumThreshold = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,6}));

%textVAR07 = Minimum possible radius of a real object (even number, in pixels)
%defaultVAR07 = 7
ErodeSize = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,7}));

%textVAR08 = Do you want to include objects touching the edge (border) of the image? (Yes or No)
%choiceVAR08 = No
%choiceVAR08 = Yes
IncludeEdge = char(handles.Settings.VariableValues{CurrentModuleNum,8});
%inputtypeVAR08 = popupmenu

%textVAR09 = What do you want to call the image of the outlines of the objects?
%infotypeVAR09 = imagegroup indep
%defaultVAR09 = Do not save
SaveOutlined = char(handles.Settings.VariableValues{CurrentModuleNum,9}); 

%textVAR10 =  What do you want to call the labeled matrix image?
%infotypeVAR10 = imagegroup indep
%defaultVAR10 = Do not save
SaveColored = char(handles.Settings.VariableValues{CurrentModuleNum,10}); 

%textVAR11 = Do you want to save the labeled matrix image in RGB or grayscale?
%choiceVAR11 = RGB
%choiceVAR11 = Grayscale
SaveMode = char(handles.Settings.VariableValues{CurrentModuleNum,11}); 
%inputtypeVAR11 = popupmenu

%textVAR12 = Enter the SmallValue1 (even number, in pixels)
%defaultVAR12 = 8
SmallValue1 = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,12}));

%textVAR13 = Enter the LargeValue1 (even number, in pixels)
%defaultVAR13 = 14
LargeValue1 = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,13}));

%textVAR14 = Enter the Value2 (integer, in pixels)
%defaultVAR14 = 3
Value2 = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,14}));

%textVAR15 = Enter the Value3 (integer, in pixels)
%defaultVAR15 = 3
Value3 = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,15}));

%textVAR16 = Enter the Value4 (integer, in pixels)
%defaultVAR16 = 6
Value4 = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,16}));

%%% Determines what the user entered for the size range.
SizeRangeNumerical = str2num(SizeRange); %#ok We want to ignore MLint error checking for this line.
MinSize = SizeRangeNumerical(1);
MaxSize = SizeRangeNumerical(2);

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Reads (opens) the image you want to analyze and assigns it to a variable,
%%% "OrigImage".
fieldname = ['',  ImageName];

%%% Checks whether the image exists in the handles structure.
if isfield(handles.Pipeline, fieldname)==0,
    error(['Image processing has been canceled. Prior to running the Identify Primary Intensity module, you must have previously run a module to load an image. You specified in the Identify Primary Intensity module that this image was called ', ImageName, ' which should have produced a field in the handles structure called ', fieldname, '. The Identify Primary Intensity module cannot find this image.']);
end
OrigImage = handles.Pipeline.(fieldname);

%%% Checks that the original image is two-dimensional (i.e. not a color
%%% image), which would disrupt several of the image functions.
if ndims(OrigImage) ~= 2
    error('Image processing was canceled because the Identify Primary Intensity module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
end

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%



%%% Diameter entry is converted to radius and made into an integer.
ErodeSize = fix(ErodeSize/2);

%%% Normalize the image.
OrigImage = OrigImage/mean(mean(OrigImage));
drawnow

%%% Invert the image so black is white.
InvertedOrigImage = imcomplement(OrigImage);

%% Enhance image for objects of a given size range
EnhancedInvertedImage = InvertedOrigImage;
disks=[SmallValue1 LargeValue1];
for i=1:length(disks)
    mask        = strel('disk',disks(i));
    top         = imtophat(InvertedOrigImage,mask);
    bot         = imbothat(InvertedOrigImage,mask);
    EnhancedInvertedImage    = imsubtract(imadd(EnhancedInvertedImage,top), bot);
    drawnow
end
figure, imshow(EnhancedInvertedImage)

%%% Determines the threshold to be used, if the user has left the Threshold
%%% variable set to 0.
if strcmp(Threshold,'Automatic')
    Threshold = CPgraythresh(EnhancedInvertedImage,handles,ImageName);
    %%% Replaced the following line to accomodate calculating the
    %%% threshold for images that have been masked.
    %    Threshold = CPgraythresh(EnhancedInvertedImage);
    Threshold = Threshold*ThresholdAdjustmentFactor;
else
    Threshold=str2double(Threshold);
end
Threshold = max(MinimumThreshold,Threshold);

%%%  1. Threshold for edges
%%% We cannot use the built in Matlab function
%%% im2bw(EnhancedInvertedImage,Threshold) to threshold the
%%% EnhancedInvertedImage image, because it does not allow using a
%%% threshold outside the range 0 to 1.  So we will use this instead:
BW = EnhancedInvertedImage;
BW(BW>Threshold) = 1;
BW(BW<=Threshold) = 0;
drawnow
figure, imshow(BW)

%%  2. Erode edges so only centers remain
BWerode = imerode(BW,strel('disk', ErodeSize));
drawnow
figure, imshow(BWerode)
%%  3. Clean it up
OrigImageMinima = imopen(BWerode,strel('disk', Value2));
drawnow

%% Segment the image with watershed
WS = watershed(imcomplement(OrigImageMinima));
figure, imshow(OrigImageMinima)
%% Watershed regions are irregularly shaped.
%% To fix the edges: Smooth BW border, then impose this border onto the WS
BWsmoothed  = imclose(BW,strel('disk',Value3));
WS          = immultiply(WS,BW);
drawnow
figure, imshow(WS)

%% Smooth the edges
PrelimLabelMatrixImage1 = imopen(WS,strel('disk', Value4));
figure, imshow(PrelimLabelMatrixImage1)
drawnow

%%% Fills holes, then identifies objects in the binary image.
%PrelimLabelMatrixImage1 = bwlabel(imfill(InvertedBinaryImage,'holes'));
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
PrelimLabelMatrixImage2 = bwlabel(im2bw(PrelimLabelMatrixImage2,.5));
%%% The large objects are overwritten with zeros.
PrelimLabelMatrixImage3 = PrelimLabelMatrixImage2;
if MaxSize ~= 99999
    PrelimLabelMatrixImage3(AreasImage > MaxSize) = 0;
end
%%% Removes objects that are touching the edge of the image, since they
%%% won't be measured properly.
if strncmpi(IncludeEdge,'N',1) == 1
    PrelimLabelMatrixImage4 = imclearborder(PrelimLabelMatrixImage3,8);
else PrelimLabelMatrixImage4 = PrelimLabelMatrixImage3;
end
%%% The PrelimLabelMatrixImage4 is converted to binary.
FinalBinaryPre = im2bw(PrelimLabelMatrixImage4,0.5);
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



fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
if any(findobj == ThisModuleFigureNumber) == 1 | strncmpi(SaveColored,'Y',1) == 1 | strncmpi(SaveOutlined,'Y',1) == 1
    %%% Calculates the ColoredLabelMatrixImage for displaying in the figure
    %%% window in subplot(2,2,2).
    %%% Note that the label2rgb function doesn't work when there are no objects
    %%% in the label matrix image, so there is an "if".
    if sum(sum(FinalLabelMatrixImage)) >= 1
        cmap = jet(max(64,max(FinalLabelMatrixImage(:))));
        ColoredLabelMatrixImage = label2rgb(FinalLabelMatrixImage, cmap, 'k', 'shuffle');
    else  ColoredLabelMatrixImage = FinalLabelMatrixImage;
    end
    %%% Calculates the object outlines, which are overlaid on the original
    %%% image and displayed in figure subplot (2,2,4).
    %%% Creates the structuring element that will be used for dilation.
    StructuringElement = strel('square',3);
    %%% Converts the FinalLabelMatrixImage to binary.
    FinalBinaryImage = im2bw(FinalLabelMatrixImage,0.5);
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

    drawnow
    CPfigure(handles,ThisModuleFigureNumber);
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,2,1); imagesc(OrigImage);CPcolormap(handles);
    title(['Input Image, Image Set # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the colored label
    %%% matrix image.
    subplot(2,2,2); imagesc(FinalLabelMatrixImage); CPcolormap(handles);title(['Segmented ',ObjectName]);
    %%% A subplot of the figure window is set to display the Overlaid image,
    %%% where the maxima are imposed on the inverted original image
    subplot(2,2,3); imagesc(EnhancedInvertedImage); CPcolormap(handles); title(['Inverted enhanced contrast image']);
    %%% A subplot of the figure window is set to display the inverted original
    %%% image with watershed lines drawn to divide up clusters of objects.
    subplot(2,2,4); imagesc(ObjectOutlinesOnOrigImage);CPcolormap(handles); title([ObjectName, ' Outlines on Input Image']);
    CPFixAspectRatio(OrigImage);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow



%%% Saves the segmented image, not edited for objects along the edges or
%%% for size, to the handles structure.
fieldname = ['UneditedSegmented',ObjectName];
handles.Pipeline.(fieldname) = PrelimLabelMatrixImage1;

%%% Saves the segmented image, only edited for small objects, to the
%%% handles structure.
fieldname = ['SmallRemovedSegmented',ObjectName];
handles.Pipeline.(fieldname) = PrelimLabelMatrixImage2;

%%% Saves the final segmented label matrix image to the handles structure.
fieldname = ['Segmented',ObjectName];
handles.Pipeline.(fieldname) = FinalLabelMatrixImage;

%%% Saves the Threshold value to the handles structure.
%%% Storing the threshold is a little more complicated than storing other measurements
%%% because several different modules will write to the handles.Measurements.Image.Threshold
%%% structure, and we should therefore probably append the current threshold to an existing structure
% First, if the Threshold fields don't exist, initialize them
if ~isfield(handles.Measurements.Image,'ThresholdFeatures')
    handles.Measurements.Image.ThresholdFeatures = {};
    handles.Measurements.Image.Threshold = {};
end
% Search the ThresholdFeatures to find the column for this object type
column = find(~cellfun('isempty',strfind(handles.Measurements.Image.ThresholdFeatures,ObjectName)));
% If column is empty it means that this particular object has not been segmented before. This will
% typically happen for the first image set. Append the feature name in the
% handles.Measurements.Image.ThresholdFeatures matrix
if isempty(column)
    handles.Measurements.Image.ThresholdFeatures(end+1) = {['Threshold ' ObjectName]};
    column = length(handles.Measurements.Image.ThresholdFeatures);
end
handles.Measurements.Image.Threshold{handles.Current.SetBeingAnalyzed}(1,column) = Threshold;


%%% Saves the ObjectCount, i.e. the number of segmented objects.
%%% See comments for the Threshold saving above
if ~isfield(handles.Measurements.Image,'ObjectCountFeatures')
    handles.Measurements.Image.ObjectCountFeatures = {};
    handles.Measurements.Image.ObjectCount = {};
end
column = find(~cellfun('isempty',strfind(handles.Measurements.Image.ObjectCountFeatures,ObjectName)));
if isempty(column)
    handles.Measurements.Image.ObjectCountFeatures(end+1) = {['ObjectCount ' ObjectName]};
    column = length(handles.Measurements.Image.ObjectCountFeatures);
end
handles.Measurements.Image.ObjectCount{handles.Current.SetBeingAnalyzed}(1,column) = max(FinalLabelMatrixImage(:));


%%% Saves the location of each segmented object
handles.Measurements.(ObjectName).LocationFeatures = {'CenterX','CenterY'};
tmp = regionprops(FinalLabelMatrixImage,'Centroid');
Centroid = cat(1,tmp.Centroid);
handles.Measurements.(ObjectName).Location(handles.Current.SetBeingAnalyzed) = {Centroid};

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
