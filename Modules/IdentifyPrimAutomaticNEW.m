function handles = IdentifyPrimAutomatic(handles)

% Help for the Identify Primary Automatic module:
% Category: Object Processing
%
% General module for identifying (segmenting) primary objects in
% grayscale images that show bright objects on a dark background. The
% module has many options which vary in terms of speed and
% sophistication.
%
% Requirements for the images to be fed into this module:
% * If the objects are dark on a light background, use the
% InvertIntensity module prior to running this module.
% * If you are working with color images, they must first be converted
% to grayscale using the RGBSplit or RGBToGray module.
%
% Settings:
%
% Typical diameter of objects, in pixel units (Min,Max):
% This is a very important parameter which tells the module what you
% are looking for. Most options within this module use this estimate
% of the size range of the objects in order to distinguish them from
% noise in the image. For example, for some of the identification
% methods, the smoothing applied to the image is based on the minimum
% size of the objects.  A comma should be placed between the minimum
% and the maximum diameters. The units here are pixels so that it is
% easy to zoom in on found objects and determine the size of what you
% think should be excluded. To measure distances easily, use the
% CellProfiler Image Tool, 'Show Or Hide Pixel Data', in any open
% window. Once this tool is activated, you can draw a line across
% objects in your image and the length of the line will be shown in
% pixel units. Note that for non-round objects, the diameter here is
% actually the 'equivalent diameter', meaning the diameter of a circle
% with the same area as the object.
%
% Discard objects outside the diameter range:
% Objects outside the specified range of diameters will be discarded.
% This allows you to exclude small objects like dust, noise, and
% debris, or large objects like clumps if desired. See also the
% FilterByAreaShape module to further discard objects based on
% some other Area or Shape measurement. During processing, the window
% for this module will show that objects outlined in green were an
% acceptable size, objects outlined in red were discarded based on
% their size, and objects outlined in yellow were discarded because
% they touch the border.
%
% Try to merge 'too small' objects with nearby larger objects:
% Use caution when choosing 'Yes' for this option! 
% This is an experimental functionality that takes objects that were
% discarded because they were smaller than the specified Minimum
% diameter and tries to merge them with other surrounding objects.
% This is helpful in cases when an object was incorrectly split into
% two objects, one of which is actually just a tiny piece of the
% larger object. However, this could be dangerous if you have selected
% poor settings which produce lots of tiny objects - the module will
% take a very long time and you won't realize that it's because the
% tiny objects are being merged. It is therefore a good idea to run
% the module first without merging objects to make sure the settings
% are reasonably effective.
%
% Discard objects touching the border of the image:
% For most applications, you do not want to make measurements of
% objects that are not fully within the field of view (because, for
% example, the area would not be accurate) so they should be
% discarded.
%
% Select thresholding method or enter a threshold:
%    The threshold affects the stringency of the lines between the
% objects and the background. You may enter an absolute number between
% 0 and 1 for the threshold (in any image window use the CellProfiler
% Image Tool, 'Show Or Hide Pixel Data', to see the pixel intensities
% for your images in the appropriate range of 0 to 1), or you may
% choose to have it automatically calculated using several methods.
% There are advantages either way.  An absolute number treats every
% image identically, but is not robust to slight changes in
% lighting/staining conditions between images. An automatically
% calculated threshold adapts to changes in lighting/staining
% conditions between images and is usually more robust/accurate, but
% it can occasionally produce a poor threshold for unusual/artifactual
% images. It also takes a small amount of time to calculate.
%    The threshold which is used for each image is recorded as a
% measurement in the output file, so if you find unusual measurements
% from one of your images, you might check whether the automatically
% calculated threshold was unusually high or low compared to the
% other images.
%    There are two methods for finding thresholds automatically,
% Otsu's method and the Mixture of Gaussian (MoG) method. The Otsu
% method uses CPgraythresh, which is a modification of the Matlab
% function 'graythresh'. Our modifications include taking into account
% the max and min values in the image and log-transforming the image
% prior to calculating the threshold. Otsu's method is probably better
% if you don't know anything about the image. But if you can supply
% the object coverage percentage the MoG can be better, especially if
% the coverage percentage differs much from 50%. Note however that the
% MoG function is experimental and has not been thoroughly validated.
%    You can also choose between global and adaptive thresholding,
% where global means that one threshold is used for the entire image
% and adaptive means that the threshold varies across the image.
% Adaptive is slower to calculate but provides more accurate edge
% determination which may help to separate clumps, especially if you
% are not using a clump-separation method (see below).
%
% Threshold correction factor:
% When the threshold is calculated automatically, it may consistently
% be too stringent or too lenient.  For example, the Otsu automatic
% thresholding inherently assumes that 50% of the image is covered by
% objects. If a larger percentage of the image is covered, the Otsu
% method will give a slightly biased threshold that may have to be
% corrected. In a future version, the 'Threshold correction factor'
% may be removed and the "Approximate percentage covered by objects"
% information used instead.  For now, however, you may need to enter an
% adjustment factor which you empirically determine is suitable for
% your images. The number 1 means no adjustment, 0 to 1 makes the
% threshold more lenient and greater than 1 (e.g. 1.3) makes the
% threshold more stringent.
%
% Lower bound on threshold:
% Can be used as a safety precaution when the threshold is calculated
% automatically. If there are no objects in the field of view, the
% automatic threshold will be unreasonably low. In such case the lower
% bound you enter here will override the automatic threshold.
%
% Approximate percentage covered by objects:
% An estimate of how much of the image is covered with objects. This
% information is currently only used in the MoG (Mixture of Gaussian)
% thresholding but may be used for other thresholding methods in the
% future (see below). This is not a very sensitive parameter so your
% estimate need not be precise.
%
% Method to distinguish clumped objects:
% * Intensity - For objects that tend to have only one peak of
% brightness per object (e.g. objects that are brighter towards their
% interiors), this option counts each intensity peak as a separate
% object. The objects can be any shape, so they need not be round and
% uniform in size as would be required for a distance-based module.
% The module is more successful when then objects have a smooth
% texture. By default, the image is automatically blurred to attempt
% to achieve appropriate smoothness (see blur option), but overriding
% the default value can improve the outcome on lumpy-textured objects
% {OLa is currently implementing this option}. Technical description:
% Object centers are defined as local intensity maxima.
% * Shape - For cases when there are definite indentations separating
% objects. This works best for objects that are round. The intensity
% patterns in the original image are irrelevant - the image is
% converted to black and white (binary) and the shape is what
% determines whether clumped objects will be distinguished. Therefore,
% the cells need not be brighter towards the interior as is required
% for the Intensity option. Technical description: The binary
% thresholded image is distance-transformed and object centers are
% defined as peaks in this image.
% * None (fastest option) - If objects are far apart and are very well
% separated, it may be unnecessary to attempt to separate clumped
% objects. Using the 'None' option, the thresholded image will be used
% to identify objects.
%
% Method to draw dividing lines between clumped objects:
% * Intensity - works best where the dividing lines between clumped
% objects are dim.
% * Distance - Dividing lines between clumped objects are halfway
% between the 'center' of each object.  The intensity patterns in the
% original image are irrelevant - the cells need not be dimmer along
% the lines between clumped objects.
% * None (fastest option) - If objects are far apart and are very well
% separated, it may be unnecessary to attempt to separate clumped
% objects. Using the 'None' option, the thresholded image will be used
% to identify objects.
%
% Size of smoothing filter, in pixel units:
% If you are distinguishing between clumped objects, the image is
% smoothed based on the specified minimum object diameter that you
% have entered, but you may want to override the automatically
% calculated value here. Reducing the texture of objects by increasing
% the smoothing increases the chance that each real, distinct object
% has only one peak of intensity but also increases the chance that
% two distinct objects will be recognized as only one object. Note
% that increasing the blur radius increases the processing time
% exponentially.
% This variable affects whether objects close by each other are
% considered a single object or multiple objects. It does not affect
% the dividing lines between an object and the background.  If you see
% too many objects merged that ought to be separate, the value should
% be lower. If you see too many objects split up that ought to be
% merged, the value should be higher.
%
% Suppress local maxima within this distance (a positive
% integer, in pixel units):
% If you are distinguishing between clumped objects, object markers
% are suppressed based on the specified minimum object diameter that
% you have entered, but you may want to override the automatically
% calculated value here. The maxima suppression distance should be
% set to be roughly equivalent to the minimum radius of a real object
% of interest. Basically, any distinct 'objects' which are found but
% are within two times this distance from each other will be assumed
% to be actually two lumpy parts of the same object, and they will be
% merged.
% This variable affects whether objects close by each other are
% considered a single object or multiple objects. It does not affect
% the dividing lines between an object and the background.  If you see
% too many objects merged that ought to be separate, the value should
% be lower. If you see too many objects split up that ought to be
% merged, the value should be higher.
%
% Speed up by using lower-resolution image to find local maxima?
% If you are distinguishing between clumped objects
% If you have entered a minimum object diameter of 10 or less, setting this
% option to Yes will have no effect.
%
% SAVING IMAGES (What do you want to call the image... ?):
%    The object outlines, the object outlines overlaid on the original
% image, and the label matrix image [in RGB(color) or grayscale
% format] can be easily saved to the hard drive by giving them a name
% here and then selecting this name in the SaveImages module.
%    This module produces several additional images which can also be
% saved using the Save Images module as follows: (1) The unedited
% segmented image, which includes objects on the edge of the image and
% objects that are outside the size range, can be saved in the
% SaveImages module by entering the name 'UneditedSegmented' +
% whatever you called the objects (e.g. UneditedSegmentedNuclei). This
% is a grayscale image where each object is a different intensity. (2)
% The segmented image which excludes objects smaller than your
% selected size range can be saved in the SaveImages module by
% entering the name 'SmallRemovedSegmented' + whatever you called the
% objects (e.g. SmallRemovedSegmented Nuclei). This is a grayscale
% image where each object is a different intensity. (3) A black and
% white (binary) image of the objects identified can be saved in the
% SaveImages module simply by entering the object name (e.g. 'Nuclei')
% in the SaveImages module.
%    Additional image(s) are calculated by this module and can be
% saved by altering the code for the module (see the SaveImages module
% help for instructions).
%
% Technical notes: The initial step of identifying local maxima is
% performed on the user-controlled heavily smoothed image, the
% foreground/background is done on a hard-coded slightly smoothed
% image, and the dividing lines between clumped objects (watershed) is
% done on the non-smoothed image.
%
% See also <nothing relevant>

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne Carpenter
%   Thouis Jones
%   In Han Kang
%
% $Revision: 1879 $

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%


%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

%%% Sets up loop for test mode.
if strcmp(char(handles.Settings.VariableValues{CurrentModuleNum,21}),'Yes')
    LocalMaximaTypeList = {'Intensity' 'Shape'};
    WatershedTransformImageTypeList = {'Intensity' 'Distance' 'None'};
else
    LocalMaximaTypeList = {char(handles.Settings.VariableValues{CurrentModuleNum,11})};
    WatershedTransformImageTypeList = {char(handles.Settings.VariableValues{CurrentModuleNum,12})};
end

for LocalMaximaTypeNumber = [1:length(LocalMaximaTypeList)]
    for WatershedTransformImageTypeNumber = [1:length(WatershedTransformImageTypeList)]
        
%%% NOTE: We cannot indent the variables or they will not be read
%%% properly.
        
%textVAR01 = What did you call the images you want to process?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the objects identified by this module?
%defaultVAR02 = Nuclei
%infotypeVAR02 = objectgroup indep
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Typical diameter of objects, in pixel units (Min,Max):
%choiceVAR03 = 10,40
SizeRange = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu custom

%textVAR04 = Discard objects outside the diameter range?
%choiceVAR04 = Yes
%choiceVAR04 = No
ExcludeSize = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%textVAR05 = Try to merge too small objects with nearby larger objects?
%choiceVAR05 = No
%choiceVAR05 = Yes
MergeChoice = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu

%textVAR06 = Discard objects touching the border of the image?
%choiceVAR06 = Yes
%choiceVAR06 = No
ExcludeBorderObjects = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%inputtypeVAR06 = popupmenu

%textVAR07 = Select thresholding method or enter a threshold in the range [0,1].
%choiceVAR07 = MoG Global
%choiceVAR07 = MoG Adaptive
%choiceVAR07 = Otsu Global
%choiceVAR07 = Otsu Adaptive
Threshold = char(handles.Settings.VariableValues{CurrentModuleNum,7});
%inputtypeVAR07 = popupmenu custom

%textVAR08 = Threshold correction factor
%defaultVAR08 = 1
ThresholdCorrection = str2num(char(handles.Settings.VariableValues{CurrentModuleNum,8}));

%textVAR09 = Lower bound on threshold in the range [0,1].
%defaultVAR09 = 0
MinimumThreshold = char(handles.Settings.VariableValues{CurrentModuleNum,9});

%textVAR10 = Approximate percentage of image covered by objects (for MoG thresholding only):
%choiceVAR10 = 10%
%choiceVAR10 = 20%
%choiceVAR10 = 30%
%choiceVAR10 = 40%
%choiceVAR10 = 50%
%choiceVAR10 = 60%
%choiceVAR10 = 70%
%choiceVAR10 = 80%
%choiceVAR10 = 90%
pObject = char(handles.Settings.VariableValues{CurrentModuleNum,10});
%inputtypeVAR10 = popupmenu

%textVAR11 = Method to distinguish clumped objects (see help for details):
%choiceVAR11 = Intensity
%choiceVAR11 = Shape
%choiceVAR11 = None
OriginalLocalMaximaType = char(handles.Settings.VariableValues{CurrentModuleNum,11});
%inputtypeVAR11 = popupmenu

LocalMaximaType = LocalMaximaTypeList{LocalMaximaTypeNumber};

%textVAR12 =  Method to draw dividing lines between clumped objects (see help for details):
%choiceVAR12 = Intensity
%choiceVAR12 = Distance
%choiceVAR12 = None
OriginalWatershedTransformImageType = char(handles.Settings.VariableValues{CurrentModuleNum,12});
%inputtypeVAR12 = popupmenu

WatershedTransformImageType = WatershedTransformImageTypeList{WatershedTransformImageTypeNumber};

%textVAR13 = Size of smoothing filter, in pixel units (if you are distinguishing between clumped objects). Enter 0 for low resolution images with small objects (~< 5 pixel diameter) to prevent any image smoothing.
%choiceVAR13 = Automatic
SizeOfSmoothingFilter = char(handles.Settings.VariableValues{CurrentModuleNum,13});
%inputtypeVAR13 = popupmenu custom

%textVAR14 = Suppress local maxima within this distance, (a positive integer, in pixel units) (if you are distinguishing between clumped objects)
%choiceVAR14 = Automatic
MaximaSuppressionSize = char(handles.Settings.VariableValues{CurrentModuleNum,14});
%inputtypeVAR14 = popupmenu custom

%textVAR15 = Speed up by using lower-resolution image to find local maxima?  (if you are distinguishing between clumped objects)
%choiceVAR15 = Yes
%choiceVAR15 = No
UseLowRes = char(handles.Settings.VariableValues{CurrentModuleNum,15});
%inputtypeVAR15 = popupmenu

%textVAR16 = Enter the following information, seperated by commas, if you would like to use Laplacian of Gaussian method: Size of neighborhood(height,width),Sigma,Minimum Area,Size for Weiner Filter(height,width),Threshold
%defaultVAR16 = /
LaplaceValues = char(handles.Settings.VariableValues{CurrentModuleNum,16});

%textVAR17 = What do you want to call the image of the outlines of the objects?
%choiceVAR17 = Do not save
%infotypeVAR17 = imagegroup indep
SaveOutlines = char(handles.Settings.VariableValues{CurrentModuleNum,17});
%inputtypeVAR17 = popupmenu custom

%textVAR18 = What do you want to call the image of the outlines of the objects, overlaid on the original image?
%choiceVAR18 = Do not save
%infotypeVAR18 = imagegroup indep
SaveOutlinedOnOriginal = char(handles.Settings.VariableValues{CurrentModuleNum,18});
%inputtypeVAR18 = popupmenu custom

%textVAR19 =  What do you want to call the labeled matrix image?
%choiceVAR19 = Do not save
%infotypeVAR19 = imagegroup indep
SaveColored = char(handles.Settings.VariableValues{CurrentModuleNum,19});
%inputtypeVAR19 = popupmenu custom

%textVAR20 = Do you want to save the labeled matrix image in RGB or grayscale?
%choiceVAR20 = RGB
%choiceVAR20 = Grayscale
SaveMode = char(handles.Settings.VariableValues{CurrentModuleNum,20});
%inputtypeVAR20 = popupmenu

%textVAR21 = Test Mode?
%choiceVAR21 = No
%choiceVAR21 = Yes
TestMode = char(handles.Settings.VariableValues{CurrentModuleNum,21});
%inputtypeVAR21 = popupmenu

%%%VariableRevisionNumber = 9

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% PRELIMINARY ERROR CHECKING & FILE HANDLING %%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%% Reads (opens) the image you want to analyze and assigns it to a variable,
        %%% "OrigImage".
        fieldname = ['',  ImageName];

        %%% Checks whether the image exists in the handles structure.
        if isfield(handles.Pipeline, fieldname)==0,
            error(['Image processing has been canceled. Prior to running the IdentifyPrimAutomatic Intensity module, you must have previously run a module to load an image. You specified in the IdentifyPrimAutomatic module that this image was called ', ImageName, ' which should have produced a field in the handles structure called ', fieldname, '. The IdentifyPrimAutomatic module cannot find this image.']);
        end
        OrigImage = handles.Pipeline.(fieldname);

        %%% Checks that the original image is two-dimensional (i.e. not a color
        %%% image), which would disrupt several of the image functions.
        if ndims(OrigImage) ~= 2
            error('Image processing was canceled because the IdentifyPrimAutomatic module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
        end
        
        %%% Checks that the Laplace parameters have valid values
        if ~strcmp(LaplaceValues,'/')
            index = strfind(LaplaceValues,',');
            if isempty(index) || (length(index) ~= 6)
                error('The LaplaceValues in the IdentifyPrimAutomatic module is invalid.');
            end
            NeighborhoodSize(1) = str2num(LaplaceValues(1:index(1)-1));
            NeighborhoodSize(2) = str2num(LaplaceValues(index(1)+1:index(2)-1));
            Sigma = str2num(LaplaceValues(index(2)+1:index(3)-1));
            MinArea = str2num(LaplaceValues(index(3)+1:index(4)-1));
            WienerSize(1) = str2num(LaplaceValues(index(4)+1:index(5)-1));
            WienerSize(2) = str2num(LaplaceValues(index(5)+1:index(6)-1));
            LaplaceThreshold = str2num(LaplaceValues(index(6)+1:end));
        end
        
        %%% Checks that the Min and Max diameter parameters have valid values
        index = strfind(SizeRange,',');
        if isempty(index),error('The Min and Max size entry in the IdentifyPrimAutomatic module is invalid.'),end
        MinDiameter = SizeRange(1:index-1);
        MaxDiameter = SizeRange(index+1:end);

        MinDiameter = str2double(MinDiameter);
        if isnan(MinDiameter) | MinDiameter < 0
            error('The Min dimater entry in the IdentifyPrimAutomatic module is invalid.')
        end
        if strcmp(MaxDiameter,'Inf') ,MaxDiameter = Inf;
        else
            MaxDiameter = str2double(MaxDiameter);
            if isnan(MaxDiameter) | MaxDiameter < 0
                error('The Max Diameter entry in the IdentifyPrimAutomatic module is invalid.')
            end
        end
        if MinDiameter > MaxDiameter, error('Min Diameter larger the Max Diameter in the IdentifyPrimAutomatic module.'),end
        Diameter = min((MinDiameter + MaxDiameter)/2,50);

        %%% Convert user-specified percentage of image covered by objects to a prior probability
        %%% of a pixel being part of an object.
        pObject = str2num(pObject(1:2))/100;

        %%% Check the MinimumThreshold entry. If no minimum threshold has been set, set it to zero.
        %%% Otherwise make sure that the user gave a valid input.
        if strcmp(MinimumThreshold,'Do not use')
            MinimumThreshold = 0;
        else
            MinimumThreshold = str2double(MinimumThreshold);
            if isnan(MinimumThreshold) |  MinimumThreshold < 0 | MinimumThreshold > 1
                error('The Minimum threshold entry in the IdentifyPrimAutomatic module is invalid.')
            end
        end

        %%% Check the smoothing filter size parameter
        if ~strcmp(SizeOfSmoothingFilter,'Automatic')
            SizeOfSmoothingFilter = str2double(SizeOfSmoothingFilter);
            if isempty(SizeOfSmoothingFilter) | SizeOfSmoothingFilter < 0 | SizeOfSmoothingFilter > min(size(OrigImage))
                error('The specified size of the smoothing filter in the IdentifyPrimAutomatic module is not valid or unreasonable.')
            end
        end

        %%% Check the maxima suppression size parameter
        if ~strcmp(MaximaSuppressionSize,'Automatic')
            MaximaSuppressionSize = str2double(MaximaSuppressionSize);
            if isempty(MaximaSuppressionSize) | MaximaSuppressionSize < 0
                error('The specified maxima suppression size in the IdentifyPrimAutomatic module is not valid or unreasonable.')
            end
        end

        %%%%%%%%%%%%%%%%%%%%%
        %%% IMAGE ANALYSIS %%%
        %%%%%%%%%%%%%%%%%%%%%

        %%% STEP 1. Find threshold and apply to image
        if strfind(Threshold,'Global')
            if strfind(Threshold,'Otsu')
                Threshold = CPgraythresh(OrigImage,handles,ImageName);
            elseif strfind(Threshold,'MoG')
                Threshold = MixtureOfGaussians(OrigImage,pObject);
            end

        elseif strfind(Threshold,'Adaptive')

            %%% Choose the block size that best covers the original image in the sense
            %%% that the number of extra rows and columns is minimal.
            %%% Get size of image
            [m,n] = size(OrigImage);

            %%% Deduce a suitable block size based on the image size and the percentage of image
            %%% covered by objects. We want blocks to be big enough to contain both background and
            %%% objects. The more uneven the ratio between background pixels and object pixels the
            %%% larger the block size need to be. The minimum block size is about 50x50 pixels.
            %%% The line below divides the image in 10x10 blocks, and makes sure that the block size is
            %%% at least 50x50 pixels.
            BlockSize = max(50,min(round(m/10),round(n/10)));

            %%% Calculates a range of acceptable block sizes as plus-minus 10% of the suggested block size.
            BlockSizeRange = floor(1.1*BlockSize):-1:ceil(0.9*BlockSize);
            [ignore,index] = min(ceil(m./BlockSizeRange).*BlockSizeRange-m + ceil(n./BlockSizeRange).*BlockSizeRange-n);
            BestBlockSize = BlockSizeRange(index);

            %%% Pads the image so that the blocks fit properly.
            RowsToAdd = BestBlockSize*ceil(m/BestBlockSize) - m;
            ColumnsToAdd = BestBlockSize*ceil(n/BestBlockSize) - n;
            RowsToAddPre = round(RowsToAdd/2);
            RowsToAddPost = RowsToAdd - RowsToAddPre;
            ColumnsToAddPre = round(ColumnsToAdd/2);
            ColumnsToAddPost = ColumnsToAdd - ColumnsToAddPre;
            PaddedImage = padarray(OrigImage,[RowsToAddPre ColumnsToAddPre],'replicate','pre');
            PaddedImage = padarray(PaddedImage,[RowsToAddPost ColumnsToAddPost],'replicate','post');

            %%% Calculates the threshold for each block in the image, and a global threshold used
            %%% to constrain the adaptive threshholds.
            if strfind(Threshold,'Otsu')
                GlobalThreshold = graythresh(OrigImage);
                Threshold = blkproc(PaddedImage,[BestBlockSize BestBlockSize],'graythresh(x)');
            elseif strfind(Threshold,'MoG')
                GlobalThreshold = MixtureOfGaussians(OrigImage,pObject);
                Threshold = blkproc(PaddedImage,[BestBlockSize BestBlockSize],@MixtureOfGaussians,pObject);
            end
            %%% Resizes the block-produced image to be the size of the padded image.
            %%% Bilinear prevents dipping below zero. The crop the image
            %%% get rid of the padding, to make the result the same size as the original image.
            Threshold = imresize(Threshold, size(PaddedImage), 'bilinear');
            Threshold = Threshold(RowsToAddPre+1:end-RowsToAddPost,ColumnsToAddPre+1:end-ColumnsToAddPost);

            %%% For any of the threshold values that is lower than the user-specified
            %%% minimum threshold, set to equal the minimum threshold.  Thus, if there
            %%% are no objects within a block (e.g. if cells are very sparse), an
            %%% unreasonable threshold will be overridden by the minimum threshold.
            Threshold(Threshold <= 0.7*GlobalThreshold) = 0.7*GlobalThreshold;
            Threshold(Threshold >= 1.5*GlobalThreshold) = 1.5*GlobalThreshold;

        else
            %%% The threshold is manually set by the user
            %%% Checks that the Threshold parameter has a valid value
            Threshold = str2double(Threshold);
            if isnan(Threshold) | Threshold > 1 | Threshold < 0
                error('The threshold entered in the IdentifyPrimAutomatic module is not a number, or is outside the acceptable range of 0 to 1.')
            end
        end

        %%% Correct the threshold using the correction factor given by the user
        %%% and make sure that the threshold is not larger than the minimum threshold
        Threshold = ThresholdCorrection*Threshold;
        Threshold = max(Threshold,MinimumThreshold);
        drawnow

        if strcmp(LaplaceValues,'/')

            %%% Apply a slight smoothing before thresholding to remove
            %%% 1-pixel objects and to smooth the edges of the objects.
            %%% Note that this smoothing is hard-coded, and not controlled
            %%% by the user, but it is omitted if the user selected 0 for
            %%% the size of the smoothing filter.
            if SizeOfSmoothingFilter == 0
                %%% No blurring is done.
                BlurredImage = OrigImage;
            else        sigma = 1;
                FiltLength = ceil(2*sigma);                                           % Determine filter size, min 3 pixels, max 61
                [x,y] = meshgrid(-FiltLength:FiltLength,-FiltLength:FiltLength);      % Filter kernel grid
                f = exp(-(x.^2+y.^2)/(2*sigma^2));f = f/sum(f(:));                    % Gaussian filter kernel
                BlurredImage = conv2(OrigImage,f,'same');                             % Blur original image
            end
            Objects = BlurredImage > Threshold;                                   % Threshold image
            Threshold = mean(Threshold(:));                                       % Use average threshold downstreams
            Objects = imfill(double(Objects),'holes');                            % Fill holes
            drawnow

            %%% STEP 2. If user wants, extract local maxima (of intensity or distance) and apply watershed transform
            %%% to separate neighboring objects.
            if ~strcmp(LocalMaximaType,'None') & ~strcmp(WatershedTransformImageType,'None')

                %%% Smooth images for maxima suppression
                if strcmp(SizeOfSmoothingFilter,'Automatic')
                    sigma = MinDiameter/3.5;                                          % Translate between minimum diameter of objects to sigma. Empirically derived formula.
                else
                    sigma = SizeOfSmoothingFilter/2.35;                               % Convert between Full Width at Half Maximum (FWHM) to sigma
                end
                if SizeOfSmoothingFilter == 0
                    %%% No blurring is done.
                else
                    FiltLength = min(30,max(1,ceil(2*sigma)));                            % Determine filter size, min 3 pixels, max 61
                    [x,y] = meshgrid(-FiltLength:FiltLength,-FiltLength:FiltLength);      % Filter kernel grid
                    f = exp(-(x.^2+y.^2)/(2*sigma^2));f = f/sum(f(:));                    % Gaussian filter kernel
                    %%% The original image is blurred. Prior to this blurring, the
                    %%% image is padded with values at the edges so that the values
                    %%% around the edge of the image are not artificially low.  After
                    %%% blurring, these extra padded rows and columns are removed.
                    BlurredImage = conv2(padarray(OrigImage, [FiltLength,FiltLength], 'replicate'),f,'same');
                    BlurredImage = BlurredImage(FiltLength+1:end-FiltLength,FiltLength+1:end-FiltLength);
                end
                %%% Get local maxima, where the definition of local depends on the
                %%% user-provided object size. This will (usually) be done in a
                %%% lower-resolution image for speed. The ordfilt2() function is
                %%% very slow for large images containing large objects.
                %%% Therefore, image is resized to a size where the smallest
                %%% objects are about 10 pixels wide. Local maxima within a radius
                %%% of 5-6 pixels are then extracted. It might be necessary to
                %%% tune this parameter. The MaximaSuppressionSize must be an
                %%% integer.  The MaximaSuppressionSize should be equal to the
                %%% minimum acceptable radius if the objects are perfectly
                %%% circular with local maxima in the center. In practice, the
                %%% MinDiameter is divided by 1.5 because this allows the local
                %%% maxima to be shifted somewhat from the center of the object.
                if strcmp(UseLowRes,'Yes') && MinDiameter > 10
                    ImageResizeFactor = 10/MinDiameter;
                    if strcmp(MaximaSuppressionSize,'Automatic')
                        MaximaSuppressionSize = 7;             % ~ 10/1.5
                    else
                        MaximaSuppressionSize = MaximaSuppressionSize*ImageResizeFactor;
                    end
                else
                    ImageResizeFactor = 1;
                    if strcmp(MaximaSuppressionSize,'Automatic')
                        MaximaSuppressionSize = round(MinDiameter/1.5);
                    else
                        MaximaSuppressionSize = round(MaximaSuppressionSize);
                    end
                end
                MaximaMask = getnhood(strel('disk', MaximaSuppressionSize));

                if strcmp(LocalMaximaType,'Intensity')

                    % Old code without image resizing
                    %MaximaMask = getnhood(strel('disk', min(50,max(1,floor(MinDiameter/1.5)))));
                    % Initialize MaximaImage
                    %MaximaImage = BlurredImage;
                    % Save only local maxima
                    %MaximaImage(BlurredImage < ...
                    %    ordfilt2(BlurredImage,sum(MaximaMask(:)),MaximaMask)) = 0;
                    % Remove dim maxima
                    %MaximaImage = MaximaImage > Threshold;

                    %%% Find local maxima in a lower resolution image
                    ResizedBlurredImage = imresize(BlurredImage,ImageResizeFactor,'bilinear');
                    %%% Initialize MaximaImage
                    MaximaImage = ResizedBlurredImage;
                    %%% Save only local maxima
                    MaximaImage(ResizedBlurredImage < ...
                        ordfilt2(ResizedBlurredImage,sum(MaximaMask(:)),MaximaMask)) = 0;
                    %%% Restore image size
                    MaximaImage = imresize(MaximaImage,size(BlurredImage),'bilinear');
                    %%% Remove dim maxima
                    MaximaImage = MaximaImage > Threshold;
                    %%% Shrink to points (needed because of the resizing)
                    MaximaImage = bwmorph(MaximaImage,'shrink',inf);
                elseif strcmp(LocalMaximaType,'Shape')
                    %%% Calculate distance transform
                    DistanceTransformedImage = bwdist(~Objects);
                    %%% Add some noise to get distinct maxima
                    DistanceTransformedImage = DistanceTransformedImage + ...
                        0.001*rand(size(DistanceTransformedImage));
                    ResizedDistanceTransformedImage = imresize(DistanceTransformedImage,ImageResizeFactor,'bilinear');
                    %%% Initialize MaximaImage
                    MaximaImage = ones(size(ResizedDistanceTransformedImage));
                    %%% Set all pixels that are not local maxima to zero
                    MaximaImage(ResizedDistanceTransformedImage < ...
                        ordfilt2(ResizedDistanceTransformedImage,sum(MaximaMask(:)),MaximaMask)) = 0;
                    %%% Restore image size
                    MaximaImage = imresize(MaximaImage,size(Objects),'bilinear');
                    %%% We are only interested in maxima within thresholded objects
                    MaximaImage(~Objects) = 0;
                    %%% Shrink to points (needed because of the resizing)
                    MaximaImage = bwmorph(MaximaImage,'shrink',inf);
                end

                %%% Overlay the maxima on either the original image or a distance
                %%% transformed image. The watershed is currently done on
                %%% non-smoothed versions of these image. We may want to try to do
                %%% the watershed in the slightly smoothed image.
                if strcmp(WatershedTransformImageType,'Intensity')
                    %%% Overlays the objects markers (maxima) on the inverted original image so
                    %%% there are black dots on top of each dark object on a white background.
                    Overlaid = imimposemin(1 - OrigImage,MaximaImage);
                elseif strcmp(WatershedTransformImageType,'Distance')
                    %%% Overlays the object markers (maxima) on the inverted DistanceTransformedImage so
                    %%% there are black dots on top of each dark object on a white background.
                    %%% We may have to calculate the distance transform:
                    if ~exist('DistanceTransformedImage','var')
                        DistanceTransformedImage = bwdist(~Objects);
                    end
                    Overlaid = imimposemin(-DistanceTransformedImage,MaximaImage);
                end

                %%% Calculate the watershed transform and cut objects along the boundaries
                WatershedBoundaries = watershed(Overlaid) > 0;
                Objects = Objects.*WatershedBoundaries;

                %%% Label the objects
                Objects = bwlabel(Objects);

                %%% Remove objects with no marker in them (this happens occasionally)
                %%% This is a very fast way to get pixel indexes for the objects
                tmp = regionprops(Objects,'PixelIdxList');
                for k = 1:length(tmp)
                    %%% If there is no maxima in these pixels, exclude object
                    if sum(MaximaImage(tmp(k).PixelIdxList)) == 0
                        Objects(index) = 0;
                    end
                end
            end
            drawnow

            %%% Label the objects
            Objects = bwlabel(Objects);

            %%% Merge small objects
            if strcmp(MergeChoice,'Yes')
                Objects = MergeObjects(Objects,OrigImage,[MinDiameter MaxDiameter]);
            end

            %%% Will be stored to the handles structure
            UneditedLabelMatrixImage = Objects;

            %%% Get diameters of objects and calculate the interval
            %%% that contains 90% of the objects
            tmp = regionprops(Objects,'EquivDiameter');
            Diameters = [0;cat(1,tmp.EquivDiameter)];
            SortedDiameters = sort(Diameters);
            NbrInTails = max(round(0.05*length(Diameters)),1);
            Lower90Limit = SortedDiameters(NbrInTails);
            Upper90Limit = SortedDiameters(end-NbrInTails+1);

            %%% Locate objects with diameter outside the specified range
            tmp = Objects;
            if strcmp(ExcludeSize,'Yes')
                %%% Create image with object intensity equal to the diameter
                DiameterMap = Diameters(Objects+1);
                %%% Remove objects that are too small
                Objects(DiameterMap < MinDiameter) = 0;
                %%% Will be stored to the handles structure
                SmallRemovedLabelMatrixImage = Objects;
                %%% Remove objects that are too big
                Objects(DiameterMap > MaxDiameter) = 0;
            else
                %%% Will be stored to the handles structure even if it's unedited.
                SmallRemovedLabelMatrixImage = Objects;
            end
            %%% Store objects that fall outside diameter range for display
            DiameterExcludedObjects = tmp - Objects;

            %%% Remove objects along the border of the image (depends on user input)
            tmp = Objects;
            if strcmp(ExcludeBorderObjects,'Yes')
                Objects = imclearborder(Objects);
            end
            %%% Store objects that touch the border for display
            BorderObjects = tmp - Objects;

            %%% Relabel the objects
            [Objects,NumOfObjects] = bwlabel(Objects > 0);
            FinalLabelMatrixImage = Objects;
        else
            %%% Creates the Laplacian of a Gaussian filter.
            rgLoG=fspecial('log',NeighborhoodSize,Sigma);
            %%% Filters the image.
            imLoGout=imfilter(double(OrigImage),rgLoG);
            %figure, imagesc(imLoGout),  title('imLoGout')
            %%% Removes noise using the weiner filter.
            imLoGoutW=wiener2(imLoGout,WienerSize);
            %figure, imagesc(imLoGoutW),  title('imLoGoutW')

            %%%%%%%%%%%%%%
            rgNegCurve = imLoGoutW < LaplaceThreshold;
            class(rgNegCurve)
            min(min(rgNegCurve))
            max(max(rgNegCurve))

            %set outsides
            rgNegCurve([1 end],1:end)=1;
            rgNegCurve(1:end,[1 end])=1;

            %Throw out noise, label regions
            rgArOpen=bwareaopen(rgNegCurve,MinArea,4);
            rgLabelled=uint16(bwlabel(rgArOpen,4));
            % rgArOpen=bwareaopen(rgNegCurve,MinArea,8); %use 8-connectivity like rest of CP
            % rgLabelled=uint16(bwlabel(rgArOpen,8));
            if max(rgLabelled(:))==1
                error('Error: No DAPI regions generated');
            end

            %Get rid of region around outsides (upper-left region gets value 1)
            rgLabelled(rgLabelled==1)=0;
            rgLabelled(rgLabelled==0)=1;
            rgLabelled=uint16(double(rgLabelled)-1);
            %disp(['Generated labelled, size-excluded regions. Time: ' num2str(toc)])

            %(Smart)closing
            % rgDilated=RgSmartDilate(rgLabelled,50); %%% IMPORTANT VARIABLE
            rgDilated=CPRgSmartDilate(rgLabelled,2); %%% IMPORTANT VARIABLE
            rgFill=imfill(rgDilated,'holes');

            %%%%SE=strel('diamond',1);
            %%%%rgErode=imerode(rgFill,SE);
            %%%%rgOut=rgErode;

            %%%%%%%%%%%%

            % InvertedBinaryImage = RgSmartDilate(rgNegCurve,1);
            InvertedBinaryImage = rgDilated;

            %%% Creates label matrix image.
            % rgLabelled2=uint16(bwlabel(imLoGoutW,4));
            % figure, imagesc(rgLabelled2),  title('rgLabelled2')
            % FinalLabelMatrixImage = bwlabel(FinalBinary,4);

            PrelimLabelMatrixImage1 = bwlabel(imfill(InvertedBinaryImage,'holes'));
            UneditedLabelMatrixImage = PrelimLabelMatrixImage1;
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
            PrelimLabelMatrixImage2(AreasImage < MinDiameter) = 0;
            SmallRemovedLabelMatrixImage = PrelimLabelMatrixImage2;
            drawnow
            %%% Relabels so that labels are consecutive. This is important for
            %%% downstream modules (IdentifySec).
            PrelimLabelMatrixImage2 = bwlabel(im2bw(PrelimLabelMatrixImage2,.5));
            %%% The large objects are overwritten with zeros.
            PrelimLabelMatrixImage3 = PrelimLabelMatrixImage2;
            if MaxDiameter ~= 99999
                PrelimLabelMatrixImage3(AreasImage > MaxDiameter) = 0;
                DiameterExcludedObjects = PrelimLabelMatrixImage3;
                BorderObjects = PrelimLabelMatrixImage3;
            end
            %%% Removes objects that are touching the edge of the image, since they
            %%% won't be measured properly.
            if strncmpi(ExcludeBorderObjects,'Y',1) == 1
                Objects = imclearborder(PrelimLabelMatrixImage3,8);
            else Objects = PrelimLabelMatrixImage3;
            end
            %%% The PrelimLabelMatrixImage4 is converted to binary.
            FinalBinaryPre = im2bw(Objects,.5);
            drawnow
            %%% Holes in the FinalBinaryPre image are filled in.
            FinalBinary = imfill(FinalBinaryPre, 'holes');
            %%% The image is converted to label matrix format. Even if the above step
            %%% is excluded (filling holes), it is still necessary to do this in order
            %%% to "compact" the label matrix: this way, each number corresponds to an
            %%% object, with no numbers skipped.
            [FinalLabelMatrixImage,NumOfObjects] = bwlabel(FinalBinary);
        end

        %%%%%%%%%%%%%%%%%%%%%%
        %%% DISPLAY RESULTS %%%
        %%%%%%%%%%%%%%%%%%%%%%

        if strcmp(OriginalLocalMaximaType,LocalMaximaType) && strcmp(OriginalWatershedTransformImageType,WatershedTransformImageType)

            if strcmp(LaplaceValues,'/')
                %%% Indicate objects in original image and color excluded objects in red
                tmp = OrigImage/max(OrigImage(:));
                OutlinedObjectsR = tmp;
                OutlinedObjectsG = tmp;
                OutlinedObjectsB = tmp;
                PerimObjects = bwperim(Objects > 0);
                PerimDiameter = bwperim(DiameterExcludedObjects > 0);
                PerimBorder = bwperim(BorderObjects > 0);
                OutlinedObjectsR(PerimObjects) = 0; OutlinedObjectsG(PerimObjects) = 1; OutlinedObjectsB(PerimObjects) = 0;
                OutlinedObjectsR(PerimDiameter) = 1; OutlinedObjectsG(PerimDiameter)   = 0; OutlinedObjectsB(PerimDiameter)   = 0;
                OutlinedObjectsR(PerimBorder) = 1; OutlinedObjectsG(PerimBorder) = 1; OutlinedObjectsB(PerimBorder) = 0;

                fieldname = ['FigureNumberForModule',CurrentModule];
                ThisModuleFigureNumber = handles.Current.(fieldname);
                if any(findobj == ThisModuleFigureNumber)
                    drawnow
                    CPfigure(handles,ThisModuleFigureNumber);
                    subplot(2,2,1)
                    ImageHandle = imagesc(OrigImage);
                    set(ImageHandle,'ButtonDownFcn','ImageTool(gco)','Tag',['Input Image, Image Set # ',num2str(handles.Current.SetBeingAnalyzed)])
                    axis image
                    title(['Input Image, Image Set # ',num2str(handles.Current.SetBeingAnalyzed)],'fontsize',handles.Current.FontSize);
                    set(gca,'fontsize',handles.Current.FontSize)
                    hx = subplot(2,2,2);
                    if sum(sum(Objects(:)))>0
                        im = CPlabel2rgb(handles,Objects);
                    else
                        im = Objects;
                    end
                    ImageHandle = image(im);
                    set(ImageHandle,'ButtonDownFcn','ImageTool(gco)','Tag',sprintf('Segmented %s',ObjectName))
                    title(sprintf('Segmented %s',ObjectName),'fontsize',handles.Current.FontSize);
                    axis image,set(gca,'fontsize',handles.Current.FontSize)

                    hy = subplot(2,2,3);
                    OutlinedObjects = cat(3,OutlinedObjectsR,OutlinedObjectsG,OutlinedObjectsB);
                    ImageHandle = image(OutlinedObjects);
                    set(ImageHandle,'ButtonDownFcn','ImageTool(gco)','Tag','Outlined objects')
                    title('Outlined objects','fontsize',handles.Current.FontSize);
                    axis image,set(gca,'fontsize',handles.Current.FontSize)

                    CPFixAspectRatio(OrigImage);

                    %%% Report numbers
                    posx = get(hx,'Position');
                    posy = get(hy,'Position');
                    bgcolor = get(ThisModuleFigureNumber,'Color');
                    uicontrol(ThisModuleFigureNumber,'Style','Text','Units','Normalized','Position',[posx(1)-0.05 posy(2)+posy(4)-0.04 posx(3)+0.1 0.04],...
                        'BackgroundColor',bgcolor,'HorizontalAlignment','Left','String',sprintf('Threshold:  %0.3f',Threshold),'FontSize',handles.Current.FontSize);
                    uicontrol(ThisModuleFigureNumber,'Style','Text','Units','Normalized','Position',[posx(1)-0.05 posy(2)+posy(4)-0.08 posx(3)+0.1 0.04],...
                        'BackgroundColor',bgcolor,'HorizontalAlignment','Left','String',sprintf('Number of segmented objects: %d',NumOfObjects),'FontSize',handles.Current.FontSize);
                    uicontrol(ThisModuleFigureNumber,'Style','Text','Units','Normalized','Position',[posx(1)-0.05 posy(2)+posy(4)-0.16 posx(3)+0.1 0.08],...
                        'BackgroundColor',bgcolor,'HorizontalAlignment','Left','String',sprintf('90%% of objects within diameter range [%0.1f, %0.1f] pixels',...
                        Lower90Limit,Upper90Limit),'FontSize',handles.Current.FontSize);
                    ObjectCoverage = 100*sum(sum(Objects > 0))/prod(size(Objects));
                    uicontrol(ThisModuleFigureNumber,'Style','Text','Units','Normalized','Position',[posx(1)-0.05 posy(2)+posy(4)-0.20 posx(3)+0.1 0.04],...
                        'BackgroundColor',bgcolor,'HorizontalAlignment','Left','String',sprintf('%0.1f%% of image consists of objects',ObjectCoverage),'FontSize',handles.Current.FontSize);
                    if ~strcmp(LocalMaximaType,'None') & ~strcmp(WatershedTransformImageType,'None')
                        uicontrol(ThisModuleFigureNumber,'Style','Text','Units','Normalized','Position',[posx(1)-0.05 posy(2)+posy(4)-0.24 posx(3)+0.1 0.04],...
                            'BackgroundColor',bgcolor,'HorizontalAlignment','Left','String',sprintf('Smoothing filter size:  %0.1f',2.35*sigma),'FontSize',handles.Current.FontSize);
                        uicontrol(ThisModuleFigureNumber,'Style','Text','Units','Normalized','Position',[posx(1)-0.05 posy(2)+posy(4)-0.28 posx(3)+0.1 0.04],...
                            'BackgroundColor',bgcolor,'HorizontalAlignment','Left','String',sprintf('Maxima suppression size:  %d',round(MaximaSuppressionSize/ImageResizeFactor)),'FontSize',handles.Current.FontSize);
                    end
                end
            else
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
                    %%% Calculates the object outlines, which are overlaid on the original
                    %%% image and displayed in figure subplot (2,2,4).
                    %%% Creates the structuring element that will be used for dilation.
                    StructuringElement = strel('square',3);
                    %%% Converts the FinalLabelMatrixImage to binary.
                    FinalBinaryImage = im2bw(FinalLabelMatrixImage,.5);
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
                    subplot(2,2,1); imagesc(OrigImage);
                    title(['Input Image, Image Set # ',num2str(handles.Current.SetBeingAnalyzed)]);
                    %%% A subplot of the figure window is set to display the colored label
                    %%% matrix image.
                    subplot(2,2,2); imagesc(ColoredLabelMatrixImage);title(['Segmented ',ObjectName]);
                    %%% A subplot of the figure window is set to display the Overlaid image,
                    %%% where the maxima are imposed on the inverted original image
                    % subplot(2,2,3); imagesc(Overlaid);  title([ObjectName, ' markers']);
                    %%% A subplot of the figure window is set to display the inverted original
                    %%% image with watershed lines drawn to divide up clusters of objects.
                    subplot(2,2,4); imagesc(ObjectOutlinesOnOrigImage); title([ObjectName, ' Outlines on Input Image']);
                    CPFixAspectRatio(OrigImage);
                end
            end

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%% SAVE DATA TO HANDLES STRUCTURE %%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %%% Saves the segmented image, not edited for objects along the edges or
            %%% for size, to the handles structure.
            fieldname = ['UneditedSegmented',ObjectName];
            handles.Pipeline.(fieldname) = UneditedLabelMatrixImage;

            %%% Saves the segmented image, only edited for small objects, to the
            %%% handles structure.
            fieldname = ['SmallRemovedSegmented',ObjectName];
            handles.Pipeline.(fieldname) = SmallRemovedLabelMatrixImage;

            %%% Saves the final segmented label matrix image to the handles structure.
            fieldname = ['Segmented',ObjectName];
            handles.Pipeline.(fieldname) = FinalLabelMatrixImage;

            %%% Saves images to the handles structure so they can be saved to the hard
            %%% drive, if the user requested.
            if ~strcmp(SaveOutlines,'Do not save')
                try    handles.Pipeline.(SaveOutlines) = PerimObjects;
                catch
                    errordlg('The object outlines were not calculated by the IdentifyPrimAutomatic module (possibly because the window is closed) so these images were not saved to the handles structure. Image processing is still in progress, but the Save Images module will fail if you attempted to save these images.')
                end
            end

            if ~strcmp(SaveOutlinedOnOriginal,'Do not save')
                try    handles.Pipeline.(SaveOutlinedOnOriginal) = OutlinedObjects;
                catch
                    errordlg('The object outlines overlaid on the original image were not calculated by the IdentifyPrimAutomatic module (possibly because the window is closed) so these images were not saved to the handles structure. Image processing is still in progress, but the Save Images module will fail if you attempted to save these images.')
                end
            end

            if ~strcmp(SaveColored,'Do not save')
                try
                    if strcmp(SaveMode,'RGB')
                        if sum(sum(FinalLabelMatrixImage)) >= 1
                            cmap = jet(max(64,max(FinalLabelMatrixImage(:))));
                            ColoredLabelMatrixImage = label2rgb(FinalLabelMatrixImage, cmap, 'k', 'shuffle');
                        else
                            ColoredLabelMatrixImage = FinalLabelMatrixImage;
                        end
                        handles.Pipeline.(SaveColored) = ColoredLabelMatrixImage;
                    else
                        handles.Pipeline.(SaveColored) = FinalLabelMatrixImage;
                    end
                catch
                    errordlg('The label matrix image was not calculated by the IdentifyPrimAutomatic module (possibly because the window is closed) so these images were not saved to the handles structure. Image processing is still in progress, but the Save Images module will fail if you attempted to save these images.')
                end
            end

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
            column = find(~cellfun('isempty',strfind(handles.Measurements.Image.ThresholdFeatures,ObjectName)));
            % If column is empty it means that this particular object has not been segmented before. This will
            % typically happen for the first image set. Append the feature name in the
            % handles.Measurements.Image.ThresholdFeatures matrix
            if isempty(column)
                handles.Measurements.Image.ThresholdFeatures(end+1) = {['Threshold ' ObjectName]};
                column = length(handles.Measurements.Image.ThresholdFeatures);
            end
            handles.Measurements.Image.Threshold{handles.Current.SetBeingAnalyzed}(1,column) = Threshold;

            %%% Saves the ObjectCount, i.e., the number of segmented objects.
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

        end

        if strcmp(TestMode,'Yes')
            if ~(LocalMaximaTypeNumber == 2 && WatershedTransformImageTypeNumber == 3)
                drawnow;
                SegmentedFigures = findobj('Tag','SegmentedFigure');
                if isempty(SegmentedFigures)
                    SegFig=CPfigure('Tag','SegmentedFigure');
                    uicontrol('style','text','units','normalized','string','SEGMENTED OBJECTS: Choosing None for either option will result in the same image, therefore only the Intensity and None option has been shown.','position',[.65 .1 .3 .4],'BackgroundColor',[.7 .7 .9])
                else
                    SegFig = CPfigure(SegmentedFigures(1));
                end

                subplot(2,3,WatershedTransformImageTypeNumber+3*(LocalMaximaTypeNumber-1));
                cmap = jet(max(64,max(Objects(:))));
                im = label2rgb(Objects, cmap, 'k', 'shuffle');
                ImageHandle = imagesc(im);

                title(sprintf('%s and %s',LocalMaximaTypeList{LocalMaximaTypeNumber},WatershedTransformImageTypeList{WatershedTransformImageTypeNumber}),'fontsize',handles.Current.FontSize);

                OutlinedFigures = findobj('Tag','OutlinedFigure');
                if isempty(OutlinedFigures)
                    OutFig=CPfigure('Tag','OutlinedFigure');
                    uicontrol('style','text','units','normalized','string','OUTLINED OBJECTS: Choosing None for either option will result in the same image, therefore only the Intensity and None option has been shown.','position',[.65 .1 .3 .4],'BackgroundColor',[.7 .7 .9])
                else
                    OutFig = CPfigure(OutlinedFigures(1));
                end

                tmp = OrigImage/max(OrigImage(:));
                OutlinedObjectsR = tmp;
                OutlinedObjectsG = tmp;
                OutlinedObjectsB = tmp;
                PerimObjects = bwperim(Objects > 0);
                PerimDiameter = bwperim(DiameterExcludedObjects > 0);
                PerimBorder = bwperim(BorderObjects > 0);
                OutlinedObjectsR(PerimObjects) = 0; OutlinedObjectsG(PerimObjects) = 1; OutlinedObjectsB(PerimObjects) = 0;
                OutlinedObjectsR(PerimDiameter) = 1; OutlinedObjectsG(PerimDiameter)   = 0; OutlinedObjectsB(PerimDiameter)   = 0;
                OutlinedObjectsR(PerimBorder) = 1; OutlinedObjectsG(PerimBorder) = 1; OutlinedObjectsB(PerimBorder) = 0;

                subplot(2,3,WatershedTransformImageTypeNumber+3*(LocalMaximaTypeNumber-1));
                OutlinedObjects = cat(3,OutlinedObjectsR,OutlinedObjectsG,OutlinedObjectsB);
                ImageHandle = imagesc(OutlinedObjects);
                title(sprintf('%s and %s',LocalMaximaTypeList{LocalMaximaTypeNumber},WatershedTransformImageTypeList{WatershedTransformImageTypeNumber}),'fontsize',handles.Current.FontSize);

            end
        end
    end
end

%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTIONS %%%
%%%%%%%%%%%%%%%%%%%

function Threshold = MixtureOfGaussians(OrigImage,pObject)
%%% This function finds a suitable threshold for the input image
%%% OrigImage. It assumes that the pixels in the image belong to either
%%% a background class or an object class. 'pObject' is an initial guess
%%% of the prior probability of an object pixel, or equivalently, the fraction
%%% of the image that is covered by objects. Essentially, there are two steps.
%%% First, a number of Gaussian distributions are estimated to match the
%%% distribution of pixel intensities in OrigImage. Currently 3 Gaussian
%%% distributions are fitted, one corresponding to a background class, one
%%% corresponding to an object class, and one distribution for an intermediate
%%% class. The distributions are fitted using the Expectation-Maximization (EM)
%%% algorithm, a procedure referred to as Mixture of Gaussians modeling. When
%%% the 3 Gaussian distributions have been fitted, it's decided whether the
%%% intermediate class models background pixels or object pixels based on the
%%% probability of an object pixel 'pObject' given by the user.

%%% The number of classes is set to 3
NumberOfClasses = 3;

%%% Transform the image into a vector. Also, if the image is (larger than 512x512),
%%% select a subset of 512^2 pixels for speed. This should be enough to capture the
%%% statistics in the image.
Intensities = OrigImage(:);
if length(Intensities) > 512^2
    indexes = randperm(length(Intensities));
    Intensities = Intensities(indexes(1:512^2));
end

%%% Get the probability for a background pixel
pBackground = 1 - pObject;

%%% Initialize mean and standard deviations of the three Gaussian distributions
%%% by looking at the pixel intensities in the original image and by considering
%%% the percentage of the image that is covered by object pixels. Class 1 is the
%%% background class and Class 3 is the object class. Class 2 is an intermediate
%%% class and we will decide later if it encodes background or object pixels.
%%% Also, for robustness the we remove 1% of the smallest and highest intensities
%%% in case there are any quantization effects that have resulted in unaturally many
%%% 0:s or 1:s in the image.
Intensities = sort(Intensities);
Intensities = Intensities(round(length(Intensities)*0.01):round(length(Intensities)*0.99));
ClassMean(1) = Intensities(round(length(Intensities)*pBackground/2));                      %%% Initialize background class
ClassMean(3) = Intensities(round(length(Intensities)*(1 - pObject/2)));                    %%% Initialize object class
ClassMean(2) = (ClassMean(1) + ClassMean(3))/2;                                            %%% Initialize intermediate class
%%% Initialize standard deviations of the Gaussians. They should be the same to avoid problems.
ClassStd(1:3) = 0.15;
%%% Initialize prior probabilities of a pixel belonging to each class. The intermediate
%%% class is gets some probability from the background and object classes.
pClass(1) = 3/4*pBackground;
pClass(2) = 1/4*pBackground + 1/4*pObject;
pClass(3) = 3/4*pObject;

%%% Apply transformation.  a < x < b, transform to log((x-a)/(b-x)).
%a = - 0.000001;
%b = 1.000001;
%Intensities = log((Intensities-a)./(b-Intensities));
%ClassMean = log((ClassMean-a)./(b - ClassMean))
%ClassStd(1:3) = [1 1 1];

%%% Expectation-Maximization algorithm for fitting the three Gaussian distributions/classes
%%% to the data. Note, the code below is general and works for any number of classes.
%%% Iterate until parameters don't change anymore.
delta = 1;
while delta > 0.001
    %%% Store old parameter values to monitor change
    oldClassMean = ClassMean;

    %%% Update probabilities of a pixel belonging to the background or object1 or object2
    for k = 1:NumberOfClasses
        pPixelClass(:,k) = pClass(k)* 1/sqrt(2*pi*ClassStd(k)^2) * exp(-(Intensities - ClassMean(k)).^2/(2*ClassStd(k)^2));
    end
    pPixelClass = pPixelClass ./ repmat(sum(pPixelClass,2) + eps,[1 NumberOfClasses]);

    %%% Update parameters in Gaussian distributions
    for k = 1:NumberOfClasses
        pClass(k) = mean(pPixelClass(:,k));
        ClassMean(k) = sum(pPixelClass(:,k).*Intensities)/(length(Intensities)*pClass(k));
        ClassStd(k)  = sqrt(sum(pPixelClass(:,k).*(Intensities - ClassMean(k)).^2)/(length(Intensities)*pClass(k))) + sqrt(eps);    % Add sqrt(eps) to avoid division by zero
    end

    %%% Calculate change
    delta = sum(abs(ClassMean - oldClassMean));
end

%%% Now the Gaussian distributions are fitted and we can describe the histogram of the pixel
%%% intensities as the sum of these Gaussian distributions. To find a threshold we first have
%%% to decide if the intermediate class 2 encodes background or object pixels. This is done by
%%% choosing the combination of class probabilities 'pClass' that best matches the user input 'pObject'.
Threshold = linspace(ClassMean(1),ClassMean(3),10000);
Class1Gaussian = pClass(1) * 1/sqrt(2*pi*ClassStd(1)^2) * exp(-(Threshold - ClassMean(1)).^2/(2*ClassStd(1)^2));
Class2Gaussian = pClass(2) * 1/sqrt(2*pi*ClassStd(2)^2) * exp(-(Threshold - ClassMean(2)).^2/(2*ClassStd(2)^2));
Class3Gaussian = pClass(3) * 1/sqrt(2*pi*ClassStd(3)^2) * exp(-(Threshold - ClassMean(3)).^2/(2*ClassStd(3)^2));
if abs(pClass(2) + pClass(3) - pObject) < abs(pClass(3) - pObject)
    %%% Intermediate class 2 encodes object pixels
    BackgroundDistribution = Class1Gaussian;
    ObjectDistribution = Class2Gaussian + Class3Gaussian;
else
    %%% Intermediate class 2 encodes background pixels
    BackgroundDistribution = Class1Gaussian + Class2Gaussian;
    ObjectDistribution = Class3Gaussian;
end

%%% Now, find the threshold at the intersection of the background distribution
%%% and the object distribution.
[ignore,index] = min(abs(BackgroundDistribution - ObjectDistribution));
Threshold = Threshold(index);

%%% Inverse transformation to log((x-a)/(b-x)) is (a+b*exp(t))/(1+exp(t))
%Threshold = (a + b*exp(Threshold))/(1+exp(Threshold));


function Objects = MergeObjects(Objects,OrigImage,Diameters)

%%% Find the object that we should try to merge with other objects. The object
%%% numbers of these objects are stored in the variable 'MergeIndex'. The objects
%%% that we will try to merge are either the ones that fall below the specified
%%% MinDiameter threshold, or relatively small objects that are above the MaxEccentricity
%%% threshold. These latter objects are likely to be cells where two maxima have been
%%% found and the watershed transform has divided cells into two parts.
MinDiameter = Diameters(1);
MaxDiameter = Diameters(2);
MaxEccentricity = 0.75;      % Empirically determined value
props = regionprops(Objects,'EquivDiameter','PixelIdxList','Eccentricity');   % Get diameters of the objects
EquivDiameters = cat(1,props.EquivDiameter);
Eccentricities = cat(1,props.Eccentricity);
IndexEccentricity = intersect(find(Eccentricities > MaxEccentricity),find(EquivDiameters < (MinDiameter + (MaxDiameter - MinDiameter)/4)));
IndexDiameter = find(EquivDiameters < MinDiameter);
MergeIndex = unique([IndexDiameter;IndexEccentricity]);

% Try to merge until there are no objects left in the 'MergeIndex' list.
[sr,sc] = size(OrigImage);
while ~isempty(MergeIndex)

    % Get next object to merge
    CurrentObjectNbr = MergeIndex(1);

    %%% Identify neighbors and put their label numbers in a list 'NeighborsNbr'
    %%% Cut a patch so we don't have to work with the entire image
    [r,c] = ind2sub([sr sc],props(CurrentObjectNbr).PixelIdxList);
    rmax = min(sr,max(r) + 3);
    rmin = max(1,min(r) - 3);
    cmax = min(sc,max(c) + 3);
    cmin = max(1,min(c) - 3);
    ObjectsPatch = Objects(rmin:rmax,cmin:cmax);
    BinaryPatch = double(ObjectsPatch == CurrentObjectNbr);
    GrownBinaryPatch = conv2(BinaryPatch,double(getnhood(strel('disk',2))),'same') > 0;
    Neighbors = ObjectsPatch .*GrownBinaryPatch;
    NeighborsNbr = setdiff(unique(Neighbors(:)),[0 CurrentObjectNbr]);


    %%% For each neighbor, calculate a set of criteria based on which we decide if to merge.
    %%% Currently, two criteria are used. The first is a Likelihood ratio that indicates whether
    %%% the interface pixels between the object to merge and its neighbor belong to a background
    %%% class or to an object class. The background class and object class are modeled as Gaussian
    %%% distributions with mean and variance estimated from the image. The Likelihood ratio determines
    %%% to which of the distributions the interface voxels most likely belong to. The second criterion
    %%% is the eccentrity of the object resulting from a merge. The more circular, i.e., the lower the
    %%% eccentricity, the better.
    LikelihoodRatio    = zeros(length(NeighborsNbr),1);
    MergedEccentricity = zeros(length(NeighborsNbr),1);
    for j = 1:length(NeighborsNbr)

        %%% Get Neigbor number
        CurrentNeighborNbr = NeighborsNbr(j);

        %%% Cut patch which contains both original object and the current neighbor
        [r,c] = ind2sub([sr sc],[props(CurrentObjectNbr).PixelIdxList;props(CurrentNeighborNbr).PixelIdxList]);
        rmax = min(sr,max(r) + 3);
        rmin = max(1,min(r) - 3);
        cmax = min(sc,max(c) + 3);
        cmin = max(1,min(c) - 3);
        ObjectsPatch = Objects(rmin:rmax,cmin:cmax);
        OrigImagePatch = OrigImage(rmin:rmax,cmin:cmax);

        %%% Identify object interiors, background and interface voxels
        BinaryNeighborPatch      = double(ObjectsPatch == CurrentNeighborNbr);
        BinaryObjectPatch        = double(ObjectsPatch == CurrentObjectNbr);
        GrownBinaryNeighborPatch = conv2(BinaryNeighborPatch,ones(3),'same') > 0;
        GrownBinaryObjectPatch   = conv2(BinaryObjectPatch,ones(3),'same') > 0;
        Interface                = GrownBinaryNeighborPatch.*GrownBinaryObjectPatch;
        Background               = ((GrownBinaryNeighborPatch + GrownBinaryObjectPatch) > 0) - BinaryNeighborPatch - BinaryObjectPatch - Interface;
        WithinObjectIndex        = find(BinaryNeighborPatch + BinaryObjectPatch);
        InterfaceIndex           = find(Interface);
        BackgroundIndex          = find(Background);

        %%% Calculate likelihood of the interface belonging to the background or to an object.
        WithinObjectClassMean   = mean(OrigImagePatch(WithinObjectIndex));
        WithinObjectClassStd    = std(OrigImagePatch(WithinObjectIndex)) + sqrt(eps);
        BackgroundClassMean     = mean(OrigImagePatch(BackgroundIndex));
        BackgroundClassStd      = std(OrigImagePatch(BackgroundIndex)) + sqrt(eps);
        InterfaceMean           = mean(OrigImagePatch(InterfaceIndex));
        LogLikelihoodObject     = -log(WithinObjectClassStd^2) - (InterfaceMean - WithinObjectClassMean)^2/(2*WithinObjectClassStd^2);
        LogLikelihoodBackground = -log(BackgroundClassStd^2) - (InterfaceMean - BackgroundClassMean)^2/(2*BackgroundClassStd^2);
        LikelihoodRatio(j)      =  LogLikelihoodObject - LogLikelihoodBackground;

        %%% Calculate the eccentrity of the object obtained if we merge the current object
        %%% with the current neighbor.
        MergedObject =  double((BinaryNeighborPatch + BinaryObjectPatch + Interface) > 0);
        tmp = regionprops(MergedObject,'Eccentricity');
        MergedEccentricity(j) = tmp(1).Eccentricity;

        %%% Get indexes for the interface pixels in original image.
        %%% These indexes are required if we need to merge the object with
        %%% the current neighbor.
        tmp = zeros(size(OrigImage));
        tmp(rmin:rmax,cmin:cmax) = Interface;
        tmp = regionprops(double(tmp),'PixelIdxList');
        OrigInterfaceIndex{j} = cat(1,tmp.PixelIdxList);
    end

    %%% Let each feature rank which neighbor to merge with. Then calculate
    %%% a score for each neighbor. If the neighbors is ranked 1st, it will get
    %%% 1 point; 2nd, it will get 2 points; and so on. The lower score the better.
    [ignore,LikelihoodRank]   = sort(LikelihoodRatio,'descend');                  % The higher the LikelihoodRatio the better
    [ignore,EccentricityRank] = sort(MergedEccentricity,'ascend');                % The lower the eccentricity the better
    NeighborScore = zeros(length(NeighborsNbr),1);
    for j = 1:length(NeighborsNbr)
        NeighborScore(j) = find(LikelihoodRank == j) +  find(EccentricityRank == j);
    end

    %%% Go through the neighbors, starting with the highest ranked, and merge
    %%% with the first neighbor for which certain basic criteria are fulfilled.
    %%% If no neighbor fulfil the basic criteria, there will be no merge.
    [ignore,TotalRank] = sort(NeighborScore);
    for j = 1:length(NeighborsNbr)
        CurrentNeighborNbr = NeighborsNbr(TotalRank(j));

        %%% To merge, the interface between objects must be more likely to belong to the object class
        %%% than the background class. The eccentricity of the merged object must also be lower than
        %%% for the original object.
        if LikelihoodRatio(TotalRank(j)) > 0 && MergedEccentricity(TotalRank(j)) < Eccentricities(CurrentObjectNbr)

            %%% OK, let's merge!
            %%% Assign the neighbor number to the current object
            Objects(props(CurrentObjectNbr).PixelIdxList) = CurrentNeighborNbr;

            %%% Assign the neighbor number to the interface pixels between the current object and the neigbor
            Objects(OrigInterfaceIndex{TotalRank(j)}) = CurrentNeighborNbr;

            %%% Add the pixel indexes to the neigbor index list
            props(CurrentNeighborNbr).PixelIdxList = cat(1,...
                props(CurrentNeighborNbr).PixelIdxList,...
                props(CurrentObjectNbr).PixelIdxList,...
                OrigInterfaceIndex{TotalRank(j)});

            %%% Remove the neighbor from the list of objects to be merged (if it's there).
            MergeIndex = setdiff(MergeIndex,CurrentNeighborNbr);
        end
    end

    %%% OK, we are done with the current object, let's go to the next
    MergeIndex = MergeIndex(2:end-1);
end

%%% Finally, relabel the objects
Objects = bwlabel(Objects > 0);