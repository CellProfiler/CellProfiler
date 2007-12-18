function handles = IdentifyPrimBasedOnOutlines(handles)

%%% NOTE: THIS MODULE IS PROBABLY NOT USEFUL> WE ENDED UP USING COUNT WORMS, BUT ALSO WE ENDED UP JUST USING IDENTIFYPRIMAUTOMATIC. SO WE CAN PROBABLY DISCARD THIS.


% Help for the Process Outlines module:
% Category: Object Processing
%
% SHORT DESCRIPTION:
% Sorry, this module is not documented... yet
% *************************************************************************
% Note: this module is beta-version and has not been thoroughly tested
%
% Takes an image with hand-drawn outlines and produces objects based on the
% outlines. It is useful for validation and when hand-outlining is
% necessary to accurately identify objects. The incoming outlined image can
% be hand drawn (e.g. using a marker on a transparency and scanning in the
% transparency) or it can be drawn in a program like photoshop.
%
% SETTINGS:
% Note that sophisticated options are not offered for thresholding, because
% the outlined image fed to this module should be essentially a black and
% white image where the background is white and the outlines are black. If
% a problem is encounted where an error states that the image might be a
% color image, run the ColorToGray module before
% IdentifyPrimBasedOnOutlines.
%
% See also IdentifyPrimAutomatic.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne E. Carpenter
%   Thouis Ray Jones
%   In Han Kang
%   Ola Friman
%   Steve Lowe
%   Joo Han Chang
%   Colin Clarke
%   Mike Lamprecht
%   Peter Swire
%   Rodrigo Ipince
%   Vicky Lay
%   Jun Liu
%   Chris Gang
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the outlines you want to process?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the objects identified by this module?
%defaultVAR02 = Nuclei
%infotypeVAR02 = objectgroup indep
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Typical diameter of objects, in pixel units (Min,Max):
%defaultVAR03 = 10,40
SizeRange = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = Are your objects typically round? If not, what is their minimum width, in pixel units?
%choiceVAR04 = Yes
RoundOrNot = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu custom

%textVAR05 = Discard objects outside the diameter range?
%choiceVAR05 = Yes
%choiceVAR05 = No
ExcludeSize = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu

%textVAR06 = Try to merge too small objects with nearby larger objects?
%choiceVAR06 = No
%choiceVAR06 = Yes
MergeChoice = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%inputtypeVAR06 = popupmenu

%textVAR07 = Discard objects touching the border of the image?
%choiceVAR07 = Yes
%choiceVAR07 = No
ExcludeBorderObjects = char(handles.Settings.VariableValues{CurrentModuleNum,7});
%inputtypeVAR07 = popupmenu

%textVAR08 = If your image is not already binary, select an automatic thresholding method or enter an absolute threshold in the range [0,1]. Choosing 'All' will use the Otsu Global method to calculate a single threshold for the entire image group. The other methods calculate a threshold for each image individually. Set interactively will allow you to manually adjust the threshold during the first cycle to determine what will work well.
%choiceVAR08 = Otsu Global
%choiceVAR08 = Otsu Adaptive
%choiceVAR08 = MoG Global
%choiceVAR08 = MoG Adaptive
%choiceVAR08 = Background Global
%choiceVAR08 = Background Adaptive
%choiceVAR08 = RidlerCalvard Global
%choiceVAR08 = RidlerCalvard Adaptive
%choiceVAR08 = All
%choiceVAR08 = Set interactively
Threshold = char(handles.Settings.VariableValues{CurrentModuleNum,8});
%inputtypeVAR08 = popupmenu custom

%textVAR09 = Threshold correction factor
%defaultVAR09 = 1
ThresholdCorrection = str2num(char(handles.Settings.VariableValues{CurrentModuleNum,9})); %#ok Ignore MLint

%textVAR10 = Lower and upper bounds on threshold, in the range [0,1]
%defaultVAR10 = 0,1
ThresholdRange = char(handles.Settings.VariableValues{CurrentModuleNum,10});

%textVAR11 = For MoG thresholding, what is the approximate percentage of image covered by objects?
%choiceVAR11 = 10%
%choiceVAR11 = 20%
%choiceVAR11 = 30%
%choiceVAR11 = 40%
%choiceVAR11 = 50%
%choiceVAR11 = 60%
%choiceVAR11 = 70%
%choiceVAR11 = 80%
%choiceVAR11 = 90%
pObject = char(handles.Settings.VariableValues{CurrentModuleNum,11});
%inputtypeVAR11 = popupmenu

%textVAR12 = Are your outlines "White" on a dark background or "Black" in a light background? (CellProfiler will make the outlines white on dark background before calculating and applying a threshold)
%choiceVAR12 = White
%choiceVAR12 = Black
OutlineColor = char(handles.Settings.VariableValues{CurrentModuleNum,12});
%inputtypeVAR12 = popupmenu

%textVAR13 = Do you want CellProfiler to close gaps in your outlines?
%choiceVAR13 = Yes
%choiceVAR13 = No
CloseGaps = char(handles.Settings.VariableValues{CurrentModuleNum,13});
%inputtypeVAR13 = popupmenu

%textVAR14 = Do you want to remove the outlines from the image once the objects are identified?
%choiceVAR14 = No
%choiceVAR14 = Yes
RemoveOutlines = char(handles.Settings.VariableValues{CurrentModuleNum,14});
%inputtypeVAR14 = popupmenu

%textVAR15 = Method to distinguish clumped objects (see help for details):
%choiceVAR15 = None
%choiceVAR15 = Shape
OriginalLocalMaximaType = char(handles.Settings.VariableValues{CurrentModuleNum,15});
%inputtypeVAR15 = popupmenu

%textVAR16 =  Method to draw dividing lines between clumped objects (see help for details):
%choiceVAR16 = None
%choiceVAR16 = Distance
OriginalWatershedTransformImageType = char(handles.Settings.VariableValues{CurrentModuleNum,16});
%inputtypeVAR16 = popupmenu

%textVAR17 = Size of smoothing filter, in pixel units (if you are distinguishing between clumped objects). Enter 0 for low resolution images with small objects (~< 5 pixel diameter) to prevent any image smoothing.
%defaultVAR17 = Automatic
SizeOfSmoothingFilter = char(handles.Settings.VariableValues{CurrentModuleNum,17});

%textVAR18 = Suppress local maxima within this distance, (a positive integer, in pixel units) (if you are distinguishing between clumped objects)
%defaultVAR18 = Automatic
MaximaSuppressionSize = char(handles.Settings.VariableValues{CurrentModuleNum,18});

%textVAR19 = Speed up by using lower-resolution image to find local maxima?  (if you are distinguishing between clumped objects)
%choiceVAR19 = Yes
%choiceVAR19 = No
UseLowRes = char(handles.Settings.VariableValues{CurrentModuleNum,19});
%inputtypeVAR19 = popupmenu

%textVAR20 = What do you want to call the outlines of the identified objects (optional)?
%defaultVAR20 = Do not save
%infotypeVAR20 = outlinegroup indep
SaveOutlines = char(handles.Settings.VariableValues{CurrentModuleNum,20});

%textVAR21 = Do you want to run in test mode where each method for distinguishing clumped objects is compared?
%choiceVAR21 = No
%choiceVAR21 = Yes
TestMode = char(handles.Settings.VariableValues{CurrentModuleNum,21});
%inputtypeVAR21 = popupmenu

%textVAR22 = Do you need to identify each object separately or just count the total amount of objects in the image?
%choiceVAR22 = Identify Separately
%choiceVAR22 = Just Count
Estimate = char(handles.Settings.VariableValues{CurrentModuleNum,22});
%inputtypeVAR22 = popupmenu

%%%VariableRevisionNumber = 2


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a variable.
try
    OrigImage = CPretrieveimage(handles,ImageName,ModuleName,'MustBeGray','DontCheckScale');
catch
    ErrorMessage = lasterr;
    error(ErrorMessage(33:end));
end

%%% Checks if a custom entry was selected for Threshold
if ~(strncmp(Threshold,'Otsu',4) || strncmp(Threshold,'MoG',3) || strncmp(Threshold,'Background',10) || strncmp(Threshold,'RidlerCalvard',13) || strcmp(Threshold,'All') || strcmp(Threshold,'Set interactively'))
    if isnan(str2double(Threshold))
        error(['Image processing was canceled in the ' ModuleName ' module because the threshold method you specified is invalid. Please select one of the available methods or specify a threshold to use (a number in the range 0-1). Your input was ' Threshold]);
    end
end

%%% Checks that the Min and Max diameter parameters have valid values
index = strfind(SizeRange,',');
if isempty(index),
    error(['Image processing was canceled in the ', ModuleName, ' module because the Min and Max size entry is invalid.'])
end
MinDiameter = SizeRange(1:index-1);
MaxDiameter = SizeRange(index+1:end);

MinDiameter = str2double(MinDiameter);
if isnan(MinDiameter) | MinDiameter < 0 %#ok Ignore MLint
    error(['Image processing was canceled in the ', ModuleName, ' module because the Min diameter entry is invalid.'])
end
if strcmpi(MaxDiameter,'Inf')
    MaxDiameter = Inf;
else
    MaxDiameter = str2double(MaxDiameter);
    if isnan(MaxDiameter) | MaxDiameter < 0 %#ok Ignore MLint
        error(['Image processing was canceled in the ', ModuleName, ' module because the Max diameter entry is invalid.'])
    end
end
if MinDiameter > MaxDiameter
    error(['Image processing was canceled in the ', ModuleName, ' module because the Min Diameter is larger than the Max Diameter.'])
end

%%% Checks Minimum Width parameter
if strcmpi(RoundOrNot,'Yes')
    RoundOrNot = 1;
    MinWidth = MinDiameter;
else
    MinWidth = str2double(RoundOrNot);
    RoundOrNot = 0;
    if isnan(MinWidth)
        error(['Image processing was canceled in the ' ModuleName ' module because the Minimum Width you specified is invalid.']);
    end
end

%%% Checks that the Min and Max threshold bounds have valid values
index = strfind(ThresholdRange,',');
if isempty(index)
    error(['Image processing was canceled in the ', ModuleName, ' module because the Min and Max threshold bounds are invalid.'])
end
%%% We do not check validity of variables now because it is done later, in CPthreshold
MinimumThreshold = ThresholdRange(1:index-1);
MaximumThreshold = ThresholdRange(index+1:end);

%%% Check the smoothing filter size parameter
if ~strcmpi(SizeOfSmoothingFilter,'Automatic')
    SizeOfSmoothingFilter = str2double(SizeOfSmoothingFilter);
    if isnan(SizeOfSmoothingFilter) | isempty(SizeOfSmoothingFilter) | SizeOfSmoothingFilter < 0 | SizeOfSmoothingFilter > min(size(OrigImage)) %#ok Ignore MLint
        error(['Image processing was canceled in the ', ModuleName, ' module because the specified size of the smoothing filter is not valid or unreasonable.'])
    end
end

%%% Check the maxima suppression size parameter
if ~strcmpi(MaximaSuppressionSize,'Automatic')
    MaximaSuppressionSize = str2double(MaximaSuppressionSize);
    if isnan(MaximaSuppressionSize) | isempty(MaximaSuppressionSize) | MaximaSuppressionSize < 0 %#ok Ignore MLint
        error(['Image processing was canceled in the ', ModuleName, ' module because the specified maxima suppression size is not valid or unreasonable.'])
    end
end

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Invert image if necessary. This module works with white outlines on dark backgrounds
if strcmp(OutlineColor,'Black')
    OrigImage = imcomplement(OrigImage);
end

%%%% Might be good to get threshold AFTER outlines have been processed...

%%% Get threshold if the image is not already binary
% if ~islogical(OrigImage)
%     [handles,OrigThreshold] = CPthreshold(handles,Threshold,pObject,MinimumThreshold,MaximumThreshold,ThresholdCorrection,OrigImage,ImageName,ModuleName);
% else
%     OrigThreshold = 0;
% end

%%% Sets up loop for test mode.
if strcmp(TestMode,'Yes')
    LocalMaximaTypeList = {'Intensity' 'Shape'};
    WatershedTransformImageTypeList = {'Intensity' 'Distance' 'None'};
else
    LocalMaximaTypeList = {OriginalLocalMaximaType};
    WatershedTransformImageTypeList = {OriginalWatershedTransformImageType};
end

%%% Start loop for test mode, fit for normal use too
for LocalMaximaTypeNumber = 1:length(LocalMaximaTypeList)
    for WatershedTransformImageTypeNumber = 1:length(WatershedTransformImageTypeList)

        %%% Get in-loop variables
        LocalMaximaType = LocalMaximaTypeList{LocalMaximaTypeNumber};
        WatershedTransformImageType = WatershedTransformImageTypeList{WatershedTransformImageTypeNumber};

        %%% Start image processing

        %%% First, process the bare outlines, regardless of whether they are binary or grayscale
        if strcmp(CloseGaps,'Yes')
            warning off MATLAB:intConvertOverflow                          % For binary images
            ImageToThreshold = imfill(OrigImage,'holes');                  % Fill whatever we can
            StructEl = strel('disk',round(MinWidth/2));                    % Create structure element
            ImageToThreshold = imclose(ImageToThreshold,StructEl);         % Close small gaps
            ImageToThreshold = imfill(ImageToThreshold,'holes');           % Fill again
            warning on MATLAB:intConvertOverflow
        else
            ImageToThreshold = imfill(double(OrigImage),'holes');
        end

        %%% Get threshold if the image is not already binary
        if LocalMaximaTypeNumber == 1  % We only want to get the threshold once
            %             if ~islogical(OrigImage)
            [handles,OrigThreshold] = CPthreshold(handles,Threshold,pObject,MinimumThreshold,MaximumThreshold,ThresholdCorrection,ImageToThreshold,ImageName,ModuleName);
            %             else
            %                 OrigThreshold = 0;
            %             end
        end

        Threshold = OrigThreshold;

        %%% Apply a slight smoothing before thresholding to remove
        %%% 1-pixel objects and to smooth the edges of the objects.
        %%% Note that this smoothing is hard-coded, and not controlled
        %%% by the user, but it is omitted if the user selected 0 for
        %%% the size of the smoothing filter.
        if SizeOfSmoothingFilter == 0 | ~islogical(ImageToThreshold) %#ok ignore MLint
            %%% No blurring is done.
            BlurredImage = ImageToThreshold;
        else
            sigma = 1;
            FiltLength = 2*sigma;                                              % Determine filter size, min 3 pixels, max 61
            [x,y] = meshgrid(-FiltLength:FiltLength,-FiltLength:FiltLength);   % Filter kernel grid
            f = exp(-(x.^2+y.^2)/(2*sigma^2));f = f/sum(f(:));                 % Gaussian filter kernel
            %%% This adjustment prevents the outer borders of the image from being
            %%% darker (due to padding with zeros), which causes some objects on the
            %%% edge of the image to not  be identified all the way to the edge of the
            %%% image and therefore not be thrown out properly.
            if islogical(ImageToThreshold)
                BlurredImage = conv2(double(ImageToThreshold),f,'same') ./ conv2(ones(size(ImageToThreshold)),f,'same');
            else
                BlurredImage = conv2(ImageToThreshold,f,'same') ./ conv2(ones(size(ImageToThreshold)),f,'same');
            end
        end

        %%%% Look in to BWMORPH for nice ways of making the thresholded image better

        %%% Threshold image
        PrelimObjects = BlurredImage > Threshold;

        %%% Correct depending on original image. When processing binary
        %%% images, it is often good to blur, threshold, and then close
        %%% gaps again with a smaller range. When processing grayscale
        %%% images, it is better not to blur, threshold, and then erode the
        %%% image a little bit because thresholding made objects wider.
        StructEl2 = strel('disk',round(MinWidth/3.5));
        if islogical(OrigImage) && strcmp(CloseGaps,'Yes')
            PrelimObjects = imclose(double(PrelimObjects),StructEl2);
        end
        PrelimObjects = imfill(double(PrelimObjects),'holes');

        %%% This should probably go after the objects have been identified (thresholded)
        if strcmp(RemoveOutlines,'Yes')
            %%% Recommended only if the outlines are very thin, and binary
            Objects = imsubtract(double(PrelimObjects),double(OrigImage));
        else
            StructEl3 = strel('disk',round(MinWidth/4));
            Objects = imerode(PrelimObjects,StructEl3);
        end

        %%% Check for CropMask
        fieldname = ['CropMask', ImageName];
        if isfield(handles.Pipeline,fieldname)
            %%% Retrieves previously selected cropping mask from handles
            %%% structure.
            BinaryCropImage = handles.Pipeline.(fieldname);
            Objects = Objects & BinaryCropImage;
        end

        Threshold = mean(Threshold(:));                                    % Use average threshold downstream
        drawnow


        %%% Now that we have masked the objects we want to Identify, we
        %%% could estimate the amount of objects by just dividing the total
        %%% area by the average object area.
        if strcmp(Estimate,'Just Count')
            %             figure,imshow(Objects)
            props = regionprops(double(Objects),'Area');
            Area = cat(1,props.Area);
            TotalArea = floor(sum(Area)/100);
            MinArea = pi*(MinDiameter.^2)/400;
            MaxArea = pi*(MaxDiameter.^2)/400;
            MeanArea = ceil(mean([MinArea,MaxArea]));
            EstimatedNumberOfObjects = TotalArea/MeanArea;
            %             disp(EstimatedNumberOfObjects);
            %             disp(2);
        end

        %%% OR, we could try skeletonizing the object mask, figure out a
        %%% way of separating the skeletons. And then use the propagation
        %%% method to separate the objects were we want them to be
        %%% separated. This could be added as another de-clumping method,
        %%% but for now let's keep it separate.

        %%% From here on, most of this code is from IdPrimAuto. I didn't
        %%% have time to check the 'saving to handles structure' part and
        %%% some of the 'display results' part.

        %%% STEP 2. If user wants, extract local maxima (of intensity or distance) and apply watershed transform
        %%% to separate neighboring objects. (De-clump)

        if ~strcmp(LocalMaximaType,'None') & ~strcmp(WatershedTransformImageType,'None') %#ok Ignore MLint

            %%% Smooth images for maxima suppression
            if islogical(OrigImage)
                % We'll just use the previous BlurredImage
                SizeOfSmoothingFilter = 0;
            elseif strcmpi(SizeOfSmoothingFilter,'Automatic')
                SizeOfSmoothingFilter=2.35*MinDiameter/3.5;
                [BlurredImage SizeOfSmoothingFilter] = CPsmooth(ImageToThreshold,'M',SizeOfSmoothingFilter,0);
            else
                [BlurredImage SizeOfSmoothingFilter] = CPsmooth(ImageToThreshold,'M',SizeOfSmoothingFilter,0);
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

            %%% Calculate ImageResizeFactor, MaximaSuppressionSize, MaximaMask
            if strcmp(UseLowRes,'Yes') && MinDiameter > 10
                ImageResizeFactor = 10/MinDiameter;
                if strcmpi(MaximaSuppressionSize,'Automatic')
                    MaximaSuppressionSize = 7;             % ~ 10/1.5
                else
                    MaximaSuppressionSize = round(MaximaSuppressionSize*ImageResizeFactor);
                end
            else
                ImageResizeFactor = 1;
                if strcmpi(MaximaSuppressionSize,'Automatic')
                    MaximaSuppressionSize = round(MinDiameter/1.5);
                else
                    MaximaSuppressionSize = round(MaximaSuppressionSize);
                end
            end
            MaximaMask = getnhood(strel('disk', MaximaSuppressionSize));

            %%% Get the local maxima using the method specified
            if strcmp(LocalMaximaType,'Intensity')
                if islogical(OrigImage)
                    error(['Image processing was canceled in the ' ModuleName ' module because you have chosen to use the Intensity method to distinguish clumped objects, but provided a binary image.']);
                end
                if strcmp(UseLowRes,'Yes')
                    %%% Find local maxima in a lower resolution image
                    ResizedBlurredImage = imresize(BlurredImage,ImageResizeFactor,'bilinear');
                else
                    ResizedBlurredImage = BlurredImage;
                end

                %%% Initialize MaximaImage
                MaximaImage = ResizedBlurredImage;
                %%% Save only local maxima
                MaximaImage(ResizedBlurredImage < ordfilt2(ResizedBlurredImage,sum(MaximaMask(:)),MaximaMask)) = 0;

                if strcmp(UseLowRes,'Yes')
                    %%% Restore image size
                    MaximaImage = imresize(MaximaImage,size(BlurredImage),'bilinear');
                end

                %%% Remove dim maxima
                MaximaImage = MaximaImage > Threshold;
                %%% Shrink to points (needed because of the resizing)
                MaximaImage = bwmorph(MaximaImage,'shrink',inf);

            elseif strcmp(LocalMaximaType,'Shape')
                %%% Calculate distance transform
                DistanceTransformedImage = bwdist(~Objects);
                %%% Add some noise to get distinct maxima
                %%% First set seed to 0, so that it is reproducible
                rand('seed',0);
                DistanceTransformedImage = DistanceTransformedImage + ...
                    0.001*rand(size(DistanceTransformedImage));
                if strcmp(UseLowRes,'Yes')
                    ResizedDistanceTransformedImage = imresize(DistanceTransformedImage,ImageResizeFactor,'bilinear');
                else
                    ResizedDistanceTransformedImage = DistanceTransformedImage;
                end
                %%% Initialize MaximaImage
                MaximaImage = ones(size(ResizedDistanceTransformedImage));
                %%% Set all pixels that are not local maxima to zero
                MaximaImage(ResizedDistanceTransformedImage < ordfilt2(ResizedDistanceTransformedImage,sum(MaximaMask(:)),MaximaMask)) = 0;
                if strcmp(UseLowRes,'Yes')
                    %%% Restore image size
                    MaximaImage = imresize(MaximaImage,size(Objects),'bilinear');
                end
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
                %%% We may have to calculate the distance transform if not already done:
                if ~exist('DistanceTransformedImage','var')
                    DistanceTransformedImage = bwdist(~Objects);
                end
                Overlaid = imimposemin(-DistanceTransformedImage,MaximaImage);
                figure, imagesc(Overlaid), title('overlaid');
                figure, imagesc(-DistanceTransformedImage), title('-DistanceTransformedImage');

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
            NumberOfObjectsBeforeMerge = max(Objects(:));
            Objects = MergeObjects(Objects,OrigImage,[MinDiameter MaxDiameter],RoundOrNot);
            NumberOfObjectsAfterMerge = max(Objects(:));
            NumberOfMergedObjects = NumberOfObjectsBeforeMerge-NumberOfObjectsAfterMerge;
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
            PrevObjects = Objects;
            Objects = CPclearborder(Objects);

            %%% TESTING CODE TO REMOVE BORDERS FROM ELLIPSE CROPPED
            %%% OBJECTS
            if sum(PrevObjects(:)) == sum(Objects(:))
                try
                    CropMask = handles.Pipeline.(['CropMask',ImageName]);
                    CropBorders = bwperim(CropMask);
                    BorderTable = sortrows(unique([CropBorders(:) Objects(:)],'rows'),1);
                    for z = 1:size(BorderTable,1)
                        if BorderTable(z,1) ~= 0 && BorderTable(z,2) ~= 0
                            Objects(Objects == BorderTable(z,2)) = 0;
                        end
                    end
                end
            end
        end

        %%% Store objects that touch the border for display
        BorderObjects = tmp - Objects;

        %%% Relabel the objects
        [Objects,NumOfObjects] = bwlabel(Objects > 0);
        FinalLabelMatrixImage = Objects;


        %%%%%%%%%%%%%%%%%%%%%%%
        %%% DISPLAY RESULTS %%%
        %%%%%%%%%%%%%%%%%%%%%%%
        drawnow

        if strcmp(OriginalLocalMaximaType,'None') || (strcmp(OriginalLocalMaximaType,LocalMaximaType) && strcmp(OriginalWatershedTransformImageType,WatershedTransformImageType))

            %%% Indicate objects in original image and color excluded objects in red
            tmp = double(OrigImage)/double(max(OrigImage(:)));
            OutlinedObjectsR = tmp;
            OutlinedObjectsG = tmp;
            OutlinedObjectsB = tmp;
            PerimObjects = bwperim(Objects > 0);
            PerimDiameter = bwperim(DiameterExcludedObjects > 0);
            PerimBorder = bwperim(BorderObjects > 0);
            OutlinedObjectsR(PerimObjects) = 0; OutlinedObjectsG(PerimObjects) = 1; OutlinedObjectsB(PerimObjects) = 0;
            OutlinedObjectsR(PerimDiameter) = 1; OutlinedObjectsG(PerimDiameter)   = 0; OutlinedObjectsB(PerimDiameter)   = 0;
            OutlinedObjectsR(PerimBorder) = 1; OutlinedObjectsG(PerimBorder) = 1; OutlinedObjectsB(PerimBorder) = 0;

            FinalOutline = false(size(OrigImage,1),size(OrigImage,2));
            FinalOutline(PerimObjects) = 1;
            FinalOutline(PerimDiameter) = 0;
            FinalOutline(PerimBorder) = 0;

            ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
            if any(findobj == ThisModuleFigureNumber)
                %%% Activates the appropriate figure window.
                CPfigure(handles,'Image',ThisModuleFigureNumber);
                if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
                    CPresizefigure(OrigImage,'TwoByTwo',ThisModuleFigureNumber)
                end
                subplot(2,2,1)
                CPimagesc(OrigImage,handles);
                title(['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
                hx = subplot(2,2,2);
                im = CPlabel2rgb(handles,Objects);
                CPimagesc(im,handles);
                title(['Identified ',ObjectName]);
                hy = subplot(2,2,3);
                OutlinedObjects = cat(3,OutlinedObjectsR,OutlinedObjectsG,OutlinedObjectsB);
                CPimagesc(OutlinedObjects,handles);
                title(['Outlined ', ObjectName]);

                %%% Report numbers
                posx = get(hx,'Position');
                posy = get(hy,'Position');
                bgcolor = get(ThisModuleFigureNumber,'Color');
                uicontrol(ThisModuleFigureNumber,'Style','Text','Units','Normalized','Position',[posx(1)-0.05 posy(2)+posy(4)-0.04 posx(3)+0.1 0.04],...
                    'BackgroundColor',bgcolor,'HorizontalAlignment','Left','String',sprintf('Threshold:  %0.3f',Threshold),'FontSize',handles.Preferences.FontSize);
                uicontrol(ThisModuleFigureNumber,'Style','Text','Units','Normalized','Position',[posx(1)-0.05 posy(2)+posy(4)-0.08 posx(3)+0.1 0.04],...
                    'BackgroundColor',bgcolor,'HorizontalAlignment','Left','String',sprintf('Number of identified objects: %d',NumOfObjects),'FontSize',handles.Preferences.FontSize);
                uicontrol(ThisModuleFigureNumber,'Style','Text','Units','Normalized','Position',[posx(1)-0.05 posy(2)+posy(4)-0.16 posx(3)+0.1 0.08],...
                    'BackgroundColor',bgcolor,'HorizontalAlignment','Left','String',sprintf('90%% of objects within diameter range [%0.1f, %0.1f] pixels',...
                    Lower90Limit,Upper90Limit),'FontSize',handles.Preferences.FontSize);
                ObjectCoverage = 100*sum(sum(Objects > 0))/numel(Objects);
                uicontrol(ThisModuleFigureNumber,'Style','Text','Units','Normalized','Position',[posx(1)-0.05 posy(2)+posy(4)-0.20 posx(3)+0.1 0.04],...
                    'BackgroundColor',bgcolor,'HorizontalAlignment','Left','String',sprintf('%0.1f%% of image consists of objects',ObjectCoverage),'FontSize',handles.Preferences.FontSize);
                if ~strcmp(LocalMaximaType,'None') & ~strcmp(WatershedTransformImageType,'None') %#ok Ignore MLint
                    uicontrol(ThisModuleFigureNumber,'Style','Text','Units','Normalized','Position',[posx(1)-0.05 posy(2)+posy(4)-0.24 posx(3)+0.1 0.04],...
                        'BackgroundColor',bgcolor,'HorizontalAlignment','Left','String',sprintf('Smoothing filter size:  %0.1f',SizeOfSmoothingFilter),'FontSize',handles.Preferences.FontSize);
                    uicontrol(ThisModuleFigureNumber,'Style','Text','Units','Normalized','Position',[posx(1)-0.05 posy(2)+posy(4)-0.28 posx(3)+0.1 0.04],...
                        'BackgroundColor',bgcolor,'HorizontalAlignment','Left','String',sprintf('Maxima suppression size:  %d',round(MaximaSuppressionSize/ImageResizeFactor)),'FontSize',handles.Preferences.FontSize);
                end
                if strcmp(MergeChoice,'Yes')
                    uicontrol(ThisModuleFigureNumber,'Style','Text','Units','Normalized','Position',[posx(1)-0.05 posy(2)+posy(4)-0.32 posx(3)+0.1 0.04],...
                        'BackgroundColor',bgcolor,'HorizontalAlignment','Left','String',sprintf('Number of Merged Objects:  %d',NumberOfMergedObjects),'FontSize',handles.Preferences.FontSize);
                end
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% SAVE DATA TO HANDLES STRUCTURE %%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        drawnow

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
        if ~strcmpi(SaveOutlines,'Do not save')
            try    handles.Pipeline.(SaveOutlines) = FinalOutline;
            catch error(['The object outlines were not calculated by the ', ModuleName, ' module, so these images were not saved to the handles structure. The Save Images module will therefore not function on these images. This is just for your information - image processing is still in progress, but the Save Images module will fail if you attempted to save these images.'])
            end
        end

        if strcmp(MergeChoice,'Yes')
            %%% Saves the NumberOfMergedObjects to the handles structure.
            %%% See comments for the Threshold saving below
            if ~isfield(handles.Measurements.Image,'NumberOfMergedObjectsFeatures')
                handles.Measurements.Image.NumberOfMergedObjectsFeatures = {};
                handles.Measurements.Image.NumberOfMergedObjects = {};
            end
            column = find(~cellfun('isempty',strfind(handles.Measurements.Image.NumberOfMergedObjectsFeatures,ObjectName)));
            if isempty(column)
                handles.Measurements.Image.NumberOfMergedObjectsFeatures(end+1) = {ObjectName};
                column = length(handles.Measurements.Image.NumberOfMergedObjectsFeatures);
            end
            handles.Measurements.Image.NumberOfMergedObjects{handles.Current.SetBeingAnalyzed}(1,column) = NumberOfMergedObjects;
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
        % typically happen for the first cycle. Append the feature name in the
        % handles.Measurements.Image.ThresholdFeatures matrix
        if isempty(column)
            handles.Measurements.Image.ThresholdFeatures(end+1) = {ObjectName};
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
            handles.Measurements.Image.ObjectCountFeatures(end+1) = {ObjectName};
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
            %%% If the test mode window does not exist, it is created, but only
            %%% if it's at the starting image set (if the user closed the window
            %%% intentionally, we don't want to pop open a new one).
            IdPrimTestModeSegmentedFigureNumber = findobj('Tag','IdPrimTestModeSegmentedFigure');
            if isempty(IdPrimTestModeSegmentedFigureNumber) && handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet;
                %%% Creates the window, sets its tag, and puts some
                %%% text in it. The first lines are meant to find a suitable
                %%% figure number for the window, so we don't choose a
                %%% figure number that is being used by another module.
                IdPrimTestModeSegmentedFigureNumber = CPfigurehandle(handles);
                CPfigure(handles,'Image',IdPrimTestModeSegmentedFigureNumber);
                set(IdPrimTestModeSegmentedFigureNumber,'Tag','IdPrimTestModeSegmentedFigure',...
                    'name',['IdentifyPrimAutomatic Test Objects Display, cycle # ']);
                uicontrol(IdPrimTestModeSegmentedFigureNumber,'style','text','units','normalized','string','Identified objects are shown here. Note: Choosing "None" for either option will result in the same image, therefore only the Intensity and None option has been shown.','position',[.65 .1 .3 .4],'BackgroundColor',[.7 .7 .9])
            end
            %%% If the figure window DOES exist now, then calculate and display items
            %%% in it.
            if ~isempty(IdPrimTestModeSegmentedFigureNumber)
                %%% Makes the window active.
                CPfigure(IdPrimTestModeSegmentedFigureNumber(1));
                %%% Updates the cycle number on the window.
                CPupdatefigurecycle(handles.Current.SetBeingAnalyzed,IdPrimTestModeSegmentedFigureNumber);

                subplot(2,3,WatershedTransformImageTypeNumber+3*(LocalMaximaTypeNumber-1));
                im = CPlabel2rgb(handles,Objects);
                CPimagesc(im,handles);
                title(sprintf('%s and %s',LocalMaximaTypeList{LocalMaximaTypeNumber},WatershedTransformImageTypeList{WatershedTransformImageTypeNumber}));
            end

            %%% Repeat what we've done for the segmented test mode window, now
            %%% for the outlined test mode window.
            IdPrimTestModeOutlinedFigureNumber = findobj('Tag','IdPrimTestModeOutlinedFigure');
            if isempty(IdPrimTestModeOutlinedFigureNumber) && handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet;
                IdPrimTestModeOutlinedFigureNumber = CPfigurehandle(handles);
                CPfigure(handles,'Image',IdPrimTestModeOutlinedFigureNumber);
                set(IdPrimTestModeOutlinedFigureNumber,'Tag','IdPrimTestModeOutlinedFigure',...
                    'name',['IdentifyPrimAutomatic Test Outlines Display, cycle # ']);
                uicontrol(IdPrimTestModeOutlinedFigureNumber,'style','text','units','normalized','string','Outlined objects are shown here. Note: Choosing "None" for either option will result in the same image, therefore only the Intensity and None option has been shown.','position',[.65 .1 .3 .4],'BackgroundColor',[.7 .7 .9]);
            end

            if ~isempty(IdPrimTestModeOutlinedFigureNumber)
                CPfigure(IdPrimTestModeOutlinedFigureNumber(1));
                CPupdatefigurecycle(handles.Current.SetBeingAnalyzed,IdPrimTestModeOutlinedFigureNumber);

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
                CPimagesc(OutlinedObjects,handles);
                title(sprintf('%s and %s',LocalMaximaTypeList{LocalMaximaTypeNumber},WatershedTransformImageTypeList{WatershedTransformImageTypeNumber}));
            end
        end
    end
end

%%% Old stuff
% %%% Saves the location of each segmented object
% handles.Measurements.(ObjectName).LocationFeatures = {'CenterX','CenterY'};
% tmp = regionprops(Objects,'Centroid');
% Centroid = cat(1,tmp.Centroid);
% handles.Measurements.(ObjectName).Location(handles.Current.SetBeingAnalyzed) = {Centroid};
%
%
% %%%%%%%%%%%%%%%%%%%%%%%
% %%% DISPLAY RESULTS %%%
% %%%%%%%%%%%%%%%%%%%%%%%
% drawnow
%
% ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
% if any(findobj == ThisModuleFigureNumber) == 1;
%     drawnow
%     CPfigure(handles,'Image',ThisModuleFigureNumber);
%     %%% Sets the width of the figure window to be appropriate (half width).
%     %%% TODO: update to new resizefigure subfunction!!
%     if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
%         originalsize = get(ThisModuleFigureNumber, 'position');
%         newsize = originalsize;
%         newsize(3) = 0.5*originalsize(3);
%         set(ThisModuleFigureNumber, 'position', newsize);
%     end
%     %%% A subplot of the figure window is set to display the original image.
%     subplot(2,1,1); CPimagesc(OrigImage,handles);
%     title(['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
%     %%% A subplot of the figure window is set to display the colored label
%     %%% matrix image.
%     subplot(2,1,2); CPimagesc(ObjectsIdentifiedImage,handles);
%     title(['Processed ',ObjectName]);
% end
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%% SAVE DATA TO HANDLES STRUCTURE %%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% drawnow
%
% %%% Saves the processed image to the handles structure.
% fieldname = ['Segmented',ObjectName];
% handles.Pipeline.(fieldname) = ObjectsIdentifiedImage;
%
% %%% Saves the ObjectCount, i.e., the number of segmented objects.
% %%% See comments for the Threshold saving above
% if ~isfield(handles.Measurements.Image,'ObjectCountFeatures')
%     handles.Measurements.Image.ObjectCountFeatures = {};
%     handles.Measurements.Image.ObjectCount = {};
% end
% column = find(~cellfun('isempty',strfind(handles.Measurements.Image.ObjectCountFeatures,ObjectName)));
% if isempty(column)
%     handles.Measurements.Image.ObjectCountFeatures(end+1) = {ObjectName};
%     column = length(handles.Measurements.Image.ObjectCountFeatures);
% end
% handles.Measurements.Image.ObjectCount{handles.Current.SetBeingAnalyzed}(1,column) = max(Objects(:));
%
% %%% Saves the Threshold value to the handles structure.
% %%% Storing the threshold is a little more complicated than storing other measurements
% %%% because several different modules will write to the handles.Measurements.Image.Threshold
% %%% structure, and we should therefore probably append the current threshold to an existing structure.
% % First, if the Threshold fields don't exist, initialize them
% if ~isfield(handles.Measurements.Image,'ThresholdFeatures')
%     handles.Measurements.Image.ThresholdFeatures = {};
%     handles.Measurements.Image.Threshold = {};
% end
% % Search the ThresholdFeatures to find the column for this object type
% column = find(~cellfun('isempty',strfind(handles.Measurements.Image.ThresholdFeatures,ObjectName)));
% % If column is empty it means that this particular object has not been segmented before. This will
% % typically happen for the first cycle. Append the feature name in the
% % handles.Measurements.Image.ThresholdFeatures matrix
% if isempty(column)
%     handles.Measurements.Image.ThresholdFeatures(end+1) = {ObjectName};
%     column = length(handles.Measurements.Image.ThresholdFeatures);
% end
% handles.Measurements.Image.Threshold{handles.Current.SetBeingAnalyzed}(1,column) = Threshold;
%
%
% %%% Saves the location of each segmented object
% handles.Measurements.(ObjectName).LocationFeatures = {'CenterX','CenterY'};
% tmp = regionprops(Objects,'Centroid');
% Centroid = cat(1,tmp.Centroid);
% handles.Measurements.(ObjectName).Location(handles.Current.SetBeingAnalyzed) = {Centroid};
% % fieldname = [ObjectName, 'Count']
% % handles.Measurements.Image.(fieldname) = NumOfObjects;
% fieldname = ['UneditedSegmented',ObjectName];
% handles.Pipeline.(fieldname) = PrevObjects;

%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTIONS %%%
%%%%%%%%%%%%%%%%%%%%

function Objects = MergeObjects(Objects,OrigImage,Diameters,RoundOrNot)

%%% The round or not variable was added so that the criteria would be
%%% different if the objects are expected to be round or not. This has not
%%% yet been implemented though.

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
    [ignore,LikelihoodRank]   = sort(LikelihoodRatio,'descend'); %#ok Ignore MLint % The higher the LikelihoodRatio the better
    [ignore,EccentricityRank] = sort(MergedEccentricity,'ascend'); %#ok Ignore MLint % The lower the eccentricity the better
    NeighborScore = zeros(length(NeighborsNbr),1);
    for j = 1:length(NeighborsNbr)
        NeighborScore(j) = find(LikelihoodRank == j) +  find(EccentricityRank == j);
    end

    %%% Go through the neighbors, starting with the highest ranked, and merge
    %%% with the first neighbor for which certain basic criteria are fulfilled.
    %%% If no neighbor fulfil the basic criteria, there will be no merge.
    [ignore,TotalRank] = sort(NeighborScore); %#ok Ignore MLint
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