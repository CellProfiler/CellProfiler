function handles = IdentifyPrimLaplacianOfGaussian(handles)

% Help for the Laplacian Of Gaussian module:
% Category: Object Identification and Modification
%
% This object identification module applies an algorithm developed by
% Perlman, Mitchison, et al. (Science, 2004 TODO: supply exact reference)
% to identify cell nuclei.
%
% TODO: update help
% Settings:
%
% Size range: You may exclude objects that are smaller or bigger than
% the size range you specify. A comma should be placed between the
% lower size limit and the upper size limit. The units here are pixels
% so that it is easy to zoom in on found objects and determine the
% size of what you think should be excluded.
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
% Maxima suppression neighborhood & blur radius: These variables
% affect whether objects close by each other are considered a single
% object or multiple objects. They do not affect the dividing lines
% between an object and the background.  If you see too many objects
% merged that ought to be separate, the values should be lower. If you
% see too many objects split up that ought to be merged, the values
% should be higher. The blur radius tries to reduce the texture of
% objects so that each real, distinct object has only one peak of
% intensity. The maxima suppression neighborhood should be set to be
% roughly equivalent to the minimum radius of a real object of
% interest. Basically, any distinct 'objects' which are found but are
% within two times this distance from each other will be assumed to be
% actually two lumpy parts of the same object, and they will be
% merged. Note that increasing the blur radius increases
% the processing time exponentially.
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
%defaultVAR02 = Cells
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Size range (in pixels) of objects to include (1,99999 = do not discard any)
%defaultVAR03 = 1,99999
SizeRange = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = Neighborhood size (height and width in pixels)
%choiceVAR04 = 9 9
NbhdSizeStr = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu custom

%textVAR05 = Sigma
%defaultVAR05 = 1.8
Sigma = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,5}));

%textVAR06 = Minimum Area
%defaultVAR06 = 3
MinArea = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,6}));

%textVAR07 = Size for Wiener filter (height and width in pixels)
%choiceVAR07 = 5 5
WienerSizeStr = char(handles.Settings.VariableValues{CurrentModuleNum,7});
%inputtypeVAR07 = popupmenu custom

%textVAR08 = Threshold
%defaultVAR08 = -.001
Threshold = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,8}));

%textVAR09 = Do you want to include objects touching the edge (border) of the image? (Yes or No)
%choiceVAR09 = No
%choiceVAR09 = Yes
IncludeEdge = char(handles.Settings.VariableValues{CurrentModuleNum,9});
%inputtypeVAR09 = popupmenu

%textVAR10 =  What do you want to call the labeled matrix image?
%infotypeVAR10 = imagegroup indep
%choiceVAR10 = Do not save
%choiceVAR10 = LabeledNuclei
SaveColored = char(handles.Settings.VariableValues{CurrentModuleNum,10}); 
%inputtypeVAR10 = popupmenu custom

%textVAR11 = Do you want to save the labeled matrix image in RGB or grayscale?
%choiceVAR11 = RGB
%choiceVAR11 = Grayscale
SaveMode = char(handles.Settings.VariableValues{CurrentModuleNum,11}); 
%inputtypeVAR11 = popupmenu

%%% Determines what the user entered for the size range.
SizeRangeNumerical = str2num(SizeRange); %#ok We want to ignore MLint error checking for this line.
MinSize = SizeRangeNumerical(1);
MaxSize = SizeRangeNumerical(2);

%%% Check NbhdSize
NeighborhoodSize = sscanf(NbhdSizeStr,'%d',[1,inf]);
if size(NeighborhoodSize(:),1) ~= 2 || max(NeighborhoodSize <= 0) > 0
    error('Image processing was canceled because the Laplacian Of Gaussian requires Neighborhood Size to be two integers, separated by a space, >0.')
end

%%% Check Sigma
if isnan(Sigma) || Sigma <= 0
    error('Image processing was canceled because the Laplacian Of Gaussian requires Sigma to be a number greater than 0.');
end

%%% Check MinArea
if isnan(MinArea) || MinArea <= 0
    error('Image processing was canceled because the Laplacian Of Gaussian requires MinArea to be a number greater than 0.');
end

%%% Check Wiener Filter Size
WienerSize = sscanf(WienerSizeStr,'%d',[1,inf]);
if size(WienerSize(:),1) ~= 2 || max(WienerSize <= 0) > 0
    error('Image processing was canceled because the Laplacian Of Gaussian requires Wiener Filter Size to be two integers, separated by a space, >0.')
end

%%% Check Threshold
if isnan(Threshold)
    error('Image processing was canceled because the Laplacian Of Gaussian requires Threshold to be a number.');
end

%%%VariableRevisionNumber = 2

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



%%% STEP 1: Finds markers for each nucleus based on local maxima in the
%%% intensity image.
drawnow
%{
if BlurRadius == 0
    BlurredImage = OrigImage;
else
%%% Blurs the image.
%%% Note: using filter2 is much faster than imfilter (e.g. 14.5 sec vs. 99.1 sec).
FiltSize = max(3,ceil(4*BlurRadius));
BlurredImage = filter2(fspecial('gaussian',FiltSize, BlurRadius), OrigImage);
end

%%% Perturbs the blurred image so that local maxima near each other with
%%% identical values will now have slightly different values.
%%% Saves off the random number generator's state, and set the state to
%%% a particular value (for repeatability)
oldstate = rand('state');
rand('state',0);
%%% Adds a random value between 0 and 0.002 to each pixel in the
%%% BlurredImage. We chose .002
BlurredImage = BlurredImage + 0.002*rand(size(BlurredImage));
%%% Restores the random number generator's state.
rand('state',oldstate);

%%% Extracts local maxima and filters them by eliminating maxima that are
%%% within a certain distance of each other.
MaximaImage = BlurredImage;
MaximaMask = strel('disk', MaximaSuppressionNeighborhood);
MaximaImage(BlurredImage < ordfilt2(BlurredImage,sum(sum(getnhood(MaximaMask))),getnhood(MaximaMask))) = 0;
%%% Determines the threshold to be used, if the user has left the Threshold
%%% variable set to 0.
if Threshold == 0
    Threshold = CPgraythresh(OrigImage);
    Threshold = Threshold*ThresholdAdjustmentFactor;
end
MinimumThreshold = str2num(MinimumThreshold);
Threshold = max(MinimumThreshold,Threshold);

%%% Thresholds the image to eliminate dim maxima.
MaximaImage(~im2bw(OrigImage, Threshold))=0;

%%% STEP 2: Performs watershed function on the original intensity
%%% (grayscale) image.
drawnow
%%% Inverts original image.
InvertedOriginal = imcomplement(OrigImage);
%%% Overlays the nuclear markers (maxima) on the inverted original image so
%%% there are black dots on top of each dark nucleus on a white background.
Overlaid = imimposemin(InvertedOriginal,MaximaImage);
%%% Identifies watershed lines.
BlackWatershedLinesPre = watershed(Overlaid);
%figure, imagesc(BlackWatershedLinesPre)
%%% Superimposes watershed lines as white (255) onto the inverted original
%%% image.
WhiteWatershedOnInvertedOrig = InvertedOriginal;
WhiteWatershedOnInvertedOrig(BlackWatershedLinesPre == 0) = 255;
%figure, imagesc(WhiteWatershedOnInvertedOrig)

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
%}

%%% Creates the Laplacian of a Gaussian filter.
rgLoG=fspecial('log',NeighborhoodSize,Sigma);
%%% Filters the image.
imLoGout=imfilter(double(OrigImage),rgLoG);
    figure, imagesc(imLoGout), colormap(gray), title('imLoGout')
%%% Removes noise using the weiner filter.
imLoGoutW=wiener2(imLoGout,WienerSize);
    figure, imagesc(imLoGoutW), colormap(gray), title('imLoGoutW')

%%%%%%%%%%%%%%
rgNegCurve = imLoGoutW < Threshold;
class(rgNegCurve)
min(min(rgNegCurve))
max(max(rgNegCurve))

%set outsides
rgNegCurve([1 end],1:end)=1;
rgNegCurve(1:end,[1 end])=1;

%disp(['Generated LoG regions. Time: ' num2str(toc)])

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
rgDilated=RgSmartDilate(rgLabelled,2); %%% IMPORTANT VARIABLE
rgFill=imfill(rgDilated,'holes');

%%%%SE=strel('diamond',1);
%%%%rgErode=imerode(rgFill,SE);
%%%%rgOut=rgErode;

%%%%%%%%%%%%

% InvertedBinaryImage = RgSmartDilate(rgNegCurve,1);
InvertedBinaryImage = rgDilated;

%%% Creates label matrix image.
% rgLabelled2=uint16(bwlabel(imLoGoutW,4));
% figure, imagesc(rgLabelled2), colormap(gray), title('rgLabelled2')
% FinalLabelMatrixImage = bwlabel(FinalBinary,4);

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
FinalBinaryPre = im2bw(PrelimLabelMatrixImage4,.5);
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
    subplot(2,2,1); imagesc(OrigImage);colormap(gray);
    title(['Input Image, Image Set # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the colored label
    %%% matrix image.
    subplot(2,2,2); imagesc(ColoredLabelMatrixImage); title(['Segmented ',ObjectName]);
    %%% A subplot of the figure window is set to display the Overlaid image,
    %%% where the maxima are imposed on the inverted original image
    % subplot(2,2,3); imagesc(Overlaid); colormap(gray); title([ObjectName, ' markers']);
    %%% A subplot of the figure window is set to display the inverted original
    %%% image with watershed lines drawn to divide up clusters of objects.
    subplot(2,2,4); imagesc(ObjectOutlinesOnOrigImage);colormap(gray); title([ObjectName, ' Outlines on Input Image']);
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
