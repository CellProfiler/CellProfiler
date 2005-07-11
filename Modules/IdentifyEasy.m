function handles = IdentifyEasy(handles)

% Help for the Identify Primary Intensity Intensity module:
% Category: Object Identification and Modification
%
% This image analysis module works best for objects that are brighter
% towards the interior; the objects can be any shape, so they need not
% be round and uniform in size as would be required for a
% distance-based module.  The dividing lines between clumped objects
% should be dim. The module is more successful when then objects have
% a smooth texture, although increasing the blur radius can improve
% the outcome on lumpy-textured objects.
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
%defaultVAR02 = Nuclei
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Min,Max diameter of objects (pixels):
%choiceVAR03 = 15,35
SizeRange = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu custom

%textVAR04 = Approximately how much of the image is covered by objects:
%choiceVAR04 = 20%
%choiceVAR04 = 50%
%choiceVAR04 = 80%
SizeRange = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu


%textVAR05 = Select thresholing method or enter a threshold in the range [0,1].
%choiceVAR05 = Global Otsu
%choiceVAR05 = Global MoG
%choiceVAR05 = Adaptive Otsu
%choiceVAR05 = Adaptive MoG
Threshold = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu custom

%textVAR06 = Threshold correction factor
%defaultVAR06 = 1.2
ThresholdCorrection = str2num(char(handles.Settings.VariableValues{CurrentModuleNum,6}));

%textVAR07 = Use maxima in intensity or distance transform as centers in Watershed transform?
%choiceVAR07 = Intensity
%choiceVAR07 = Distance
%choiceVAR07 = Do not apply
LocalMaximaType = char(handles.Settings.VariableValues{CurrentModuleNum,7});
%inputtypeVAR07 = popupmenu

%textVAR08 = Try to merge too small objects into larger objects?
%choiceVAR08 = Yes
%choiceVAR08 = No
MergeChoice = char(handles.Settings.VariableValues{CurrentModuleNum,8});
%inputtypeVAR08 = popupmenu

%%%VariableRevisionNumber = 4

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY ERROR CHECKING & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

%%% Checks that the Min and Max diameter parameters have valid values
index = strfind(SizeRange,',');
if isempty(index),error('The Min and Max size entry in the IdentifyEasy module is invalid.'),end
MinDiameter = SizeRange(1:index-1);
MaxDiameter = SizeRange(index+1:end);

MinDiameter = str2double(MinDiameter);
if isnan(MinDiameter) | MinDiameter < 0
    error('The Min dimater entry in the IdentifyEasy module is invalid.')
end

if strcmp(MaxDiameter,'Inf') ,MaxDiameter = Inf;
else
    MaxDiameter = str2double(MaxDiameter);
    if isnan(MaxDiameter) | MaxDiameter < 0
        error('The Max Diameter entry in the IdentifyEasy module is invalid.')
    end
end
if MinDiameter > MaxDiameter, error('Min Diameter larger the Max Diameter in the IdentifyEasy module.'),end
Diameter = min((MinDiameter + MaxDiameter)/2,50);

%%% Checks that the Threshold parameter has a valid value
%if ~strcmp(Threshold,'Automatic')
%    Threshold = str2double(Threshold);
%    if isnan(Threshold) | Threshold > 1 | Threshold < 0
%        error('The threshold entered in the IdentifyEasy module is out of range.')
%    end
%end


%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%

%%% STEP 1. Find threshold and apply to image
if strfind(Threshold,'Global')
    if strfind(Threshold,'Otsu')
        Threshold = CPgraythresh(OrigImage,handles,ImageName);
    elseif strfind(Threshold,'MoG')
        Threshold = MixtureOfGaussian(OrigImage,pObject)
    end
elseif strfind(Threshold,'Adaptive')

    %%% Choose the block size that best covers the original image in the sense
    %%% that the number of extra rows and columns is minimal.
    %%% Get size of image
    [m,n] = size(OrigImage);

    %%% Calculates the MinimumThreshold automatically as 0.7 times the
    %%% global threshold calculated by Otsu's method
    MinimumThreshold = 0.7*graythresh(OrigImage);

    %%% Calculates a range of acceptable block sizes as plus minus 10% of the suggested block size.
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

    if strfind(Threshold,'Otsu')
        %%% Calculates the threshold for each block in the image.
        Threshold = blkproc(PaddedImage,[BestBlockSize BestBlockSize],'graythresh(x)');


    elseif strfind(Threshold,'MoG')
    end
    %%% Resizes the block-produced image to be the size of the padded image.
    %%% Bilinear prevents dipping below zero. The crop the image
    %%% get rid of the padding, to make the result the same size as the original image.
    ThresholdImage = imresize(ThresholdImage, size(PaddedImage), 'bilinear');
    ThresholdImage = ThresholdImage(RowsToAddPre+1:end-RowsToAddPost,ColumnsToAddPre+1:end-ColumnsToAddPost);

    %%% For any of the threshold values that is lower than the user-specified
    %%% minimum threshold, set to equal the minimum threshold.  Thus, if there
    %%% are no objects within a block (e.g. if cells are very sparse), an
    %%% unreasonable threshold will be overridden by the minimum threshold.
    ThresholdImage(ThresholdImage <= MinimumThreshold) = MinimumThreshold;

end

%%% Correct the threshold using the correction factor given by the user
Threshold = ThresholdCorrection*Threshold;


%%% Apply threshold
Objects = OrigImage > Threshold;                              % Threshold image
Objects = imfill(Objects,'holes');                            % Fill holes

%%% STEP 2. Extract local maxima and apply watershed transform
if ~strcmp(LocalMaximaType,'Do not apply')
    MaximaMask = getnhood(strel('disk', max(1,floor(MinDiameter/1.5))));

    if strcmp(LocalMaximaType,'Intensity')
        %%% Find maxima in a blurred version of the original image
        sigma = (MinDiameter/4)/2.35;                                              % Translate between minimum diamter of objects to sigma
        FiltLength = max(1,ceil(3*sigma));                                         % Determine filter size
        [x,y] = meshgrid(-FiltLength:FiltLength,-FiltLength:FiltLength);           % Filter kernel grid
        f = exp(-(x.^2+y.^2)/(2*sigma^2));f = f/sum(f(:));                         % Gaussian filter kernel
        BlurredImage = conv2(OrigImage,f,'same');                                  % Blur original image
        MaximaImage = BlurredImage;                                                % Initialize MaximaImage
        MaximaImage(BlurredImage < ...                                             % Save only local maxima
            ordfilt2(BlurredImage,sum(MaximaMask(:)),MaximaMask)) = 0;
        MaximaImage = MaximaImage > Threshold;                                     % Remove dim maxima

    elseif strcmp(LocalMaximaType,'Distance')
        DistanceTransformedImage = bwdist(~Objects);                               % Calculate distance transform
        DistanceTransformedImage = DistanceTransformedImage + ...                  % Add some noise to get distinct maxima
            0.001*rand(size(DistanceTransformedImage));
        MaximaImage = ones(size(OrigImage));                                       % Initialize MaximaImage
        MaximaImage(DistanceTransformedImage < ...                                 % Set all pixels that are not local maxima to zero
            ordfilt2(DistanceTransformedImage,sum(MaximaMask(:)),MaximaMask)) = 0;
        MaximaImage(~Objects) = 0;                                                 % We are only interested in maxima within thresholded objects
    end

    %%% Overlays the nuclear markers (maxima) on the inverted original image so
    %%% there are black dots on top of each dark nucleus on a white background.
    Overlaid = imimposemin(1 - OrigImage,MaximaImage);

    %%% Calculate the watershed transform and cut objects along the boundaries
    WatershedBoundaries = watershed(Overlaid) > 0;
    Objects = Objects.*WatershedBoundaries;

    %%% Label the objects
    Objects = bwlabel(Objects);

    %%% Remove objects with no marker in them (this happens occasionally)
    tmp = regionprops(Objects,'PixelIdxList');                              % This is a very fast way to get pixel indexes for the objects
    for k = 1:length(tmp)
        if sum(MaximaImage(tmp(k).PixelIdxList)) == 0                       % If there is no maxima in these pixels, exclude object
            Objects(index) = 0;
        end
    end
end

%%% Label the objects
Objects = bwlabel(Objects);

%%% Merge small objects
if strcmp(MergeChoice,'Yes')
    Objects = MergeObjects(Objects,OrigImage,[MinDiameter MaxDiameter]);
end

%%% Will be stored to the handles structure
PrelimLabelMatrixImage1 = Objects;

%%% Get diameters of objects and calculate the interval
%%% that contains 90% of the objects
tmp = regionprops(Objects,'EquivDiameter');
Diameters = [0;cat(1,tmp.EquivDiameter)];
SortedDiameters = sort(Diameters);
NbrInTails = max(round(0.05*length(Diameters)),1);
Lower90Limit = SortedDiameters(NbrInTails);
Upper90Limit = SortedDiameters(end-NbrInTails+1);

%%% Locate objects with diameter outside the specified range
DiameterMap = Diameters(Objects+1);                                   % Create image with object intensity equal to the diameter
tmp = Objects;
Objects(DiameterMap > MaxDiameter) = 0;                               % Remove objects that are too big
Objects(DiameterMap < MinDiameter) = 0;                               % Remove objects that are too small
DiameterExcludedObjects = tmp - Objects ;                             % Store objects that fall outside diameter range for display
NumOfDiameterObjects = length(unique(DiameterExcludedObjects(:)))-1;  % Count the objects

%%% Will be stored to the handles structure
PrelimLabelMatrixImage2 = Objects;

%%% Remove objects along the border of the image
tmp = Objects;
Objects = imclearborder(Objects);
BorderObjects = tmp - Objects;
NumOfBorderObjects = length(unique(BorderObjects(:)))-1;

%%% Relabel the objects
[Objects,NumOfObjects] = bwlabel(Objects > 0);
FinalLabelMatrixImage = Objects;


%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
if any(findobj == ThisModuleFigureNumber)

    drawnow
    CPfigure(handles,ThisModuleFigureNumber);

    subplot(2,2,1)
    ImageHandle = imagesc(OrigImage);colormap(gray)
    set(ImageHandle,'ButtonDownFcn','ImageTool(gco)','Tag',['Input Image, Image Set # ',num2str(handles.Current.SetBeingAnalyzed)])
    axis image
    title(['Input Image, Image Set # ',num2str(handles.Current.SetBeingAnalyzed)],'fontsize',8);
    set(gca,'fontsize',8)

    hx = subplot(2,2,2);
    ImageHandle = image(label2rgb(Objects, 'jet', 'k', 'shuffle'));
    set(ImageHandle,'ButtonDownFcn','ImageTool(gco)','Tag',sprintf('Segmented %s',ObjectName))
    title(sprintf('Segmented %s',ObjectName),'fontsize',8);
    axis image,set(gca,'fontsize',8)

    % Indicate objects in original image and color excluded objects in red
    tmp = OrigImage/max(OrigImage(:));
    OutlinedObjectsR = tmp;
    OutlinedObjectsG = tmp;
    OutlinedObjectsB = tmp;
    PerimObjects = bwperim(Objects > 0);
    PerimDiameter   = bwperim(DiameterExcludedObjects > 0);
    PerimBorder = bwperim(BorderObjects > 0);
    OutlinedObjectsR(PerimObjects) = 0; OutlinedObjectsG(PerimObjects) = 1; OutlinedObjectsB(PerimObjects) = 0;
    OutlinedObjectsR(PerimDiameter) = 1; OutlinedObjectsG(PerimDiameter)   = 0; OutlinedObjectsB(PerimDiameter)   = 0;
    OutlinedObjectsR(PerimBorder) = 1; OutlinedObjectsG(PerimBorder) = 1; OutlinedObjectsB(PerimBorder) = 0;
    hy = subplot(2,2,3);
    ImageHandle = image(cat(3,OutlinedObjectsR,OutlinedObjectsG,OutlinedObjectsB));
    set(ImageHandle,'ButtonDownFcn','ImageTool(gco)','Tag','Outlined objects')
    title('Outlined objects','fontsize',8);
    axis image,set(gca,'fontsize',8)

    CPFixAspectRatio(OrigImage);

    % Report numbers
    posx = get(hx,'Position');
    posy = get(hy,'Position');
    bgcolor = get(ThisModuleFigureNumber,'Color');
    uicontrol(ThisModuleFigureNumber,'Style','Text','Units','Normalized','Position',[posx(1)-0.05 posy(2)+posy(4)-0.04 posx(3)+0.1 0.04],...
        'BackgroundColor',bgcolor,'HorizontalAlignment','Left','String',sprintf('Threshold:  %0.3f',Threshold),'FontSize',handles.Current.FontSize);
    uicontrol(ThisModuleFigureNumber,'Style','Text','Units','Normalized','Position',[posx(1)-0.05 posy(2)+posy(4)-0.08 posx(3)+0.1 0.04],...
        'BackgroundColor',bgcolor,'HorizontalAlignment','Left','String',sprintf('Number of segmented objects: %d',NumOfObjects),'FontSize',handles.Current.FontSize);
    uicontrol(ThisModuleFigureNumber,'Style','Text','Units','Normalized','Position',[posx(1)-0.05 posy(2)+posy(4)-0.16 posx(3)+0.1 0.08],...
        'BackgroundColor',bgcolor,'HorizontalAlignment','Left','String',sprintf('90%% of objects within diameter range[%0.1f, %0.1f] pixels',Lower90Limit,Upper90Limit),'FontSize',handles.Current.FontSize);

    uicontrol(ThisModuleFigureNumber,'Style','Text','Units','Normalized','Position',[posx(1)-0.05 posy(2)+posy(4)-0.20 posx(3)+0.1 0.04],...
        'BackgroundColor',bgcolor,'HorizontalAlignment','Left','String',sprintf('Number of border objects: %d',NumOfBorderObjects),'FontSize',handles.Current.FontSize);
    uicontrol(ThisModuleFigureNumber,'Style','Text','Units','Normalized','Position',[posx(1)-0.05 posy(2)+posy(4)-0.24 posx(3)+0.1 0.04],...
        'BackgroundColor',bgcolor,'HorizontalAlignment','Left','String',sprintf('Number of small objects: %d',NumOfDiameterObjects),'FontSize',handles.Current.FontSize);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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


%%% Saves images to the handles structure so they can be saved to the hard
%%% drive, if the user requested.
% try
%     if ~strcmp(SaveColored,'Do not save')
%         if strcmp(SaveMode,'RGB')
%             handles.Pipeline.(SaveColored) = ColoredLabelMatrixImage;
%         else
%             handles.Pipeline.(SaveColored) = FinalLabelMatrixImage;
%         end
%     end
%     if ~strcmp(SaveOutlined,'Do not save')
%         handles.Pipeline.(SaveOutlined) = ObjectOutlinesOnOrigImage;
%     end
% catch errordlg('The object outlines or colored objects were not calculated by an identify module (possibly because the window is closed) so these images were not saved to the handles structure. The Save Images module will therefore not function on these images. This is just for your information - image processing is still in progress, but the Save Images module will fail if you attempted to save these images.')
% end




function Threshold = MixtureOfGaussians(OrigImage,pObject)

%%% Transform the image into a vector
OrigImage = OrigImage(:);

%%% Get the probability for a background pixel
pBackground = 1 - pObject;

%%% Initialize mean and standard deviations of the two Gaussian distributions
%%% by looking at the pixel intensities in the original image and by considering
%%% the percentage of the image that is covered by object pixels. The means of
%%% the Object class and Background class are calculated as the (1-pObject/2) and 
%%% pBackground/2 percentiles of the original pixel intensities respectively. The
%%% initial standard deviations are then initialized so that the distributions
%%% don't overlap too much
SortedIntensities = sort(OrigImage);
MeanObject = SortedIntensities(length(OrigImage)*(1 - pObject/2));
MeanBackground = SortedIntensities(length(OrigImage)*pBackground/2);
StdObject = (MeanObject - MeanBackground)/4;
StdBackground = (MeanObject - MeanBackground)/4;

%%% Expectation-Maximization algorithm for fitting the two Gaussian distributions
%%% to the data. Iterate until parameters don't change anymore.
delta = 1;
while delta > 0.001
    %%% Store old parameter value to monitor change
    oldMeanObject = MeanObject;
    
    %%% Update probabilities of a pixel belonging to the background or object
    pPixelBackground = pBackground * 1/sqrt(2*pi*StdBackground^2) * exp(-(OrigImage - MeanBackground).^2/(2*StdBackground^2));
    pPixelObject     = pObject * 1/sqrt(2*pi*StdObject^2) * exp(-(OrigImage - MeanObject).^2/(2*StdObject^2));
    pPixelBackground = pPixelBackground./(pPixelBackground+pPixelObject);
    pPixelObject     = pPixelObject./(pPixelBackground+pPixelObject);

    %%% Update parameters in Gaussian distributions
    MeanBackground = sum(pPixelBackground.*OrigImage)/sum(pPixelBackground);
    MeanObject     = sum(pPixelObject.*OrigImage)/sum(pPixelObject);
    StdBackground  = sqrt(sum(pPixelBackground.*(OrigImage - MeanBackground).^2)/sum(pPixelBackground));
    StdObject      = sqrt(sum(pPixelObject.*(OrigImage - MeanObject).^2)/sum(pPixelObject));
    pBackground = mean(pPixelBackground);
    pObject     = mean(pPixelObject);

    %%% Calculate change
    delta = abs(MeanBackground - oldMeanBackground);
end

Threshold = 









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
        WithinObjectClassStd    = std(OrigImagePatch(WithinObjectIndex));
        BackgroundClassMean     = mean(OrigImagePatch(BackgroundIndex));
        BackgroundClassStd      = std(OrigImagePatch(BackgroundIndex));
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



