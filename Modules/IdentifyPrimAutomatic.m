function handles = IdentifyEasy(handles)

% Help for the Identify Primary Intensity Intensity module:
% Category: Object Identification and Modification
%
% General module for segmenting primary objects. The objects
% must be bright on a dark background, otherwise the image intensity
% should be inverted.
%
% Settings:
%
% Min,Max diameter of objects: 
% The smoothing applied to the image is based on the minimum size of 
% the objects. Objects outside the entered range will be excluded.
% You may exclude objects that are smaller or bigger than
% the size range you specify. A comma should be placed between the
% lower size limit and the upper size limit. The units here are pixels
% so that it is easy to zoom in on found objects and determine the
% size of what you think should be excluded.
%
% Approximate percentage covered by objects:
% An estimate of how much of the image is covered with objects. This
% information is currently only used in the Mixture of Gaussian 
% thresholding, see below.
%
% Select thresholding method:
% The threshold affects the stringency of the lines between
% the objects and the background. You may enter an absolute number
% between 0 and 1 for the threshold (use 'Show pixel data' to see the
% pixel intensities for your images in the appropriate range of 0 to
% 1), or you may have it calculated determined automatically.
% There are advantages either way.  An absolute number
% treats every image identically, but an automatically calculated
% threshold is more realistic/accurate, though occasionally subject to
% artifacts.  The threshold which is used for each image is recorded
% as a measurement in the output file, so if you find unusual
% measurements from one of your images, you might check whether the
% automatically calculated threshold was unusually high or low
% compared to the remaining images. There are two methods for finding
% thresholds automatically, the Otsu method and the Mixture of Gaussian
% method. The user can also choose between global and adaptive thresholding,
% where global means that one threshold is used for the entire image and
% adaptive means that the threshold can vary across the image.
%
% Threshold correction factor:
% When an automatic threshold is
% selected, it may consistently be too stringent or too lenient, so an
% adjustment factor can be entered as well. The number 1 means no
% adjustment, 0 to 1 makes the threshold more lenient and greater than
% 1 (e.g. 1.3) makes the threshold more stringent. This information is
% essentially equivalent to the information given in "Approximate percentage
% covered by objects" above. For example, the Otsu automatic thresholding
% inherently assumes that 50% of the image is covered by objects. If a larger
% percentage of the image is covered, the Otsu method will give a slightly
% biased threshold that may have to be corrected. In a future version,
% the 'Threshold correction factor' may be removed and the "Approximate percentage
% covered by objects" information used instead.
%
% Lower bound on threshold:
% Can be used as a safety precaution when automatic thresholding is utilized. If
% there are no objects in the field of view, the automatic threshold will be
% unreasonably low. In such case the lower bound will override this threshold.
%
% Use intensity or distance transform maxima as sources in Watershed transform:
% This functionality allows neighboring objects to be separated. There are three 
% options: 'Do not use', 'Intensity' and 'Distance'. If the 'Do not use' option is 
% chosen a plain thresholding is applied to the image. If 'Intensity' is chosen,
% local bright maxima will be used as sources in a Watershed transform. If 'Distance'
% is chosen, a distance transform is applied to the binary thresholded image, and
% the maxima within segemented objects are assumed to be centers in individual objects
% and used as sources in the Watershed transform.
%
% Apply Watershed transform to Intensity image or Distance tranformed image:
% This option determines if the Watershed transform will be applied to the original
% 'Intensity' image or to the 'Distance' transformed binary image obtained after the 
% tresholding.
% 
%
% Try to merge too small objects into larger objects:
% This is an experimental functionality that tries to merge objects that fall below
% the Minimum diameter bound with other surrounding objects. For example, it happens
% that the watershed transform divides objects into two halves if two local maxima
% are found within the same object.
%
% Exclude objects touching the border of the image:
% Measurements extracted from objects that are not fully within the field of view
% are uncertain. Such objects should in general be excluded from further processing.
%


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
%choiceVAR03 = 10,30
SizeRange = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu custom

%textVAR04 = Approximate percentage of image covered by objects:
%choiceVAR04 = 10%
%choiceVAR04 = 20%
%choiceVAR04 = 30%
%choiceVAR04 = 40%
%choiceVAR04 = 50%
%choiceVAR04 = 60%
%choiceVAR04 = 70%
%choiceVAR04 = 80%
%choiceVAR04 = 90%
pObject = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%textVAR05 = Select thresholding method or enter a threshold in the range [0,1].
%choiceVAR05 = Otsu Global
%choiceVAR05 = Otsu Adaptive
%choiceVAR05 = MoG Global
%choiceVAR05 = MoG Adaptive
Threshold = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu custom

%textVAR06 = Threshold correction factor
%defaultVAR06 = 1
ThresholdCorrection = str2num(char(handles.Settings.VariableValues{CurrentModuleNum,6}));

%textVAR07 = Lower bound on threshold in the range [0,1].
%defaultVAR07 = 0
MinimumThreshold = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = Use intensity or distance transform maxima as sources in Watershed transform?
%choiceVAR08 = Do not use
%choiceVAR08 = Intensity
%choiceVAR08 = Distance
LocalMaximaType = char(handles.Settings.VariableValues{CurrentModuleNum,8});
%inputtypeVAR08 = popupmenu

%textVAR09 = Apply Watershed transform to Intensity image or Distance tranformed image?
%choiceVAR09 = Do not use
%choiceVAR09 = Intensity
%choiceVAR09 = Distance
WatershedTransformImageType = char(handles.Settings.VariableValues{CurrentModuleNum,9});
%inputtypeVAR09 = popupmenu

%textVAR10 = Try to merge too small objects into larger objects?
%choiceVAR10 = No
%choiceVAR10 = Yes
MergeChoice = char(handles.Settings.VariableValues{CurrentModuleNum,10});
%inputtypeVAR10 = popupmenu

%textVAR11 = Exclude objects touching the border of the image?
%choiceVAR11 = Yes
%choiceVAR11 = No
ExcludeBorderObjects = char(handles.Settings.VariableValues{CurrentModuleNum,11});
%inputtypeVAR11 = popupmenu

%textVAR12 = What do you want to call the image of the outlines of the objects?
%choiceVAR12 = Do not save
SaveOutlined = char(handles.Settings.VariableValues{CurrentModuleNum,12});
%inputtypeVAR12 = popupmenu custom

%textVAR13 =  What do you want to call the labeled matrix image?
%choiceVAR13 = Do not save
SaveColored = char(handles.Settings.VariableValues{CurrentModuleNum,13});
%inputtypeVAR13 = popupmenu custom

%textVAR14 = Do you want to save the labeled matrix image in RGB or grayscale?
%choiceVAR14 = RGB
%choiceVAR14 = Grayscale
SaveMode = char(handles.Settings.VariableValues{CurrentModuleNum,14});
%inputtypeVAR14 = popupmenu



%%%VariableRevisionNumber = 6

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY ERROR CHECKING & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Reads (opens) the image you want to analyze and assigns it to a variable,
%%% "OrigImage".
fieldname = ['',  ImageName];

%%% Checks whether the image exists in the handles structure.
if isfield(handles.Pipeline, fieldname)==0,
    error(['Image processing has been canceled. Prior to running the IdentifyEasy Intensity module, you must have previously run a module to load an image. You specified in the IdentifyEasy module that this image was called ', ImageName, ' which should have produced a field in the handles structure called ', fieldname, '. The IdentifyEasy module cannot find this image.']);
end
OrigImage = handles.Pipeline.(fieldname);

%%% Checks that the original image is two-dimensional (i.e. not a color
%%% image), which would disrupt several of the image functions.
if ndims(OrigImage) ~= 2
    error('Image processing was canceled because the IdentifyEasy module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
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
        error('The Minimum threshold entry in the IdentifyEasy module is invalid.')
    end
end

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%

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
        errordlg('The threshold entered in the IdentifyEasy module is out of range.')
    end
end

%%% Correct the threshold using the correction factor given by the user
%%% and make sure that the threshold is not larger than the minimum threshold
Threshold = ThresholdCorrection*Threshold;
Threshold = max(Threshold,MinimumThreshold);

%%% Smooth images slightly and apply threshold
sigma = MinDiameter/4;                                                % Translate between minimum diamter of objects to sigma. 
FiltLength = min(30,max(1,ceil(3*sigma)));                            % Determine filter size, min 3 pixels, max 61 
[x,y] = meshgrid(-FiltLength:FiltLength,-FiltLength:FiltLength);      % Filter kernel grid
f = exp(-(x.^2+y.^2)/(2*sigma^2));f = f/sum(f(:));                    % Gaussian filter kernel
BlurredImage = conv2(OrigImage,f,'same');                             % Blur original image
Objects = BlurredImage > Threshold;                                   % Threshold image
Threshold = mean(Threshold(:));                                       % Use average threshold downstreams
Objects = imfill(double(Objects),'holes');                            % Fill holes




%%% STEP 2. If user wants, extract local maxima and apply watershed transform
%%% to separate neighboring objects.
if ~strcmp(LocalMaximaType,'Do not use') & ~strcmp(WatershedTransformImageType,'Do not use')
    
    %%% NOTE: If the image is big and the objects are big, the maxima suppression
    %%% takes forever! The image should be resized with imresize() and the maxima
    %%% suppression should be made in the lower resolution image!
    %%% Below the max MaximaMask is set to max 50, but it will still take time.
    MaximaMask = getnhood(strel('disk', min(50,max(1,floor(MinDiameter/1.5)))));     % Local maxima defined in this neighborhood

    %%% Get local maxima
    if strcmp(LocalMaximaType,'Intensity')
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

    %%% Overlay the maxima on either the original image or a distance transformed image
    if strcmp(WatershedTransformImageType,'Intensity')
        %%% Overlays the nuclear markers (maxima) on the inverted original image so
        %%% there are black dots on top of each dark nucleus on a white background.
        Overlaid = imimposemin(1 - OrigImage,MaximaImage);
    
    elseif strcmp(WatershedTransformImageType,'Distance')
        %%% Overlays the nuclear markers (maxima) on the inverted DistanceTransformedImage so
        %%% there are black dots on top of each dark nucleus on a white background.
        %%% We may have to calculated the distance transform:
        if ~exist('DistanceTransformedImage','var')
            DistanceTransformedImage = bwdist(~Objects);                               
        end
        figure,imagesc(DistanceTransformedImage)
        Overlaid = imimposemin(-DistanceTransformedImage,MaximaImage);
        figure,imagesc(Overlaid)
    end
    
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

%%% Will be stored to the handles structure
PrelimLabelMatrixImage2 = Objects;

%%% Remove objects along the border of the image (depends on user input)
tmp = Objects;
if strcmp(ExcludeBorderObjects,'Yes')
    Objects = imclearborder(Objects);
end
BorderObjects = tmp - Objects;

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
    cmap = jet(max(Objects(:)));
    im = label2rgb(Objects, cmap, 'k', 'shuffle');
    ImageHandle = image(im);
    set(ImageHandle,'ButtonDownFcn','ImageTool(gco)','Tag',sprintf('Segmented %s',ObjectName))
    title(sprintf('Segmented %s',ObjectName),'fontsize',8);
    axis image,set(gca,'fontsize',8)

    % Indicate objects in original image and color excluded objects in red
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
    hy = subplot(2,2,3);
    OutlinedObjects = cat(3,OutlinedObjectsR,OutlinedObjectsG,OutlinedObjectsB);
    ImageHandle = image(OutlinedObjects);
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
        'BackgroundColor',bgcolor,'HorizontalAlignment','Left','String',sprintf('90%% of objects within diameter range [%0.1f, %0.1f] pixels',...
        Lower90Limit,Upper90Limit),'FontSize',handles.Current.FontSize);
    ObjectCoverage = 100*sum(sum(Objects > 0))/prod(size(Objects));
    uicontrol(ThisModuleFigureNumber,'Style','Text','Units','Normalized','Position',[posx(1)-0.05 posy(2)+posy(4)-0.20 posx(3)+0.1 0.04],...
        'BackgroundColor',bgcolor,'HorizontalAlignment','Left','String',sprintf('%0.1f%% of image consists of objects',ObjectCoverage),'FontSize',handles.Current.FontSize);
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

%%% Store outlines of objects in the handles structure
handles.PipelineObjectOutlines = PerimObjects;
    

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
try
    if ~strcmp(SaveColored,'Do not save')
        if strcmp(SaveMode,'RGB')
            if sum(sum(FinalLabelMatrixImage)) >= 1
                ColoredLabelMatrixImage = label2rgb(FinalLabelMatrixImage, 'jet', 'k', 'shuffle');
            else
                ColoredLabelMatrixImage = FinalLabelMatrixImage;
            end
            handles.Pipeline.(SaveColored) = ColoredLabelMatrixImage;
        else
            handles.Pipeline.(SaveColored) = FinalLabelMatrixImage;
        end
    end
    if ~strcmp(SaveOutlined,'Do not save')
        handles.Pipeline.(SaveOutlined) = OutlinedObjects;
    end
catch
    errordlg('The object outlines or colored objects were not calculated by an identify module (possibly because the window is closed) so these images were not saved to the handles structure. The Save Images module will therefore not function on these images. This is just for your information - image processing is still in progress, but the Save Images module will fail if you attempted to save these images.')
end


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
ClassStd(1:3) = 0.1;
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
        ClassStd(k)  = sqrt(sum(pPixelClass(:,k).*(Intensities - ClassMean(k)).^2)/(length(Intensities)*pClass(k)));
    end

    %%% Calculate change
    delta = sum(abs(ClassMean - oldClassMean));
    [ClassMean' ClassStd'];
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



