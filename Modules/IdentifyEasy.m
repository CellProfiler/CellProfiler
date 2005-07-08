function handles = IdentifyEasy(handles)
% Category: Object Identification

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
%choiceVAR03 = 1,Inf
SizeRange = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu custom

%textVAR04 = Threshold in the range [0,1].
%choiceVAR04 = Automatic
Threshold = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu custom

%textVAR05 = Minimum eccentricity of objects in the range [0,1]
%defaultVAR05 = 0.8
MinEccentricity = str2num(char(handles.Settings.VariableValues{CurrentModuleNum,5}));


%%%VariableRevisionNumber = 2

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

%%% Checks that the Diameter has a valid value
if isnan(MinEccentricity) | isempty(MinEccentricity) | MinEccentricity < 0 | MinEccentricity > 1
    error('The MinEccentricity parameter in the IdentifyEasy module is invalid.')
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
if ~strcmp(Threshold,'Automatic')
    Threshold = str2double(Threshold);
    if isnan(Threshold) | Threshold > 1 | Threshold < 0
        error('The threshold entered in the IdentifyEasy module is out of range.')
    end
end

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%

%%% Blurs the image using a separable Gaussian filtering.
sigma = (Diameter/4)/2.35;                                                 % Convert from FWHM to sigma
FiltLength = max(1,ceil(3*sigma));                                         % Determine filter length
[x,y] = meshgrid(-FiltLength:FiltLength,-FiltLength:FiltLength);
f = exp(-(x.^2+y.^2)/(2*sigma^2));f = f/sum(f(:));
fx = x.*f;
fy = y.*f;
BlurredImage = conv2(OrigImage,f,'same');
EdgeImage = abs(conv2(OrigImage,fx)) + abs(conv2(OrigImage,fy));

%%% Extract object markers by finding local maxima in the blurred image
MaximaImage = BlurredImage;
MaximaMask = getnhood(strel('disk', max(1,floor(Diameter/8))));
MaximaImage(BlurredImage < ordfilt2(BlurredImage,sum(MaximaMask(:)),MaximaMask)) = 0;

%%% Thresholds the image to eliminate dim maxima.
if strcmp(Threshold,'Automatic'),
    Threshold = CPgraythresh(OrigImage,handles,ImageName);
end
MaximaImage = MaximaImage > Threshold;


%%% Overlays the nuclear markers (maxima) on the inverted original image so
%%% there are black dots on top of each dark nucleus on a white background.
Overlaid = imimposemin(1 - BlurredImage,MaximaImage);
WatershedBoundaries = watershed(Overlaid) > 0;
Objects = OrigImage > Threshold;                              % Threshold image
Objects = imfill(Objects,'holes');                            % Fill holes
Objects = Objects.*WatershedBoundaries;                       % Cut objects along the watershed lines
Objects = bwlabel(Objects);                                   % Label the objects

% Try to merge objects
%MergeObjects(Objects,OrigImage,[MinDiameter MaxDiameter],MinEccentrity)


%%% Locate objects with diameter outside the specified range
MedianDiameter = median(Diameters);
Diameters = [0;cat(1,tmp.EquivDiameter)];
DiameterMap = Diameters(Objects+1);                                   % Create image with object intensity equal to the diameter
tmp = Objects;
Objects(DiameterMap > MaxDiameter) = 0;                               % Remove objects that are too big
Objects(DiameterMap < MinDiameter) = 0;                               % Remove objects that are too small
DiameterExcludedObjects = tmp - Objects ;                             % Store objects that fall outside diameter range for display
NumOfDiameterObjects = length(unique(DiameterExcludedObjects(:)))-1;  % Count the objects

%%% Remove objects along the border of the image
tmp = Objects;
Objects = imclearborder(Objects);
BorderObjects = tmp - Objects;
NumOfBorderObjects = length(unique(BorderObjects(:)))-1;

%%% Remove objects with no marker in them
tmp = regionprops(Objects,'PixelIdxList');                              % This is a very fast way to get pixel indexes for the objects
for k = 1:length(tmp)
    if sum(MaximaImage(tmp(k).PixelIdxList)) == 0                       % If there is no maxima in these pixels, exclude object
        Objects(index) = 0;
    end
end

%%% Relabel the objects
[Objects,NumOfObjects] = bwlabel(Objects > 0);


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
    OutlinedObjectsR(PerimDiameter)   = 1; OutlinedObjectsG(PerimDiameter)   = 0; OutlinedObjectsB(PerimDiameter)   = 0;
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
    uicontrol(ThisModuleFigureNumber,'Style','Text','Units','Normalized','Position',[posx(1)-0.05 posy(2)+posy(4)-0.12 posx(3)+0.1 0.04],...
        'BackgroundColor',bgcolor,'HorizontalAlignment','Left','String',['Median diameter (pixels): ' num2str(MedianDiameter)],'FontSize',handles.Current.FontSize);

    uicontrol(ThisModuleFigureNumber,'Style','Text','Units','Normalized','Position',[posx(1)-0.05 posy(2)+posy(4)-0.20 posx(3)+0.1 0.04],...
        'BackgroundColor',bgcolor,'HorizontalAlignment','Left','String',sprintf('Number of border objects: %d',NumOfBorderObjects),'FontSize',handles.Current.FontSize);
    uicontrol(ThisModuleFigureNumber,'Style','Text','Units','Normalized','Position',[posx(1)-0.05 posy(2)+posy(4)-0.24 posx(3)+0.1 0.04],...
        'BackgroundColor',bgcolor,'HorizontalAlignment','Left','String',sprintf('Number of objects outside diameter range: %d',NumOfDiameterObjects),'FontSize',handles.Current.FontSize);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Saves the final segmented label matrix image to the handles structure.
fieldname = ['Segmented',ObjectName];
handles.Pipeline.(fieldname) = Objects;

%%% Saves the Threshold value to the handles structure.
fieldname = ['ImageThreshold', ObjectName];
handles.Measurements.(fieldname)(handles.Current.SetBeingAnalyzed) = {Threshold};










function Objects = MergeObjects(Objects,OrigImage,Diameters,MinEccentricity)

%%% Find the object that we should try to merge with other objects. The object
%%% numbers of these objects are stored in the variable 'MergeIndex'.
props = regionprops(Objects,'EquivDiameter','PixelIdxList','Eccentricity');   % Get diameters of the objects
EquivDiameters = cat(1,props.EquivDiameter);
Eccentricities = cat(1,props.Eccentricity);
IndexDiameter = find(EquivDiameters < MinDiameter);
IndexEccentrity = find(Eccentricities < MinEccentricity)
MergeIndex = unique([IndexDiameter;IndexEccentricity]);

% Repeat until there are no more objects left to merge
[sr,sc] = size(OrigImage);
while ~isempty(MergeIndex)

    % Get next object to merge
    ObjectNbr = MergeIndex(1);

    %%% Identify neighbors
    %%% Cut a patch so we don't have to work with the entire image
    [r,c] = ind2sub([sr sc],props(ObjectNbr).PixelIdxList);
    rmax = min(sr,max(r) + 3);
    rmin = max(1,min(r) - 3);
    cmax = min(sc,max(c) + 3);
    cmin = max(1,min(c) - 3);
    ObjectsPatch = Objects(rmin:rmax,cmin:cmax);
    BinaryPatch = double(ObjectsPatch == ObjectNbr);
    GrownBinaryPatch = conv2(BinaryPatch,double(getnhood(strel('disk',2))),'same') > 0;
    Neighbors = ObjectsPatch .*GrownBinaryPatch;
    NeighborsNbr = setdiff(unique(Neighbors(:)),[0 ObjectNbr]);


    %%% For each neighbor, calculate a set of criteria based on which we decide if to merge
    LikelihoodRatio    = zeros(length(NeighborsNbr),1);
    MergedEccentricity = zeros(length(NeighborsNbr),1);
    for j = 1:length(NeighborsNbr)

        %%% Get Neigbor number
        NeighborNbr = NeighborsNbr(j);

        %%% Cut patch which contains both original object and the current neighbor
        [r,c] = ind2sub([sr sc],[props(ObjectNbr).PixelIdxList;props(NeighborNbr).PixelIdxList]);
        rmax = min(sr,max(r) + 3);
        rmin = max(1,min(r) - 3);
        cmax = min(sc,max(c) + 3);
        cmin = max(1,min(c) - 3);
        ObjectsPatch = Objects(rmin:rmax,cmin:cmax);
        OrigImagePatch = OrigImage(rmin:rmax,cmin:cmax);

        %%% Identify object interiors, background and interface voxels
        BinaryNeighborPatch      = double(ObjectsPatch == NeighborNbr);
        BinaryObjectPatch        = double(ObjectsPatch == ObjectNbr);
        GrownBinaryNeighborPatch = conv2(BinaryNeighborPatch,ones(3),'same') > 0;
        GrownBinaryObjectPatch   = conv2(BinaryObjectPatch,ones(3),'same') > 0;
        Interface                = GrownObject.*GrownNeighbor;
        Background               = ((GrownObject + GrownNeighbor) > 0) - BinaryNeighborPatch - BinaryObjectPatch - Interface;
        WithinObjectIndex        = find(BinaryNeighborPatch + BinaryObjectPatch);
        InterfaceIndex           = find(Interface);
        BackgroundIndex          = find(Background);

        %%% Calculate likelihood of the interface belonging to the background or to an object.
        WithinObjectClassMean   = mean(OrigImagePatch(WithinObjectIndex));
        WithinObjectClassStd    = std(OrigImagePatch(WithinObjectIndex));
        BackgroundClassMean     = mean(OrigImagePatch(BackgroundIndex));
        BackgroundClassStd      = std(OrigImagePatch(BackgroundIndex));
        InterfaceMean           = mean(OrigImagePatch(InterfaceIndex));
        LogLikelihoodObject     = -log(WithinbjectClassStd^2) - (InterfaceMean - WithinObjectClassMean)^2/(2*WithinObjectClassStd^2);
        LogLikelihoodBackground = -log(BackgroundClassStd^2) - (InterfaceMean - BackgroundClassMean)^2/(2*BackgroundClassStd^2);
        LikelihoodRatio(j)      =  LogLikelihoodObject - LogLikelihoodBackground;

        %%% Calculate the eccentrity of the object obtained if we merge the current object
        %%% with the current neighbor.
        MergedObject =  (BinaryNeighborPatch + BinaryObjectPatch + Interface) > 0;
        tmp = regionprops(MergedObject,'Eccentricity');
        MergedEccentricity(j) = tmp(1).Eccentricity;
    
        %%% Get indexes for the interface pixels in original image.
        %%% These indexes are required if we need to merge the object with
        %%% the current neighbor.
        tmp = zeros(size(OrigImage));
        tmp(rmin:rmax,cmin:cmax) = Interface;
        tmp = regionprops(tmp,'PixelIdxList');
        OrigInterfaceIndex{j} = cat(1,tmp.PixelIdxList);
    end

    %%% Let each feature rank which neighbor to merge with. Then calculate
    %%% a score for each neighbor. If the neighbors is ranked 1st, it will get
    %%% 1 point; 2nd, it will get 2 points; and so on. The lower score the better.
    [ignore,LikelihoodRank]   = sort(LikelihoodRatio,'descend');                  % The higher the LikelihoodRatio the better
    [ignore,EccentricityRank] = sort(MergedEccentricity,'ascend');                % The lower the eccentricity the better
    for j = 1:length(NeighborsNbr)
        NeighborScore(j) = find(LikelihoodRank == j) +  find(EccentricityRank == j);
    end

    %%% Go through the neighbors, starting with the highest ranked, and merge
    %%% with the first neighbor for which certain basic criteria are fulfilled.
    %%% If no neighbor fulfil the basic criteria, there will be no merge.
    [ignore,TotalRank] = sort(NeigborScore);
    for j = 1:length(NeighborsNbr)
        NeighborNbr = NeighborsNbr(TotalRank(j));

        %%% To merge, the interface between objects must be more likely to belong to the object class
        %%% than the background class. The eccentricity of the merged object must also be lower than
        %%% for the original object.
        if LikelihoodRatio(TotalRank(j)) > 0 && MergedEccentricity(TotalRank(j)) < Eccentricities(ObjectNbr)
            
            %%% OK, let's merge!
            %%% Assign the neighbor number to the current object
            Objects(props(ObjectNbr).PixelIdxList) = NeighborNbr;
            
            %%% Assign the neighbor number to the interface pixels between the current object and the neigbor
            Objects(OrigInterfaceIndex{TotalRank(j)}) = NeighborNbr;
            
            %%% Add the pixel indexes to the neigbor index list
            props(NeighborNbr).PixelIdxList = cat(1,...
                props(Neighbor).PixelIdxList,...
                props(ObjectNbr).PixelIdxList,...
                OrigInterfaceIndex{TotalRank(j)});
            
            %%% Remove the neighbor from the list of objects to be merged (if it's there).
            MergeIndex = setdiff(MergeIndex,MergeNbr);
        end
    end
    
    %%% OK, we are done with the current object, let's go to the next
    MergeIndex = MergeIndex(2:end-1);
end

%%% Finally, relabel the objects
Objects = bwlabel(Objects > 0);



