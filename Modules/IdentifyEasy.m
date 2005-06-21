function handles = IdentifyEasy(handles)
% Sorry, no help yet

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%

%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

%infotypeVAR01 = imagegroup
%textVAR01 = What did you call the images you want to process?
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%infotypeVAR02 = objectgroup indep
%textVAR02 = What do you want to call the objects identified by this module?
%defaultVAR02 = Nuclei
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Approximate diameter of objects (pixels)
%defaultVAR03 = 10
Diameter = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = Min,Max area of objects (pixels):
%choiceVAR04 = 1,Inf
SizeRange = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu custom

%textVAR05 = Threshold in the range [0,1].
%choiceVAR05 = Automatic
Threshold = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu custom

%%%VariableRevisionNumber = 1

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
Diameter = str2double(Diameter);
if isnan(Diameter) | Diameter <= 0
    error('The Diameter parameter in the Segmentation module is invalid.')
end

%%% Checks that the Min and Max area parameters have valid values
index = strfind(SizeRange,',');
if isempty(index),error('The Min and Max size entry in the Segmentation module is invalid.'),end
MinArea = SizeRange(1:index-1);
MaxArea = SizeRange(index+1:end);
  
MinArea = str2double(MinArea);
if isnan(MinArea) | MinArea < 0
    error('The Min area entry in the Segmentation module is invalid.')
end

if strcmp(MaxArea,'Inf') ,MaxArea = Inf;
else    
    MaxArea = str2double(MaxArea);
    if isnan(MaxArea) | MaxArea < 0
        error('The Max area entry in the Segmentation module is invalid.')
    end
end
if MinArea > MaxArea, error('Min area larger the Max area in the Segmentation module.'),end

%%% Checks that the Threshold parameter has a valid value
if ~strcmp(Threshold,'Automatic')
        Threshold = str2double(Threshold);
        if isnan(Threshold) | Threshold > 1 | Threshold < 0
            error('The threshold entered in the Segmentation module is out of range.')
        end
end

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%

%%% Blurs the image using a separable Gaussian filtering.
sigma = (Diameter/8)/2.35;                                                 % Convert from FWHM to sigma
FiltLength = max(1,ceil(3*sigma));                                         % Determine filter length
f = exp(-linspace(-FiltLength,FiltLength,2*FiltLength+1).^2/(2*sigma^2));  % 1D Gaussian filter kernel
f = f/sum(f(:));                                                           % Normalize
BlurredImage = conv2(f,f,OrigImage,'same');                            % Separable filtering

%%% Extract object markers by finding local maxima in the blurred image
MaximaImage = BlurredImage;
MaximaMask = getnhood(strel('disk', max(1,floor(Diameter/4))));
MaximaImage(BlurredImage < ordfilt2(BlurredImage,sum(MaximaMask(:)),MaximaMask)) = 0;


%%% Thresholds the image to eliminate dim maxima.
if strcmp(Threshold,'Automatic'),
    Threshold = CPgraythresh(OrigImage,handles,ImageName);
    %%% Replaced the following line to accomodate calculating the
    %%% threshold for images that have been masked.
    % Threshold = CPgraythresh(OrigImage);
end
MaximaImage = MaximaImage > Threshold;

%%% Overlays the nuclear markers (maxima) on the inverted original image so
%%% there are black dots on top of each dark nucleus on a white background.
Overlaid = imimposemin(1 - BlurredImage,MaximaImage);
WatershedBoundaries = watershed(Overlaid) > 0;

Objects = OrigImage > Threshold;                          % Threshold image
Objects = imfill(Objects,'holes');                            % Fill holes
Objects = Objects.*WatershedBoundaries;                       % Cut objects along the watershed lines
Objects = bwlabel(Objects);                                   % Label the objects

%%% Remove objects with area outside the specified range
tmp = regionprops(Objects,'Area');                            % Get areas of the objects
areas = [0;cat(1,tmp.Area)];                      
MedianArea = median(areas);
AreaMap = areas(Objects+1);                                   % Create image with object intensity equal to the area
tmp = Objects;
Objects(AreaMap > MaxArea) = 0;                               % Remove objects that are too big
Objects(AreaMap < MinArea) = 0;                               % Remove objects that are too small
AreaExcludedObjects = tmp - Objects ;                         % Store objects that fall outside area range for display
NumOfAreaObjects = length(unique(AreaExcludedObjects(:)))-1;  % Count the objects 

%%% Remove objects along the border of the image
tmp = Objects;
Objects = imclearborder(Objects);                       
BorderObjects = tmp - Objects;                      
NumOfBorderObjects = length(unique(BorderObjects(:)))-1;

%%% Remove objects with no marker in it
List = setdiff(unique(Objects(:)),0);
for k = 1:length(List)
    index = find(Objects==List(k));                       % Get index for Object nr k pixels
    if sum(MaximaImage(index)) == 0                       % If there is no maxima in these pixels, exclude object
        Objects(index) = 0;
    end
end

%%% Relabel the objects
[Objects,NumOfObjects] = bwlabel(Objects > 0);


%%% Merge objects
%for k = 1:NumOfObjects
%    NeighborIndex = setdiff(unique(bwmorph(Objects==k,'dilate').*Objects),[0 k]);
%end


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
    PerimArea   = bwperim(AreaExcludedObjects > 0);
    PerimBorder = bwperim(BorderObjects > 0);
    OutlinedObjectsR(PerimObjects) = 0; OutlinedObjectsG(PerimObjects) = 1; OutlinedObjectsB(PerimObjects) = 0;
    OutlinedObjectsR(PerimArea)   = 1; OutlinedObjectsG(PerimArea)   = 0; OutlinedObjectsB(PerimArea)   = 0;
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
        'BackgroundColor',bgcolor,'HorizontalAlignment','Left','String',sprintf('Threshold:  %0.3f',Threshold));
    uicontrol(ThisModuleFigureNumber,'Style','Text','Units','Normalized','Position',[posx(1)-0.05 posy(2)+posy(4)-0.08 posx(3)+0.1 0.04],...
        'BackgroundColor',bgcolor,'HorizontalAlignment','Left','String',sprintf('Number of segmented objects: %d',NumOfObjects));
    uicontrol(ThisModuleFigureNumber,'Style','Text','Units','Normalized','Position',[posx(1)-0.05 posy(2)+posy(4)-0.12 posx(3)+0.1 0.04],...
        'BackgroundColor',bgcolor,'HorizontalAlignment','Left','String',['Median area (pixels): ' num2str(MedianArea)]);
    
    uicontrol(ThisModuleFigureNumber,'Style','Text','Units','Normalized','Position',[posx(1)-0.05 posy(2)+posy(4)-0.20 posx(3)+0.1 0.04],...
        'BackgroundColor',bgcolor,'HorizontalAlignment','Left','String',sprintf('Number of border objects: %d',NumOfBorderObjects));
    uicontrol(ThisModuleFigureNumber,'Style','Text','Units','Normalized','Position',[posx(1)-0.05 posy(2)+posy(4)-0.24 posx(3)+0.1 0.04],...
        'BackgroundColor',bgcolor,'HorizontalAlignment','Left','String',sprintf('Number of objects outside area range: %d',NumOfAreaObjects));
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
