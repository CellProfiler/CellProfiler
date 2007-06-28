function handles = CountWorms(handles)

% Help for the Count Worms module:
% Category: Object Processing
%
% SHORT DESCRIPTION:
% Estimates the amount of worms in a given image.
% *************************************************************************
%
%%% THIS MODULE HAS BEEN REPLACED. You can replicate it by using FindEdges
%%% plus IdentifyPrimAutomatic. The only unique things that we might want
%%% to incorporate into IdentifyPrimAutomatic someday are the bits of code
%%% that try to fill gaps (when you have objects that are outlines rather
%%% than solid objects).IdentifyPrimAutomatic already fills holes in
%%% completely closed objects, but this goes a bit further in the case
%%% where objects are not completely closed circles. These sections are
%%% marked %%% THIS PART SEEMS TO BE UNIQUE >>>




% Given that CellProfiler currently has difficulty identifying weird shaped
% objects, this module only estimates the amount of objects in an image,
% rather than trying to identify them separately.
%
%
% See also IdentifyPrimAutomatic, IdentifyPrimBasedOnOutlines.

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the image you want to process?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the objects identified by this module?
%defaultVAR02 = Worms
%infotypeVAR02 = objectgroup indep
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Typical diameter of objects, in pixel units (Min,Max):
%defaultVAR03 = 55,75
SizeRange = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = What is the minimum width of objects, in pixel units?
%defaultVAR04 = 12
MinWidth = str2double(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = If you have an accurate estimate of an objects average area (in pixels),
%please specify it here. Otherwise it will be estimated from the minimum
%and maximum diameteres.
%defaultVAR05 = /
MeanArea = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = Select an automatic thresholding method or enter an absolute threshold in the range [0,1]. Choosing 'All' will use the Otsu Global method to calculate a single threshold for the entire image group. The other methods calculate a threshold for each image individually. Set interactively will allow you to manually adjust the threshold during the first cycle to determine what will work well.
%choiceVAR06 = Otsu Global
%choiceVAR06 = Otsu Adaptive
%choiceVAR06 = MoG Global
%choiceVAR06 = MoG Adaptive
%choiceVAR06 = Background Global
%choiceVAR06 = Background Adaptive
%choiceVAR06 = RidlerCalvard Global
%choiceVAR06 = RidlerCalvard Adaptive
%choiceVAR06 = All
%choiceVAR06 = Set interactively
ThresholdMethod = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%inputtypeVAR06 = popupmenu custom

%textVAR07 = What is the Threshold Correction Factor? The Threshold can be automatically found, then multiplied by this factor.
%defaultVAR07 = 1
ThresholdCorrection = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,7}));

%textVAR08 = Lower and upper bounds on threshold, in the range [0,1]
%defaultVAR08 = 0,1
ThresholdRange = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%textVAR09 = For MoG thresholding, what is the approximate percentage of image covered by objects?
%choiceVAR09 = 10%
%choiceVAR09 = 20%
%choiceVAR09 = 30%
%choiceVAR09 = 40%
%choiceVAR09 = 50%
%choiceVAR09 = 60%
%choiceVAR09 = 70%
%choiceVAR09 = 80%
%choiceVAR09 = 90%
pObject = char(handles.Settings.VariableValues{CurrentModuleNum,9});
%inputtypeVAR09 = popupmenu

%textVAR10 = Specify the filter size (in pixels) that you would like to use.
%defaultVAR10 = 8
SizeOfSmoothingFilter = str2double(handles.Settings.VariableValues{CurrentModuleNum,10});

%textVAR11 = Do you want to try our other method too? (to compare)
%choiceVAR11 = No
%choiceVAR11 = Yes
TryOtherMethodToo = char(handles.Settings.VariableValues{CurrentModuleNum,11});
%inputtypeVAR11 = popupmenu

%%%VariableRevisionNumber = 1


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
if ~(strncmp(ThresholdMethod,'Otsu',4) || strncmp(ThresholdMethod,'MoG',3) || strncmp(ThresholdMethod,'Background',10) || strncmp(ThresholdMethod,'RidlerCalvard',13) || strcmp(ThresholdMethod,'All') || strcmp(ThresholdMethod,'Set interactively'))
    if isnan(str2double(ThresholdMethod))
        error(['Image processing was canceled in the ' ModuleName ' module because the threshold method you specified is invalid. Please select one of the available methods or specify a threshold to use (a number in the range 0-1). Your input was ' ThresholdMethod]);
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

%%% Check parameter MeanArea
if strcmp(MeanArea,'/')
    MinArea = pi*(MinDiameter.^2)/4;
    MaxArea = pi*(MaxDiameter.^2)/4;
    MeanArea = mean([MinArea,MaxArea]);
    MeanAreaToUse = ceil(MeanArea/100);
else
    MeanArea = str2double(MeanArea);
    if isnan(MeanArea) || MeanArea<0
        error(['Image processing has been canceled in the ' ModuleName ' module because the average object area you specified is invalid.']);
    else
        MeanAreaToUse = ceil(MeanArea/100);
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
if isnan(SizeOfSmoothingFilter)
    error(['Image processing was canceled in the ' ModuleName ' module because the size of smoothing filter you specified is invalid.']);
else
    SizeOfSmoothingFilter = min(30,max(1,floor(SizeOfSmoothingFilter)));
end

if isnan(MinWidth)
    error(['Image processing was canceled in the ' ModuleName ' module because the minimum width you specified is invalid.']);
end


%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;

%%% Get grayscale edges of image
Sq = CPsmooth(OrigImage,'Q',SizeOfSmoothingFilter,0);
Mn = CPsmooth(OrigImage,'S',SizeOfSmoothingFilter,0);
EdgedImage = Mn./Sq;
%%% The ratio image has really weird numbers, put it in the 0-1 range:
[handles, EdgedImage] = CPrescale(handles,EdgedImage,'S',[]);

%%% Get threshold. Graythresh is currently hardcoded to determine edges, as
%%% it seemed to give a better threshold than the other methods.
OrigThreshold = graythresh(EdgedImage);

%%% Get binary edges of image
EdgedImage = im2bw(EdgedImage, OrigThreshold);



%%% THIS PART SEEMS TO BE UNIQUE >>>>

warning off MATLAB:intConvertOverflow                          % For binary images not to give warnings
ImageToThreshold = imfill(EdgedImage,'holes');                 % Fill whatever we can
StructEl = strel('disk',round(MinWidth/2));                    % Create structure element
ImageToThreshold = imclose(ImageToThreshold,StructEl);         % Close small gaps
ImageToThreshold = imfill(ImageToThreshold,'holes');           % Fill again

%%% Get new threshold. This will make the objects a little bit wider
[handles, Threshold] = CPthreshold(handles,ThresholdMethod,pObject,MinimumThreshold,MaximumThreshold,ThresholdCorrection,ImageToThreshold,ImageName,ModuleName);

%%% <<< THIS PART SEEMS TO BE UNIQUE




%%% Apply a slight smoothing before thresholding to remove
%%% 1-pixel objects and to smooth the edges of the objects.
%%% Note that this smoothing is hard-coded, and not controlled
%%% by the user.
sigma = 1;
FiltLength = 2*sigma;                                              % Determine filter size, min 3 pixels, max 61
[x,y] = meshgrid(-FiltLength:FiltLength,-FiltLength:FiltLength);   % Filter kernel grid
f = exp(-(x.^2+y.^2)/(2*sigma^2));f = f/sum(f(:));                 % Gaussian filter kernel
%%% This adjustment prevents the outer borders of the image from being
%%% darker (due to padding with zeros), which causes some objects on the
%%% edge of the image to not  be identified all the way to the edge of the
%%% image and therefore not be thrown out properly.
BlurredImage = conv2(double(ImageToThreshold),f,'same') ./ conv2(ones(size(ImageToThreshold)),f,'same');

%%% Threshold image again
PrelimObjects = BlurredImage > Threshold;






%%% THIS PART SEEMS TO BE UNIQUE >>>>

%%% Clean up
StructEl2 = strel('disk',round(MinWidth/3.5));
PrelimObjects = imclose(double(PrelimObjects),StructEl2);
PrelimObjects = imfill(double(PrelimObjects),'holes');

StructEl3 = strel('disk',round(MinWidth/4));
Objects = imerode(PrelimObjects,StructEl3);


%%% <<< THIS PART SEEMS TO BE UNIQUE







%%% Check for CropMask
fieldname = ['CropMask', ImageName];
if isfield(handles.Pipeline,fieldname)
    %%% Retrieves previously selected cropping mask from handles
    %%% structure.
    BinaryCropImage = handles.Pipeline.(fieldname);
    Objects = Objects & BinaryCropImage;
end

%%% Estimate number of objects
props = regionprops(double(Objects),'Area');
Area = cat(1,props.Area);
TotalArea = sum(Area);
TotalAreaToUse = floor(TotalArea/100);
EstimatedNumberOfObjects = TotalAreaToUse/MeanAreaToUse;



%%% THIS PART SEEMS TO BE UNIQUE >>>>

if strcmp(TryOtherMethodToo,'Yes')
    %%% Get original image and apply traditional threshold to it
    [handles, Threshold] = CPthreshold(handles,ThresholdMethod,pObject,MinimumThreshold,MaximumThreshold,ThresholdCorrection,OrigImage,ImageName,ModuleName);
    BlurredImage = conv2(OrigImage,f,'same') ./ conv2(ones(size(OrigImage)),f,'same');
    %%% Threshold image
    PrelimObjects = BlurredImage > Threshold;
    
    %%% Get skeletons of traditional masking
    SkelTrad = bwmorph(PrelimObjects,'skel',Inf);
    %%% Get skeletons of new masking
    SkelNew = bwmorph(Objects,'skel',Inf);
    
    %%% Superimpose them and get common points
    BothSkels = SkelTrad & SkelNew;
    
    %%% Process these new 'midlines'
    BothSkels = bwmorph(BothSkels,'bridge');
    BothSkels = imdilate(BothSkels,StructEl2);
	
    %%% Put together with outlines
    PrelimObjects = BothSkels | EdgedImage;
    
    %%% Clean up
    Objects2 = imfill(PrelimObjects,'holes');
    Objects2 = imclose(Objects2,StructEl2);
    Objects2 = imfill(Objects2,'holes');
    StructEl4 = strel('disk',round(MinWidth/5));
    Objects2 = imerode(Objects2,StructEl4);
    
    %%% Estimate number of objects
    props = regionprops(double(Objects2),'Area');
    Area2 = cat(1,props.Area);
    TotalArea2 = sum(Area2);
    TotalAreaToUse2 = floor(TotalArea2/100);
    EstimatedNumberOfObjects2 = TotalAreaToUse2/MeanAreaToUse;
end
warning on MATLAB:intConvertOverflow


%%% <<<< THIS PART SEEMS TO BE UNIQUE




%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

FontSize = handles.Preferences.FontSize;
ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule', CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    %%% Activates display window
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure(OrigImage,'TwoByTwo',ThisModuleFigureNumber)
    end
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,2,1);
    CPimagesc(OrigImage,handles);
    title(['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the masked objects image.
    subplot(2,2,3);
    CPimagesc(Objects,handles);
    title(['Masked ' ObjectName]);
    if strcmp(TryOtherMethodToo,'Yes')
        subplot(2,2,2);
        CPimagesc(Objects2,handles);
        title(['Masked ' ObjectName ' by other method']);

        % Text for Total Area in masked image
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.52 0.22 0.45 0.03],...
            'HorizontalAlignment','left','Backgroundcolor',[.7 .7 .9],'fontname','Helvetica',...
            'fontsize',FontSize,'fontweight','bold','string','Other Method:','UserData',SetBeingAnalyzed);

        % Text for Total Area in masked image
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.52 0.22 0.45 0.03],...
            'HorizontalAlignment','left','Backgroundcolor',[.7 .7 .9],'fontname','Helvetica',...
            'fontsize',FontSize,'string',sprintf('Total Area: %.0f',TotalArea2),'UserData',SetBeingAnalyzed);

        % Text for Equivalent Area
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.52 0.18 0.45 0.03],...
            'HorizontalAlignment','left','BackgroundColor',[.7 .7 .9],'fontname','Helvetica',...
            'fontsize',FontSize,'string',sprintf('Area is equivalent to that of %.2f %s',EstimatedNumberOfObjects2,lower(ObjectName)),'UserData',SetBeingAnalyzed);

        % Text for Estimated number of objects
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.52 0.14 0.45 0.03],...
            'HorizontalAlignment','left','BackgroundColor',[.7 .7 .9],'fontname','Helvetica',...
            'fontsize',FontSize,'string',sprintf('Estimated number of objects: %.0f',round(EstimatedNumberOfObjects2)),'UserData',SetBeingAnalyzed);
    end
    
    % Text for Mean Area used
    uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.52 0.42 0.45 0.03],...
        'HorizontalAlignment','left','BackgroundColor',[.7 .7 .9],'fontname','Helvetica',...
        'fontsize',FontSize,'string',sprintf('Mean Area: %.0f',MeanArea),'UserData',SetBeingAnalyzed);
    
    % Text for Total Area in masked image
    uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.52 0.36 0.45 0.03],...
        'HorizontalAlignment','left','Backgroundcolor',[.7 .7 .9],'fontname','Helvetica',...
        'fontsize',FontSize,'string',sprintf('Total Area: %.0f',TotalArea),'UserData',SetBeingAnalyzed);

    % Text for Equivalent Area
    uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.52 0.32 0.45 0.03],...
        'HorizontalAlignment','left','BackgroundColor',[.7 .7 .9],'fontname','Helvetica',...
        'fontsize',FontSize,'string',sprintf('Area is equivalent to that of %.2f %s',EstimatedNumberOfObjects,lower(ObjectName)),'UserData',SetBeingAnalyzed);
    
    % Text for Estimated number of objects
    uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.52 0.28 0.45 0.03],...
        'HorizontalAlignment','left','BackgroundColor',[.7 .7 .9],'fontname','Helvetica',...
        'fontsize',FontSize,'string',sprintf('Estimated number of objects: %.0f',round(EstimatedNumberOfObjects)),'UserData',SetBeingAnalyzed);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the adjusted image to the handles structure so it can be used by
%%% subsequent modules.

handles = CPaddmeasurements(handles,'Image',ModuleName,'AreaEquivTo',EstimatedNumberOfObjects);
handles = CPaddmeasurements(handles,'Image',ModuleName,'NumberOfObjects',round(EstimatedNumberOfObjects));

if strcmp(TryOtherMethodToo,'Yes')
    handles = CPaddmeasurements(handles,'Image',ModuleName,'AreaEquivTo2',EstimatedNumberOfObjects2);
    handles = CPaddmeasurements(handles,'Image',ModuleName,'NumberOfObjects2',round(EstimatedNumberOfObjects2));
end