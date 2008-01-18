function handles = TrackObjects(handles)

% Help for the Track Objects module:
% Category: Object Processing
%
% SHORT DESCRIPTION:
% Allows tracking objects throughout sequential frames of a movie, so that
% each object has a stable number in the output measurements.
% *************************************************************************
% Note: this module is beta-version. It is very simple and has not
% been thoroughly checked. Improvements to the code are welcome!
%
% This module must be run after objects have been identified using an
% identify module.
%
% Settings:
%
% Tracking Method:
% Choose between the methods based on which is most consistent from frame
% to frame of your movie:
%
%       Distance - Usually the best choice, this method will compare the
%       distance between each identified object in the previous frame with
%       the current frame. Closest objects to each other will be assigned
%       the same label.
%
%       Size - Each object will be compared to objects in the next frame
%       that are within the "neighborhood" (as defined by the next
%       variable) and the object with the closest size will be selected as
%       a match and will be assigned the same label.
%
%       Intensity - Each object will be compared to objects in the next
%       frame that are within the "neighborhood" (as defined by the next
%       variable) and the object with the closest total intensity will be
%       selected as a match and will be assigned the same label.
%
% Neighborhood:
% This indicates the neighborhood (in pixels) within which objects in the
% next frame are to be compared. To determine pixel distances, you can look
% at the markings on the side of each image (these are in pixel units) and
% you can also look at the values revealed using the Show Pixel Data Image
% tool (in the CellProfiler Image Tools menu of figure windows). This
% setting is only required for the methods of Size and Intensity.
%
% Intensity image:
% When using the Intensity option, you must specify the original image
% whose intensity values you want to use for comparison. Note that this
% image must be grayscale, not a color image.
%
% Statistics:
% This option is not yet available.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the objects you want to track?
%infotypeVAR01 = objectgroup
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the resulting image with tracked, color-coded objects?
%defaultVAR02 = TrackedDataDisp
%infotypeVAR02 = imagegroup indep
DataImage = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Choose a tracking method:(SORRY, ONLY THE DISTANCE METHOD IS CURRENTLY FUNCTIONING.)
%choiceVAR03 = Distance
%choiceVAR03 = Size
%choiceVAR03 = Intensity
%inputtypeVAR03 = popupmenu
TrackMethod = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = For SIZE or INTENSITY, choose the neighborhood (in pixels) within which objects will be evaluated to find a potential match.
%defaultVAR04 = 50
PixelRadius = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,4}));

%textVAR05 = For INTENSITY, what did you call the intensity image you want to use for tracking?
%infotypeVAR05 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu

%textVAR06 = How do you want to display the tracked objects?
%choiceVAR06 = Color and Number
%choiceVAR06 = Grayscale and Number
%inputtypeVAR06 = popupmenu
DisplayType = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = Do you want to calculate statistics? (SORRY, THIS OPTION IS NOT AVAILABLE YET)
%choiceVAR07 = Not available yet
%%% %choiceVAR07 = No
%%% %choiceVAR07 = Yes
%inputtypeVAR07 = popupmenu
Stats = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = Choose a text-labeling method for displaying tracked obejcts:
%choiceVAR08 = Object ID
%choiceVAR08 = Progeny ID
%inputtypeVAR08 = popupmenu
LabelMethod = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%%%VariableRevisionNumber = 3

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

try
    handles.Pipeline.TrackObjects.(ObjectName).Previous = handles.Pipeline.TrackObjects.(ObjectName).Current;
end

%%% I THINK THIS LINE IS NEEDED FOR INTENSITY ONLY:
handles.Pipeline.TrackObjects.(ObjectName).Current.(ImageName) = handles.Pipeline.(ImageName);
%%% Saves the final segmented label matrix image to the handles structure.
SegmentedObjectName = ['Segmented' ObjectName];
handles.Pipeline.TrackObjects.(ObjectName).Current.SegmentedImage = handles.Pipeline.(SegmentedObjectName);
%%% Saves the location of each segmented object
handles.Pipeline.TrackObjects.(ObjectName).Current.Locations = handles.Measurements.(ObjectName).Location{handles.Current.SetBeingAnalyzed};

% ObjectProperties = handles.Pipeline.TrackObjects.(ObjectName);
CurrLocations = handles.Pipeline.TrackObjects.(ObjectName).Current.Locations;
CurrSegImage = handles.Pipeline.TrackObjects.(ObjectName).Current.SegmentedImage;

if ~(handles.Current.SetBeingAnalyzed == 1)
    %%% Extracts data from the handles structure
    %%% I THINK THIS LINE IS NEEDED FOR INTENSITY ONLY:
    PrevImage = handles.Pipeline.TrackObjects.(ObjectName).Previous.(ImageName);
    PrevLocations = handles.Pipeline.TrackObjects.(ObjectName).Previous.Locations;
    PrevLabels = handles.Pipeline.TrackObjects.(ObjectName).Previous.Labels;
    PrevSegImage = handles.Pipeline.TrackObjects.(ObjectName).Previous.SegmentedImage;
    PrevHeaders = handles.Pipeline.TrackObjects.(ObjectName).Previous.Headers;

    %%% I THINK THIS LINE IS NEEDED FOR INTENSITY ONLY:
    CurrImage = CPretrieveimage(handles,ImageName,ModuleName,'MustBeGray','DontCheckScale'); %#ok Ignore MLint

    CurrLocations = handles.Pipeline.TrackObjects.(ObjectName).Current.Locations;
    CurrSegImage = handles.Pipeline.TrackObjects.(ObjectName).Current.SegmentedImage;

    switch TrackMethod
        case 'Distance'
            [CurrLabels, CurrHeaders] = ClosestXY(PrevLocations, CurrLocations, PrevLabels, PrevHeaders);
        otherwise
            % For the choices of 'Intensity' and 'Size', PixelRadius doesn't seem to work properly - fix it! 
           ObjectsToEval = FindObjectsToEval(PrevLocations, CurrLocations, PixelRadius);
           [CurrLabels, CurrHeaders] = CompareImages( handles.Pipeline.TrackObjects.(ObjectName), ObjectsToEval, ImageName, TrackMethod, PixelRadius, PrevLabels, PrevHeaders);
    end
      %%% Make Stats work and uncomment the following code
%     if strcmp(Stats, 'Yes')
%         PrevNumObj = max(PrevSegImage(:));
%         CurrSegObj = max(CurrSegImage(:));
%         if PrevNumObj < CurrSegObj
%             CellsEntered = CurrSegObj - PrevNumObj;
%             handles.TrackObjects.(ObjectName).CellsEnteredCount = CellsEntered + handles.TrackObjects.(ObjectName).CellsEnteredCount;
%         elseif PrevNumObj > CurrSegObj
%             CellsExited = PrevNumObj - CurrSegObj;
%             handles.TrackObjects.(ObjectName).CellsExitedCount = CellsExited + handles.TrackObjects.(ObjectName).CellsExitedCount;
%         end
% 
%         if handles.Current.SetBeingAnalyzed == handles.Current.NumberOfImageSets
%             for i = 1:length(handles.Pipeline.TrackObjects.(ObjectName).Stats.FirstObjSize)
%                 a = find(CurrLabels == i);
%                 if isempty(a)
%                     LastIsolatedObjSize(i) = NaN;
%                 else
%                     [IsolatedObject, border]=IsolateImage(handles.Pipeline.TrackObjects.(ObjectName).Current.SegmentedImage, a);
%                     LastIsolatedObjSize(i) = length(find(~(IsolatedObject == 0)));
%                 end
%             end
%             %%%%%%%%%%%%SOMETHING HERE IS WRONG%%%%%%%%%%%%%%%%%
%             FirstIsolatedObjSize = handles.Pipeline.TrackObjects.(ObjectName).Stats.FirstObjSize;
%             ObjectSizeChange = FirstIsolatedObjSize./LastIsolatedObjSize;
%         end
%     end

else
    len = length(handles.Pipeline.TrackObjects.(ObjectName).Current.Locations);
    CurrLabels = 1:len;
    for i = 1:len
        CurrHeaders{i} = '';
    end 
      %%% Make Stats work and uncomment the following code
%     if strcmp(Stats, 'Yes')
%         handles.Pipeline.TrackObjects.(ObjectName).Stats.CellsEnteredCount = 0;
%         handles.Pipeline.TrackObjects.(ObjectName).Stats.CellsExitedCount = 0;
% 
%         for i = 1:max(handles.Pipeline.TrackObjects.(ObjectName).Current.SegmentedImage(:))
%             [IsolatedObject, border] = IsolateImage(handles.Pipeline.TrackObjects.(ObjectName).Current.SegmentedImage, i);
%             IsolatedObjSize(i) = length(find(~(IsolatedObject == 0)));
%         end
%         handles.Pipeline.TrackObjects.(ObjectName).Stats.FirstObjSize = IsolatedObjSize;
%     end
end

%Create colored image
ColoredImage = LabelByColor(handles.Pipeline.TrackObjects.(ObjectName), CurrLabels);
[ColoredImage,handles] = TrackCPlabel2rgb(handles, ColoredImage);

if strcmp(DisplayType, 'Grayscale and Number')
    DisplayImage = CurrSegImage;
elseif strcmp(DisplayType, 'Color and Number')
    DisplayImage = ColoredImage;
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    %%% Activates the appropriate figure window.
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    CPimagesc(DisplayImage, handles);
    title('Tracked Objects');
end

CStringObjectID = cellstr(num2str((CurrLabels)'));
CStringProgenyID = strcat(CurrHeaders', cellstr(num2str((CurrLabels)')));
if strcmp(LabelMethod, 'Object ID') 
    CStringOfMeas = CStringObjectID;
elseif strcmp(LabelMethod, 'Progeny ID')
    CStringOfMeas = CStringProgenyID;
end
% PutTextInImage is designed to put texts in an image as pixels
% but, more adjustment work is required here.
%[DisplayImage, TextHandles] = PutTextInImage(DisplayImage,CurrLocations,CStringOfMeas)
TextHandles = text(CurrLocations(:,1) , CurrLocations(:,2) , CStringOfMeas,...
    'HorizontalAlignment','center', 'color', [.6 .6 .6],'fontsize',10,...%handles.Preferences.FontSize,...
    'fontweight','bold');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% DOES THIS STUFF REALLY NEED TO BE STORED IN THE HANDLES STRUCTURE, OR
%%% IS IT ONLY USED DURING THE CURRENT CYCLE?
handles.Pipeline.TrackObjects.(DataImage).Info.ListOfMeasurements = CurrLabels;
handles.Pipeline.TrackObjects.(DataImage).Info.TextHandles = TextHandles;
Info = handles.Pipeline.TrackObjects.(DataImage).Info;

handles.Pipeline.TrackObjects.(ObjectName).Current.Labels = CurrLabels;
handles.Pipeline.TrackObjects.(ObjectName).Current.Headers = CurrHeaders;
handles.Pipeline.(DataImage)=DisplayImage;

%%% Saves the Object-ID and Progeny-ID of each tracked object
handles.Measurements.(ObjectName).TrackingDescription = {'Object-ID','Progeny-ID'};
handles.Measurements.(ObjectName).Tracking{handles.Current.SetBeingAnalyzed}= [CStringObjectID, CStringProgenyID];

%%
%%%%%%%%%%%%%%%%%%%%%%
%%%% SUBFUNCTIONS %%%%
%%%%%%%%%%%%%%%%%%%%%%

%%% pixeldistance is not being used here - why?
function [CurrLabels, CurrHeaders] = CompareImages(handles, Info, ImageName, Method, pixeldistance, PrevLabels, PrevHeaders)
%Compares the objects based on the method chosen

PrevMaskedImage = handles.Previous.SegmentedImage;
CurrentMaskedImage = handles.Current.SegmentedImage;

%%% I THINK THIS LINE IS NEEDED FOR INTENSITY ONLY:
PrevOrigImage = handles.Previous.(ImageName);
CurrOrigImage = handles.Current.(ImageName);

PrevLocations = handles.Previous.Locations;
CurrLocations = handles.Previous.Locations;
CurrNumOfObjects = length(CurrLocations);

EvalArray = Info.EvaluationArray;
Size = size(EvalArray);
%RelationData = zeros(Size); <-- this will make all entries 0.
RelationData = 99999999999999999999*ones(Size);

for i = 1:Size(1)
    [PrevMaskedObj, PrevObjLoc] = IsolateImage(PrevMaskedImage, i);
    for j = 1:Size(2)
        if EvalArray(i,j) == 1
            [CurrMaskedObj, CurrObjLoc] = IsolateImage(CurrentMaskedImage, j);

            [NormPrevMask, NormCurrMask] = NormalizeSizes(PrevMaskedObj, CurrMaskedObj);

            %only when you need the orig isolatedimage
            %%% I THINK THIS LINE IS NEEDED FOR INTENSITY ONLY:
            PrevIsoObj= PrevOrigImage(PrevObjLoc(1):PrevObjLoc(2), PrevObjLoc(3):PrevObjLoc(4));
            %%% I THINK THIS LINE IS NEEDED FOR INTENSITY ONLY:
            CurrIsoObj= CurrOrigImage(CurrObjLoc(1):CurrObjLoc(2), CurrObjLoc(3):CurrObjLoc(4));

            [NormPrevIso, NormCurrIso] = NormalizeSizes(PrevIsoObj, CurrIsoObj);
            NormPrevIso(NormPrevMask==0) = 0;
            NormCurrIso(NormCurrMask==0) = 0;

            switch Method
                case 'Intensity'
                    RelationData(i,j) = CompareIntensity(PrevIsoObj, CurrIsoObj);
                case 'Size'
                    RelationData(i,j) = CompareSize(NormPrevIso, NormCurrIso);
            end
        end
    end
end

[CurrLabels, CurrHeaders] = AssignLabels(RelationData, PrevLabels, PrevHeaders);
%%
function [CurrLabels, CurrHeaders] = ClosestXY(PrevLocations, CurrLocations, PrevLabels, PrevHeaders)
%finds the closest objects in the two consectutive images and labels them
%the same.

PrevXLocations = PrevLocations(:,1);
PrevYLocations = PrevLocations(:,2);
CurrXLocations = CurrLocations(:,1);
CurrYLocations = CurrLocations(:,2);

%%% Calculates the distance that each object are from each other and stores it in an array.
PrevNumOfObjects = length(PrevXLocations);
CurrNumOfObjects = length(CurrXLocations);
DistanceArray = zeros(PrevNumOfObjects, CurrNumOfObjects);

for i= 1:PrevNumOfObjects
    for j = 1:CurrNumOfObjects
        Distance = sqrt((abs(CurrXLocations(j)-PrevXLocations(i)))^2+(abs(CurrYLocations(j)-PrevYLocations(i)))^2);
        DistanceArray(i,j) = Distance;
    end
end

[CurrLabels, CurrHeaders] = AssignLabels(DistanceArray, PrevLabels, PrevHeaders);
%%
function Objects = FindObjectsToEval(PrevLocations, CurrLocations, NeighborDist)
%Finds which objects that are in the pixelradius vicinity and evaluates
%their intensity or size correlation

EvalArray = zeros(length(PrevLocations), length(CurrLocations));

PrevXLocations = PrevLocations(:,1);
PrevYLocations = PrevLocations(:,2);
CurrXLocations = CurrLocations(:,1);
CurrYLocations = CurrLocations(:,2);

%%% Determines the neighbors for each object.
PrevNumOfObjects = length(PrevXLocations);
CurrNumOfObjects = length(CurrXLocations);

EvalArray = zeros(PrevNumOfObjects, CurrNumOfObjects);

for i= 1:PrevNumOfObjects
    counter = 0;
    ShortestDistance = Inf;
    for j = 1:CurrNumOfObjects

        Distance = sqrt((abs(CurrXLocations(j)-PrevXLocations(i)))^2+(abs(CurrYLocations(j)-PrevYLocations(i)))^2);

        if Distance < ShortestDistance
            ShortestDistance = Distance;
            ClosestCellArray = zeros(1,CurrNumOfObjects);
            ClosestCellArray(j) = 1;
        end
        if Distance < NeighborDist
            counter = counter+1;
            EvalArray(i,j) = 1;
        else
            EvalArray(i,j) = 0;
        end
    end
    if counter == 0
        EvalArray(i,:) = ClosestCellArray;
    end
end
Objects.EvaluationArray = EvalArray;
%%
function [Image, border] = IsolateImage(IncomingLabelMatrixImage, counter)
%Isolates each object

[sr,sc] = size(IncomingLabelMatrixImage);
IdentityOfNeighbors = cell(max(IncomingLabelMatrixImage(:)),1);
props = regionprops(IncomingLabelMatrixImage,'PixelIdxList');

% Cut patch
[r,c] = ind2sub([sr sc],props(counter).PixelIdxList);
rmax = min(sr,max(r))+1;
rmin = max(1,min(r))-1;
cmax = min(sc,max(c))+1;
cmin = max(1,min(c))-1;
Image = IncomingLabelMatrixImage(rmin:rmax,cmin:cmax);
Image = CPclearborder(Image);
border = [rmin rmax cmin cmax];


%%
function SizeCorrelation = CompareSize(Image1, Image2)
%Finds the size correlation

PixelSpace1 = find(~(Image1 == 0));
PixelSpace2 = find(~(Image2 == 0));
TotalSize1 = size(Image1);
TotalSize2 = size(Image2);

AreaOcc1 = length(PixelSpace1)/(TotalSize1(1)*TotalSize1(2));
AreaOcc2 = length(PixelSpace2)/(TotalSize2(1)*TotalSize2(2));

if AreaOcc2 < AreaOcc1
    SizeRatio = AreaOcc2/AreaOcc1;
else
    SizeRatio = AreaOcc1/AreaOcc2;
end
SizeCorrelation = 1-SizeRatio;

function IntensityCorrelation = CompareIntensity(Image1, Image2)
%Finds the Intenisity Correlation
[x1, y1] = find(~(Image1 == 0));
[x2, y2] = find(~(Image2 == 0));

TotalIntensity1 = sum(sum(Image1));
AvgIntense1 = TotalIntensity1/ length(x1);

TotalIntensity2 = sum(sum(Image2));
AvgIntense2 = TotalIntensity2/ length(x2);

if AvgIntense2 < AvgIntense1
    IntensityRatio = AvgIntense2/AvgIntense1;
else
    IntensityRatio = AvgIntense1/AvgIntense2;
end
IntensityCorrelation = 1-IntensityRatio;
%%
function [NormPrev, NormCurr] = NormalizeSizes(PrevIsoObj, CurrIsoObj)
%Make isolated images the same size
NormPrev = PrevIsoObj;
NormCurr = CurrIsoObj;
Psize = size(NormPrev);
Csize = size(NormCurr);
sizediff = Psize - Csize;

if abs(rem(sizediff(1),2)) == 1
    if sizediff(1) < 0
        NormPrev = padarray(NormPrev, [1 0], 'post');
    else
        NormCurr = padarray(NormCurr, [1 0], 'post');
    end
end

if abs(rem(sizediff(2),2)) == 1
    if sizediff(2) < 0
        NormPrev = padarray(NormPrev, [0 1], 'post');
    else
        NormCurr = padarray(NormCurr, [0 1], 'post');
    end
end

Psize = size(NormPrev);
Csize = size(NormCurr);
padsize = (Psize-Csize)/2;

if padsize(1)<0
    NormPrev = padarray(NormPrev, [abs(padsize(1)) 0]);
else
    NormCurr = padarray(NormCurr, [abs(padsize(1)) 0]);
end
if padsize(2)<0
    NormPrev = padarray(NormPrev, [0 abs(padsize(2))]);
else
    NormCurr = padarray(NormCurr, [0 abs(padsize(2))]);
end
%%
function [UpdatedLabels, UpdatedHeaders] = AssignLabels(DataArray, PrevLabels, PrevHeaders)
% Assigns each object the label corresponding to its previous label

Size = size(DataArray);
UpdatedLabels = zeros(1, Size(2));
copyDataArray = DataArray;
while ~(length(find(DataArray == Inf)) == numel(DataArray))
    [i,j] = find(DataArray == min(min(DataArray)));
    UpdatedLabels(j(1)) = PrevLabels(i(1));
    UpdatedHeaders{j(1)} = PrevHeaders{i(1)};
    Array(i,j) = 1;
    ObjectName = ['Object' int2str(i(1))];
    DataArray(i(1),:) = Inf(1,Size(2));
    DataArray(:,j(1)) = Inf(1,Size(1));
end

for i= 1:Size(2)
    if UpdatedLabels(i)==0
        UpdatedLabels(i) = max(UpdatedLabels)+1;
        [val,idx] = min(copyDataArray(:,i));
        UpdatedHeaders{i} = strcat(PrevHeaders{idx},int2str(PrevLabels(idx)),'-');
    end
end
%%
function ColoredImage = LabelByColor(handles, CurrLabel)
%Relabel the label matrix so that the labels in the matrix are consistent
%with the text labels.

IncomingLabelMatrixImage = handles.Current.SegmentedImage;
[sr,sc] = size(IncomingLabelMatrixImage);
ColoredImage = zeros(sr,sc);

props = regionprops(IncomingLabelMatrixImage,'PixelIdxList');
NumberOfObjects=max(IncomingLabelMatrixImage(:));
for k = 1:NumberOfObjects
    [r,c] = ind2sub([sr sc],props(k).PixelIdxList);
    ColoredImage(sub2ind([sr sc],r,c)) = CurrLabel(k);
end

function [im, handles] = TrackCPlabel2rgb(handles, image)
numregion = double(max(image(:)));
cmap = eval([handles.Preferences.LabelColorMap '(255)']);
try
    if numregion>length(handles.newcmap)
        newregions = numregions-length(handles.newcmap);
        newindex = round(rand(1,newregions)*255);
        index = [index newindex];
        handles.newcmap = cmap(index,:,:);
    end
catch
    S = rand('state');
    rand('state', 0);
    index = round(rand(1,numregion)*255);
    handles.Pipeline.TrackObjects.Colormap = cmap(index,:,:);
    rand('state', S);
end
im = label2rgb(image, handles.Pipeline.TrackObjects.Colormap, 'k', 'noshuffle');
%%
function [im,TextHandles] = PutTextInImage(im,CurrLocations,CStringOfMeas)

% Create the text mask
% Make an image the same size and put text in it
hf = figure('color','white','units','normalized','position',[.1 .1 .8 .8]);
image(ones(size(im)));
set(gca,'units','pixels','position',[5 5 size(im,2)-1 size(im,1)-1],'visible','off')

% Text at arbitrary position
TextHandles = text(CurrLocations(:,1), size(im,1)-CurrLocations(:,2), CStringOfMeas, 'units','pixels','fontsize',5,...
    'fontweight','bold','HorizontalAlignment','center', 'color', [0 0 0]);

%TextHandles = text(CurrLocations(:,1), CurrLocations(:,2), CStringOfMeas, 'units','pixels','fontsize',10,...
%    'fontweight','bold','HorizontalAlignment','center', 'color', [0 0 0]);

%TextHandles = text(size(im,1)-CurrLocations(:,1), CurrLocations(:,2), CStringOfMeas, 'units','pixels','fontsize',10,...
%    'fontweight','bold','HorizontalAlignment','center', 'color', [0 0 0]);

% Capture the text image
% Note that the size will have changed by about 1 pixel
tim = getframe(gca);
close(hf)

% Extract the cdata
tim2 = tim.cdata;

% Make a mask with the negative of the text
tmask = tim2==0;

% Place white text
% Replace mask pixels with UINT8 max
im(tmask) = 255;%uint8(255);
