function handles = TrackObjects(handles)

% Help for the Track Objects module:
% Category: Object Processing
%
% SHORT DESCRIPTION:
% The TrackObjects module allows tracking objects throughout
% the duration of a video file, so that each object has a stable number 
% in the output measurements.
% *************************************************************************
% Note: this module is still beta-version and has not been thoroughly 
% checked. Improvements to the code are welcome!
% 
% This module must be run after objects have been identified.  The image
% that objects must appear in cannot be a color image.  For color video
% files, run the ColorToGray module before this.  
% 
% Settings:
% Objects To Track- Select the objects that you wish to track.  They must
%       already be identified before this module.
%
% Track Method -   Distance, Size, Intensity
%
%       Distance - This method will compare the distance between each
%                  identified object in the previous frame with the current frame.
%                  Closest objects to each other will be assigned the same label.
%
%       Size -     For this setting, the user must specify the pixel distance
%                  where the objects of the next frame are to be compared to the
%                  first.  After objects that are identified to be within the distance
%                  specified, the object to be tracked will be compared to the
%                  objects in question based on size.  The objects with the most
%                  closely related size among the potential objects will be assigned the same label.
%
%       Intensity - For this setting, the user must also specify the pixel distance
%                  of where the objects of the next frame are to be compared to the
%                  first.  After objects that are identified to be within the distance
%                  specified, the object to be tracked will be compared to the
%                  objects in question based on size.  The objects with the most
%                  closely related size among the potential objects will be
%                  be assigned the same label.
%
% Pixel Radius - This setting is only required for the methods of Size and
%                Intensity.  This indicates the vicinity where objects in the next frame
%                are to be compared.
%
% Image with Object -   Select the original image of where the objects to be
%                       tracked were identified in.  Note that this image must not be a color
%                       image.
%
% Tracked Display Name - Specify the name of what the image with the
%                        labeled object is to be stored. 
%
% Suggestions:  The Distance method generally works the best; however, if
% the objects move very quickly, try the size or intensity based on which
% property is most consistent.  How well this module works depends largely
% on how well the objects were identified in the Identify module you are 
% using.

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
% $Revision: 3899 $

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What objects do you want to track?
%infotypeVAR01 = objectgroup
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = Choose a tracking method:
%choiceVAR02 = Size
%choiceVAR02 = Intensity
%choiceVAR02 = Distance
%inputtypeVAR02 = popupmenu
TrackMethod = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = For SIZE or INTENSITY, choose the neighborhood (in pixels) within which objects will be evaluated.
%defaultVAR03 = 100
PixelRadius = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,3}));

%textVAR04 = What did you call the image with the object you wish to track?
%infotypeVAR04 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%textVAR05 = What do you want to call the generated image with data?
%defaultVAR05 = TrackedDataDisp
%infotypeVAR05 = imagegroup indep
DataImage = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = How do you want to display the tracked objects?
%choiceVAR06 = Color and Number
%choiceVAR06 = Grayscale and Number
%inputtypeVAR06 = popupmenu
DisplayType = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = Do you want to calculate stats? (THIS OPTION IS NOT COMPLETE AT THE MOMENT)
%defaultVAR07 = No
%choiceVAR07 = Yes
%choiceVAR07 = No
%inputtypeVAR07 = popupmenu
Stats = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

try
    handles.Pipeline.TrackObjects.(ObjectName).Previous = handles.Pipeline.TrackObjects.(ObjectName).Current;
end

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
    PrevImage = handles.Pipeline.TrackObjects.(ObjectName).Previous.(ImageName);
    PrevLocations = handles.Pipeline.TrackObjects.(ObjectName).Previous.Locations;
    PrevLabels = handles.Pipeline.TrackObjects.(ObjectName).Previous.Labels;
    PrevSegImage = handles.Pipeline.TrackObjects.(ObjectName).Previous.SegmentedImage;
    
    CurrImage = CPretrieveimage(handles,ImageName,ModuleName,'MustBeGray','DontCheckScale'); %#ok Ignore MLint

    CurrLocations = handles.Pipeline.TrackObjects.(ObjectName).Current.Locations;
    CurrSegImage = handles.Pipeline.TrackObjects.(ObjectName).Current.SegmentedImage;
    
    switch TrackMethod
        case 'Distance'
            CurrLabels = ClosestXY(PrevLocations, CurrLocations, PrevLabels);
        otherwise
            ObjectsToEval = FindObjectsToEval(PrevLocations, CurrLocations, PixelRadius);
            CurrLabels = CompareImages( handles.Pipeline.TrackObjects.(ObjectName), ObjectsToEval, ImageName, TrackMethod, PixelRadius, PrevLabels);
    end
    
    if strcmp(Stats, 'Yes')
        PrevNumObj = max(PrevSegImage(:));
        CurrSegObj = max(CurrSegImage(:));
        if PrevNumObj < CurrSegObj
            CellsEntered = CurrSegObj - PrevNumObj;
            handles.TrackObjects.(ObjectName).CellsEnteredCount = CellsEntered + handles.TrackObjects.(ObjectName).CellsEnteredCount;
        elseif PrevNumObj > CurrSegObj
            CellsExited = PrevNumObj - CurrSegObj;
            handles.TrackObjects.(ObjectName).CellsExitedCount = CellsExited + handles.TrackObjects.(ObjectName).CellsExitedCount;
        end
        
        if handles.Current.SetBeingAnalyzed == handles.Current.NumberOfImageSets
            for i = 1:length(handles.Pipeline.TrackObjects.(ObjectName).Stats.FirstObjSize)
                a = find(CurrLabels == i);
                if isempty(a)
                    LastIsolatedObjSize(i) = NaN;
                else
                    [IsolatedObject, border]=IsolateImage(handles.Pipeline.TrackObjects.(ObjectName).Current.SegmentedImage, a);
                    LastIsolatedObjSize(i) = length(find(~(IsolatedObject == 0)));
                end
            end
            %%%%%%%%%%%%SOMETHING HERE IS WRONG%%%%%%%%%%%%%%%%%
            FirstIsolatedObjSize = handles.Pipeline.TrackObject.(ObjectName).Stats.FirstObjSize;
            ObjectSizeChange = FirstIsolatedObjSize./LastIsolatedObjSize;
        end
    end
    
else
    CurrLabels = 1:length(handles.Pipeline.TrackObjects.(ObjectName).Current.Locations);
    if strcmp(Stats, 'Yes')
        handles.Pipeline.TrackObjects.(ObjectName).Stats.CellsEnteredCount = 0;
        handles.Pipeline.TrackObjects.(ObjectName).Stats.CellsExitedCount = 0;
        
        for i = 1:max(handles.Pipeline.TrackObjects.(ObjectName).Current.SegmentedImage(:))
            [IsolatedObject, border] = IsolateImage(handles.Pipeline.TrackObjects.(ObjectName).Current.SegmentedImage, i);
            IsolatedObjSize(i) = length(find(~(IsolatedObject == 0)));
        end
        handles.Pipeline.TrackObjects.(ObjectName).Stats.FirstObjSize = IsolatedObjSize;      
    end
end
   

CStringOfMeas = cellstr(num2str((CurrLabels)'));

%Create colored image
ColoredImage = LabelByColor(handles.Pipeline.TrackObjects.(ObjectName), CurrLabels);
[ColoredImage,handles] = TrackCPlabel2rgb(handles, ColoredImage);

if strcmp(DisplayType, 'Grayscale and Number')
    DisplayImage = CurrSegImage;
elseif strcmp(DisplayType, 'Color and Number')
    DisplayImage = ColoredImage;
end
%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%

drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    
    %%% Activates the appropriate figure window.
    DataHandle = CPfigure(handles,'Image',ThisModuleFigureNumber);
    CPimagesc(DisplayImage, handles);
    title('Tracked Objects');
    TextHandles = text(CurrLocations(:,1) , CurrLocations(:,2) , CStringOfMeas,...
    'HorizontalAlignment','center', 'color', [1 1 0],'fontsize',handles.Preferences.FontSize);
end

Info = get(DataHandle,'UserData');
Info.ListOfMeasurements = CurrLabels;
Info.TextHandles = TextHandles;
set(DataHandle,'UserData',Info);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if any(findobj == ThisModuleFigureNumber)
    handles.Pipeline.TrackObjects.(ObjectName).Current.Labels = CurrLabels;
    OneFrame = getframe(DataHandle);
    handles.Pipeline.(DataImage)=OneFrame.cdata;
end

%%%%%%%%%%%%%%%%%%%%%%
%%%% SUBFUNCTIONS %%%%
%%%%%%%%%%%%%%%%%%%%%%

function CurrLabels = CompareImages(handles, Info, ImageName, Method, pixeldistance, PrevLabels)
%Compares the objects based on the method chosen

PrevMaskedImage = handles.Previous.SegmentedImage;
CurrentMaskedImage = handles.Current.SegmentedImage;

PrevOrigImage = handles.Previous.(ImageName);
CurrOrigImage = handles.Current.(ImageName);

PrevLocations = handles.Previous.Locations;
CurrLocations = handles.Previous.Locations;
CurrNumOfObjects = length(CurrLocations);

EvalArray = Info.EvaluationArray;
Size = size(EvalArray);
RelationData = zeros(Size);

for i = 1:Size(1)
    [PrevMaskedObj, PrevObjLoc] = IsolateImage(PrevMaskedImage, i);
    for j = 1:Size(2)
        if EvalArray(i,j) == 1
            [CurrMaskedObj, CurrObjLoc] = IsolateImage(CurrentMaskedImage, j);
            
            [NormPrevMask, NormCurrMask] = NormalizeSizes(PrevMaskedObj, CurrMaskedObj);
    
            %only when you need the orig isolatedimage
            PrevIsoObj= PrevOrigImage(PrevObjLoc(1):PrevObjLoc(2), PrevObjLoc(3):PrevObjLoc(4));
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

CurrLabels = AssignLabels(RelationData, PrevLabels);

function CurrLabels = ClosestXY(PrevLocations, CurrLocations, PrevLabels)
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

CurrLabels = AssignLabels(DistanceArray, PrevLabels);

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

function UpdatedLabels = AssignLabels(DataArray, PrevLabels)
% Assigns each object the label corresponding to its previous label

Size = size(DataArray);
UpdatedLabels = zeros(1, Size(2));

while ~(length(find(DataArray == Inf)) == numel(DataArray))
    [i,j] = find(DataArray == min(min(DataArray)));
    UpdatedLabels(j(1)) = PrevLabels(i(1));
    Array(i,j) = 1;
    ObjectName = ['Object' int2str(i(1))];
    DataArray(i(1),:) = Inf(1,Size(2));
    DataArray(:,j(1)) = Inf(1,Size(1));
end

for i= 1:Size(2)
    if UpdatedLabels(i)==0
        UpdatedLabels(i) = max(UpdatedLabels)+1;
    end
end

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