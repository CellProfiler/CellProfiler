function handles = MeasureObjectNeighbors(handles)

% Help for the Measure Object Neighbors module:
% Category: Measurement
%
% SHORT DESCRIPTION:
% Calculates how many neighbors each object has.
% *************************************************************************
%
% Given an image with objects identified (e.g. nuclei or cells), this
% module determines how many neighbors each object has. The user selects
% the distance within which objects should be considered neighbors. The
% module can measure the number of neighbors each object has if every
% object were expanded up until the point where it hits another object. To
% use this option, enter 0 (the number zero) for the pixel distance.
%
% How it works:
% Retrieves objects in label matrix format. The objects are expanded by the
% number of pixels the user specifies, and then the module counts up how
% many other objects the object is overlapping.

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
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the objects whose neighbors you want to measure?
%infotypeVAR01 = objectgroup
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = Objects are considered neighbors if they are within this distance, in pixels. If you want your objects to be touching before you count neighbors (for instance, in an image of tissue), use the ExpandOrShrink module to expand your objects:
%defaultVAR02 = 0
NeighborDistance = str2double(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = What do you want to call the objects colored by number of neighbors, which are compatible for converting to a color image using the Convert To Image and Save Images modules?
%defaultVAR03 = Do not save
%infotypeVAR03 = objectgroup indep
ColoredNeighborsName = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = What do you want to call the image of the objects with grayscale values corresponding to the number of neighbors, which is compatible for saving in .mat format using the Save Images module for further analysis in Matlab?
%defaultVAR04 = Do not save
%infotypeVAR04 = imagegroup indep
GrayscaleNeighborsName = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = Do you want to calculate the extra measures?
%choiceVAR05 = No
%choiceVAR05 = Yes
%inputtypeVAR05 = popupmenu
ExtraMeasures = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%%%VariableRevisionNumber = 5

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a variable,
%%% "OrigImage".
IncomingLabelMatrixImage = CPretrieveimage(handles,['Segmented', ObjectName],ModuleName,'MustBeGray','DontCheckScale');

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines the neighbors for each object.
d = max(2,NeighborDistance+1);
[sr,sc] = size(IncomingLabelMatrixImage);
ImageOfNeighbors = -ones(sr,sc);
NumberOfNeighbors = zeros(max(IncomingLabelMatrixImage(:)),1);
IdentityOfNeighbors = cell(max(IncomingLabelMatrixImage(:)),1);
se = strel('disk',d,0);
if strcmp(ExtraMeasures,'Yes')
    ese = strel('disk',2,0);
    XLocations=handles.Measurements.(ObjectName).Location{handles.Current.SetBeingAnalyzed}(:,1);
    YLocations=handles.Measurements.(ObjectName).Location{handles.Current.SetBeingAnalyzed}(:,2);
end
props = regionprops(IncomingLabelMatrixImage,'PixelIdxList');
NumberOfObjects=max(IncomingLabelMatrixImage(:));
for k = 1:NumberOfObjects
    % Cut patch
    [r,c] = ind2sub([sr sc],props(k).PixelIdxList);
    rmax = min(sr,max(r) + (d+1));
    rmin = max(1,min(r) - (d+1));
    cmax = min(sc,max(c) + (d+1));
    cmin = max(1,min(c) - (d+1));
    p = IncomingLabelMatrixImage(rmin:rmax,cmin:cmax);
    % Extend cell boundary
    pextended = imdilate(p==k,se,'same');
    overlap = p.*pextended;
    if strcmp(ExtraMeasures,'Yes')
        %%% PERCENT TOUCHING %%%
        epextended = imdilate(p,ese,'same');
        x=bwperim(bwlabel(p==k));
        State = warning;
        warning off Matlab:DivideByZero
        y=(imdilate(p~=k & p~=0,ese,'same')+(p==k))./(imdilate(p~=k & p~=0,ese,'same')+(p==k));
        warning(State);
        y(find(isnan(y)))=0;
        z1=[zeros(1,size(y,2));y(1:end-1,:)];
        z2=[y(2:end,:);zeros(1,size(y,2))];
        z3=[zeros(size(y,1),1),y(:,1:end-1)];
        z4=[y(:,2:end),zeros(size(y,1),1)];
        Combined1=z1-x;
        Combined2=z2-x;
        Combined3=z3-x;
        Combined4=z4-x;
        EdgePixels=find(Combined1==-1 | Combined2==-1 | Combined3==-1 | Combined4==-1);
        PercentTouching(k) = ((sum(sum(x))-length(EdgePixels))/sum(sum(x)))*100;

        if NumberOfObjects >= 3
            %%% CLOSEST NEIGHBORS %%%
            CurrentX=XLocations(k);
            CurrentY=YLocations(k);
            XLocationsMinusCurrent=XLocations;
            XLocationsMinusCurrent(k)=[];
            YLocationsMinusCurrent=YLocations;
            YLocationsMinusCurrent(k)=[];
            FirstClosest = dsearch(XLocationsMinusCurrent,YLocationsMinusCurrent,delaunay(XLocationsMinusCurrent,YLocationsMinusCurrent),CurrentX,CurrentY);
            XLocationsMinusFirstClosest=XLocationsMinusCurrent;
            XLocationsMinusFirstClosest(FirstClosest)=[];
            YLocationsMinusFirstClosest=YLocationsMinusCurrent;
            YLocationsMinusFirstClosest(FirstClosest)=[];
            SecondClosest = dsearch(XLocationsMinusFirstClosest,YLocationsMinusFirstClosest,delaunay(XLocationsMinusFirstClosest,YLocationsMinusFirstClosest),CurrentX,CurrentY);
            FirstXVector(k)=XLocationsMinusCurrent(FirstClosest)-CurrentX;
            FirstYVector(k)=YLocationsMinusCurrent(FirstClosest)-CurrentY;
            FirstObjectNumber(k)=IncomingLabelMatrixImage(round(YLocationsMinusCurrent(FirstClosest)),round(XLocationsMinusCurrent(FirstClosest)));
            SecondXVector(k)=XLocationsMinusFirstClosest(SecondClosest)-CurrentX;
            SecondYVector(k)=YLocationsMinusFirstClosest(SecondClosest)-CurrentY;
            SecondObjectNumber(k)=IncomingLabelMatrixImage(round(YLocationsMinusFirstClosest(SecondClosest)),round(XLocationsMinusFirstClosest(SecondClosest)));
        else
            FirstObjectNumber(k)=0;
            FirstXVector(k)=0;
            FirstYVector(k)=0;
            SecondObjectNumber(k)=0;
            SecondXVector(k)=0;
            SecondYVector(k)=0;
        end
    end
    IdentityOfNeighbors{k} = setdiff(unique(overlap(:)),[0,k]);
    NumberOfNeighbors(k) = length(IdentityOfNeighbors{k});
    ImageOfNeighbors(sub2ind([sr sc],r,c)) = NumberOfNeighbors(k);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% MAKE MEASUREMENTS & SAVE TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

if strcmp(ExtraMeasures,'Yes')
    %%% Saves neighbor measurements to handles structure.
    handles.Measurements.(ObjectName).NumberNeighbors(handles.Current.SetBeingAnalyzed) = {[NumberOfNeighbors PercentTouching' FirstObjectNumber' FirstXVector' FirstYVector' SecondObjectNumber' SecondXVector' SecondYVector']};
    handles.Measurements.(ObjectName).NumberNeighborsFeatures = {'Number of neighbors' 'Percent Touching' 'First Closest Object Number' 'First Closest X Vector' 'First Closest Y Vector' 'Second Object Number' 'Second Closest X Vector' 'Second Closest Y Vector'};
else
    %%% Saves neighbor measurements to handles structure.
    handles.Measurements.(ObjectName).NumberNeighbors(handles.Current.SetBeingAnalyzed) = {[NumberOfNeighbors]};
    handles.Measurements.(ObjectName).NumberNeighborsFeatures = {'Number of neighbors'};
end

% This field is different from the usual measurements. To avoid problems with export modules etc we don't
% add a IdentityOfNeighborsFeatures field. It will then be "invisible" to
% export modules, which look for fields with 'Features' in the name.
handles.Measurements.Neighbors.IdentityOfNeighbors(handles.Current.SetBeingAnalyzed) = {IdentityOfNeighbors};

%%% Example: To extract the number of neighbor for objects called Cells, use code like this:
%%% handles.Measurements.Neighbors.IdentityOfNeighborsCells{1}{3}
%%% where 1 is the image number and 3 is the object number. This
%%% yields a list of the objects who are neighbors with Cell object 3.

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    %%% Calculates the ColoredIncomingObjectsImage for displaying in the figure
    %%% window and saving to the handles structure.
    ColoredIncomingObjectsImage = CPlabel2rgb(handles,IncomingLabelMatrixImage);

    %%% Activates the appropriate figure window.
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure(IncomingLabelMatrixImage,'TwoByOne',ThisModuleFigureNumber)
    end
    subplot(2,1,1);
    CPimagesc(ColoredIncomingObjectsImage,handles);
    title(ObjectName)
    subplot(2,1,2);
    CPimagesc(ImageOfNeighbors,handles);
    colormap(handles.Preferences.LabelColorMap)
    colorbar('SouthOutside')
    title([ObjectName,' colored by number of neighbors'])
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE IMAGES TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the grayscale version of objects to the handles structure so
%%% they can be saved to the hard drive, if the user requests. Here, the
%%% background is -1, and the objects range from 0 (if it has no neighbors)
%%% up to the highest number of neighbors. The -1 value makes it
%%% incompatible with the Convert To Image module which expects a label
%%% matrix starting at zero.
if ~strcmpi(GrayscaleNeighborsName,'Do not save')
    handles.Pipeline.(GrayscaleNeighborsName) = ImageOfNeighbors;
end

%%% Saves the grayscale version of objects to the handles structure so
%%% they can be saved to the hard drive, if the user requests. Here, the
%%% scalar value 1 is added to every pixel so that the background is zero
%%% and the objects are from 1 up to the highest number of neighbors, plus
%%% one. This makes the objects compatible with the Convert To Image
%%% module.
if ~strcmpi(ColoredNeighborsName,'Do not save')
    handles.Pipeline.(['Segmented',ColoredNeighborsName]) = ImageOfNeighbors + 1;
end