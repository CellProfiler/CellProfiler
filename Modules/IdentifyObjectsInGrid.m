function handles = IdentifyObjectsInGrid(handles)

% Help for the Identify Objects In Grid module:
% Category: Object Processing
%
% SHORT DESCRIPTION: After a grid has been established by DefineGrid, this
% module will dientify objects within each section of the grid.
% *************************************************************************
%
% This module identifies objects what are in a grid.  It requires that you
% create a grid in an earlier module using the DefineGrid module.  If you
% are using natural shape, the module need to use objects that are already
% identified.  It will merge objects if they were accidentally seperated,
% and properly numbers the objects according to how the grid was originally
% defined.  Note that if an object does not exist, it will create an object
% with a single pixel in the middle of the grid square.
%
% Special note on saving images: Using the settings in this module, object
% outlines can be passed along to the module OverlayOutlines and then saved
% with SaveImages. Objects themselves can be passed along to the object
% processing module ConvertToImage and then saved with SaveImages. This
% module produces several additional types of objects with names that are
% automatically passed along with the following naming structure: (1) The
% unedited segmented image, which includes objects on the edge of the image
% and objects that are outside the size range, can be saved using the name:
% UneditedSegmented + whatever you called the objects (e.g.
% UneditedSegmentedNuclei). (2) The segmented image which excludes objects
% smaller than your selected size range can be saved using the name:
% SmallRemovedSegmented + whatever you called the objects (e.g.
% SmallRemovedSegmented Nuclei).
%
% See also DefineGrid.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne Carpenter
%   Thouis Jones
%   In Han Kang
%   Ola Friman
%   Steve Lowe
%   Joo Han Chang
%   Colin Clarke
%   Mike Lamprecht
%   Susan Ma
%   Wyman Li
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

%%% Reads the current module number, because this is needed to find the
%%% variable values that the user entered.
[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What is the already defined grid?
%infotypeVAR01 = gridgroup
GridName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What would you like to call the newly identified objects?
%defaultVAR02 = Spots
%infotypeVAR02 = objectgroup indep
NewObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Would you like the object to retain their natural shape, be circles with a particular diameter, or rectangles that fill the entire grid?
%choiceVAR03 = Natural Shape
%choiceVAR03 = Circle Natural Location
%choiceVAR03 = Circle Forced Location
%choiceVAR03 = Rectangle
Shape = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 = If you are using natural shape, a Circle with a Natural Location, or using a circle with an automatically calculated diameter, what did you call the objects that were already identified?
%infotypeVAR04 = objectgroup
OldObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%textVAR05 = If you selected the object to be circles, what is the diameter of each object (in pixels)?
%defaultVAR05 = Automatic
Diameter = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = Would you like to save the image of the outlines of the objects? and if so, what would you like to call them?
%defaultVAR06 = Do not save
%infotypeVAR06 = outlinegroup indep
OutlineName = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = If the grid fails, would you like to use the previous grid which worked?
%choiceVAR07 = No
%choiceVAR07 = Yes
FailedGridChoice = char(handles.Settings.VariableValues{CurrentModuleNum,7});
%inputtypeVAR07 = popupmenu

%%%VariableRevisionNumber = 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

try
    Grid = handles.Pipeline.(['Grid_' GridName]);
catch
    error(['Image processing was canceled in the ', ModuleName, ' module because it is unable to find the grid you specified, ', GridName, '.  Make sure you properly defined it using the DefineGrid module earlier.']);
end

TotalHeight = Grid.TotalHeight;
TotalWidth = Grid.TotalWidth;
Cols = Grid.Columns;
Rows = Grid.Rows;
YDiv = Grid.YSpacing;
XDiv = Grid.XSpacing;
Topmost = Grid.YLocationOfLowestYSpot;
Leftmost = Grid.XLocationOfLowestXSpot;
SpotTable = Grid.SpotTable;
VertLinesX = Grid.VertLinesX;
VertLinesY = Grid.VertLinesY;
HorizLinesX = Grid.HorizLinesX;
HorizLinesY = Grid.HorizLinesY;

if strcmp(Shape,'Natural Shape') || strcmp(Shape,'Circle Natural Location') || strcmp(Shape,'Circle Forced Location') && strcmp(Diameter,'Automatic')
    Image = handles.Pipeline.(['Segmented' OldObjectName]);
end

if strmatch('Circle',Shape)
    if strcmp(Diameter,'Automatic')
        tmp = regionprops(Image,'Area');
        Area = cat(1,tmp.Area);
        radius = floor(sqrt(median(Area)/pi));
    else
        radius = floor(str2double(Diameter)/2);
    end

    if strcmp(FailedGridChoice,'Yes')
        measfield = [GridName,'Info'];
        if handles.Current.SetBeingAnalyzed == 1
            featfield = [GridName,'InfoFeatures'];
            handles.Measurements.Image.(featfield){12} = 'GridFailed';
        end
        if (2*radius > YDiv) || (2*radius > XDiv) || (VertLinesX(1,1) < 0) || (HorizLinesY(1,1) < 0)
            if handles.Current.SetBeingAnalyzed == 1
                error(['Image processing was canceled in the ', ModuleName, ' module because the grid you have designed is not working, please check the pipeline.']);
            else
                FailCheck = 1;
                SetNum = 1;
                while FailCheck >= 1
                    PreviousGrid = handles.Measurements.Image.(measfield){handles.Current.SetBeingAnalyzed - SetNum};
                    FailCheck = PreviousGrid(1,12);
                    SetNum = SetNum + 1;
                end
                GridInfo.XLocationOfLowestXSpot = PreviousGrid(1,1);
                GridInfo.YLocationOfLowestYSpot = PreviousGrid(1,2);
                GridInfo.XSpacing = PreviousGrid(1,3);
                GridInfo.YSpacing = PreviousGrid(1,4);
                GridInfo.Rows = PreviousGrid(1,5);
                GridInfo.Columns = PreviousGrid(1,6);
                GridInfo.TotalHeight = TotalHeight;
                GridInfo.TotalWidth = TotalWidth;
                if PreviousGrid(1,9) == 1
                    GridInfo.LeftOrRight = 'Left';
                else
                    GridInfo.LeftOrRight = 'Right';
                end
                if PreviousGrid(1,10) == 1
                    GridInfo.TopOrBottom = 'Top';
                else
                    GridInfo.TopOrBottom = 'Bottom';
                end
                if PreviousGrid(1,11) == 1
                    GridInfo.RowsOrColumns = 'Rows';
                else
                    GridInfo.RowsOrColumns = 'Columns';
                end

                Grid = CPmakegrid(GridInfo);

                Leftmost = PreviousGrid(1,1);
                Topmost = PreviousGrid(1,2);
                XDiv = PreviousGrid(1,3);
                YDiv = PreviousGrid(1,4);
                Rows = PreviousGrid(1,5);
                Cols = PreviousGrid(1,6);
                VertLinesX = Grid.VertLinesX;
                VertLinesY = Grid.VertLinesY;
                HorizLinesX = Grid.HorizLinesX;
                HorizLinesY = Grid.HorizLinesY;
                SpotTable = Grid.SpotTable;
            end
            handles.Measurements.Image.(measfield){handles.Current.SetBeingAnalyzed}(1,12) = 1;
        else
            handles.Measurements.Image.(measfield){handles.Current.SetBeingAnalyzed}(1,12) = 0;
        end
    else
        if (2*radius > YDiv) || (2*radius > XDiv) || (VertLinesX(1,1) < 0) || (HorizLinesY(1,1) < 0)
            error(['Image processing was canceled in the ', ModuleName, ' module because your grid failed. Please check the DefineGrid module to see if your objects were properly identified and the grid looks correct. You MUST have an identified object on each side (right, left, top, bottom) of the grid to work properly. Also, there must be no "extra" objects identified near the edges of the image or it will fail.']);
        end
    end
end

FinalLabelMatrixImage = zeros(TotalHeight,TotalWidth);

for i=1:Cols
    for j=1:Rows
        subregion = FinalLabelMatrixImage(max(1,Topmost - floor(YDiv/2) + (j-1)*YDiv+1):min(Topmost - floor(YDiv/2) + j*YDiv,end),max(1,Leftmost - floor(XDiv/2) + (i-1)*XDiv+1):min(Leftmost - floor(XDiv/2) + i*XDiv,end));
        if strcmp(Shape,'Natural Shape')
            subregion = Image(max(1,Topmost - floor(YDiv/2) + (j-1)*YDiv+1):min(Topmost - floor(YDiv/2) + j*YDiv,end),max(1,Leftmost - floor(XDiv/2) + (i-1)*XDiv+1):min(Leftmost - floor(XDiv/2) + i*XDiv,end));
            subregion=bwlabel(subregion>0);
            props = regionprops(subregion,'Centroid');
            loc = cat(1,props.Centroid);
            for k = 1:size(loc,1)
                if loc(k,1) < size(subregion,2)*.1 || loc(k,1) > size(subregion,2)*.9 || loc(k,2) < size(subregion,1)*.1 || loc(k,2) > size(subregion,1)*.9
                    subregion(subregion == subregion(floor(loc(k,2)),floor(loc(k,1)))) = 0;
                end
            end
            if max(max(subregion))==0
                subregion(floor(end/2),floor(end/2)) = SpotTable(j,i);
            else
                subregion(subregion>0) = SpotTable(j,i);
            end
        elseif strcmp(Shape,'Circle Forced Location')
            subregion(floor(end/2)-radius:floor(end/2)+radius,floor(end/2)-radius:floor(end/2)+radius)=SpotTable(j,i)*getnhood(strel('disk',radius,0));
        elseif strcmp(Shape,'Circle Natural Location')
            subregion = Image(max(1,Topmost - floor(YDiv/2) + (j-1)*YDiv+1):min(Topmost - floor(YDiv/2) + j*YDiv,end),max(1,Leftmost - floor(XDiv/2) + (i-1)*XDiv+1):min(Leftmost - floor(XDiv/2) + i*XDiv,end));
            subregion=bwlabel(subregion>0);
            props = regionprops(subregion,'Centroid');
            loc = cat(1,props.Centroid);
            for k = 1:size(loc,1)
                if loc(k,1) < size(subregion,2)*.1 || loc(k,1) > size(subregion,2)*.9 || loc(k,2) < size(subregion,1)*.1 || loc(k,2) > size(subregion,1)*.9
                    subregion(subregion == subregion(floor(loc(k,2)),floor(loc(k,1)))) = 0;
                end
            end
            if max(max(subregion))==0
                subregion(floor(end/2)-radius:floor(end/2)+radius,floor(end/2)-radius:floor(end/2)+radius)=SpotTable(j,i)*getnhood(strel('disk',radius,0));
            else
                subregion(subregion>0)=1;
                props = regionprops(subregion,'Centroid');
                circle = SpotTable(j,i)*getnhood(strel('disk',radius,0));
                Ymin = max(1,floor(props.Centroid(2))-radius);
                Ymax = min(size(subregion,1),floor(props.Centroid(2))+radius);
                Xmin = max(1,floor(props.Centroid(1))-radius);
                Xmax = min(size(subregion,2),floor(props.Centroid(1))+radius);
                subregion(:,:) = 0;
                subregion(Ymin:Ymax,Xmin:Xmax)=circle(radius-floor(props.Centroid(2))+1+Ymin:radius-floor(props.Centroid(2))+1+Ymax,radius-floor(props.Centroid(1))+1+Xmin:radius-floor(props.Centroid(1))+1+Xmax);
            end
        elseif strcmp(Shape,'Rectangle')
            subregion(:,:) = SpotTable(j,i);
        else
            error(['Image processing was canceled in the ', ModuleName, ' module because the value of Shape is not recognized.']);
        end
        FinalLabelMatrixImage(max(1,Topmost - floor(YDiv/2) + (j-1)*YDiv+1):min(Topmost - floor(YDiv/2) + j*YDiv,end),max(1,Leftmost - floor(XDiv/2) + (i-1)*XDiv+1):min(Leftmost - floor(XDiv/2) + i*XDiv,end))=subregion;
    end
end

%%% Indicate objects in original image and color excluded objects in red
OutlinedObjects1 = bwperim(mod(FinalLabelMatrixImage,2));
OutlinedObjects2 = bwperim(mod(floor(FinalLabelMatrixImage/Rows),2));
OutlinedObjects3 = bwperim(mod(floor(FinalLabelMatrixImage/Cols),2));
OutlinedObjects4 = bwperim(FinalLabelMatrixImage>0);
FinalOutline = OutlinedObjects1 + OutlinedObjects2 + OutlinedObjects3 + OutlinedObjects4;
FinalOutline = logical(FinalOutline>0);

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);

if any(findobj == ThisModuleFigureNumber)
    drawnow
    CPfigure(handles,ThisModuleFigureNumber);
    ColoredLabelMatrixImage = CPlabel2rgb(handles,FinalLabelMatrixImage);
    subplot(2,1,1); CPimagesc(ColoredLabelMatrixImage);
    line(VertLinesX,VertLinesY);
    line(HorizLinesX,HorizLinesY);
    title(sprintf('Identified %s',NewObjectName),'fontsize',handles.Preferences.FontSize);
    subplot(2,1,2); CPimagesc(FinalOutline);
    line(VertLinesX,VertLinesY);
    line(HorizLinesX,HorizLinesY);
    title('Outlined objects','fontsize',handles.Preferences.FontSize);
    set(findobj('type','line'), 'color',[.15 .15 .15])
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

handles.Pipeline.(['Segmented',NewObjectName]) = FinalLabelMatrixImage;
handles.Pipeline.(['UneditedSegmented',NewObjectName]) = FinalLabelMatrixImage;
handles.Pipeline.(['SmallRemovedSegmented',NewObjectName]) = FinalLabelMatrixImage;

%%% Saves the ObjectCount, i.e. the number of segmented objects.
%%% See comments for the Threshold saving above
if ~isfield(handles.Measurements.Image,'ObjectCountFeatures')
    handles.Measurements.Image.ObjectCountFeatures = {};
    handles.Measurements.Image.ObjectCount = {};
end
column = find(~cellfun('isempty',strfind(handles.Measurements.Image.ObjectCountFeatures,NewObjectName)));
if isempty(column)
    handles.Measurements.Image.ObjectCountFeatures(end+1) = {['ObjectCount ', NewObjectName]};
    column = length(handles.Measurements.Image.ObjectCountFeatures);
end
handles.Measurements.Image.ObjectCount{handles.Current.SetBeingAnalyzed}(1,column) = max(FinalLabelMatrixImage(:));

%%% Saves the location of each segmented object
handles.Measurements.(NewObjectName).LocationFeatures = {'CenterX','CenterY'};
tmp = regionprops(FinalLabelMatrixImage,'Centroid');
Centroid = cat(1,tmp.Centroid);
handles.Measurements.(NewObjectName).Location(handles.Current.SetBeingAnalyzed) = {Centroid};

if ~strcmp(OutlineName,'Do not save')
    handles.Pipeline.(OutlineName) = FinalOutline;
end