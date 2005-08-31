function handles = DefineGrid(handles)

% Help for the Define Grid module:
% Category: Other
%
% This module defines a grid that can be used by modules downstream.  If
% you would like the grid to be identified automatically, then you need to
% identify objects before.  If you are using manual, you still need some
% type of picture to get height and width of the image.  Note that
% automatic mode will create a skewed grid if there is an object on the far
% left or far right that is not supposed to be there.
%
% See also IdentifyObjectsInGrid.

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

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%


%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

%textVAR01 = What would you like to call the grid?
%defaultVAR01 = GridBlue
%infotypeVAR01 = gridgroup indep
GridName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = Would you like to create the grid manually, or automatic.  Also, if the grid is defined manual, would you like to create a different grid for each image, or use the same grid for all of them?
%choiceVAR02 = Automatic
%choiceVAR02 = Manual (all)
%choiceVAR02 = Manual (each)
AutoOrManual = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 = If you specified manual above, what is the original image?
%infotypeVAR03 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 = If you specified automatic above, what are the identified objects?
%infotypeVAR04 = objectgroup
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%textVAR05 = How many rows and columns?
%defaultVAR05 = 8,12
RowsCols = char(handles.Settings.VariableValues{CurrentModuleNum,5});
try
    RowsCols = str2num(RowsCols);
    Rows = RowsCols(1);
    Cols = RowsCols(2);
catch
    error('There is an invalid input for the number of rows and columns.  You need two integers seperated by a comma, such as "5,5".');
end

%textVAR06 = If you selected manual (all), what is the vertical and horizontal spacing?
%defaultVAR06 = 57,57
HorizVertSpacing = char(handles.Settings.VariableValues{CurrentModuleNum,6});
try
    HorizVertSpacingNumerical = str2num(HorizVertSpacing);%#ok We want to ignore MLint error checking for this line.
    VertSpacing = HorizVertSpacingNumerical(1);
    HorizSpacing = HorizVertSpacingNumerical(2);
catch error('Image processing was canceled because your entry for the spacing between rows, columns (vertical spacing, horizontal spacing) was not understood.')
end

%textVAR07 = If using manual mode, how would you like to specify where the control spot is?
%choiceVAR07 = Coordinates
%choiceVAR07 = Mouse
ControlSpotMode = char(handles.Settings.VariableValues{CurrentModuleNum,7});
%inputtypeVAR07 = popupmenu

%textVAR08 = If manual (all) and coordinates, here is the control spot?
%defaultVAR08 = 57,57
ControlSpot = char(handles.Settings.VariableValues{CurrentModuleNum,8});
try
    ControlSpot = str2num(ControlSpot);%#ok We want to ignore MLint error checking for this line.
    XControlSpot = ControlSpot(1);
    YControlSpot = ControlSpot(2);
catch
    error('There was an invalid value for the location of the control spot.  The value needs to be two integers seperated by a comma.');
end


%textVAR09 = If you are using manual (all) and coordinates, are you going to be specifying the distance (the next option) in units or pixels?
%choiceVAR09 = Spots
%choiceVAR09 = Pixels
DistanceUnits = char(handles.Settings.VariableValues{CurrentModuleNum,9});
%inputtypeVAR09 = popupmenu

%textVAR10 = If you are using manual (all) and coordinates, what is the distance to the top left spot from the control spot?
%defaultVAR10 = 0,0
HorizVertOffset = char(handles.Settings.VariableValues{CurrentModuleNum,10});
try
    HorizVertOffset = str2num(HorizVertOffset);%#ok We want to ignore MLint error checking for this line.
    HorizOffset = HorizVertOffset(1);
    VertOffset = HorizVertOffset(2);
catch
    error('There was an invalid value for the distance from the control spot to the top left spot.  The value needs to be two integers seperated by a comma.');
end

%textVAR11 = For number purposes, is the first spot at the left or right?
%choiceVAR11 = Left
%choiceVAR11 = Right
LeftOrRight = char(handles.Settings.VariableValues{CurrentModuleNum,11});
%inputtypeVAR11 = popupmenu

%textVAR12 = For numbering purposes, is the first spot on the top or bottom
%choiceVAR12 = Top
%choiceVAR12 = Bottom
TopOrBottom = char(handles.Settings.VariableValues{CurrentModuleNum,12});
%inputtypeVAR12 = popupmenu

%textVAR13 = Would you like to count by rows or columns?
%choiceVAR13 = Rows
%choiceVAR13 = Columns
RowsOrColumns = char(handles.Settings.VariableValues{CurrentModuleNum,13});
%inputtypeVAR13 = popupmenu



%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if strcmp(AutoOrManual,'Automatic')
    FinalLabelMatrixImage = handles.Pipeline.(['Segmented' ObjectName]);
    tmp = regionprops(FinalLabelMatrixImage,'Area');
    Area = cat(1,tmp.Area);

    tmp = regionprops(FinalLabelMatrixImage,'Centroid');
    Location = cat(1,tmp.Centroid);

else
    FinalLabelMatrixImage = handles.Pipeline.(ImageName);
end


if strcmp(AutoOrManual,'Automatic') %Rightmost and Lowermost should not be used later.
    x = sort(Location(:,1));
    y = sort(Location(:,2));

    Leftmost = floor(min(x));
    Topmost = floor(min(y));
    Rightmost = floor(max(x));
    Lowermost = floor(max(y));

    %Assuming the user declared the number of rows and cols
    XDiv = floor((Rightmost - Leftmost)/(Cols - 1));
    YDiv = floor((Lowermost - Topmost)/(Rows - 1));

elseif strcmp(AutoOrManual,'Manual (all)')
    YDiv = VertSpacing;
    XDiv = HorizSpacing;
    Leftmost = XControlSpot + HorizOffset;
    Topmost = YControlSpot + VertOffset;
elseif strcmp(AutoOrManual,'Manual (each)')
    answers = inputdlg({'Enter the control spot' 'Enter the distance from the control spot to the top left corner' 'Enter the vertical and horizontal spacing'});
    ControlSpot = str2num(answers{1});
    Offset = str2num(answers{2});
    try
        Topmost = ControlSpot(1)+Offset(1);
        Leftmost = ControlSpot(2)+Offset(2);
    catch
        error('One of the values did not make sense (either control spot or offset).  Note that you need a comma in both.');
    end
    Spacing = str2num(answers{3});
    
    try
        YDiv = Spacing(1);
        xDiv = Spacing(2);
    catch
        error('The value of spacing did not make sense.  Note that you need two values seperated by a comma.');
    end
else
    error('The value of AutoOrManual is not recognized');
end

LinearNumbers = 1:Rows*Cols;
if strcmp(RowsOrColumns,'Columns')
    SpotTable = reshape(LinearNumbers,Rows,Cols);
elseif strcmp(RowsOrColumns,'Rows')
    SpotTable = reshape(LinearNumbers,Cols,Rows);
    SpotTable = SpotTable';
else
    error('The value of RowsOrColumns is not recognized');
end
if strcmp(LeftOrRight,'Right')
    SpotTable = fliplr(SpotTable);
end
if strcmp(TopOrBottom,'Bottom')
    SpotTable = flipud(SpotTable);
end

%%% Calculates the locations for all the sample labelings (whether it is
%%% numbers, spot identifying information, or coordinates).
GridXLocations = SpotTable;
for g = 1:size(GridXLocations,2)
    GridXLocations(:,g) = Leftmost + (g-1)*XDiv - floor(XDiv/2);
end
%%% Converts to a single column.
XLocations = reshape(GridXLocations, 1, []);

% %%% Shifts if necessary.
% if strcmp(LeftOrRight,'R') == 1
% XLocations = XLocations - (NumberRows-1)*VertSpacing;
% end
%%% Same routine for Y.
GridYLocations = SpotTable;
for h = 1:size(GridYLocations,1)
    GridYLocations(h,:) = Topmost + (h-1)*YDiv - floor(YDiv/2);
end
YLocations = reshape(GridYLocations, 1, []);
    
TotalHeight = size(FinalLabelMatrixImage,1);
TotalWidth = size(FinalLabelMatrixImage,2);

%%% Calculates the lines.
VertLinesX(1,:) = [GridXLocations(1,:),GridXLocations(1,end)+XDiv];
VertLinesX(2,:) = [GridXLocations(1,:),GridXLocations(1,end)+XDiv];
VertLinesY(1,:) = repmat(0,1,size(GridXLocations,2)+1);
VertLinesY(2,:) = repmat(TotalHeight,1,size(GridXLocations,2)+1);
HorizLinesY(1,:) = [GridYLocations(:,1)',GridYLocations(end,1)+YDiv];
HorizLinesY(2,:) = [GridYLocations(:,1)',GridYLocations(end,1)+YDiv];
HorizLinesX(1,:) = repmat(0,1,size(GridXLocations,1)+1);
HorizLinesX(2,:) = repmat(TotalWidth,1,size(GridXLocations,1)+1);

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%


fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);

if any(findobj == ThisModuleFigureNumber)    
    CPfigure(handles,ThisModuleFigureNumber);    
    imagesc(FinalLabelMatrixImage);CPcolormap(handles);  
    %%% Draws the lines.
    line(VertLinesX,VertLinesY);
    line(HorizLinesX,HorizLinesY);
    set(findobj('type','line'), 'color',[.15 .15 .15])
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

temp.Leftmost = Leftmost;
temp.Topmost = Topmost;
temp.XDiv = XDiv;
temp.YDiv = YDiv;
temp.Rows = Rows;
temp.Cols = Cols;
temp.VertLinesX = VertLinesX;
temp.VertLinesY = VertLinesY;
temp.HorizLinesX = HorizLinesX;
temp.HorizLinesY = HorizLinesY;
temp.SpotTable = SpotTable;
temp.TotalHeight = TotalHeight;
temp.TotalWidth = TotalWidth;


handles.Pipeline.(['Grid_' GridName]) = temp;