function handles = DefineGrid(handles)

% Help for the Define Grid module:
% Category: Other
%
% SHORT DESCRIPTION:
% Produces a grid of desired specifications either manually, or
% automatically based on previously identified objects. The grid can then
% be used to make measurements (using Identify Objects in Grid) or to
% display text information (using Display Grid Info) within each
% compartment of the grid.
% *************************************************************************
%
% This module defines the location of a grid that can be used by modules
% downstream. When used in combination with IdentifyObjectsInGrid, it
% allows the measurement of the size, shape, intensity and texture of each
% object in a grid. The grid is defined by the location of marker spots
% (control spots) in the grid, which are either indicated manually or are
% found automatically using previous modules in the pipeline.
%
% If you are using images of plastic plates, it may be useful to precede
% this module with an IdentifyPrimAtuomatic module to find the plastic
% plate, followed by a Crop module to remove the plastic edges of the
% plate, so that the grid can be defined within the smooth portion of the
% plate only. If the plates are not centered in exactly the same position
% from one image to the next, this allows the plates to be identified
% automatically and then cropped so that the interior of the plates, upon
% which the grids will be defined, are always in precise alignment with
% each other. 
%
% Features measured:      Feature Number:
% XLocationOfLowestXSpot |      1
% YLocationOfLowestYSpot |      2
% XSpacing               |      3 
% YSpacing               |      4 
% Rows                   |      5
% Columns                |      6 
% TotalHeight            |      7
% TotalWidth             |      8
% LeftOrRightNum         |      9
% TopOrBottomNum         |     10
% RowsOrColumnsNum       |     11
%
%
% Settings: Most are self-explanatory.
% 
% EACH CYCLE or ONCE: If all of your images are perfectly aligned with each
% other (due to very consistent image acquisition, consistent grid location
% within the plate, and/or automatic cropping precisely within each plate),
% you can define the location of the marker spots ONCE for all of the image
% cycles; if the location of the grid will vary from one image cycle to the
% next then you should define the location of the marker spots for EACH
% CYCLE independently.
%
% MANUAL mode: In MANUAL mode, you manually indicate known locations of
% marker spots in the grid and have the rest of the positions calculated
% from those marks, no matter what the image itself looks like. This mode
% requires manually clicking on a marker spot with the MOUSE for each image
% (if it is not visible you have to click where it ought to be). Or, you
% can type in the pixel COORDINATES of the marker spots. DISTANCE from the
% control spot to the top left corner of the grid: If the marker spot you
% have chosen to use is the top left corner spot of the grid, then specify
% 0,0. As another example, if the control spot is in the same row as the
% grid and is one spot distance to the left of the grid, you could specify
% that the spot is -1,0 spot units away from the top left spot. You can
% also us marker spots that are within the grid. For example, if the 3rd
% row, 3rd column spot is the one you choose to be the marker spot because
% it is always present in images, then you would enter 3,3.
%
% AUTOMATIC mode: If you would like the grid to be defined automatically,
% an IdentifyPrimAutomatic module must be run prior to this module to
% identify the objects which will be used to define the grid. The
% left-most, right-most, top-most, and bottom-most object will be used to
% define the edges of the grid and the rows and columns will be evenly
% spaced between these edges. Note that automatic mode requires that the
% incoming objects are nicely defined - for example, if there is an object
% at the edge of the images that is not really an object that ought to be
% in the grid, a skewed grid will result. You might wish to use a
% FilterByObjectMeasurement module to clean up badly identified objects
% prior to defining the grid. If the spots are slightly out of alignment
% with each other from one image cycle to the next, this allows the
% identification to be a bit flexible and adapt to the real location of the
% spots.
%
% SAVING IMAGES: You can save the grid and numbering produced by this
% module as a color image, on top of the original image. You can then save
% this image using the SaveImages module or use the ColorToGray module to
% separate out the color of interest to combine with other images as
% desired.
%
% See also IdentifyObjectsInGrid, DisplayGridInfo.

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

%textVAR01 = What would you like to call the grid that you define in this module?
%defaultVAR01 = GridBlue
%infotypeVAR01 = gridgroup indep
GridName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = How many rows and columns are in the grid (not counting control spots outside the grid itself)?
%defaultVAR02 = 8,12
RowsCols = char(handles.Settings.VariableValues{CurrentModuleNum,2});
try
    RowsCols = str2num(RowsCols); %#ok Ignore MLint
    Rows = RowsCols(1);
    Columns = RowsCols(2);
catch
    error(['Image processing was canceled in the ', ModuleName, ' module because there is an invalid input for the number of rows and columns.  You need two integers separated by a comma, such as "5,5".']);
end

%textVAR03 = For numbering purposes, is the first spot at the left or right?
%choiceVAR03 = Left
%choiceVAR03 = Right
LeftOrRight = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 = For numbering purposes, is the first spot on the top or bottom?
%choiceVAR04 = Top
%choiceVAR04 = Bottom
TopOrBottom = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%textVAR05 = Would you like to count across first (by rows) or up/down first (by columns)?
%choiceVAR05 = Rows
%choiceVAR05 = Columns
RowsOrColumns = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu

%textVAR06 = Would you like to define a new grid for each image cycle, or define a grid once and use it for all images?
%choiceVAR06 = Each cycle
%choiceVAR06 = Once
EachOrOnce = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%inputtypeVAR06 = popupmenu

%textVAR07 = Would you like to define the grid automatically, based on objects you have identified in a previous module?
%choiceVAR07 = Automatic
%choiceVAR07 = Manual
AutoOrManual = char(handles.Settings.VariableValues{CurrentModuleNum,7});
%inputtypeVAR07 = popupmenu

%textVAR08 = For AUTOMATIC, what are the previously identified objects you want to use to define the grid?
%infotypeVAR08 = objectgroup
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,8});
%inputtypeVAR08 = popupmenu

%textVAR09 = For MANUAL, how would you like to specify where the control spot is?
%choiceVAR09 = Coordinates
%choiceVAR09 = Mouse
ControlSpotMode = char(handles.Settings.VariableValues{CurrentModuleNum,9});
%inputtypeVAR09 = popupmenu

%textVAR10 = For MANUAL or if you are saving an RGB image, what is the original image on which to mark/display the grid?
%infotypeVAR10 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,10});
%inputtypeVAR10 = popupmenu

%textVAR11 = For MANUAL + MOUSE, what is the distance from the control spot to the top left spot in the grid? (X,Y: specify spot units or pixels below)
%defaultVAR11 = 0,0
HorizVertOffset = char(handles.Settings.VariableValues{CurrentModuleNum,11});
try
    HorizVertOffset = str2num(HorizVertOffset);%#ok We want to ignore MLint error checking for this line.
    XOffsetFromControlToTopLeft = HorizVertOffset(1);
    YOffsetFromControlToTopLeft = HorizVertOffset(2);
catch
    error(['Image processing was canceled in the ', ModuleName, ' module because there was an invalid value for the distance from the control spot to the top left spot.  The value needs to be two integers seperated by a comma.']);
end

%textVAR12 = For MANUAL + MOUSE, did you specify the distance to the control spot (above) in spot units or pixels?
%choiceVAR12 = Spot Units
%choiceVAR12 = Pixels
DistanceUnits = char(handles.Settings.VariableValues{CurrentModuleNum,12});
%inputtypeVAR12 = popupmenu

%textVAR13 = For MANUAL + ONCE or MANUAL + MOUSE, what is the spacing, in pixels, between columns (horizontal = X) and rows (vertical = Y)?
%defaultVAR13 = 57,57
HorizVertSpacing = char(handles.Settings.VariableValues{CurrentModuleNum,13});
try
    HorizVertSpacingNumerical = str2num(HorizVertSpacing);%#ok We want to ignore MLint error checking for this line.
    XSpacing = HorizVertSpacingNumerical(1);
    YSpacing = HorizVertSpacingNumerical(2);
catch error(['Image processing was canceled in the ', ModuleName, ' module because your entry for the spacing between columns, rows (horizontal spacing, vertical spacing) was not understood.'])
end

%textVAR14 = For MANUAL + ONCE + COORDINATES, where is the center of the control spot (X,Y pixel location)?
%defaultVAR14 = 57,57
ControlSpot = char(handles.Settings.VariableValues{CurrentModuleNum,14});
try
    ControlSpot = str2num(ControlSpot);%#ok We want to ignore MLint error checking for this line.
    XControlSpot = ControlSpot(1);
    YControlSpot = ControlSpot(2);
catch
    error(['Image processing was canceled in the ', ModuleName, ' module because there was an invalid value for the location of the control spot.  The value needs to be two integers seperated by a comma.']);
end

%textVAR15 - What would you like to call an RGB image with R = the image, G = grid lines, and B = text?
%defaultVAR15 = Do not save
%infotypeVAR15 = imagegroup indep
RGBname = char(handles.Settings.VariableValues{CurrentModuleNum,15});

%textVAR16 = If the gridding fails, would you like to use a previous grid that worked?
%choiceVAR16 = No
%choiceVAR16 = Any Previous
%choiceVAR16 = The First
FailedGridChoice = char(handles.Settings.VariableValues{CurrentModuleNum,16});
%inputtypeVAR16 = popupmenu

%%%VariableRevisionNumber = 3

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% The image to display is retrieved from the handles.
if strcmp(AutoOrManual,'Automatic')
%%% For automatic mode, the previously identified objects are
%%% retrieved from the handles structure.
    try OrigImage = CPretrieveimage(handles,['Segmented' ObjectName],ModuleName);
    catch error(['Image processing was canceled in the ', ModuleName, ' module because you specified automatic mode using the objects you called "', ObjectName, '" and these objects were not found by CellProfiler. Please adjust your pipeline to produce the objects "', ObjectName, '" prior to this ', ModuleName, ' module.'])
    end
elseif strcmp(AutoOrManual,'Manual')
    OrigImage = CPretrieveimage(handles,ImageName,ModuleName);
end

%%% If we are in 'Once' mode and this is not the first image cycle,
%%% then we only need to retrieve values from the handles structure
%%% rather than calculating/asking the user for the values.
if strncmp(EachOrOnce,'Once',4) && handles.Current.SetBeingAnalyzed ~= 1
    GridInfo = handles.Pipeline.(['Grid_' GridName]);
    XLocationOfLowestXSpot = GridInfo.XLocationOfLowestXSpot;
    YLocationOfLowestYSpot = GridInfo.YLocationOfLowestYSpot;
    XSpacing = GridInfo.XSpacing;
    YSpacing = GridInfo.YSpacing;
    Rows = GridInfo.Rows;
    Columns = GridInfo.Columns;
    VertLinesX = GridInfo.VertLinesX;
    VertLinesY = GridInfo.VertLinesY;
    HorizLinesX = GridInfo.HorizLinesX;
    HorizLinesY = GridInfo.HorizLinesY;
    SpotTable = GridInfo.SpotTable;
    TotalHeight = GridInfo.TotalHeight;
    TotalWidth = GridInfo.TotalWidth;
    GridXLocations = GridInfo.GridXLocations;
    GridYLocations = GridInfo.GridYLocations;
    YLocations = GridInfo.YLocations;
    XLocations = GridInfo.XLocations;

%%% Otherwise, everything needs to be asked to the user and/or
%%% calculated. Each of these options should ultimately produce these
%%% variables: XLocationOfLowestXSpot, YLocationOfLowestYSpot,
%%% XSpacing, YSpacing
else
    %%% In Automatic mode, the objects' locations are used to define
    %%% the outer edges of the grid and the proper spacing.
    if strcmp(AutoOrManual,'Automatic') && strcmp(FailedGridChoice,'No')
        tmp = regionprops(OrigImage,'Centroid');
        Location = cat(1,tmp.Centroid);
        %%% Chooses the coordinates of the objects at the farthest edges
        %%% of the incoming image.
        if size(Location,1) > 1
        x = sort(Location(:,1));
        y = sort(Location(:,2));
        else
            error('Either no objects are identified or only one object is identified in the DefineGrid module, neither of which will work. Please make sure your objects are correctly identified.');
        end
        XLocationOfLowestXSpot = floor(min(x));
        YLocationOfLowestYSpot = floor(min(y));
        XLocationOfHighestXSpot = ceil(max(x));
        YLocationOfHighestYSpot = ceil(max(y));
        %%% Calculates the horizontal and vertical spacing based on the
        %%% grid extending from the furthest objects in each direction.
        XSpacing = round((XLocationOfHighestXSpot - XLocationOfLowestXSpot)/(Columns - 1));
        YSpacing = round((YLocationOfHighestYSpot - YLocationOfLowestYSpot)/(Rows - 1));
    elseif strcmp(AutoOrManual,'Automatic') && ~strcmp(FailedGridChoice,'No')
        tmp = regionprops(OrigImage,'Centroid');
        Location = cat(1,tmp.Centroid);
        %%% Chooses the coordinates of the objects at the farthest edges
        %%% of the incoming image.
        if size(Location,1) > 1
            x = sort(Location(:,1));
            y = sort(Location(:,2));
            XLocationOfLowestXSpot = floor(min(x));
            YLocationOfLowestYSpot = floor(min(y));
            XLocationOfHighestXSpot = ceil(max(x));
            YLocationOfHighestYSpot = ceil(max(y));
            %%% Calculates the horizontal and vertical spacing based on the
            %%% grid extending from the furthest objects in each direction.
            XSpacing = round((XLocationOfHighestXSpot - XLocationOfLowestXSpot)/(Columns - 1));
            YSpacing = round((YLocationOfHighestYSpot - YLocationOfLowestYSpot)/(Rows - 1));
        else
            measfield = [GridName,'Info'];
            FailCheck = 1;
            SetNum = 1;
            if strcmp(FailedGridChoice,'The First')
                PreviousGrid = handles.Measurements.Image.(measfield){1};
            end
            if strcmp(FailedGridChoice,'Any Previous')
                while FailCheck >= 1
                    try PreviousGrid = handles.Measurements.Image.(measfield){handles.Current.SetBeingAnalyzed - SetNum};
                    catch error(['Image processing was canceled in the ', ModuleName, ' module because the module went looking for previous functioning grid(s) and could not find it, please check the pipeline.']);
                    end
                    FailCheck = PreviousGrid(1,12);
                    SetNum = SetNum + 1;
                end
            end
            XLocationOfLowestXSpot = PreviousGrid(1,1);
            YLocationOfLowestYSpot = PreviousGrid(1,2);
            XSpacing = PreviousGrid(1,3);
            YSpacing = PreviousGrid(1,4);
         end
    elseif strcmp(AutoOrManual,'Manual')
        if strcmp(ControlSpotMode,'Coordinates')
            if strncmp(EachOrOnce,'Once',4)
                %%% In 'Manual' + 'Coordinates' + 'Once' mode,
                %%% the values from the main GUI are used for
                %%% XControlSpot and YControlSpot and XSpacing and
                %%% YSpacing, so they aren't retrieved here.
                %%% Converts units to pixels if they were given in spot
                %%% units.
                if strcmp(DistanceUnits,'Spot Units') == 1
                    XOffsetFromControlToTopLeft = XOffsetFromControlToTopLeft * XSpacing;
                    YOffsetFromControlToTopLeft = YOffsetFromControlToTopLeft * YSpacing;
                end
                XLocationOfLowestXSpot = XControlSpot + XOffsetFromControlToTopLeft;
                YLocationOfLowestYSpot = YControlSpot + YOffsetFromControlToTopLeft;
            elseif strncmp(EachOrOnce,'Each',4)
                answers = inputdlg({'Enter the X,Y location of the center of the top, left spot in the grid.' 'Enter the vertical and horizontal spacing, separated by a comma.'});
                ControlSpot = str2num(answers{1});
                try
                    XLocationOfLowestXSpot = ControlSpot(1);
                    YLocationOfLowestYSpot = ControlSpot(2);
                catch
                    error(['Image processing was canceled in the ', ModuleName, ' module because the values entered for the the X,Y location of the center of the bottom, left spot in the grid do not make sense.  Note that you need two values separated by a comma.']);
                end
                Spacing = str2num(answers{2});
                try
                    YSpacing = Spacing(1);
                    XSpacing = Spacing(2);
                catch
                    error(['Image processing was canceled in the ', ModuleName, ' module because the values entered for the spacing do not make sense.  Note that you need two values separated by a comma.']);
                end
            end
        elseif strcmp(ControlSpotMode,'Mouse')
            %%% Opens the figure and displays the image so user can
            %%% click on it to mark the control spot.
            ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
            CPfigure(handles,'Image',ThisModuleFigureNumber);
            CPimagesc(OrigImage,handles);
            %%% Sets the top, left of the grid based on mouse clicks.
            title({'Click on the center of the top left control spot, then press Enter.','If you make an error, the Delete or Backspace key will delete the previously selected point.','If multiple points are clicked, the last point clicked will be used.', 'Use Edit > Colormap to adjust the contrast of the image. BE PATIENT!'})
            drawnow
            %pixval
            [x,y] = getpts(ThisModuleFigureNumber);
            if length(x) < 1
                error(['Image processing was canceled in the ', ModuleName, ' module because you must click on one point then press enter.'])
            end
            title('')
            XControlSpot = x(end);
            YControlSpot = y(end);
            %%% Converts units to pixels if they were given in spot
            %%% units.
            if strcmp(DistanceUnits,'Spot Units') == 1
                XOffsetFromControlToTopLeft = XOffsetFromControlToTopLeft * XSpacing;
                YOffsetFromControlToTopLeft = YOffsetFromControlToTopLeft * YSpacing;
            end
            XLocationOfLowestXSpot = XControlSpot + XOffsetFromControlToTopLeft;
            YLocationOfLowestYSpot = YControlSpot + YOffsetFromControlToTopLeft;
        end
    end

    TotalHeight = size(OrigImage,1);
    TotalWidth = size(OrigImage,2);

    GridInfo.XLocationOfLowestXSpot = XLocationOfLowestXSpot;
    GridInfo.YLocationOfLowestYSpot = YLocationOfLowestYSpot;
    GridInfo.XSpacing = XSpacing;
    GridInfo.YSpacing = YSpacing;
    GridInfo.Rows = Rows;
    GridInfo.Columns = Columns;
    GridInfo.TotalHeight = TotalHeight;
    GridInfo.TotalWidth = TotalWidth;
    GridInfo.LeftOrRight = LeftOrRight;
    GridInfo.TopOrBottom = TopOrBottom;
    GridInfo.RowsOrColumns = RowsOrColumns;

    %%% Send data to subfunction to get grid lines.
    Grid = CPmakegrid(GridInfo);

    VertLinesX = Grid.VertLinesX;
    VertLinesY = Grid.VertLinesY;
    HorizLinesX = Grid.HorizLinesX;
    HorizLinesY = Grid.HorizLinesY;
    SpotTable = Grid.SpotTable;
    GridXLocations = Grid.GridXLocations;
    GridYLocations = Grid.GridYLocations;
    YLocations = Grid.YLocations;
    XLocations = Grid.XLocations;
    
    if ~strcmp(RGBname,'Do not save')
        
        tempfig = figure('Color','black','CloseRequestFcn','','visible','off');
        colormap(gray);
        imagesc(OrigImage);  %makes sure the axes are right
        delete(findobj(tempfig,'type','image'));
        
        line(VertLinesX,VertLinesY,'Color','white');
        line(HorizLinesX,HorizLinesY,'Color','white');
        set(gca,'color','black');
        saveas(gcf,'Temp.jpg');
        imageG = imread('Temp.jpg');
        imageG = round(sum(imageG,3)/3);
        imageG = imageG/max(max(imageG));
        imageG = imageG(69:802,157:1088);
        imageG = imresize(imageG,size(OrigImage));
        imageG = im2bw(imageG,graythresh(imageG));
        imageG = ~imageG;
        
        delete(get(gca,'children'));
        TextHandles = text((floor(XLocations+XSpacing/6)),(YLocations+floor(YSpacing/2)),cellstr(num2str(reshape(SpotTable,1,[])'))','color','white','fontsize',handles.Preferences.FontSize);
        set(gca,'visible','off');
        saveas(gcf,'Temp.jpg');
        imageB = imread('Temp.jpg');
        delete(tempfig);
        delete('Temp.jpg');
        imageB = round(sum(imageB,3)/3);
        imageB = imageB/max(max(imageB));
        imageB = imageB(69:802,157:1088);
        imageB = imresize(imageB,size(OrigImage));
        imageB = im2bw(imageB,graythresh(imageB));
        imageB = ~imageB;
        
        imageR = sum(OrigImage,3);
        
        ColorImage(:,:,1) = imageR;
        ColorImage(:,:,2) = imageG;
        ColorImage(:,:,3) = imageB;    
    end
   
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);

if any(findobj == ThisModuleFigureNumber)
    %%% Recreates the figure or Activates the appropriate figure window.
    FigHandle = CPfigure(handles,'Image',ThisModuleFigureNumber);
    %%% Usually this image should be fairly large, so we are pretending it's a
    %%% 2x2 figure window rather than 1x1.
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure(OrigImage,'TwoByTwo',ThisModuleFigureNumber);
    end
    CPimagesc(OrigImage,handles);
    %%% Draws the lines.
    line(VertLinesX,VertLinesY);
    line(HorizLinesX,HorizLinesY);
    set(findobj(FigHandle,'type','line'), 'color',[1 0 0])
    TextToShow = cellstr(num2str(reshape(SpotTable,1,[])'))';
    TextHandles = text((floor(XLocations+XSpacing/6)),(YLocations+floor(YSpacing/2)),TextToShow,'color','yellow','fontsize',handles.Preferences.FontSize);
    drawnow

    %%% Sets the location of Tick marks.
    set(gca, 'XTick', GridXLocations(1,:)+floor(XSpacing/2))
    set(gca, 'YTick', GridYLocations(:,1)+floor(YSpacing/2))

    %%% Sets the Tick Labels.
    if strcmp(LeftOrRight,'Right')
        set(gca, 'XTickLabel',fliplr(1:Columns))
    else
        set(gca, 'XTickLabel',{1:Columns})
    end
    if strcmp(TopOrBottom,'Bottom')
        set(gca, 'YTickLabel',{fliplr(1:Rows)})
    else
        set(gca, 'YTickLabel',{1:Rows})
    end
    set(gca,'XAxisLocation','top');
    set(gca,'YAxisLocation','right');
    TextButtonCallback = [...
        'if strcmp(get(gcbo,''String''),''Hide Text''),'...
        'set(gcbo,''String'',''Show Text'');'...
        'set(get(gcbo,''UserData''),''visible'',''off'');'...
        'else,'...
        'set(gcbo,''String'',''Hide Text'');'...
        'set(get(gcbo,''UserData''),''visible'',''on'');'...
        'end;'];

    GridButtonCallback = [...
        'if strcmp(get(gcbo,''String''),''Hide Grid''),'...
        'set(gcbo,''String'',''Show Grid'');'...
        'set(findobj(gca,''type'',''line''), ''visible'',''off'');'...
        'else,'...
        'set(gcbo,''String'',''Hide Grid'');'...
        'set(findobj(gca,''type'',''line''), ''visible'',''on'');'...
        'end;'];

    TextColorButtonCallback = [...
        'Colors = get(gcbo,''string'');'...
        'Value = get(gcbo,''value'');'...
        'set(get(gcbo,''UserData''),''color'',Colors{Value});'];

    GridColorButtonCallback = [...
        'Colors = get(gcbo,''string'');'...
        'Value = get(gcbo,''value'');'...
        'set(findobj(gca,''type'',''line''),''color'',Colors{Value});'];

    TextButtonHandle = uicontrol(FigHandle,...
        'Units','normalized',...
        'Position',[.5 .02 .13 .04],...
        'String','Hide Text',...
        'BackgroundColor',[.7 .7 .9],...
        'FontSize',handles.Preferences.FontSize,...
        'UserData',TextHandles,...
        'Callback',TextButtonCallback);

    GridButtonHandle = uicontrol(FigHandle,...
        'Units','normalized',...
        'Position',[.15 .02 .13 .04],...
        'String','Hide Grid',...
        'BackgroundColor',[.7 .7 .9],...
        'FontSize',handles.Preferences.FontSize,...
        'Callback',GridButtonCallback);

    Colors = {'Yellow' 'Magenta' 'Cyan' 'Red' 'Green' 'Blue' 'White' 'Black'};

    TextColorButtonHandle = uicontrol(FigHandle,...
        'Units','normalized',...
        'Style','popupmenu',...
        'Position',[.65 .02 .13 .04],...
        'String',Colors,...
        'BackgroundColor',[.7 .7 .9],...
        'FontSize',handles.Preferences.FontSize,...
        'Value',1,...
        'UserData',TextHandles,...
        'Tag','TextColorButton',...
        'Callback',TextColorButtonCallback);

    GridColorButtonHandle = uicontrol(FigHandle,...
        'Units','normalized',...
        'Style','popupmenu',...
        'Position',[.3 .02 .13 .04],...
        'String',Colors,...
        'BackgroundColor',[.7 .7 .9],...
        'FontSize',handles.Preferences.FontSize,...
        'Value',4,...
        'Tag','GridColorButton',...
        'Callback',GridColorButtonCallback);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

GridInfo.XLocationOfLowestXSpot = XLocationOfLowestXSpot;
GridInfo.YLocationOfLowestYSpot = YLocationOfLowestYSpot;
GridInfo.XSpacing = XSpacing;
GridInfo.YSpacing = YSpacing;
GridInfo.Rows = Rows;
GridInfo.Columns = Columns;
GridInfo.VertLinesX = VertLinesX;
GridInfo.VertLinesY = VertLinesY;
GridInfo.HorizLinesX = HorizLinesX;
GridInfo.HorizLinesY = HorizLinesY;
GridInfo.SpotTable = SpotTable;
GridInfo.TotalHeight = TotalHeight;
GridInfo.TotalWidth = TotalWidth;
GridInfo.GridXLocations = GridXLocations;
GridInfo.GridYLocations = GridYLocations;
GridInfo.YLocations = YLocations;
GridInfo.XLocations = XLocations;
GridInfo.LeftOrRight = LeftOrRight;
GridInfo.TopOrBottom = TopOrBottom;

handles.Pipeline.(['Grid_' GridName]) = GridInfo;

%%% We need some way to keep track of these values in
%%% handles.Measurements.Image.GridInfo so we must convert them to numbers.
%%% If you retrieve these values anywhere else, you must convert them back
%%% to strings if you want to send them to CPmakegrid(GridInfo).
if strcmp(LeftOrRight,'Left')
    LeftOrRightNum = 1;
else
    LeftOrRightNum = 0;
end
if strcmp(TopOrBottom,'Top')
    TopOrBottomNum = 1;
else
    TopOrBottomNum = 0;
end
if strcmp(RowsOrColumns,'Rows')
    RowsOrColumnsNum = 1;
else
    RowsOrColumnsNum = 0;
end
GridInfoList = [XLocationOfLowestXSpot,YLocationOfLowestYSpot,XSpacing,YSpacing,Rows,Columns,TotalHeight,TotalWidth,LeftOrRightNum,TopOrBottomNum,RowsOrColumnsNum];

featfield = [GridName,'InfoFeatures'];
measfield = [GridName,'Info'];
GridFeatures = {'XLocationOfLowestXSpot' 'YLocationOfLowestYSpot' 'XSpacing' 'YSpacing' 'Rows' 'Columns' 'TotalHeight' 'TotalWidth' 'LeftOrRightNum' 'TopOrBottomNum' 'RowsOrColumnsNum'};
handles.Measurements.Image.(featfield) = GridFeatures;
handles.Measurements.Image.(measfield){handles.Current.SetBeingAnalyzed} = GridInfoList;

if ~strcmp(RGBname,'Do not save')
    handles.Pipeline.(RGBname) = ColorImage;
end