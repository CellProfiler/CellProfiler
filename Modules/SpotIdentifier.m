function handles = AlgSpotIdentifier(handles)

% Help for the Spot Identifier module:
% Sorry, this module has not yet been documented.  
%
% It works for our basic needs right now, but improvements/fixes need to be
% made:
% (0) I have not tested all of the various ways to rotate and mark the top,
% left corner spot.
% (1) I am not confident that the offsets and flipping left/right and
% top/bottom are accurate at the moment.
% (2) Loading gene names from an Excel file works only for a PC at the
% moment, although I think it can be adjusted by skipping or adding an
% entry to the imported data.
% (3) Right now the numbers are in columns, and I want to add the option of
% rows.
% (4) When clicking "Show coordinates" or "Show sample info", all the numbers across the whole
% image are displayed, which takes a while and slows down zoom functions.
% I would like to be able to click on the image and show the coordinates
% for that spot.
% (5) We would like to be able to show several images within the same
% figure window and toggle between them, so that controls and different
% wavelength images can be compared.

% The contents of this file are subject to the Mozilla Public License Version
% 1.1 (the "License"); you may not use this file except in compliance with
% the License. You may obtain a copy of the License at
% http://www.mozilla.org/MPL/
%
% Software distributed under the License is distributed on an "AS IS" basis,
% WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
% for the specific language governing rights and limitations under the
% License.
%
%
% The Original Code is the Spot Identifier module.
%
% The Initial Developer of the Original Code is
% Whitehead Institute for Biomedical Research
% Portions created by the Initial Developer are Copyright (C) 2003,2004
% the Initial Developer. All Rights Reserved.
%
% Contributor(s):
%   Anne Carpenter <carpenter@wi.mit.edu>
%   Thouis Jones   <thouis@csail.mit.edu>
%   In Han Kang    <inthek@mit.edu>
%
% $Revision$

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow

%%% Reads the current algorithm number, since this is needed to find
%%% the variable values that the user entered.
CurrentAlgorithm = handles.currentalgorithm;
CurrentAlgorithmNum = str2double(handles.currentalgorithm);

%textVAR01 = What did you call the image to be rotated and labeled with spot information?
%defaultVAR01 = OrigBlue
ImageName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,1});

%textVAR02 = Do you want to rotate the image? (No, C = Coordinates, M = Mouse)
%defaultVAR02 = No
RotateMethod = char(handles.Settings.Vvariable{CurrentAlgorithmNum,2});

%textVAR03 = What do you want to call the rotated image?
%defaultVAR03 = RotatedImage
RotatedImageName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,3});

%textVAR04 = Mark the top, left corner of the grid by coordinates (C), or by mouse (M)?
%defaultVAR04 = C
MarkingMethod = char(handles.Settings.Vvariable{CurrentAlgorithmNum,4});

%textVAR05 = Enter the number of rows, columns
%defaultVAR05 = 40,140
RowsColumns = char(handles.Settings.Vvariable{CurrentAlgorithmNum,5});
%%% Extracts the rows and columns from the user's input.
try
    RowsColumnsNumerical = str2double(RowsColumns);
    NumberRows = RowsColumnsNumerical(1);
    NumberColumns = RowsColumnsNumerical(2);
catch error('Image processing was canceled because your entry for rows, columns in the Spot Identifier module was not understood.')
end

%textVAR06 = Enter the spacing between rows, columns (vertical spacing, horizontal spacing)
%defaultVAR06 = 57,57
HorizVertSpacing = char(handles.Settings.Vvariable{CurrentAlgorithmNum,6});
%%% Extracts the vertical and horizontal spacing from the user's input.
try
    HorizVertSpacingNumerical = str2double(HorizVertSpacing);
    VertSpacing = HorizVertSpacingNumerical(1);
    HorizSpacing = HorizVertSpacingNumerical(2);
catch error('Image processing was canceled because your entry for the spacing between rows, columns (vertical spacing, horizontal spacing) in the Spot Identifier module was not understood.')
end

%textVAR07 = Enter the distance from the top left marker to the center of the nearest spot (vertical, horizontal)
%defaultVAR07 = 57,0
HorizVertOffset = char(handles.Settings.Vvariable{CurrentAlgorithmNum,7});
%%% Extracts the vertical and horizontal offset from the user's input.
try
    HorizVertOffsetNumerical = str2double(HorizVertOffset);
    VertOffset = HorizVertOffsetNumerical(1);
    HorizOffset = HorizVertOffsetNumerical(2);
catch error('Image processing was canceled because your entry for the distance from the top left marker to the center of the nearest spot (vertical, horizontal) in the Spot Identifier module was not understood.')
end

%textVAR08 = Is the first spot at the Left or Right?
%defaultVAR08 = L
LeftOrRight = char(handles.Settings.Vvariable{CurrentAlgorithmNum,8});
LeftOrRight = upper(LeftOrRight);

%textVAR09 = Is the first spot at the Bottom or Top?
%defaultVAR09 = B
TopOrBottom = char(handles.Settings.Vvariable{CurrentAlgorithmNum,9});
TopOrBottom = upper(TopOrBottom);

%textVAR10 = Do you want to load spot information from a file (e.g. gene names)?
%defaultVAR10 = N
LoadSpotIdentifiers = char(handles.Settings.Vvariable{CurrentAlgorithmNum,10});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a variable,
%%% "OrigImageToBeAnalyzed".
fieldname = ['dOT', ImageName];
%%% Checks whether the image exists in the handles structure.
if isfield(handles, fieldname) == 0
    error(['Image processing has been canceled. Prior to running the Spot Identifier module, you must have previously run an algorithm to load an image. You specified in the Spot Identifier module that this image was called ', ImageName, ' which should have produced a field in the handles structure called ', fieldname, '. The Spot Identifier module cannot find this image.']);
end
OriginalImage = handles.(fieldname);

if handles.setbeinganalyzed == 1
    %%% Determines the figure number to display in.
    fieldname = ['figurealgorithm',CurrentAlgorithm];
    ThisAlgFigureNumber = handles.(fieldname);
    FigureHandle = figure(ThisAlgFigureNumber); ImageHandle = imagesc(OriginalImage); colormap(gray), axis image, pixval %#ok We want to ignore MLint error checking for this line.
else FigureHandle = figure; ImageHandle = imagesc(OriginalImage); colormap(gray), axis image, pixval %#ok We want to ignore MLint error checking for this line.
end
drawnow
RotateMethod = upper(RotateMethod);
if strncmp(RotateMethod, 'N',1) == 1
    RotatedImage = OriginalImage;
elseif strncmp(RotateMethod, 'M',1) == 1
    Answer2 = questdlg('After closing this window by clicking OK, click on the lower left marker in the image, then the lower right marker, then press the Enter key. If you make an error, the Delete or Backspace key will delete the previously selected point.','Rotate image using the mouse','OK','Cancel','OK');
    waitfor(Answer2)
    if strcmp(Answer2, 'Cancel') == 1
        error('Image processing was canceled during the Spot Identifier module.')
    end
    [x,y] = getpts(FigureHandle);
    if length(x) ~=2
        error('The Spot Identifier was canceled because you must click on two points then press enter.')
    end
    LowerLeftX = x(1);
    LowerRightX = x(2);
    LowerLeftY = y(1);
    LowerRightY = y(2);
    HorizLeg = LowerRightX - LowerLeftX;
    VertLeg = LowerLeftY - LowerRightY;
    Hypotenuse = sqrt(HorizLeg^2 + VertLeg^2);
    AngleToRotateRadians = asin(VertLeg/Hypotenuse);
    AngleToRotateDegrees = AngleToRotateRadians*180/pi;
    PatienceHandle = msgbox('Please be patient; Image rotation in progress');
    drawnow
    RotatedImage = imrotate(OriginalImage, -AngleToRotateDegrees);
    figure(FigureHandle); ImageHandle = imagesc(RotatedImage), colormap(gray), axis image;
    title('Rotated Image'), pixval
    try %#ok We want to ignore MLint error checking for this line.
        delete(PatienceHandle)
    end 
elseif strncmp(RotateMethod, 'C',1) == 1
    %%% Rotates the image based on user-entered coordinates.
    Prompts = {'Enter the X coordinate of the lower left marker', 'Enter the Y coordinate of the lower left marker', 'Enter the X coordinate of the lower right marker', 'Enter the Y coordinate of the lower right marker'};
    Height = size(OriginalImage,1);
    Width = size(OriginalImage,2);
    Defaults = {'0', num2str(Height), num2str(Width), num2str(Height)};
    Answers = inputdlg(Prompts, 'Enter coordinates', 1, Defaults);
    if isempty(Answers) == 1
        error('Image processing was canceled during the Spot Identifier module.')
    end
    LowerLeftX = str2double(Answers{1});
    LowerLeftY = str2double(Answers{2});
    LowerRightX = str2double(Answers{3});
    LowerRightY = str2double(Answers{4});
    HorizLeg = LowerRightX - LowerLeftX;
    VertLeg = LowerLeftY - LowerRightY;
    Hypotenuse = sqrt(HorizLeg^2 + VertLeg^2);
    AngleToRotateRadians = asin(VertLeg/Hypotenuse);
    AngleToRotateDegrees = AngleToRotateRadians*180/pi;
    PatienceHandle = msgbox('Please be patient; Image rotation in progress');
    drawnow
    RotatedImage = imrotate(OriginalImage, -AngleToRotateDegrees);
    figure(FigureHandle); ImageHandle = imagesc(RotatedImage), colormap(gray), axis image
    title('Rotated Image'), pixval
    try, delete(PatienceHandle), end
else
    error('Image processing was canceled because your entry relating to image rotation was not one of the options: No, C, or M.')
end
drawnow
if strncmp(MarkingMethod,'C',1) == 1
    %%% Sets the top, left of the grid based on user-entered coordinates.
    Prompts = {'Enter the X coordinate of the top left marker', 'Enter the Y coordinate of the top left marker'};
    Defaults = {'0', '0'};
    Answers = inputdlg(Prompts, 'Top left marker', 1, Defaults);
    if isempty(Answers) == 1
        error('Image processing was canceled during the Spot Identifier module.')
    end
    TopLeftX = str2double(Answers{1});
    TopLeftY = str2double(Answers{2});
elseif strncmp(MarkingMethod,'M',1) == 1
    %%% Sets the top, left of the grid based on mouse clicks.
    Answer3 = questdlg('After closing this window by clicking OK, click on the top left marker in the image then press the Enter key. If you make an error, the Delete or Backspace key will delete the previously selected point.', 'Choose marker point', 'OK', 'Cancel', 'OK');
    if strcmp(Answer3, 'Cancel') == 1
        error('Image processing was canceled during the Spot Identifier module.')
    end
    pixval
    [x,y] = getpts(gcf);
    if length(x) ~=1
        error('SpotIdentifier was canceled because you must click on one point then press enter.')
    end
    TopLeftX = x(1);
    TopLeftY = y(1);
else
    error('Image processing was canceled because your entry relating to marking the top, left corner of the grid was not one of the options: C or M.')
end
drawnow
%%% Calculates the numbers to be displayed on the image and shapes them
%%% properly for display.
OriginX = TopLeftX + HorizOffset;
OriginY = TopLeftY + VertOffset;
NumberSpots = NumberRows*NumberColumns;
Numbers = 1:NumberSpots;
NumbersGridShape = reshape(Numbers, NumberRows, NumberColumns);
if strcmp(TopOrBottom,'B') == 1
    NumbersGridShape = flipud(NumbersGridShape);
end
if strcmp(LeftOrRight,'R') == 1
    NumbersGridShape = fliplr(NumbersGridShape);
end
%%% Converts to a single column.
LinearNumbers = reshape(NumbersGridShape, 1, NumberSpots);
%%% Converts to text for display purposes.
for i = 1:length(LinearNumbers)
    PositionList{i} = num2str(LinearNumbers(i));,
end
drawnow
%%% Calculates the locations for all the sample labelings (whether it is
%%% numbers, spot identifying information, or coordinates).
GridXLocations = NumbersGridShape;
for g = 1:NumberColumns
    GridXLocations(:,g) = OriginX + (g-1)*HorizSpacing;
end
%%% Converts to a single column.
XLocations = reshape(GridXLocations, 1, NumberSpots);
% %%% Shifts if necessary.
% if strcmp(LeftOrRight,'R') == 1
% XLocations = XLocations - (NumberRows-1)*VertSpacing;
% end
%%% Same routine for Y.
GridYLocations = NumbersGridShape;
for h = 1:NumberRows
    GridYLocations(h,:) = OriginY + (h-1)*VertSpacing;
end
YLocations = reshape(GridYLocations, 1, NumberSpots);
% %%% Shifts if necessary.
% if strcmp(TopOrBottom,'B') == 1
% YLocations = YLocations - (NumberColumns-1)*HorizSpacing;
% end

%%% Draws the Numbers, though they are initially invisible until the
%%% user clicks "Show".
PositionListHandles = text(XLocations, YLocations, PositionList, ...
    'HorizontalAlignment','center', 'Color', 'red','visible','off', ...
    'UserData','PositionListHandles');
drawnow
%%% Retrieves the spot identifying info from a file, if requested.
if strcmp(upper(LoadSpotIdentifiers),'Y') == 1
    [FileName,PathName] = uigetfile('*.*', 'Choose the file containing the spot identifying information.');
    if FileName == 0
        error('Image processing was canceled during the Spot Identifier module.')
    end
    Answer = inputdlg('What is the name of the Excel sheet with the data of interest?');
    if isempty(Answer) == 1
        error('Image processing was canceled during the Spot Identifier module.')
    end
    cd(PathName)
    SheetName = Answer{1};
    [data_numbers,SpotIdentifyingInfo]=xlsread(FileName,SheetName);
    SpotIdentifyingInfo = SpotIdentifyingInfo(:,2:end);
    SpotIdentifyingInfo = SpotIdentifyingInfo(2:end,:);
    %%% Determines the number of rows and columns for later use.
    [NumberRowsSpotIdentifyingInfo, NumberColumnsSpotIdentifyingInfo] = size(SpotIdentifyingInfo);
    if NumberRowsSpotIdentifyingInfo ~= NumberRows
        warndlg('NumberRowsSpotIdentifyingInfo does not match NumberRows.')
        error(['There were ', num2str(NumberRowsSpotIdentifyingInfo), ' rows and ', num2str(NumberColumnsSpotIdentifyingInfo), ' columns of data imported, but you specified that there are ', num2str(NumberRows), ' rows and ', num2str(NumberColumns), ' columns.']);
    end
    if NumberColumnsSpotIdentifyingInfo ~= NumberColumns
        warndlg('NumberColumnsSpotIdentifyingInfo does not match NumberColumns.')
        error(['There were ', num2str(NumberRowsSpotIdentifyingInfo), ' rows and ', NumberColumnsSpotIdentifyingInfo, ' columns of data imported, but you specified that there are ', NumberRows, ' rows and ', NumberColumns, ' .']);
    end
    %%% Draws the SpotIdentifyingInfo, though they are initially invisible until the
    %%% user clicks "Show".
    SpotIdentifyingInfoHandles = text(XLocations, YLocations, SpotIdentifyingInfo, ...
        'HorizontalAlignment','center', 'Color', 'white','visible','off', ...
        'UserData','SpotIdentifyingInfoHandles');
end

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%

%%% The "drawnow" function executes any pending figure window-related
%%% commands.  In general, Matlab does not update figure windows
%%% until breaks between image analysis modules, or when a few select
%%% commands are used. "figure" and "drawnow" are two of the commands
%%% that allow Matlab to pause and carry out any pending figure window-
%%% related commands (like zooming, or pressing timer pause or cancel
%%% buttons or pressing a help button.)  If the drawnow command is not
%%% used immediately prior to the figure(FigureHandle) line,
%%% then immediately after the figure line executes, the other commands
%%% that have been waiting are executed in the other windows.  Then,
%%% when Matlab returns to this module and goes to the subplot line,
%%% the figure which is active is not necessarily the correct one.
%%% This results in strange things like the subplots appearing in the
%%% timer window or in the wrong figure window, or in help dialog boxes.
drawnow
%%% Sets the figure to take up most of the screen.
ScreenSize = get(0,'ScreenSize');
NewFigureSize = [30,60, ScreenSize(3)-60, ScreenSize(4)-150];
set(FigureHandle, 'Position', NewFigureSize)
axis image
ShowGridButtonFunction = 'Handles = findobj(''type'',''line''); set(Handles,''visible'',''on''); clear Handles';
uicontrol('Style', 'pushbutton', ...
    'String', 'Show', 'Position', [10 6 45 20], ...
    'Callback', ShowGridButtonFunction, 'parent',gcf);
HideGridButtonFunction = 'Handles = findobj(''type'',''line''); set(Handles,''visible'',''off''); clear Handles';
uicontrol('Style', 'pushbutton', ...
    'String', 'Hide', 'Position', [60 6 45 20], ...
    'Callback', HideGridButtonFunction, 'parent',gcf);
ChangeGridButtonFunction = 'Handles = findobj(''type'',''line''); try, propedit(Handles), catch, msgbox(''A bug in Matlab is preventing this function from working. Service Request #1-RR6M1''), end; clear Handles';
uicontrol('Style', 'pushbutton', ...
    'String', 'Change', 'Position', [110 6 45 20], ...
    'Callback', ChangeGridButtonFunction, 'parent',gcf);
ShowCoordinatesButtonFunction = 'Handles = findobj(''UserData'',''PositionListHandles''); set(Handles,''visible'',''on''); clear Handles';
uicontrol('Style', 'pushbutton', ...
    'String', 'Show', 'Position', [170 6 45 20], ...
    'Callback', ShowCoordinatesButtonFunction, 'parent',gcf);
HideCoordinatesButtonFunction = 'Handles = findobj(''UserData'',''PositionListHandles''); set(Handles,''visible'',''off''); clear Handles';
uicontrol('Style', 'pushbutton', ...
    'String', 'Hide', 'Position', [220 6 45 20], ...
    'Callback', HideCoordinatesButtonFunction, 'parent',gcf);
ChangeCoordinatesButtonFunction = 'Handles = findobj(''UserData'',''PositionListHandles''); try, propedit(Handles), catch, msgbox(''A bug in Matlab is preventing this function from working. Service Request #1-RR6M1''), end; clear Handles';
uicontrol('Style', 'pushbutton', ...
    'String', 'Change', 'Position', [270 6 45 20], ...
    'Callback', ChangeCoordinatesButtonFunction, 'parent',gcf);

ShowSpotIdentifyingInfoButtonFunction = 'Handles = findobj(''UserData'',''SpotIdentifyingInfoHandles''); if isempty(Handles) == 1, warndlg(''No Spot Identifying information was loaded.''), else set(Handles,''visible'',''on''); end, clear Handles';
uicontrol('Style', 'pushbutton', ...
    'String', 'Show', 'Position', [330 6 45 20], ...
    'Callback', ShowSpotIdentifyingInfoButtonFunction, 'parent',gcf);
HideSpotIdentifyingInfoButtonFunction = 'Handles = findobj(''UserData'',''SpotIdentifyingInfoHandles''); if isempty(Handles) == 1, warndlg(''No Spot Identifying information was loaded.''), else set(Handles,''visible'',''off''); end, clear Handles';
uicontrol('Style', 'pushbutton', ...
    'String', 'Hide', 'Position', [380 6 45 20], ...
    'Callback', HideSpotIdentifyingInfoButtonFunction, 'parent',gcf);
ChangeSpotIdentifyingInfoButtonFunction = 'Handles = findobj(''UserData'',''SpotIdentifyingInfoHandles''); if isempty(Handles) == 1, warndlg(''No Spot Identifying information was loaded.''), else try, propedit(Handles), catch, msgbox(''A bug in Matlab is preventing this function from working. Service Request #1-RR6M1''), end; end, clear Handles';
uicontrol('Style', 'pushbutton', ...
    'String', 'Change', 'Position', [430 6 45 20], ...
    'Callback', ChangeSpotIdentifyingInfoButtonFunction, 'parent',gcf);

ChangeColormapButtonFunction = 'ImageHandle = findobj(gca, ''type'',''image''); if strcmp(get(ImageHandle,''UserData''),''Color'') == 1, msgbox(''This image was loaded as a color image, so the colormap cannot be changed. You can use an RGB Split or RGB to Grayscale module to change the format of the image prior to running the Spot Identifier module.''), else propedit(ImageHandle), end';
uicontrol('Style', 'pushbutton', ...
    'String', 'Change', 'Position', [490 6 45 20], ...
    'Callback', ChangeColormapButtonFunction, 'parent',gcf);
%%% Text1
uicontrol('Parent',gcf, ...
    'BackgroundColor',get(gcf,'Color'), ...
    'Position',[10 28 145 14], ...
    'HorizontalAlignment','center', ...
    'String','Gridlines:', ...
    'Style','text');
%%% Text2
uicontrol('Parent',gcf, ...
    'BackgroundColor',get(gcf,'Color'), ...
    'Position',[170 28 145 14], ...
    'HorizontalAlignment','center', ...
    'String','Coordinates:', ...
    'Style','text');
%%% Text3
uicontrol('Parent',gcf, ...
    'BackgroundColor',get(gcf,'Color'), ...
    'Position',[330 28 145 14], ...
    'HorizontalAlignment','center', ...
    'String','Spot identifying info:', ...
    'Style','text');
%%% Text4
uicontrol('Parent',gcf, ...
    'BackgroundColor',get(gcf,'Color'), ...
    'Position',[485 28 55 14], ...
    'HorizontalAlignment','center', ...
    'String','Colormap:', ...
    'Style','text');

SizeOfImage = size(OriginalImage);
TotalHeight = SizeOfImage(1);
TotalWidth = SizeOfImage(2);

if length(SizeOfImage) == 3
    set(ImageHandle,'UserData','Color')
else set(ImageHandle,'UserData','Gray')
end

%%% Draws the grid on the image.  The 0.5 accounts for the fact that
%%% pixels are labeled where the middle of the pixel is a whole number,
%%% and the left hand side of each pixel is 0.5.

%%% Draws the Vertical Lines.
VertLinesX(1,:) = [GridXLocations(1,:),GridXLocations(1,end)+HorizSpacing];
VertLinesX(2,:) = [GridXLocations(1,:),GridXLocations(1,end)+HorizSpacing];
VertLinesX = VertLinesX - HorizSpacing/2;
VertLinesY(1,:) = repmat(0,1,NumberColumns+1);
VertLinesY(2,:) = repmat(TotalHeight,1,NumberColumns+1);
line(VertLinesX,VertLinesY)

%%% Draws the Horizontal Lines.
HorizLinesY(1,:) = [GridYLocations(:,1)',GridYLocations(end,1)+VertSpacing];
HorizLinesY(2,:) = [GridYLocations(:,1)',GridYLocations(end,1)+VertSpacing];
HorizLinesY = HorizLinesY - VertSpacing/2;
HorizLinesX(1,:) = repmat(0,1,NumberRows+1);
HorizLinesX(2,:) = repmat(TotalWidth,1,NumberRows+1);
line(HorizLinesX,HorizLinesY)

%%% Sets the line color.
Handles = findobj('type','line');
set(Handles, 'color',[.15 .15 .15])

%%% Sets the location of Tick marks.
set(gca, 'XTick', GridXLocations(1,:))
set(gca, 'YTick', GridYLocations(:,1))

%%% Sets the Tick Labels.
if strcmp(LeftOrRight,'R') == 1
    set(gca, 'XTickLabel',[fliplr(1:NumberColumns)])
else
    set(gca, 'XTickLabel',{1:NumberColumns})
end
if strcmp(TopOrBottom,'B') == 1
    set(gca, 'YTickLabel',{fliplr(1:NumberRows)})
else
    set(gca, 'YTickLabel',{1:NumberRows})
end

%%% Adds the toolbar to the figure, which is lost after some of the
%%% steps above.
set(FigureHandle,'toolbar','figure')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the adjusted image to the handles structure so it can be used by
%%% subsequent algorithms.
fieldname = ['dOT', RotatedImageName];
handles.(fieldname) = RotatedImage;

if strncmp(RotateMethod, 'N', 1) ~= 1
%%% Saves the Rotation coordinates to the handles structure so they are
%%% saved in the measurements file.
fieldname = ['dMTRotationLowerLeftX', ImageName];
handles.(fieldname)(handles.setbeinganalyzed) = {LowerLeftX};
fieldname = ['dMTRotationLowerRightX', ImageName];
handles.(fieldname)(handles.setbeinganalyzed) = {LowerRightX};
fieldname = ['dMTRotationLowerLeftY', ImageName];
handles.(fieldname)(handles.setbeinganalyzed) = {LowerLeftY};
fieldname = ['dMTRotationLowerRightY', ImageName];
handles.(fieldname)(handles.setbeinganalyzed) = {LowerRightY};
end

%%% Saves the top, left marker locations to the handles structure so they are
%%% saved in the measurements file.
fieldname = ['dMTTopLeftX', ImageName];
handles.(fieldname)(handles.setbeinganalyzed) = {TopLeftX};
fieldname = ['dMTTopLeftY', ImageName];
handles.(fieldname)(handles.setbeinganalyzed) = {TopLeftY};