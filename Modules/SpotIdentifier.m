function handles = AlgSpotIdentifier(handles)

%%% Reads the current algorithm number, since this is needed to find
%%% the variable values that the user entered.
CurrentAlgorithm = handles.currentalgorithm;

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow

%textVAR01 = Do you want to load spot information from a file?
%defaultVAR01 = N
fieldname = ['Vvariable',CurrentAlgorithm,'_01'];
LoadSpotIdentifiers = handles.(fieldname);

%textVAR09 = To save the resulting rotated image, enter a filename (no extension)
%defaultVAR09 = N
fieldname = ['Vvariable',CurrentAlgorithm,'_09'];
SaveImage = handles.(fieldname);
%textVAR10 =  Otherwise, leave as "N". To save or display other images, press Help button
%textVAR11 = In what file format do you want to save the image? Do not include a period
%defaultVAR11 = tif
fieldname = ['Vvariable',CurrentAlgorithm,'_11'];
FileFormat = handles.(fieldname);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow
%%% Checks whether the file format the user entered is readable by Matlab.
IsFormat = imformats(FileFormat);
if isempty(IsFormat) == 1
    error('The image file type entered in the Spot Identifier module is not recognized by Matlab. Or, you may have entered a period in the box. For a list of recognizable image file formats, type "imformats" (no quotes) at the command line in Matlab.','Error')
end

%%% The following code prevents the warning message in the Matlab
%%% main window: "Warning: Image is too big to fit on screen":
%%% This warning appears due to the truesize command which
%%% rescales an image so that it fits on the screen.  Truesize is often
%%% called by imshow.
iptsetpref('TruesizeWarning','off')

[FileName,PathName] = uigetfile('*.*', 'Choose the image');
if FileName == 0
    return
end
FileLocation = strcat(PathName,FileName);
try
    OriginalImage = imread(FileLocation);
catch error('There was a problem importing the image you selected')
end
%%% Determines the figure number to display in.
fieldname = ['figurealgorithm',CurrentAlgorithm];
ThisAlgFigureNumber = handles.(fieldname);
FigureHandle = figure(ThisAlgFigureNumber); imagesc(OriginalImage); colormap(gray), axis image
Answer = questdlg('Does the image need to be rotated prior to labeling the spots?', 'Rotate?', 'Yes', 'No', 'Cancel', 'Yes');
if strcmp(Answer, 'Cancel') == 1
    return
end
if strcmp(Answer, 'Yes') == 1
    Answer2 = questdlg('After closing this window by clicking OK, click on the lower left marker in the image, then the lower right marker, then press the Enter key. If you make an error, the Delete or Backspace key will delete the previously selected point.', 'Choose marker points', 'OK', 'Cancel', 'OK');
    if strcmp(Answer2, 'Cancel') == 1
        return
    end
    [x,y] = getpts(FigureHandle);
    if length(x) ~=2
        errordlg('SpotIdentifier was canceled because you must click on two points then press enter.')
        return
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

    PatienceHandle = waitbar(0,'Please be patient; Image rotation in progress');
    drawnow
    RotatedImage = imrotate(OriginalImage, -AngleToRotateDegrees);
    figure(ThisAlgFigureNumber); imagesc(RotatedImage), colormap(gray), axis image
        title('Rotated Image'), pixval
    delete(PatienceHandle)
end % Goes with questdlg regarding image rotation.


Prompts = {'Enter the X coordinate of the top left marker', 'Enter the Y coordinate of the top left marker'};
Defaults = {'0', '0'};
Answers = inputdlg(Prompts, 'Top left marker', 1, Defaults);
TopLeftX = str2num(Answers{1});
TopLeftY = str2num(Answers{2});

% Answer3 = questdlg('After closing this window by clicking OK, click on the top left marker in the image then press the Enter key. If you make an error, the Delete or Backspace key will delete the previously selected point.', 'Choose marker point', 'OK', 'Cancel', 'OK');
% if strcmp(Answer3, 'Cancel') == 1
%     return
% end
% 
% pixval
% [x,y] = getpts(gcf);
% if length(x) ~=1
%     errordlg('SpotIdentifier was canceled because you must click on one point then press enter.')
%     return
% end
% TopLeftX = x(1);
% TopLeftY = y(1);

%%% Asks the user about the slide layout.
Prompts = {'Enter the number of rows to be marked', 'Enter the number of columns to be marked', 'Enter the spacing between columns (horizontal spacing)', 'Enter the spacing between rows (vertical spacing)', 'Enter the horizontal distance from the top left marker to the nearestwhit spot.', 'Enter the vertical distance from the top left marker to the nearest spot.','Is the first spot at the Left or Right?', 'Is the first spot at the Top or Bottom?'};
Defaults = {'40', '140', '57', '57', '57', '0','L','B'};
Answers = inputdlg(Prompts, 'Specifications for this slide', 1, Defaults);
NumberRows = str2num(Answers{1});
NumberColumns = str2num(Answers{2});
HorizSpacing = str2num(Answers{3});
VertSpacing = str2num(Answers{4});
HorizOffset = str2num(Answers{5});
VertOffset = str2num(Answers{6});
LeftOrRight = upper(Answers{7});
TopOrBottom = upper(Answers{8});

OriginX = TopLeftX + HorizOffset;
OriginY = TopLeftY + VertOffset;
NumberSpots = NumberRows*NumberColumns;

%%% Calculates the numbers and shapes them properly for display.
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

         
%%% Retrieves the spot identifying info from a file, if requested.
if strcmp(upper(LoadSpotIdentifiers),'Y') == 1
    [FileName,PathName] = uigetfile('*.*', 'Choose the file containing the spot identifying information.');
    if FileName == 0
        return
    end
    Answer = inputdlg('What is the name of the excel sheet with the data of interest?');
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
    %%% used immediately prior to the figure(ThisAlgFigureNumber) line,
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
    set(ThisAlgFigureNumber, 'Position', NewFigureSize)
    axis image
    ShowGridButtonFunction = 'Handles = findobj(''type'',''line''); set(Handles,''visible'',''on''); clear Handles';
    ShowGridButton_handle = uicontrol('Style', 'pushbutton', ...
        'String', 'Show', 'Position', [10 6 45 20], ...
        'Callback', ShowGridButtonFunction, 'parent',ThisAlgFigureNumber);
    HideGridButtonFunction = 'Handles = findobj(''type'',''line''); set(Handles,''visible'',''off''); clear Handles';
    HideGridButton_handle = uicontrol('Style', 'pushbutton', ...
        'String', 'Hide', 'Position', [60 6 45 20], ...
        'Callback', HideGridButtonFunction, 'parent',ThisAlgFigureNumber);
    ChangeGridButtonFunction = 'Handles = findobj(''type'',''line''); propedit(Handles); clear Handles';
    ChangeGridButton_handle = uicontrol('Style', 'pushbutton', ...
        'String', 'Change', 'Position', [110 6 45 20], ...
        'Callback', ChangeGridButtonFunction, 'parent',ThisAlgFigureNumber);
    ShowCoordinatesButtonFunction = 'Handles = findobj(''UserData'',''PositionListHandles''); set(Handles,''visible'',''on''); clear Handles';
    ShowCoordinatesButton_handle = uicontrol('Style', 'pushbutton', ...
        'String', 'Show', 'Position', [170 6 45 20], ...
        'Callback', ShowCoordinatesButtonFunction, 'parent',ThisAlgFigureNumber);
    HideCoordinatesButtonFunction = 'Handles = findobj(''UserData'',''PositionListHandles''); set(Handles,''visible'',''off''); clear Handles';
    HideCoordinatesButton_handle = uicontrol('Style', 'pushbutton', ...
        'String', 'Hide', 'Position', [220 6 45 20], ...
        'Callback', HideCoordinatesButtonFunction, 'parent',ThisAlgFigureNumber);
    ChangeCoordinatesButtonFunction = 'Handles = findobj(''UserData'',''PositionListHandles''); propedit(Handles); clear Handles';
    ChangeCoordinatesButton_handle = uicontrol('Style', 'pushbutton', ...
        'String', 'Change', 'Position', [270 6 45 20], ...
        'Callback', ChangeCoordinatesButtonFunction, 'parent',ThisAlgFigureNumber);

    ShowSpotIdentifyingInfoButtonFunction = 'Handles = findobj(''UserData'',''SpotIdentifyingInfoHandles''); if isempty(Handles) == 1, warndlg(''No Spot Identifying information was loaded.''), else set(Handles,''visible'',''on''); end, clear Handles';
    ShowSpotIdentifyingInfoButton_handle = uicontrol('Style', 'pushbutton', ...
        'String', 'Show', 'Position', [330 6 45 20], ...
        'Callback', ShowSpotIdentifyingInfoButtonFunction, 'parent',ThisAlgFigureNumber);
    HideSpotIdentifyingInfoButtonFunction = 'Handles = findobj(''UserData'',''SpotIdentifyingInfoHandles''); if isempty(Handles) == 1, warndlg(''No Spot Identifying information was loaded.''), else set(Handles,''visible'',''off''); end, clear Handles';
    HideSpotIdentifyingInfoButton_handle = uicontrol('Style', 'pushbutton', ...
        'String', 'Hide', 'Position', [380 6 45 20], ...
        'Callback', HideSpotIdentifyingInfoButtonFunction, 'parent',ThisAlgFigureNumber);
    ChangeSpotIdentifyingInfoButtonFunction = 'Handles = findobj(''UserData'',''SpotIdentifyingInfoHandles''); if isempty(Handles) == 1, warndlg(''No Spot Identifying information was loaded.''), else propedit(Handles); end, clear Handles';
    ChangeSpotIdentifyingInfoButton_handle = uicontrol('Style', 'pushbutton', ...
        'String', 'Change', 'Position', [430 6 45 20], ...
        'Callback', ChangeSpotIdentifyingInfoButtonFunction, 'parent',ThisAlgFigureNumber);


    ChangeColormapButtonFunction = 'ImageHandle = findobj(gca, ''type'',''image''); propedit(ImageHandle)';
    ChangeColormapButton_handle = uicontrol('Style', 'pushbutton', ...
        'String', 'Change', 'Position', [490 6 45 20], ...
        'Callback', ChangeColormapButtonFunction, 'parent',ThisAlgFigureNumber);
    TextHandle1 = uicontrol('Parent',ThisAlgFigureNumber, ...
        'BackgroundColor',get(ThisAlgFigureNumber,'Color'), ...
        'Position',[10 28 145 14], ...
        'HorizontalAlignment','center', ...
        'String','Gridlines:', ...
        'Style','text');
    TextHandle2 = uicontrol('Parent',ThisAlgFigureNumber, ...
        'BackgroundColor',get(ThisAlgFigureNumber,'Color'), ...
        'Position',[170 28 145 14], ...
        'HorizontalAlignment','center', ...
        'String','Coordinates:', ...
        'Style','text');
    TextHandle3 = uicontrol('Parent',ThisAlgFigureNumber, ...
        'BackgroundColor',get(ThisAlgFigureNumber,'Color'), ...
        'Position',[330 28 145 14], ...
        'HorizontalAlignment','center', ...
        'String','Spot identifying info:', ...
        'Style','text');
    TextHandle4 = uicontrol('Parent',ThisAlgFigureNumber, ...
        'BackgroundColor',get(ThisAlgFigureNumber,'Color'), ...
        'Position',[485 28 55 14], ...
        'HorizontalAlignment','center', ...
        'String','Colormap:', ...
        'Style','text');

    [TotalHeight,TotalWidth] = size(OriginalImage(:,:,1));
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
        set(gca, 'XTickLabel',[fliplr([1:NumberColumns])])
    else
        set(gca, 'XTickLabel',{[1:NumberColumns]})
    end
    if strcmp(TopOrBottom,'B') == 1
        set(gca, 'YTickLabel',{fliplr([1:NumberRows])})
    else
        set(gca, 'YTickLabel',{[1:NumberRows]})
    end

    %%% Adds the toolbar to the figure, which is lost after some of the
    %%% steps above.
    set(ThisAlgFigureNumber,'toolbar','figure')

    %%% Executes pending figure-related commands so that the results are
    %%% displayed.
    drawnow
%%% The following code turns the warning message back on that
%%% I turned off when the GUI was launched.
%%% "Warning: Image is too big to fit on screen":
%%% This warning appears due to the truesize command which
%%% rescales an image so that it fits on the screen.  Truesize is often
%%% called by imshow.
iptsetpref('TruesizeWarning','on')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE PROCESSED IMAGE TO HARD DRIVE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Determine whether the user wanted to save the corrected image
%%% by comparing their entry "SaveImage" with "N" (after
%%% converting SaveImage to uppercase).
if strcmp(upper(SaveImage),'N') ~= 1
    %%% Save the image to the hard drive.
    imwrite(RotatedImage, [SaveImage,'.',FileFormat], FileFormat);
end

drawnow

%%%%%%%%%%%
%%% HELP %%%
%%%%%%%%%%%

%%%%% Help for the Spot Identifier module: 

%%%%% . 
%%%%% DISPLAYING AND SAVING PROCESSED IMAGES 
%%%%% PRODUCED BY THIS IMAGE ANALYSIS MODULE:
%%%%% Note: Images saved using the boxes in the main CellProfiler window
%%%%% will be saved in the default directory specified at the top of the
%%%%% CellProfiler window.
%%%%% .
%%%%% If you want to save other processed images, open the m-file for this 
%%%%% image analysis module, go to the line in the
%%%%% m-file where the image is generated, and there should be 2 lines
%%%%% which have been inactivated.  These are green comment lines that are
%%%%% indented. To display an image, remove the percent sign before
%%%%% the line that says "figure, imshow...". This will cause the image to
%%%%% appear in a fresh display window for every image set. To save an
%%%%% image to the hard drive, remove the percent sign before the line
%%%%% that says "imwrite..." and adjust the file type and appendage to the
%%%%% file name as desired.  When you have finished removing the percent
%%%%% signs, go to File > Save As and save the m file with a new name.
%%%%% Then load the new image analysis module into the CellProfiler as
%%%%% usual.
%%%%% Please note that not all of these imwrite lines have been checked for
%%%%% functionality: it may be that you will have to alter the format of
%%%%% the image before saving.  Try, for example, adding the uint8 command:
%%%%% uint8(Image) surrounding the image prior to using the imwrite command
%%%%% if the image is not saved correctly.