function handles = AlgImageTiler5(handles)

%%% Reads the current algorithm number, since this is needed to find 
%%% the variable values that the user entered.
CurrentAlgorithm = handles.currentalgorithm;
CurrentAlgorithmNum = str2num(handles.currentalgorithm);

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow

%textVAR01 = What did you call the images to be tiled?
%defaultVAR01 = OrigBlue
ImageName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,1});
%textVAR02 = What do you want to call the tiled image?
%defaultVAR02 = TiledImage
TiledImageName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,2});
%textVAR03 = Number of rows to display (leave "A" to calculate automatically)
%defaultVAR03 = A
NumberRows = char(handles.Settings.Vvariable{CurrentAlgorithmNum,3});
%textVAR04 = Number of columns to display (leave "A" to calculate automatically)
%defaultVAR04 = A
NumberColumns = char(handles.Settings.Vvariable{CurrentAlgorithmNum,4});
%textVAR05 = Are the first two images arranged in a row or a column?
%defaultVAR05 = C
RowOrColumn = char(handles.Settings.Vvariable{CurrentAlgorithmNum,5});
%textVAR06 = Is the first image at the bottom or the top?
%defaultVAR06 = T
TopOrBottom = char(handles.Settings.Vvariable{CurrentAlgorithmNum,6});
%textVAR07 = Is the first image at the left or the right?
%defaultVAR07 = L
LeftOrRight = char(handles.Settings.Vvariable{CurrentAlgorithmNum,7});
%textVAR08 = What fraction should the images be sized (the resolution will be changed)?
%defaultVAR08 = .1
SizeChange = char(handles.Settings.Vvariable{CurrentAlgorithmNum,8});
%textVAR09 = To save the resulting tiled image, enter a filename (no extension) 
%defaultVAR09 = N
SaveImage = char(handles.Settings.Vvariable{CurrentAlgorithmNum,9});
%textVAR10 =  Otherwise, leave as "N". To save or display other images, press Help button
%textVAR11 = In what file format do you want to save the image? Do not include a period
%defaultVAR11 = tif
FileFormat = char(handles.Settings.Vvariable{CurrentAlgorithmNum,11});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow
%%% Checks whether the file format the user entered is readable by Matlab.
IsFormat = imformats(FileFormat);
if isempty(IsFormat) == 1
    error('The image file type entered in the Image Tiler module is not recognized by Matlab. Or, you may have entered a period in the box. For a list of recognizable image file formats, type "imformats" (no quotes) at the command line in Matlab.','Error')
end
%%% Read (open) the image you want to analyze and assign it to a variable,
%%% "OrigImage".
fieldname = ['dOT', ImageName];
%%% Check whether the image to be analyzed exists in the handles structure.
if isfield(handles, fieldname) == 0
    %%% If the image is not there, an error message is produced.  The error
    %%% is not displayed: The error function halts the current function and
    %%% returns control to the calling function (the analyze all images
    %%% button callback.)  That callback recognizes that an error was
    %%% produced because of its try/catch loop and breaks out of the image
    %%% analysis loop without attempting further modules.
    error(['Image processing was canceled because the Image Tiler module could not find the input image.  It was supposed to be named ', ImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end

%%% If the resulting tiled image is to be saved, check whether the file
%%% name of images will result in overwriting a file, or in a file name
%%% that contains spaces.
if strcmp(upper(SaveImage),'N') ~= 1
    A = isspace(SaveImage);
    if any(A) == 1
        error('Image processing was canceled because you have entered one or more spaces for the tiled image file name in the Image Tiler module.  If you do not want to save the object outlines image to the hard drive, type "N" into the appropriate box.')
        return
    end
    %%% Checks whether the specified file name will overwrite an
    %%% existing file. 
    OutputFileOverwrite = exist([cd,'/',SaveImage]);
    if OutputFileOverwrite ~= 0
        Answer = questdlg(['A file with the name ', SaveImage, ' already exists. Do you want to overwrite it?'],'Confirm file overwrite','Yes','No','No');
        if strcmp(Answer,'No') == 1;
            error('Image processing was canceled.')
            return
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%

%%% The first time the module is run, the tiling is performed.
if handles.setbeinganalyzed == 1
    %%% Make note of the current directory so the module can return to it
    %%% at the end of this module.
    CurrentDirectory = cd;
    
    %%% Checks whether any sample info has been loaded.
    if isfield(handles, 'headings') == 1
        %%% Retrieves the Sample Info (only the first column is used).
        SampleInfo = handles.headings(1);
    else SampleInfo = [];
    end
    
    %%% Retrieve the path where the images are stored from the handles
    %%% structure.
    fieldname = ['dOTPathName', ImageName];
    try PathName = handles.(fieldname);
    catch error('Image processing was canceled because the Image Tiler module must be run using images straight from a load images module (i.e. the images cannot have been altered by other image processing modules). This is because the Image Tiler module calculates an illumination correction image based on all of the images before correcting each individual image as CellProfiler cycles through them. One solution is to process the entire batch of images using the image analysis modules preceding this module and save the resulting images to the hard drive, then start a new stage of processing from this Image Tiler module onward.')
    end
    %%% Change to that directory.
    cd(PathName)
    %%% Retrieve the list of filenames where the images are stored from the
    %%% handles structure.
    fieldname = ['dOTFileList', ImageName];
    FileList = handles.(fieldname);
    %%% Checks whether the number of entries in the SampleInfo is equal to the
    %%% number of images.
    if isempty(SampleInfo) ~= 1
        if length(SampleInfo) ~= length(FileList)
            error(['You have ', num2str(length(SampleInfo)), ' lines of sample information loaded, but ', num2str(length(FileList)), ' images loaded.'])
            return
        end
    end
    NumberOfImages = length(FileList);
    if strcmp(upper(NumberRows),'A') == 1 & strcmp(upper(NumberColumns),'A')== 1
        %%% Calculates the square root in order to determine the dimensions
        %%% of the display grid.
        SquareRoot = sqrt(NumberOfImages);
        %%% Converts the result to an integer.
        NumberRows = fix(SquareRoot);
        NumberColumns = ceil((NumberOfImages)/NumberRows);
    elseif strcmp(upper(NumberRows),'A') == 1
        NumberColumns = str2num(NumberColumns);
        NumberRows = ceil((NumberOfImages)/NumberColumns);
    elseif strcmp(upper(NumberColumns),'A') == 1 
        NumberRows = str2num(NumberRows);
        NumberColumns = ceil((NumberOfImages)/NumberRows);
    else NumberColumns = str2num(NumberColumns);
        NumberRows = str2num(NumberRows);
    end
    if NumberRows*NumberColumns > NumberOfImages;
        Answer = questdlg(['You have specified ', num2str(NumberRows), ' rows and ', num2str(NumberColumns), ' columns (=',num2str(NumberRows*NumberColumns),' images), but there are ', num2str(length(FileList)), ' images loaded. The image locations at the end of the grid for which there is no image data will be displayed as black. Do you want to continue?'],'Continue?','Yes','No','Yes');
        if strcmp(Answer,'No') == 1
            %%% This line will "cancel" processing after the first time through this
            %%% module.  Without the following cancel line, the module will run X
            %%% times, where X is the number of files in the current directory.  
            set(handles.timertexthandle,'string','Cancel')
            return
        end
        FileList(length(FileList)+1:NumberRows*NumberColumns) = {'none'};
        SampleInfo(length(FileList)+1:NumberRows*NumberColumns) = {'none'};
    elseif NumberRows*NumberColumns < NumberOfImages;
        Answer = questdlg(['You have specified ', num2str(NumberRows), ' rows and ', num2str(NumberColumns), ' columns (=',num2str(NumberRows*NumberColumns),' images), but there are ', num2str(length(FileList)), ' images loaded. Images at the end of the list will not be displayed. Do you want to continue?'],'Continue?','Yes','No','Yes');
        if strcmp(Answer,'No') == 1
            %%% This line will "cancel" processing after the first time through this
            %%% module.  Without the following cancel line, the module will run X
            %%% times, where X is the number of files in the current directory.  
            set(handles.timertexthandle,'string','Cancel')
            return
        end
        FileList(NumberRows*NumberColumns+1:NumberOfImages) = [];
        if isempty(SampleInfo) ~= 1
            SampleInfo(length(FileList)+1:NumberRows*NumberColumns) = {'none'};
        end
    end
    
    if strcmp(upper(RowOrColumn),'R') == 1
        NewFileList = reshape(FileList, NumberColumns, NumberRows);
        NewFileList = NewFileList';
        if isempty(SampleInfo) ~= 1
            NewSampleInfo = reshape(SampleInfo, NumberColumns, NumberRows);
            NewSampleInfo = NewSampleInfo';
        end
    elseif strcmp(upper(RowOrColumn),'C') == 1
        NewFileList = reshape(FileList, NumberRows, NumberColumns);
        if isempty(SampleInfo) ~= 1
            
            NewSampleInfo = reshape(SampleInfo, NumberRows, NumberColumns);
        end
    else error('You must enter "R" or "C" to select whether the first two images are in a row or a column relative to each other')
        return
    end
    NumberOfImages = NumberColumns*NumberRows;
    global CancelButton_handle
    CancelButtonFunction = 'global CancelButton_handle, set(CancelButton_handle, ''string'', ''Canceling''), clear CancelButton_handle';
    WaitbarHandle = waitbar(0,'Tiling images...');
    WaitbarPosition = get(WaitbarHandle,'position');
    set(WaitbarHandle, 'CloseRequestFcn', 'global CancelButton_handle, set(CancelButton_handle, ''string'', ''Canceling''), clear CancelButton_handle')
    CancelButton_handle = uicontrol('Style', 'pushbutton', ...
        'String', 'Cancel', 'Position', [10 WaitbarPosition(4)-22 100 20], ...
        'Callback', CancelButtonFunction, 'parent',WaitbarHandle);
    ImageSize = size(imresize(im2double(imread(char(NewFileList(1,1)))),SizeChange));
    ImageHeight = ImageSize(1);
    ImageWidth = ImageSize(2);
    TotalWidth = NumberColumns*ImageWidth;
    TotalHeight = NumberRows*ImageHeight;
    TiledImage(TotalHeight,TotalWidth) = 0;  

    for i = 1:NumberRows,
        for j = 1:NumberColumns,
            FileName = NewFileList(i,j);
            %%% In case there are more image slots than there are images,
            %%% the 'none' images are displayed as all black (zeros).
            if strcmp(char(FileName),'none') == 1
                CurrentImage = imresize(zeros(size(CurrentImage)),SizeChange);
            else
                CurrentImage = im2double(imresize(imread(char(FileName)),SizeChange));
                %%% Flips the image left to right or top to bottom if
                %%% necessary.  The entire image will be flipped at the
                %%% end.
                if strcmp(LeftOrRight,'R') == 1
                    CurrentImage = fliplr(CurrentImage);
                end
                if strcmp(TopOrBottom,'B') == 1
                    CurrentImage = flipud(CurrentImage);
                end
            end
            TiledImage((ImageHeight*(i-1))+(1:ImageHeight),(ImageWidth*(j-1))+(1:ImageWidth)) = CurrentImage;
            
            ImageNumber = (i-1)*NumberColumns + j;
            ImagesRemaining = NumberOfImages - ImageNumber;
            waitbar(ImageNumber/NumberOfImages, WaitbarHandle)
            CurrentText = get(CancelButton_handle, 'string');
            if strcmp(CurrentText, 'Canceling') == 1
                delete(WaitbarHandle)
                %%% Determines the figure number to delete.
                fieldname = ['figurealgorithm',CurrentAlgorithm];
                ThisAlgFigureNumber = handles.(fieldname);
                delete(ThisAlgFigureNumber)
                error('Image tiling was canceled')
                return
            end
        end
    end
    if strcmp(LeftOrRight,'R') == 1
        NewFileList = fliplr(NewFileList);
        if isempty(SampleInfo) ~= 1
            
            NewSampleInfo = fliplr(NewSampleInfo);
        end
        TiledImage = fliplr(TiledImage);
    end
    if strcmp(TopOrBottom,'B') == 1
        NewFileList = flipud(NewFileList);
        if isempty(SampleInfo) ~= 1
            NewSampleInfo = flipud(NewSampleInfo);
        end
        TiledImage = flipud(TiledImage);
    end
    
    delete(WaitbarHandle)
    %%% This line will "cancel" processing after the first time through this
    %%% module.  Without the following cancel line, the module will run X
    %%% times, where X is the number of files in the current directory.  
    set(handles.timertexthandle,'string','Cancel')
    
    %%% Returns to the original directory.
    cd(CurrentDirectory)
end
%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Note: Everything between the "if" and "end" is not carried out if the 
%%% user has closed
%%% the figure window, so do not do any important calculations here.
%%% Otherwise an error message will be produced if the user has closed the
%%% window but you have attempted to access data that was supposed to be
%%% produced by this part of the code.

%%% Determines the figure number to display in.
fieldname = ['figurealgorithm',CurrentAlgorithm];
ThisAlgFigureNumber = handles.(fieldname);
%%% Check whether that figure is open. This checks all the figure handles
%%% for one whose handle is equal to the figure number for this algorithm.
if any(findobj == ThisAlgFigureNumber) == 1;
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
    %%% Activates the appropriate figure window.
    figure(ThisAlgFigureNumber);
    %%% Displays the image.
    imagesc(TiledImage)
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
    ShowFileNamesButtonFunction = 'Handles = findobj(''UserData'',''FileNameTextHandles''); set(Handles,''visible'',''on''); clear Handles';
    ShowFileNamesButton_handle = uicontrol('Style', 'pushbutton', ...
        'String', 'Show', 'Position', [170 6 45 20], ...
        'Callback', ShowFileNamesButtonFunction, 'parent',ThisAlgFigureNumber);
    HideFileNamesButtonFunction = 'Handles = findobj(''UserData'',''FileNameTextHandles''); set(Handles,''visible'',''off''); clear Handles';
    HideFileNamesButton_handle = uicontrol('Style', 'pushbutton', ...
        'String', 'Hide', 'Position', [220 6 45 20], ...
        'Callback', HideFileNamesButtonFunction, 'parent',ThisAlgFigureNumber);
    ChangeFileNamesButtonFunction = 'Handles = findobj(''UserData'',''FileNameTextHandles''); propedit(Handles); clear Handles';
    ChangeFileNamesButton_handle = uicontrol('Style', 'pushbutton', ...
        'String', 'Change', 'Position', [270 6 45 20], ...
        'Callback', ChangeFileNamesButtonFunction, 'parent',ThisAlgFigureNumber);

    ShowSampleInfoButtonFunction = 'Handles = findobj(''UserData'',''SampleInfoTextHandles''); if isempty(Handles) == 1, warndlg(''No sample information was loaded.''), else set(Handles,''visible'',''on''); end, clear Handles';
    ShowSampleInfoButton_handle = uicontrol('Style', 'pushbutton', ...
        'String', 'Show', 'Position', [330 6 45 20], ...
        'Callback', ShowSampleInfoButtonFunction, 'parent',ThisAlgFigureNumber);
    HideSampleInfoButtonFunction = 'Handles = findobj(''UserData'',''SampleInfoTextHandles''); if isempty(Handles) == 1, warndlg(''No sample information was loaded.''), else set(Handles,''visible'',''off''); end, clear Handles';
    HideSampleInfoButton_handle = uicontrol('Style', 'pushbutton', ...
        'String', 'Hide', 'Position', [380 6 45 20], ...
        'Callback', HideSampleInfoButtonFunction, 'parent',ThisAlgFigureNumber);
    ChangeSampleInfoButtonFunction = 'Handles = findobj(''UserData'',''SampleInfoTextHandles''); if isempty(Handles) == 1, warndlg(''No sample information was loaded.''), else propedit(Handles); end, clear Handles';
    ChangeSampleInfoButton_handle = uicontrol('Style', 'pushbutton', ...
        'String', 'Change', 'Position', [430 6 45 20], ...
        'Callback', ChangeSampleInfoButtonFunction, 'parent',ThisAlgFigureNumber);

    
    ChangeColormapButtonFunction = 'ImageHandle = findobj(gca, ''type'',''image''); propedit(ImageHandle)';
    ChangeColormapButton_handle = uicontrol('Style', 'pushbutton', ...
        'String', 'Change', 'Position', [490 6 45 20], ...
        'Callback', ChangeColormapButtonFunction, 'parent',ThisAlgFigureNumber);
    FolderButtonFunction = 'PathName = uigetdir('''',''Choose the directory where images are stored''); if PathName ~= 0, set(findobj(''UserData'',''PathNameTextDisplay''), ''String'', PathName), cd(PathName), end';
    
    FolderButton_handle = uicontrol('Style', 'pushbutton', ...
        'String', 'Change', 'Position', [550 6 45 20], ...
        'Callback', FolderButtonFunction, 'parent',ThisAlgFigureNumber);
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
        'String','File names:', ...
        'Style','text');
    TextHandle3 = uicontrol('Parent',ThisAlgFigureNumber, ...
        'BackgroundColor',get(ThisAlgFigureNumber,'Color'), ...
        'Position',[330 28 145 14], ...
        'HorizontalAlignment','center', ...
        'String','Sample names:', ...
        'Style','text');
    TextHandle4 = uicontrol('Parent',ThisAlgFigureNumber, ...
        'BackgroundColor',get(ThisAlgFigureNumber,'Color'), ...
        'Position',[485 28 55 14], ...
        'HorizontalAlignment','center', ...
        'String','Colormap:', ...
        'Style','text');
    TextHandle5 = uicontrol('Parent',ThisAlgFigureNumber, ...
        'BackgroundColor',get(ThisAlgFigureNumber,'Color'), ...
        'Position',[545 28 75 14], ...
        'HorizontalAlignment','left', ...
        'String','Image folder:', ...
        'Style','text');
    TextHandle6 = uicontrol('Parent',ThisAlgFigureNumber, ...
        'BackgroundColor',get(ThisAlgFigureNumber,'Color'), ...
        'Position',[625 28 NewFigureSize(3)-625 14], ...
        'HorizontalAlignment','left', ...
        'String',pwd, ...
        'UserData', 'PathNameTextDisplay', ...
        'Style','text');
    
    %%% Draws the grid on the image.  The 0.5 accounts for the fact that
    %%% pixels are labeled where the middle of the pixel is a whole number,
    %%% and the left hand side of each pixel is 0.5.
    X(1:2,:) = [(0.5:ImageWidth:TotalWidth+0.5);(0.5:ImageWidth:TotalWidth+0.5)];
    NumberVerticalLines = size(X');
    NumberVerticalLines = NumberVerticalLines(1);
    Y(1,:) = repmat(0,1,NumberVerticalLines);
    Y(2,:) = repmat(TotalHeight,1,NumberVerticalLines);
    line(X,Y)
    
    NewY(1:2,:) = [(0.5:ImageHeight:TotalHeight+0.5);(0.5:ImageHeight:TotalHeight+0.5)];
    NumberHorizontalLines = size(NewY');
    NumberHorizontalLines = NumberHorizontalLines(1);
    NewX(1,:) = repmat(0,1,NumberHorizontalLines);
    NewX(2,:) = repmat(TotalWidth,1,NumberHorizontalLines);
    line(NewX,NewY)
    
    Handles = findobj('type','line'); 
    set(Handles, 'color',[.15 .15 .15])
    
    %%% Sets the location of Tick marks.
    set(gca, 'XTick', [ImageWidth/2:ImageWidth:TotalWidth-ImageWidth/2])
    set(gca, 'YTick', [ImageHeight/2:ImageHeight:TotalHeight-ImageHeight/2])
    
    %%% Sets the Tick Labels.
    if strcmp(LeftOrRight,'R') == 1
        set(gca, 'XTickLabel',fliplr([1:NumberColumns]))
    else
        set(gca, 'XTickLabel',[1:NumberColumns])
    end
    if strcmp(TopOrBottom,'B') == 1
        set(gca, 'YTickLabel',fliplr([1:NumberRows]))
    else
        set(gca, 'YTickLabel',[1:NumberRows])
    end
    
    %%% Calculates where to display the file names on the tiled image.
    %%% Provides the i,j coordinates of the file names.  The
    %%% cellfun(length) part is just a silly way to get a number for every
    %%% entry in the NewFileList so that the find function can find it.
    %%% find does not work directly on strings in cell arrays.
    [i,j,ignore] = find(cellfun('length',NewFileList));
    YLocations = i*ImageHeight - ImageHeight/2;
    XLocations = j*ImageWidth - ImageWidth/2;
    OneColumnNewFileList = reshape(NewFileList,[],1);
    PrintableOneColumnNewFileList = strrep(OneColumnNewFileList,'_','\_');
    FileNameTextHandles = text(XLocations, YLocations, PrintableOneColumnNewFileList,...
        'HorizontalAlignment','center', 'color', 'white','visible','off', ...
        'UserData','FileNameTextHandles');
    if isempty(SampleInfo) ~= 1
        OneColumnNewSampleInfo = reshape(NewSampleInfo,[],1);
        PrintableOneColumnNewSampleInfo = strrep(OneColumnNewSampleInfo,'_','\_');
        SampleInfoTextHandles = text(XLocations, YLocations, PrintableOneColumnNewSampleInfo,...
            'HorizontalAlignment','center', 'color', 'white','visible','off', ...
            'UserData','SampleInfoTextHandles');
    end
    set(ThisAlgFigureNumber,'toolbar','figure')
end
%%% Executes pending figure-related commands so that the results are
%%% displayed.
drawnow

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% The tiled image is saved to the
%%% handles structure so it can be used by subsequent algorithms.
fieldname = ['dOT', TiledImageName];
handles.(fieldname) = TiledImage;
%%% Removed for parallel: guidata(gcbo, handles);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE PROCESSED IMAGE TO HARD DRIVE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Determine whether the user wanted to save the corrected image
%%% by comparing their entry "SaveImage" with "N" (after
%%% converting SaveImage to uppercase).
if strcmp(upper(SaveImage),'N') ~= 1
%%% Save the image to the hard drive.    
imwrite(TiledImage, [SaveImage,'.',FileFormat], FileFormat);
end

drawnow

%%%%%%%%%%%
%%% HELP %%%
%%%%%%%%%%%

%%%%% Help for the Image Tiler module: 

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
% The Original Code is the ______________________.
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
