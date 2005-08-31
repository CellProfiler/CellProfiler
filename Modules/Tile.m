function handles = Tile(handles)

% Help for the Image Tiler module:
% Category: Image Processing
%
% Allows many images to be viewed simultaneously, in a grid layout you
% specify (e.g. in the actual layout in which the images were
% collected).
%
% If you want to view a large number of images, you will generate an
% extremely large file (roughly the MB of all the images added
% together) which, even if it could be created by Matlab, could not be
% opened by any image software anyway. Matlab has a limit to the
% amount of data it can open which prevents you from creating such a
% gigantic, high resolution file.  There are several ways to allow a
% larger image to be produced, given memory limitations: (1) Decrease
% the resolution of each image tile by entering a fraction where
% requested. Then, in the window which pops open after Tile
% finishes, you can use the 'Get high res image' button to retrieve
% the original high resolution image. (This button is not yet
% functional). (2) Use the SpeedUpCellProfiler module to clear out
% images that are stored in memory. Place this module just prior to
% the Tile module and ask it to retain only those images which
% are needed for downstream modules.  (3) Rescale the images to 8 bit
% format by putting in the RescaleImages module just prior to the
% Tile module. Normally images are stored in memory as class
% "double" which takes about 10 times the space of class "uint8" which
% is 8 bits.  You will lose resolution in terms of the number of
% different graylevels - this will be limited to 256 - but you will
% not lose spatial resolution.
%
% The file name (automatic) and sample info (optional) can be
% displayed on each image using buttons in the final figure window.
%
% SAVING IMAGES: The tiled image produced by this module can be easily
% saved using the Save Images module, using the name you assign. If
% you want to save other intermediate images, alter the code for this
% module to save those images to the handles structure (see the
% SaveImages module help) and then use the Save Images module.
%
% See also SPOTIDENTIFIER.

% CellProfiler is distributed under the GNU contGeneral Public License.
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

drawnow

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%

%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

%textVAR01 = What did you call the images to be tiled?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What were the images called when the were originally loaded?
%infotypeVAR02 = imagegroup
OrigImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 = What do you want to call the tiled image?
%defaultVAR03 = TiledImage
%infotypeVAR03 = imagegroup indep
TiledImageName = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = Number of rows to display.
%choiceVAR04 = Automatic
NumberRows = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu custom

%textVAR05 = Number of columns to display.
%choiceVAR05 = Automatic
NumberColumns = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu custom

%textVAR06 = Are the first two images arranged in a row or a column?
%choiceVAR06 = Column
%choiceVAR06 = Row
RowOrColumn = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%inputtypeVAR06 = popupmenu

%textVAR07 = Is the first image at the bottom or the top?
%choiceVAR07 = Top
%choiceVAR07 = Bottom
TopOrBottom = char(handles.Settings.VariableValues{CurrentModuleNum,7});
%inputtypeVAR07 = popupmenu

%textVAR08 = Is the first image at the left or the right?
%choiceVAR08 = Left
%choiceVAR08 = Right
LeftOrRight = char(handles.Settings.VariableValues{CurrentModuleNum,8});
%inputtypeVAR08 = popupmenu

%textVAR09 = What fraction should the images be sized (the resolution will be changed)?
%defaultVAR09 = .1
SizeChange = char(handles.Settings.VariableValues{CurrentModuleNum,9});
SizeChange = str2num(SizeChange);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a
%%% variable.

%%% Checks whether the image to be analyzed exists in the handles structure.
if isfield(handles.Pipeline, ImageName) == 0
    %%% If the image is not there, an error message is produced.  The error
    %%% is not displayed: The error function halts the current function and
    %%% returns control to the calling function (the analyze all images
    %%% button callback.)  That callback recognizes that an error was
    %%% produced because of its try/catch loop and breaks out of the image
    %%% analysis loop without attempting further modules.
    error(['Image processing was canceled because the Image Tiler module could not find the input image.  It was supposed to be named ', ImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow

if handles.Current.SetBeingAnalyzed == 1
    %%% Retrieves the path where the images are stored from the handles
    %%% structure.
    fieldname = ['Pathname', OrigImageName];
    try Pathname = handles.Pipeline.(fieldname);
    catch error('Image processing was canceled because the Image Tiler module must be run using images straight from a load images module (i.e. the images cannot have been altered by other image processing modules). This is because the Image Tiler module calculates an illumination correction image based on all of the images before correcting each individual image as CellProfiler cycles through them. One solution is to process the entire batch of images using the image analysis modules preceding this module and save the resulting images to the hard drive, then start a new stage of processing from this Image Tiler module onward.')
    end
    %%% Retrieves the list of filenames where the images are stored from the
    %%% handles structure.
    fieldname = ['FileList', OrigImageName];
    FileList = handles.Pipeline.(fieldname);
    NumberOfImages = length(FileList);
    if strcmp(NumberRows,'Automatic') == 1 && strcmp(NumberColumns,'Automatic')== 1
        %%% Calculates the square root in order to determine the dimensions
        %%% of the display grid.
        SquareRoot = sqrt(NumberOfImages);
        %%% Converts the result to an integer.
        NumberRows = fix(SquareRoot);
        NumberColumns = ceil((NumberOfImages)/NumberRows);
    elseif strcmp(NumberRows,'Automatic')
        NumberColumns = str2double(NumberColumns);
        NumberRows = ceil((NumberOfImages)/NumberColumns);
    elseif strcmp(NumberColumns,'Automatic')
        NumberRows = str2double(NumberRows);
        NumberColumns = ceil((NumberOfImages)/NumberRows);
    else NumberColumns = str2double(NumberColumns);
        NumberRows = str2double(NumberRows);
    end
    if NumberRows*NumberColumns > NumberOfImages;
        Answer = CPquestdlg(['You have specified ', num2str(NumberRows), ' rows and ', num2str(NumberColumns), ' columns (=',num2str(NumberRows*NumberColumns),' images), but there are ', num2str(length(FileList)), ' images loaded. The image locations at the end of the grid for which there is no image data will be displayed as black. Do you want to continue?'],'Continue?','Yes','No','Yes');
        if strcmp(Answer,'No') == 1
            return
        end
        FileList(length(FileList)+1:NumberRows*NumberColumns) = {'none'};
    elseif NumberRows*NumberColumns < NumberOfImages;
        Answer = CPquestdlg(['You have specified ', num2str(NumberRows), ' rows and ', num2str(NumberColumns), ' columns (=',num2str(NumberRows*NumberColumns),' images), but there are ', num2str(length(FileList)), ' images loaded. Images at the end of the list will not be displayed. Do you want to continue?'],'Continue?','Yes','No','Yes');
        if strcmp(Answer,'No') == 1
            return
        end
        FileList(NumberRows*NumberColumns+1:NumberOfImages) = [];
    end
    
    if strcmp(RowOrColumn,'Row')
        NewFileList = reshape(FileList,NumberColumns,NumberRows);
        NewFileList = NewFileList';
    elseif strcmp(RowOrColumn,'Column')
        NewFileList = reshape(FileList,NumberRows,NumberColumns);
    end
    if strcmp(LeftOrRight,'Right')
        NewFileList = fliplr(NewFileList);
    end
    if strcmp(TopOrBottom,'Bottom')
        NewFileList = flipud(NewFileList);
    end    
    
    NumberOfImages = NumberColumns*NumberRows;
    
    LoadedImage = handles.Pipeline.(ImageName);
    ImageSize = size(imresize(LoadedImage,SizeChange));
    ImageHeight = ImageSize(1);
    ImageWidth = ImageSize(2);
    TotalWidth = NumberColumns*ImageWidth;
    TotalHeight = NumberRows*ImageHeight;
    %%% Packs the workspace to free up memory since a large variable is about to be produced.
    pack;
    %%% Preallocates the array to improve speed. The data class for
    %%% the tiled image is set to match the incoming image's class.
    TiledImage = zeros(TotalHeight,TotalWidth,size(LoadedImage,3),class(LoadedImage));
    
    TileDataToSave.NumberColumns = NumberColumns;
    TileDataToSave.NumberRows = NumberRows;
    TileDataToSave.ImageHeight = ImageHeight;
    TileDataToSave.ImageWidth = ImageWidth;
    TileDataToSave.NewFileList = NewFileList;
    TileDataToSave.TotalWidth = TotalWidth;
    TileDataToSave.TotalHeight = TotalHeight; 
    TileDataToSave.TiledImage = TiledImage; 
    
    %stores data in handles
    handles.Pipeline.TileData.(['Module' handles.Current.CurrentModuleNumber]) = TileDataToSave;
end

%gets data from handles
RetrievedTileData = handles.Pipeline.TileData.(['Module' handles.Current.CurrentModuleNumber]);

TiledImage = RetrievedTileData.TiledImage;
NumberColumns = RetrievedTileData.NumberColumns;
ImageHeight = RetrievedTileData.ImageHeight;
ImageWidth = RetrievedTileData.ImageWidth;
NumberColumns = RetrievedTileData.NumberColumns;
NumberRows = RetrievedTileData.NumberRows;

CurrentImage = handles.Pipeline.(ImageName);
CurrentImage = imresize(CurrentImage,SizeChange);

if strcmp(RowOrColumn,'Column')
    HorzPos = floor((handles.Current.SetBeingAnalyzed-1)/NumberRows);
    VertPos = handles.Current.SetBeingAnalyzed - HorzPos*NumberRows-1;
elseif strcmp(RowOrColumn,'Row')
    VertPos = floor((handles.Current.SetBeingAnalyzed-1)/NumberColumns);
    HorzPos = handles.Current.SetBeingAnalyzed - VertPos*NumberColumns-1;
end

if strcmp(TopOrBottom,'Bottom')
    VertPos = NumberRows - VertPos-1;
end

if strcmp(LeftOrRight,'Right')
    HorzPos = NumberColumns - HorzPos-1;
end

%%% Memory errors can occur here if the tiled image is too big.
TiledImage((ImageHeight*VertPos)+(1:ImageHeight),(ImageWidth*HorzPos)+(1:ImageWidth),:) = CurrentImage(:,:,:);
handles.Pipeline.TileData.(['Module' handles.Current.CurrentModuleNumber]).TiledImage = TiledImage;

if handles.Current.SetBeingAnalyzed == handles.Current.NumberOfImageSets

    %%%%%%%%%%%%%%%%%%%%%%
    %%% DISPLAY RESULTS %%%
    %%%%%%%%%%%%%%%%%%%%%%
    drawnow

    %gets data from handles
    RetrievedTileData = handles.Pipeline.TileData.(['Module' handles.Current.CurrentModuleNumber]);
    TiledImage = RetrievedTileData.TiledImage;
    NumberColumns = RetrievedTileData.NumberColumns;
    ImageHeight = RetrievedTileData.ImageHeight;
    ImageWidth = RetrievedTileData.ImageWidth;
    NumberColumns = RetrievedTileData.NumberColumns;
    NumberRows = RetrievedTileData.NumberRows;
    TotalWidth = RetrievedTileData.TotalWidth;
    TotalHeight = RetrievedTileData.TotalHeight;
    NewFileList = RetrievedTileData.NewFileList;

    fieldname = ['FigureNumberForModule',CurrentModule];
    ThisModuleFigureNumber = handles.Current.(fieldname);
    if any(findobj == ThisModuleFigureNumber) == 1;

        drawnow
        %%% Activates the appropriate figure window.
        CPfigure(handles,ThisModuleFigureNumber);
        CPcolormap(handles);
        %%% Displays the image.
        imagesc(TiledImage)
        %%% Sets the figure to take up most of the screen.
        ScreenSize = get(0,'ScreenSize');
        Font = handles.Current.FontSize;
        NewFigureSize = [60,250, ScreenSize(3)-200, ScreenSize(4)-350];
        set(ThisModuleFigureNumber, 'Position', NewFigureSize)
        axis image

        ToggleGridButtonFunction = ...
        ['Handles = findobj(''type'',''line'');'...
            'button = findobj(''Tag'',''ToggleGridButton'');'...
            'if strcmp(get(button,''String''),''Hide''),'...
                'set(button,''String'',''Show'');'...
                'set(Handles,''visible'',''off'');'...
            'else,'...
                'set(button,''String'',''Hide'');'...
                'set(Handles,''visible'',''on'');'...
            'end,'...
            'clear Handles button'];
        uicontrol('Style', 'pushbutton', ...
            'String', 'Hide', 'Position', [10 6 45 20], 'BackgroundColor',[.7 .7 .9],...
            'Callback', ToggleGridButtonFunction, 'parent',ThisModuleFigureNumber,'FontSize',Font,'Tag','ToggleGridButton');
        ChangeGridButtonFunction = 'Handles = findobj(''type'',''line''); propedit(Handles); clear Handles';
        uicontrol('Style', 'pushbutton', ...
            'String', 'Change', 'Position', [60 6 45 20],'BackgroundColor',[.7 .7 .9], ...
            'Callback', ChangeGridButtonFunction, 'parent',ThisModuleFigureNumber,'FontSize',Font);

        ToggleFileNamesButtonFunction = ...
        ['Handles = findobj(''UserData'',''FileNameTextHandles'');'...
            'button = findobj(''Tag'',''ToggleFileNamesButton'');'...
            'if strcmp(get(button,''String''),''Hide''),'...
                'set(button,''String'',''Show'');'...
                'set(Handles,''visible'',''off'');'...
            'else,'...
                'set(button,''String'',''Hide'');'...
                'set(Handles,''visible'',''on'');'...
            'end,'...
            'clear Handles button'];
        uicontrol('Style', 'pushbutton', ...
            'String', 'Show', 'Position', [120 6 45 20], 'BackgroundColor',[.7 .7 .9],...
            'Callback', ToggleFileNamesButtonFunction, 'parent',ThisModuleFigureNumber,'FontSize',Font,'Tag','ToggleFileNamesButton');
        ChangeFileNamesButtonFunction = 'Handles = findobj(''UserData'',''FileNameTextHandles''); propedit(Handles); clear Handles';
        uicontrol('Style', 'pushbutton', 'BackgroundColor',[.7 .7 .9],...
            'String', 'Change', 'Position', [170 6 45 20], ...
            'Callback', ChangeFileNamesButtonFunction, 'parent',ThisModuleFigureNumber,'FontSize',Font);

        ChangeColormapButtonFunction = 'ImageHandle = findobj(gca, ''type'',''image''); propedit(ImageHandle)';
        uicontrol('Style', 'pushbutton', ...
            'String', 'Change', 'Position', [230 6 45 20], 'BackgroundColor',[.7 .7 .9],...
            'Callback', ChangeColormapButtonFunction, 'parent',ThisModuleFigureNumber,'FontSize',Font);

        uicontrol('Parent',ThisModuleFigureNumber, ...
            'BackgroundColor',get(ThisModuleFigureNumber,'Color'), ...
            'Position',[10 28 95 14], ...
            'HorizontalAlignment','center', ...
            'String','Gridlines:', ...
            'Style','text', ...
            'FontSize',Font);
        uicontrol('Parent',ThisModuleFigureNumber, ...
            'BackgroundColor',get(ThisModuleFigureNumber,'Color'), ...
            'Position',[120 28 95 14], ...
            'HorizontalAlignment','center', ...
            'String','File names:', ...
            'Style','text', ...
            'FontSize',Font);
        uicontrol('Parent',ThisModuleFigureNumber, ...
            'BackgroundColor',get(ThisModuleFigureNumber,'Color'), ...
            'Position',[230 28 55 14], ...
            'HorizontalAlignment','center', ...
            'String','Colormap:', ...
            'Style','text', ...
            'FontSize',Font);


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
        set(gca, 'XTick', ImageWidth/2:ImageWidth:TotalWidth-ImageWidth/2)
        set(gca, 'YTick', ImageHeight/2:ImageHeight:TotalHeight-ImageHeight/2)

        %%% Sets the Tick Labels.
        if strcmp(LeftOrRight,'Right') == 1
            set(gca, 'XTickLabel',fliplr(1:NumberColumns))
        else
            set(gca, 'XTickLabel', 1:NumberColumns)
        end
        if strcmp(TopOrBottom,'Bottom') == 1
            set(gca, 'YTickLabel',fliplr(1:NumberRows))
        else
            set(gca, 'YTickLabel', 1:NumberRows)
        end

        %%% Calculates where to display the file names on the tiled image.
        %%% Provides the i,j coordinates of the file names.  The
        %%% cellfun(length) part is just a silly way to get a number for every
        %%% entry in the NewFileList so that the find function can find it.
        %%% find does not work directly on strings in cell arrays.
        [i,j] = find(cellfun('length',NewFileList));
        YLocations = i*ImageHeight - ImageHeight/2;
        XLocations = j*ImageWidth - ImageWidth/2;
        OneColumnNewFileList = reshape(NewFileList,[],1);
        PrintableOneColumnNewFileList = strrep(OneColumnNewFileList,'_','\_');
        %%% Creates FileNameText
        text(XLocations, YLocations, PrintableOneColumnNewFileList,...
            'HorizontalAlignment','center', 'color', 'white','visible','off', ...
            'UserData','FileNameTextHandles') 
        set(ThisModuleFigureNumber,'toolbar','figure')
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% SAVE DATA TO HANDLES STRUCTURE %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    drawnow

    %%% Saves the tiled image to the handles structure so it can be used by
    %%% subsequent modules.
    handles.Pipeline.(TiledImageName) = TiledImage;
end
