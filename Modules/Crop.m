function handles = AlgCrop(handles)

% Help for the Crop module: 
% Sorry, this module has not yet been documented.

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
% The Original Code is the Crop module.
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

%textVAR01 = What did you call the image to be cropped?
%defaultVAR01 = OrigBlue
ImageName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,1});

%textVAR02 = What do you want to call the cropped image?
%defaultVAR02 = CropBlue
CroppedImageName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,2});

%textVAR03 = For rectangular cropping, type R. For a shape from a file, type F
%textVAR04 = To draw an ellipse on each image, type EE; draw one ellipse for all images: EA
%defaultVAR04 = R
Shape = upper(char(handles.Settings.Vvariable{CurrentAlgorithmNum,4}));

%textVAR05 = Rectangular: enter the pixel position for the left (X), top (Y) corner (with comma)
%defaultVAR05 = 1,1
LeftTop = char(handles.Settings.Vvariable{CurrentAlgorithmNum,5});

%textVAR06 = Rectangular: enter the pixel position for the right (X), bottom (Y) corner (with comma)
%defaultVAR06 = 100,100
RightBottom = char(handles.Settings.Vvariable{CurrentAlgorithmNum,6});

%textVAR10 = Other shape cropping: To crop to another shape, type the location and file name of 
%textVAR11 = the binary image to guide the cropping (Zero values will be removed).  Type carefully!  
%defaultVAR11 = /
BinaryCropImageName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,11});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image to be used for cropping and assigns it to a
%%% variable. There is an error catching mechanism in case the image cannot
%%% be found or opened.
if Shape == 'F'
    try
    BinaryCropImage = imread(BinaryCropImageName);
    catch error(['Image processing was canceled because the Crop module could not find the binary cropping image.  It was supposed to be here: ', BinaryCropImageName, ', but a readable image does not exist at that location or with that name.  Perhaps there is a typo.'])
    end
end

%%% Reads (opens) the image to be analyzed and assigns it to a variable,
%%% "OrigImage".
fieldname = ['dOT', ImageName];
%%% Checks whether the image to be analyzed exists in the handles structure.
if isfield(handles, fieldname) == 0
    %%% If the image is not there, an error message is produced.  The error
    %%% is not displayed: The error function halts the current function and
    %%% returns control to the calling function (the analyze all images
    %%% button callback.)  That callback recognizes that an error was
    %%% produced because of its try/catch loop and breaks out of the image
    %%% analysis loop without attempting further modules.
    error(['Image processing was canceled because the Crop module could not find the input image.  It was supposed to be named ', ImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
%%% Reads the image.
OrigImage = handles.(fieldname);
        
%%% For cropping from a file image, checks whether the crop image is the
%%% same size as the image to be processed.
if Shape == 'F'
    if size(OrigImage(:,:,1)) ~= size(BinaryCropImage)
        error('Image processing was canceled because the binary image selected for cropping in the Crop module is not the same size as the image to be cropped.  The pixel dimensions must be identical.')
    end 
end

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow

if Shape == 'F'
    %%% Sets pixels in the original image to zero if those pixels are zero in
    %%% the binary image file.
    PrelimCroppedImage = OrigImage;
    ImagePixels = size(PrelimCroppedImage,1)*size(PrelimCroppedImage,2);
    for Channel = 1:size(PrelimCroppedImage,3),
        PrelimCroppedImage((Channel-1)*ImagePixels + find(BinaryCropImage == 0)) = 0;
    end
    drawnow
    %%% Removes Rows and Columns that are completely blank.
    ColumnTotals = sum(BinaryCropImage);
    RowTotals = sum(BinaryCropImage');
    warning off all
    ColumnsToDelete = ~logical(ColumnTotals);
    RowsToDelete = ~logical(RowTotals);
    warning on all
    drawnow
    CroppedImage = PrelimCroppedImage;
    CroppedImage(:,ColumnsToDelete,:) = [];
    CroppedImage(RowsToDelete,:,:) = [];

elseif Shape == 'R'
    %%% Extracts the top, left, bottom, right pixel positions from the user's
    %%% input.
    LeftTopNumerical = str2num(LeftTop);
    Left = LeftTopNumerical(1);
    Top = LeftTopNumerical(2);
    RightBottomNumerical = str2num(RightBottom);
    Right = RightBottomNumerical(1);
    Bottom = RightBottomNumerical(2);
    if Left == 0 | Right == 0 | Bottom == 0 | Top ==0
        error('There was a problem in the Cropping module. One of the values entered for the rectangular cropping pixel positions was zero: all values must be integers greater than zero.')
    end
    if Left >= Right
        error('There was a problem in the Cropping module. The value entered for the right corner is less than or equal to the value entered for the left corner.')
    end
    if Top >= Bottom
        error('There was a problem in the Cropping module. The value entered for the bottom corner is less than or equal to the value entered for the top corner.')
    end
    try
        CroppedImage = OrigImage(Top:Bottom, Left:Right,:);
    catch error('There was a problem in the Cropping module. The values entered for the rectangular cropping pixel positions are not valid.')
    end

elseif Shape == 'EA' | Shape == 'EE'
    if handles.setbeinganalyzed == 1 | Shape =='EE'
        if Shape == 'EA'
            %%% Asks the user to open an image file upon which to draw the
            %%% ellipse.
            CurrentDirectory = cd;
            Directory = handles.Vpathname;
            cd(Directory)
            %%% Opens a user interface window which retrieves a file name and path
            %%% name for the image to be used as a test image.
            [CroppingFileName,CroppingPathName] = uigetfile('*.*','Select the image to use for cropping');
            %%% If the user presses "Cancel", the FileName will = 0 and an error
            %%% message results.
            if CroppingFileName == 0
                error('Image processing was canceled because you did not select an image to use for cropping in the Crop module.')
            else
                ImageToBeCropped = imread([CroppingPathName,'/',CroppingFileName]);
            end
            cd(CurrentDirectory)
        else ImageToBeCropped = OrigImage;
        end
        %%% Displays the image and asks the user to choose points for the
        %%% ellipse.
        CroppingFigureHandle = figure;
        imshow(ImageToBeCropped), colormap('gray');pixval
        title('Click on 5 or more points to be used to create a cropping ellipse & then press Enter.')
        [Pre_x,Pre_y] = getpts(CroppingFigureHandle);
        close(CroppingFigureHandle)
        x = Pre_y;
        y = Pre_x;
        drawnow
        %%% Removes bias of the ellipse - to make matrix inversion more
        %%% accurate. (will be added later on) (Not really sure what this
        %%% is doing).
        mean_x = mean(x);
        mean_y = mean(y);
        New_x = x-mean_x;
        New_y = y-mean_y;
        %%% the estimation for the conic equation of the ellipse
        X = [New_x.^2, New_x.*New_y, New_y.^2, New_x, New_y ];
        params = sum(X)/(X'*X);
        masksize = size(ImageToBeCropped);
        [X,Y] = meshgrid(1:masksize(1), 1:masksize(2));
        X = X - mean_x;
        Y = Y - mean_y;
        drawnow
        %%% Produces the BinaryCropImage.
        BinaryCropImage = ((params(1) * (X .* X) + ...
            params(2) * (X .* Y) + ...
            params(3) * (Y .* Y) + ...
            params(4) * X + ...
            params(5) * Y) < 1);
        %%% Need to flip X and Y.
        BinaryCropImage = BinaryCropImage';
        %%% Displays the result in a new figure window.
        figure;
        imagesc(BinaryCropImage);title('Cropping Mask')
        colormap(gray)
        hold on
        plot(Pre_x,Pre_y, 'r.')
        drawnow
        %%% The Binary Crop image is saved to the handles
        %%% structure so it can be used to crop subsequent image sets.
        fieldname = ['dOTCropping', CroppedImageName];
        handles.(fieldname) = BinaryCropImage;
    end
    %%% Retrieves previously selected cropping ellipse from handles
    %%% structure.
    fieldname = ['dOTCropping', CroppedImageName];
    BinaryCropImage = handles.(fieldname);
    if size(OrigImage(:,:,1)) ~= size(BinaryCropImage(:,:,1))
        error('Image processing was canceled because an image you wanted to analyze is not the same size as the image used for cropping in the Crop module.  The pixel dimensions must be identical.')
    end
    %%% Sets pixels in the original image to zero if those pixels are zero in
    %%% the binary image file.
    PrelimCroppedImage = OrigImage;
    ImagePixels = size(PrelimCroppedImage,1)*size(PrelimCroppedImage,2);
    for Channel = 1:size(PrelimCroppedImage,3),
        PrelimCroppedImage((Channel-1)*ImagePixels + find(BinaryCropImage == 0)) = 0;
    end
    %%% Removes Rows and Columns that are completely blank.
    ColumnTotals = sum(BinaryCropImage);
    RowTotals = sum(BinaryCropImage');
    warning off all
    ColumnsToDelete = ~logical(ColumnTotals);
    RowsToDelete = ~logical(RowTotals);
    warning on all
    CroppedImage = PrelimCroppedImage;
    drawnow
    CroppedImage(:,ColumnsToDelete,:) = [];
    drawnow
    CroppedImage(RowsToDelete,:,:) = [];
else error('You must choose rectangular cropping (R) or cropping from a file (F) or drawing an ellipse (EE or EA) to use the Crop module.')
end

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines the figure number to display in.
fieldname = ['figurealgorithm',CurrentAlgorithm];
ThisAlgFigureNumber = handles.(fieldname);
%%% Check whether that figure is open. This checks all the figure handles
%%% for one whose handle is equal to the figure number for this algorithm.
%%% Note: Everything between the "if" and "end" is not carried out if the
%%% user has closed the figure window, so do not do any important
%%% calculations here. Otherwise an error message will be produced if the
%%% user has closed the window but you have attempted to access data that
%%% was supposed to be produced by this part of the code.
if any(findobj == ThisAlgFigureNumber) == 1;
    %%% Sets the window to be half as wide as usual.
    originalsize = get(ThisAlgFigureNumber, 'position');
    newsize = originalsize;
    newsize(3) = 250;
    set(ThisAlgFigureNumber, 'position', newsize);
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
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,1,1); imagesc(OrigImage);colormap(gray);
    title(['Input Image, Image Set # ',num2str(handles.setbeinganalyzed)]);
    %%% A subplot of the figure window is set to display the adjusted
    %%%  image.
    subplot(2,1,2); imagesc(CroppedImage); title('Cropped Image');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the adjusted image to the handles structure so it can be used by
%%% subsequent algorithms.
fieldname = ['dOT', CroppedImageName];
handles.(fieldname) = CroppedImage;

%%% Determines the filename of the image to be analyzed.
fieldname = ['dOTFilename', ImageName];
FileName = handles.(fieldname)(handles.setbeinganalyzed);
%%% Saves the original file name to the handles structure in a
%%% field named after the adjusted image name.
fieldname = ['dOTFilename', CroppedImageName];
handles.(fieldname)(handles.setbeinganalyzed) = FileName;