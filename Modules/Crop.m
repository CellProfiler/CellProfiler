function handles = AlgCrop3(handles)
%%% Reads the current algorithm number, since this is needed to find 
%%% the variable values that the user entered.
CurrentAlgorithm = handles.currentalgorithm;
CurrentAlgorithmNum = str2num(handles.currentalgorithm);

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow

%textVAR01 = What did you call the image to be cropped?
%defaultVAR01 = OrigBlue
fieldname = ['Vvariable',CurrentAlgorithm,'_01'];
ImageName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,1});
%textVAR02 = What do you want to call the cropped image?
%defaultVAR02 = CropBlue
fieldname = ['Vvariable',CurrentAlgorithm,'_02'];
CroppedImageName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,2});
%textVAR03 = To save the adjusted image, enter text to append to the image name 
%defaultVAR03 = N
fieldname = ['Vvariable',CurrentAlgorithm,'_03'];
SaveImage = char(handles.Settings.Vvariable{CurrentAlgorithmNum,3});
%textVAR04 =  Otherwise, leave as "N". To save or display other images, press Help button
%textVAR05 = In what file format do you want to save images? Do not include a period
%defaultVAR05 = tif
fieldname = ['Vvariable',CurrentAlgorithm,'_05'];
FileFormat = char(handles.Settings.Vvariable{CurrentAlgorithmNum,5});
%textVAR06 = For rectangular cropping, type R. For a shape from a file, type F
%defaultVAR06 = R
%textVAR07 = To draw an ellipse on each image, type EE; draw one ellipse for all images: EA
fieldname = ['Vvariable',CurrentAlgorithm,'_06'];
Shape = upper(char(handles.Settings.Vvariable{CurrentAlgorithmNum,6}));
%textVAR08 = Rectangular: enter the pixel position for the left (X), top (Y) corner (with comma between)
%defaultVAR08 = 1,1
fieldname = ['Vvariable',CurrentAlgorithm,'_08'];
LeftTop = char(handles.Settings.Vvariable{CurrentAlgorithmNum,8});
%textVAR09 = Rectangular: enter the pixel position for the right (X), bottom (Y) corner (with comma)
%defaultVAR09 = 100,100
fieldname = ['Vvariable',CurrentAlgorithm,'_09'];
RightBottom = char(handles.Settings.Vvariable{CurrentAlgorithmNum,9});
%textVAR10 = Other shape cropping: To crop to another shape, type the location and file name of 
%textVAR11 = the binary image to guide the cropping (Zero values will be removed).  Type carefully!  
%defaultVAR11 = /
fieldname = ['Vvariable',CurrentAlgorithm,'_11'];
BinaryCropImageName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,11});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Checks whether the file format the user entered is readable by Matlab.
IsFormat = imformats(FileFormat);
if isempty(IsFormat) == 1
    error('The image file type entered in the Crop module is not recognized by Matlab. Or, you may have entered a period in the box. For a list of recognizable image file formats, type "imformats" (no quotes) at the command line in Matlab.','Error')
end
%%% Read (open) the image to be used for cropping and assign it to a
%%% variable. There is an error catching mechanism in case the image cannot
%%% be found or opened.
if Shape == 'F'
    try
    BinaryCropImage = imread(BinaryCropImageName);
       % figure, imshow(BinaryCropImage), title('BinaryCropImage')
    catch error(['Image processing was canceled because the Crop module could not find the binary cropping image.  It was supposed to be here: ', BinaryCropImageName, ', but a readable image does not exist at that location or with that name.  Perhaps there is a typo.'])
    end
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
    error(['Image processing was canceled because the Crop module could not find the input image.  It was supposed to be named ', ImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
%%% Read the image.
OrigImage = handles.(fieldname);
        % figure, imshow(OrigImage), title('OrigImage')
        
%%% For cropping from a file image, checks whether the crop image is the
%%% same size as the image to be processed.
if Shape == 'F'
    if size(OrigImage(:,:,1)) ~= size(BinaryCropImage)
        error('Image processing was canceled because the binary image selected for cropping in the Crop module is not the same size as the image to be cropped.  The pixel dimensions must be identical.')
    end 
end

%%% Check whether the appendages to be added to the file names of images
%%% will result in overwriting the original file, or in a file name that
%%% contains spaces.
%%% Determine the filename of the image to be analyzed.
fieldname = ['dOTFilename', ImageName];
FileName = handles.(fieldname)(handles.setbeinganalyzed);
%%% Find and remove the file format extension within the original file
%%% name, but only if it is at the end. Strip the original file format extension 
%%% off of the file name, if it is present, otherwise, leave the original
%%% name intact.
CharFileName = char(FileName);
PotentialDot = CharFileName(end-3:end-3);
if strcmp(PotentialDot,'.') == 1
    BareFileName = CharFileName(1:end-4);
else BareFileName = CharFileName;
end

%%% Assemble the new image name.
NewImageName = [BareFileName,SaveImage,'.',FileFormat];
%%% Check whether the new image name is going to result in a name with
%%% spaces.
A = isspace(SaveImage);
if any(A) == 1
    error('Image processing was canceled because you have entered one or more spaces in the box of text to append to the cropped image name in the Crop module.  If you do not want to save the cropped image to the hard drive, type "N" into the appropriate box.')
    return
end
%%% Check whether the new image name is going to result in overwriting the
%%% original file.
B = strcmp(upper(CharFileName), upper(NewImageName));
if B == 1
    error('Image processing was canceled because you have not entered text to append to the cropped image name in the Crop module.  If you do not want to save the cropped image to the hard drive, type "N" into the appropriate box.')
    return
end

drawnow

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%

if Shape == 'F'
    %%% Sets pixels in the original image to zero if those pixels are zero in
    %%% the binary image file.
    PrelimCroppedImage = OrigImage;
    ImagePixels = size(PrelimCroppedImage,1)*size(PrelimCroppedImage,2);
    for Channel = 1:size(PrelimCroppedImage,3),
        PrelimCroppedImage((Channel-1)*ImagePixels + find(BinaryCropImage == 0)) = 0;
    end
        % figure, imshow(CroppedImage), title('PreliminaryCroppedImage')
    %%% Removes Rows and Columns that are completely blank.
    ColumnTotals = sum(BinaryCropImage);
    RowTotals = sum(BinaryCropImage');
    warning off all
    ColumnsToDelete = ~logical(ColumnTotals);
    RowsToDelete = ~logical(RowTotals);
    warning on all
    CroppedImage = PrelimCroppedImage;
    CroppedImage(:,ColumnsToDelete,:) = [];
    CroppedImage(RowsToDelete,:,:) = [];
    % figure, imshow(CroppedImage), title('CroppedImage')
    
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
        %%% The Binary Crop image is saved to the handles
        %%% structure so it can be used to crop subsequent image sets.
        fieldname = ['dOTCropping', CroppedImageName];
        handles.(fieldname) = BinaryCropImage;
        %%% Removed for parallel: guidata(gcbo, handles);
    % figure, imshow(BinaryCropImage)
    end
    %%% Retrieve previously selected cropping ellipse from handles
    %%% structure.
    fieldname = ['dOTCropping', CroppedImageName];
    BinaryCropImage = handles.(fieldname);
        % figure, imshow(BinaryCropImage)
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
    % figure, imshow(CroppedImage), title('PreliminaryCroppedImage')
    %%% Removes Rows and Columns that are completely blank.
    ColumnTotals = sum(BinaryCropImage);
    RowTotals = sum(BinaryCropImage');
    warning off all
    ColumnsToDelete = ~logical(ColumnTotals);
    RowsToDelete = ~logical(RowTotals);
    warning on all
    CroppedImage = PrelimCroppedImage;
    CroppedImage(:,ColumnsToDelete,:) = [];
    CroppedImage(RowsToDelete,:,:) = [];
    % figure, imshow(CroppedImage), title('CroppedImage')
else error('You must choose rectangular cropping (R) or cropping from a file (F) or drawing an ellipse (EE or EA) to use the Crop module.')
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
    %%% Sets the window to be only 300 pixels wide.
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
%%% Executes pending figure-related commands so that the results are
%%% displayed.
drawnow

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% The adjusted image is saved to the
%%% handles structure so it can be used by subsequent algorithms.
fieldname = ['dOT', CroppedImageName];
handles.(fieldname) = CroppedImage;

%%% The original file name is saved to the handles structure in a
%%% field named after the adjusted image name.
fieldname = ['dOTFilename', CroppedImageName];
handles.(fieldname)(handles.setbeinganalyzed) = FileName;
%%% Removed for parallel: guidata(gcbo, handles);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE PROCESSED IMAGE TO HARD DRIVE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Determine whether the user wanted to save the adjusted image
%%% by comparing their entry "SaveImage" with "N" (after
%%% converting SaveImage to uppercase).
if strcmp(upper(SaveImage),'N') ~= 1
%%% Save the image to the hard drive.    
imwrite(CroppedImage, NewImageName, FileFormat);
end
drawnow

%%%%%%%%%%%
%%% HELP %%%
%%%%%%%%%%%

%%%%% Help for the Crop module: 
%%%%% 


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