function handles = AlgRGBMerge(handles)


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
% The Original Code is the Merge RGB Module.
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

%%% Reads the current algorithm number, since this is needed to find 
%%% the variable values that the user entered.
CurrentAlgorithm = handles.currentalgorithm;

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow

%textVAR01 = What did you call the image to be colored blue?
%defaultVAR01 = OrigBlue
fieldname = ['Vvariable',CurrentAlgorithm,'_01'];
BlueImageName = handles.(fieldname);
%textVAR02 = What did you call the image to be colored green?
%defaultVAR02 = OrigGreen
fieldname = ['Vvariable',CurrentAlgorithm,'_02'];
GreenImageName = handles.(fieldname);
%textVAR03 = What did you call the image to be colored red?
%defaultVAR03 = OrigRed
fieldname = ['Vvariable',CurrentAlgorithm,'_03'];
RedImageName = handles.(fieldname);
%textVAR04 = Type "N" in any slots above to leave that color black.
%textVAR05 = What do you want to call the resulting image?
%defaultVAR05 = RGBImage
fieldname = ['Vvariable',CurrentAlgorithm,'_05'];
RGBImageName = handles.(fieldname);
%textVAR06 = Enter the adjustment factor for the blue image
%defaultVAR06 = 1
fieldname = ['Vvariable',CurrentAlgorithm,'_06'];
BlueAdjustmentFactor = handles.(fieldname);
%textVAR07 = Enter the adjustment factor for the green image
%defaultVAR07 = 1
fieldname = ['Vvariable',CurrentAlgorithm,'_07'];
GreenAdjustmentFactor = handles.(fieldname);
%textVAR08 = Enter the adjustment factor for the red image
%defaultVAR08 = 1
fieldname = ['Vvariable',CurrentAlgorithm,'_08'];
RedAdjustmentFactor = handles.(fieldname);
%textVAR09 = In what file format do you want to save images? Do not include a period
%defaultVAR09 = tif
fieldname = ['Vvariable',CurrentAlgorithm,'_09'];
FileFormat = handles.(fieldname);
%textVAR10 = To save the adjusted image, enter text to append to the image name 
%defaultVAR10 = N
fieldname = ['Vvariable',CurrentAlgorithm,'_10'];
SaveImage = handles.(fieldname);
%textVAR11 =  Otherwise, leave as "N". To save or display other images, press Help button

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Checks whether the file format the user entered is readable by Matlab.
IsFormat = imformats(FileFormat);
if isempty(IsFormat) == 1
    error('The image file type entered in the MergeRGB module is not recognized by Matlab. Or, you may have entered a period in the box. For a list of recognizable image file formats, type "imformats" (no quotes) at the command line in Matlab.','Error')
end

%%% Determines whether the user has specified an image to be loaded in
%%% blue.
if strcmp(upper(BlueImageName), 'N') == 0
    %%% Read (open) the images and assign them to variables.
    fieldname = ['dOT', BlueImageName];
    %%% Check whether the image to be analyzed exists in the handles structure.
    if isfield(handles, fieldname) == 0
        %%% If the image is not there, an error message is produced.  The error
        %%% is not displayed: The error function halts the current function and
        %%% returns control to the calling function (the analyze all images
        %%% button callback.)  That callback recognizes that an error was
        %%% produced because of its try/catch loop and breaks out of the image
        %%% analysis loop without attempting further modules.
        error(['Image processing was canceled because the MergeRGB module could not find the input image.  It was supposed to be named ', BlueImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
    end
    %%% Read the image.
    BlueImage = handles.(fieldname);
    % figure, imshow(BlueImage), title('BlueImage')
    BlueImageExists = 1;
else BlueImageExists = 0;
end

drawnow
%%% Repeat for Green and Red.
if strcmp(upper(GreenImageName), 'N') == 0
    fieldname = ['dOT', GreenImageName];
    if isfield(handles, fieldname) == 0
        error(['Image processing was canceled because the MergeRGB module could not find the input image.  It was supposed to be named ', GreenImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
    end
    GreenImage = handles.(fieldname);
    % figure, imshow(GreenImage), title('GreenImage')
    GreenImageExists = 1;
else GreenImageExists = 0;
end
if strcmp(upper(RedImageName), 'N') == 0
    fieldname = ['dOT', RedImageName];
    if isfield(handles, fieldname) == 0
        error(['Image processing was canceled because the MergeRGB module could not find the input image.  It was supposed to be named ', RedImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
    end
    RedImage = handles.(fieldname);
    % figure, imshow(RedImage), title('RedImage')
    RedImageExists = 1;
else RedImageExists = 0;
end
drawnow

%%% If any of the colors are to be left black, the appropriate image is
%%% created.
if BlueImageExists == 0 & RedImageExists == 0 & GreenImageExists == 0
    error('Image processing was canceled because you have not selected any images to be merged in the MergeRGB module.')
end
if BlueImageExists == 0 & RedImageExists == 0 & GreenImageExists == 1
    BlueImage = zeros(size(GreenImage));
    RedImage = zeros(size(GreenImage));
end
if BlueImageExists == 0 & RedImageExists == 1 & GreenImageExists == 0
    BlueImage = zeros(size(RedImage));
    GreenImage = zeros(size(RedImage));
end
if BlueImageExists == 1 & RedImageExists == 0 & GreenImageExists == 0
    RedImage = zeros(size(BlueImage));
    GreenImage = zeros(size(BlueImage));
end
if BlueImageExists == 1 & RedImageExists == 1 & GreenImageExists == 0
    GreenImage = zeros(size(BlueImage));
end
if BlueImageExists == 0 & RedImageExists == 1 & GreenImageExists == 1
    BlueImage = zeros(size(GreenImage));
end
if BlueImageExists == 1 & RedImageExists == 0 & GreenImageExists == 1
    RedImage = zeros(size(BlueImage));
end

%%% Checks whether the three images are the same size.
try
    if size(BlueImage) ~= size(GreenImage)
        error('Image processing was canceled because the three images selected for the MergeRGB module are not the same size.  The pixel dimensions must be identical.')
    end 
    if size(RedImage) ~= size(GreenImage)
        error('Image processing was canceled because the three images selected for the MergeRGB module are not the same size.  The pixel dimensions must be identical.')
    end 
catch error('Image processing was canceled because there was a problem with one of three images selected for the MergeRGB module. Most likely one of the images is not in the same format as the others - for example, one of the images might already be in RGB format.')
end
%%% Check whether the appendages to be added to the file names of images
%%% will result in overwriting the original file, or in a file name that
%%% contains spaces.
%%% Determine the filename of the image to be analyzed.
if BlueImageExists == 1
    fieldname = ['dOTFilename', BlueImageName];
elseif GreenImageExists == 1
    fieldname = ['dOTFilename', GreenImageName];
elseif RedImageExists == 1
    fieldname = ['dOTFilename', RedImageName];
end
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
    error('Image processing was canceled because you have entered one or more spaces in the box of text to append to the RGB image name in the MergeRGB module.  If you do not want to save the RGB image to the hard drive, type "N" into the appropriate box.')
    return
end
%%% Check whether the new image name is going to result in overwriting the
%%% original file.
B = strcmp(upper(CharFileName), upper(NewImageName));
if B == 1
    error('Image processing was canceled because you have not entered text to append to the RGB image name in the MergeRGB module.  If you do not want to save the RGB image to the hard drive, type "N" into the appropriate box.')
    return
end
drawnow

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%

RGBImage(:,:,1) = immultiply(RedImage,str2num(RedAdjustmentFactor));
RGBImage(:,:,2) = immultiply(GreenImage,str2num(GreenAdjustmentFactor));;
RGBImage(:,:,3) = immultiply(BlueImage,str2num(BlueAdjustmentFactor));;
% figure, imshow(RGBImage), title('RGBImage')

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
    
    %%% A subplot of the figure window is set to display the Merged RGB
    %%% image.  Using imagesc or image instead of imshow doesn't work when
    %%% some of the pixels are saturated.
    subplot(2,2,1); imagesc(RGBImage);
    title(['Merged RGB Image, Image Set # ',num2str(handles.setbeinganalyzed)]);
    %%% A subplot of the figure window is set to display the blue image.
    subplot(2,2,2); imagesc(BlueImage); colormap(gray), title('Blue Image');
    %%% A subplot of the figure window is set to display the green image.
    subplot(2,2,3); imagesc(GreenImage); colormap(gray), title('Green Image');
    %%% A subplot of the figure window is set to display the red image.
    subplot(2,2,4); imagesc(RedImage); colormap(gray), title('Red Image');
end
%%% Executes pending figure-related commands so that the results are
%%% displayed.
drawnow

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% The adjusted image is saved to the
%%% handles structure so it can be used by subsequent algorithms.
fieldname = ['dOT', RGBImageName];
handles.(fieldname) = RGBImage;
%%% Removed for parallel: guidata(gcbo, handles);

%%% The original file name is saved to the handles structure in a
%%% field named after the adjusted image name.
fieldname = ['dOTFilename', RGBImageName];
handles.(fieldname)(handles.setbeinganalyzed) = FileName;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE PROCESSED IMAGE TO HARD DRIVE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Determine whether the user wanted to save the adjusted image
%%% by comparing their entry "SaveImage" with "N" (after
%%% converting SaveImage to uppercase).
if strcmp(upper(SaveImage),'N') ~= 1
    %%% Save the image to the hard drive.    
    imwrite(RGBImage, NewImageName, FileFormat);
end

drawnow

%%%%%%%%%%%
%%% HELP %%%
%%%%%%%%%%%

%%%%% Help for the MergeRGB module: 
%%%%% .
