function handles = AlgRGBMerge(handles)

% Help for the RGB Merge module: 
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
% The Original Code is the RGB Merge Module.
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
CurrentAlgorithmNum = str2num(handles.currentalgorithm);

%textVAR01 = What did you call the image to be colored blue?
%defaultVAR01 = OrigBlue
BlueImageName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,1});

%textVAR02 = What did you call the image to be colored green?
%defaultVAR02 = OrigGreen
GreenImageName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,2});

%textVAR03 = What did you call the image to be colored red?
%defaultVAR03 = OrigRed
RedImageName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,3});

%textVAR04 = Type "N" in any slots above to leave that color black.
%textVAR05 = What do you want to call the resulting image?
%defaultVAR05 = RGBImage
RGBImageName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,5});

%textVAR06 = Enter the adjustment factor for the blue image
%defaultVAR06 = 1
BlueAdjustmentFactor = char(handles.Settings.Vvariable{CurrentAlgorithmNum,6});

%textVAR07 = Enter the adjustment factor for the green image
%defaultVAR07 = 1
GreenAdjustmentFactor = char(handles.Settings.Vvariable{CurrentAlgorithmNum,7});

%textVAR08 = Enter the adjustment factor for the red image
%defaultVAR08 = 1
RedAdjustmentFactor = char(handles.Settings.Vvariable{CurrentAlgorithmNum,8});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines whether the user has specified an image to be loaded in
%%% blue.
if strcmp(upper(BlueImageName), 'N') == 0
    %%% Read (open) the images and assign them to variables.
    fieldname = ['dOT', BlueImageName];
    %%% Checks whether the image to be analyzed exists in the handles structure.
    if isfield(handles, fieldname) == 0
        %%% If the image is not there, an error message is produced.  The error
        %%% is not displayed: The error function halts the current function and
        %%% returns control to the calling function (the analyze all images
        %%% button callback.)  That callback recognizes that an error was
        %%% produced because of its try/catch loop and breaks out of the image
        %%% analysis loop without attempting further modules.
        error(['Image processing was canceled because the RGB Merge module could not find the input image.  It was supposed to be named ', BlueImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
    end
    %%% Reads the image.
    BlueImage = handles.(fieldname);
    BlueImageExists = 1;
else BlueImageExists = 0;
end

drawnow
%%% Repeat for Green and Red.
if strcmp(upper(GreenImageName), 'N') == 0
    fieldname = ['dOT', GreenImageName];
    if isfield(handles, fieldname) == 0
        error(['Image processing was canceled because the RGB Merge module could not find the input image.  It was supposed to be named ', GreenImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
    end
    GreenImage = handles.(fieldname);
    GreenImageExists = 1;
else GreenImageExists = 0;
end
if strcmp(upper(RedImageName), 'N') == 0
    fieldname = ['dOT', RedImageName];
    if isfield(handles, fieldname) == 0
        error(['Image processing was canceled because the RGB Merge module could not find the input image.  It was supposed to be named ', RedImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
    end
    RedImage = handles.(fieldname);
    RedImageExists = 1;
else RedImageExists = 0;
end
drawnow

%%% If any of the colors are to be left black, creates the appropriate
%%% image.
if BlueImageExists == 0 & RedImageExists == 0 & GreenImageExists == 0
    error('Image processing was canceled because you have not selected any images to be merged in the RGB Merge module.')
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
        error('Image processing was canceled because the three images selected for the RGB Merge module are not the same size.  The pixel dimensions must be identical.')
    end 
    if size(RedImage) ~= size(GreenImage)
        error('Image processing was canceled because the three images selected for the RGB Merge module are not the same size.  The pixel dimensions must be identical.')
    end 
catch error('Image processing was canceled because there was a problem with one of three images selected for the RGB Merge module. Most likely one of the images is not in the same format as the others - for example, one of the images might already be in RGB format.')
end

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%

RGBImage(:,:,1) = immultiply(RedImage,str2num(RedAdjustmentFactor));
RGBImage(:,:,2) = immultiply(GreenImage,str2num(GreenAdjustmentFactor));;
RGBImage(:,:,3) = immultiply(BlueImage,str2num(BlueAdjustmentFactor));;

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines the figure number to display in.
fieldname = ['figurealgorithm',CurrentAlgorithm];
ThisAlgFigureNumber = handles.(fieldname);
%%% Checks whether that figure is open. This checks all the figure handles
%%% for one whose handle is equal to the figure number for this algorithm.
%%% Note: Everything between the "if" and "end" is not carried out if the
%%% user has closed the figure window, so do not do any important
%%% calculations here. Otherwise an error message will be produced if the
%%% user has closed the window but you have attempted to access data that
%%% was supposed to be produced by this part of the code.
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the adjusted image to the handles structure so it can be used by
%%% subsequent algorithms.
fieldname = ['dOT', RGBImageName];
handles.(fieldname) = RGBImage;

%%% Determines the filename of the image to be analyzed. Only one of the
%%% original file names is chosen to name this field.
if BlueImageExists == 1
    fieldname = ['dOTFilename', BlueImageName];
elseif GreenImageExists == 1
    fieldname = ['dOTFilename', GreenImageName];
elseif RedImageExists == 1
    fieldname = ['dOTFilename', RedImageName];
end
FileName = handles.(fieldname)(handles.setbeinganalyzed);
%%% Saves the original file name to the handles structure in a
%%% field named after the adjusted image name.
fieldname = ['dOTFilename', RGBImageName];
handles.(fieldname)(handles.setbeinganalyzed) = FileName;