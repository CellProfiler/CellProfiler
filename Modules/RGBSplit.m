function handles = AlgRGBSplit(handles)

% Help for the RGB Split module: 
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
% The Original Code is the Split RGB Module.
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

%textVAR01 = What did you call the image to be split into black and white images?
%defaultVAR01 = OrigRGB
RGBImageName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,1});

%textVAR02 = What do you want to call the image that was red?
%defaultVAR02 = OrigRed
RedImageName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,2});

%textVAR03 = What do you want to call the image that was green?
%defaultVAR03 = OrigGreen
GreenImageName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,3});

%textVAR04 = What do you want to call the image that was blue?
%defaultVAR04 = OrigBlue
BlueImageName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,4});

%textVAR05 = Type "N" in any slots above to ignore that color.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Retrieves the RGB image from the handles structure.
fieldname = ['dOT', RGBImageName];
%%% Checks whether the image to be analyzed exists in the handles structure.
if isfield(handles, fieldname) == 0
    %%% If the image is not there, an error message is produced.  The error
    %%% is not displayed: The error function halts the current function and
    %%% returns control to the calling function (the analyze all images
    %%% button callback.)  That callback recognizes that an error was
    %%% produced because of its try/catch loop and breaks out of the image
    %%% analysis loop without attempting further modules.
    error(['Image processing was canceled because the RGB Split module could not find the input image.  It was supposed to be named ', RGBImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
%%% Reads the image.
RGBImage = handles.(fieldname);

Size = size(RGBImage);
if length(Size) ~= 3
    error(['Image processing was canceled because the RGB image you specified in the RGB Split module could not be separated into three layers of image data.  Is it a color image?  This module was only tested with TIF and BMP images.'])
end
if Size(3) ~= 3
    error(['Image processing was canceled because the RGB image you specified in the RGB Split module could not be separated into three layers of image data.  This module was only tested with TIF and BMP images.'])
end

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines whether the user has specified an image to be loaded in
%%% blue.
if strcmp(upper(RedImageName), 'N') == 0
    RedImage = RGBImage(:,:,1);
else RedImage = zeros(size(RGBImage(:,:,1)));
end
if strcmp(upper(GreenImageName), 'N') == 0
    GreenImage = RGBImage(:,:,2);
else GreenImage = zeros(size(RGBImage(:,:,1)));
end
if strcmp(upper(BlueImageName), 'N') == 0
    BlueImage = RGBImage(:,:,3);
else BlueImage = zeros(size(RGBImage(:,:,1)));
end

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
    
    %%% A subplot of the figure window is set to display the Splitd RGB
    %%% image.  Using imagesc or image instead of imshow doesn't work when
    %%% some of the pixels are saturated.
    subplot(2,2,1); imagesc(RGBImage);
    title(['Input RGB Image, Image Set # ',num2str(handles.setbeinganalyzed)]);
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
fieldname = ['dOT', RedImageName];
handles.(fieldname) = RedImage;
fieldname = ['dOT', GreenImageName];
handles.(fieldname) = GreenImage;
fieldname = ['dOT', BlueImageName];
handles.(fieldname) = BlueImage;

%%% Determines the filename of the image to be analyzed.
fieldname = ['dOTFilename', RGBImageName];
FileName = handles.(fieldname)(handles.setbeinganalyzed);
%%% Saves the original file name to the handles structure in a
%%% field named after the adjusted image name.
fieldname = ['dOTFilename', RGBImageName];
handles.(fieldname)(handles.setbeinganalyzed) = FileName;
fieldname = ['dOTFilename', RedImageName];
handles.(fieldname)(handles.setbeinganalyzed) = FileName;
fieldname = ['dOTFilename', GreenImageName];
handles.(fieldname)(handles.setbeinganalyzed) = FileName;
fieldname = ['dOTFilename', BlueImageName];
handles.(fieldname)(handles.setbeinganalyzed) = FileName;