function handles = AlgSubtractImages(handles)

% Help for Subtract Images module:
%
% Sorry, this module has not yet been documented. It was written for a
% very specific purpose and it allows blurring and subtracting images.
%
% SPEED OPTIMIZATION: Note that increasing the blur radius increases
% the processing time exponentially.
% 
% See also <nothing relevant>

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
% The Original Code is the Subtract Images Module.
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

%textVAR01 = Subtract this image (enter the name here)
%defaultVAR01 = tubulinw3
SubtractImageName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,1});

%textVAR02 = From this image (enter the name here)
%defaultVAR02 = NHSw1
BasicImageName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,2});

%textVAR03 = What do you want to call the resulting image?
%defaultVAR03 = SubtractedCellStain
ResultingImageName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,3});

%textVAR04 = Enter the factor to multiply the subtracted image by:
%defaultVAR04 = 1
MultiplyFactor = str2double(char(handles.Settings.Vvariable{CurrentAlgorithmNum,4}));

%textVAR05 = Contrast stretch the resulting image?
%defaultVAR05 = N
Stretch = char(handles.Settings.Vvariable{CurrentAlgorithmNum,5});

%textVAR06 = Blur radius for the basic image
%defaultVAR06 = 3
BlurRadius = str2double(char(handles.Settings.Vvariable{CurrentAlgorithmNum,6}));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the images you want to analyze and assigns them to
%%% variables.
fieldname = ['dOT', BasicImageName];
%%% Checks whether the image exists in the handles structure.
if isfield(handles, fieldname) == 0
    error(['Image processing has been canceled. Prior to running the Subtract Images module, you must have previously run an algorithm to load an image. You specified in the Subtract Images module that this image was called ', BasicImageName, ' which should have produced a field in the handles structure called ', fieldname, '. The Subtract Images module cannot find this image.']);
end
BasicImage = handles.(fieldname);
fieldname = ['dOT', SubtractImageName];
%%% Checks whether the image exists in the handles structure.
if isfield(handles, fieldname) == 0
    error(['Image processing has been canceled. Prior to running the Subtract Images module, you must have previously run an algorithm to load an image. You specified in the Subtract Images module that this image was called ', SubtractImageName, ' which should have produced a field in the handles structure called ', fieldname, '. The Subtract Images module cannot find this image.']);
end
SubtractImage = handles.(fieldname);

%%% Checks that the original images are two-dimensional (i.e. not a color
%%% image), which would disrupt several of the image functions.
if ndims(BasicImage) ~= 2
    error('Image processing was canceled because the Subtract Images module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
end
if ndims(SubtractImage) ~= 2
    error('Image processing was canceled because the Subtract Images module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
end

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow

AdjustedSubtractImage = MultiplyFactor*SubtractImage;
if BlurRadius ~= 0
    %%% Blurs the image.
    %%% Note: using filter2 is much faster than imfilter (e.g. 14.5 sec vs. 99.1 sec).
    FiltSize = max(3,ceil(4*BlurRadius));
    BasicImage = filter2(fspecial('gaussian',FiltSize, BlurRadius), BasicImage);
end
ResultingImage = imsubtract(BasicImage,AdjustedSubtractImage);
if strcmp(upper(Stretch),'Y') == 1
    ResultingImage = imadjust(ResultingImage,stretchlim(ResultingImage,[.01 .99]));
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
    figure(ThisAlgFigureNumber);
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,2,1); imagesc(BasicImage);colormap(gray);
    title([BasicImageName, ' input Image, Image Set # ',num2str(handles.setbeinganalyzed)]);
    %%% A subplot of the figure window is set to display the colored label
    %%% matrix image.
    subplot(2,2,2); imagesc(SubtractImage); title([SubtractImageName, ' input image']);
    subplot(2,2,3); imagesc(ResultingImage); title([BasicImageName,' minus ',SubtractImageName,' = ',ResultingImageName]);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the processed image to the handles structure.
fieldname = ['dOT',ResultingImageName];
handles.(fieldname) = ResultingImage;

%%% Determines the filename of the image to be analyzed.
fieldname = ['dOTFilename', BasicImageName];
FileName = handles.(fieldname)(handles.setbeinganalyzed);
%%% Saves the original file name to the handles structure in a
%%% field named after the adjusted image name.
fieldname = ['dOTFilename', ResultingImageName];
handles.(fieldname)(handles.setbeinganalyzed) = FileName;