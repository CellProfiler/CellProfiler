function handles = AlgIdentifyTertiarySubregion(handles)

% Help for the Identify Tertiary Subregion module: 
%
% This module will take the identified objects specified in the first
% box and remove from them the identified objects specified in the
% second box. For example, "subtracting" the nuclei from the cells
% will leave just the cytoplasm, the properties of which can then be
% measured by Measure modules. The first objects should therefore be
% equal in size or larger than the second objects and must completely
% contain the second objects.  Both images should be the result of a
% segmentation process, not grayscale images. Note that creating
% subregions using this module can result in objects that are not
% contiguous, which does not cause problems when running the Measure
% Intensity and Texture module, but does cause problems when running
% the Measure Area Shape Intensity Texture module because calculations
% of the perimeter, aspect ratio, solidity, etc. cannot be made for
% noncontiguous objects.
%
% See also <nothing relevant>.

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
% The Original Code is the Identify Tertiary Subregion module.
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

%textVAR01 = What did you call the larger identified objects?
%defaultVAR01 = Cells
SecondaryObjectName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,1});

%textVAR02 = What did you call the smaller identified objects?
%defaultVAR02 = Nuclei
PrimaryObjectName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,2});

%textVAR03 = What do you want to call the new subregions?
%defaultVAR03 = Cytoplasm
SubregionObjectName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,3});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a variable,
%%% "OrigImage".
fieldname = ['dOTSegmented', PrimaryObjectName];
%%% Checks whether the image to be analyzed exists in the handles structure.
if isfield(handles, fieldname) == 0
    %%% If the image is not there, an error message is produced.  The error
    %%% is not displayed: The error function halts the current function and
    %%% returns control to the calling function (the analyze all images
    %%% button callback.)  That callback recognizes that an error was
    %%% produced because of its try/catch loop and breaks out of the image
    %%% analysis loop without attempting further modules.
    error(['Image processing was canceled because the Identify Tertiary Subregion module could not find the input image.  It was supposed to be named ', PrimaryObjectName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
PrimaryObjectImage = handles.(fieldname);

%%% Retrieves the Secondary object segmented image.
fieldname = ['dOTSegmented', SecondaryObjectName];
if isfield(handles, fieldname) == 0
    error(['Image processing was canceled because the Identify Tertiary Subregion module could not find the input image.  It was supposed to be named ', SecondaryObjectName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
SecondaryObjectImage = handles.(fieldname);
       
%%% Checks that these images are two-dimensional (i.e. not a color
%%% image), which would disrupt several of the image functions.
if ndims(PrimaryObjectImage) ~= 2
    error('Image processing was canceled because the Identify Tertiary Subregion module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
end
if ndims(SecondaryObjectImage) ~= 2
    error('Image processing was canceled because the Identify Tertiary Subregion module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
end

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Erodes the primary object image and then subtracts it from the
%%% secondary object image.  This prevents the subregion from having zero
%%% pixels (which cannot be measured in subsequent measure modules) in the
%%% cases where the secondary object is exactly the same size as the
%%% primary object.
ErodedPrimaryObjectImage = imerode(PrimaryObjectImage, ones(3));
SubregionObjectImage = SecondaryObjectImage - ErodedPrimaryObjectImage;

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
    %%% THE FOLLOWING CALCULATIONS ARE FOR DISPLAY PURPOSES ONLY: The
    %%% resulting images are shown in the figure window (if open), or saved
    %%% to the hard drive (if desired).  To speed execution, all of this
    %%% code has been moved to within the if statement in the figure window
    %%% display section and then after starting image analysis, the figure
    %%% window can be closed.  Just remember that when the figure window is
    %%% closed, nothing within the if loop is carried out, so you would not
    %%% be able to save images depending on these lines to the hard drive,
    %%% for example.  If you plan to save images, these lines should be
    %%% moved outside this if statement.

    %%% Converts the label matrix to a colored label matrix for display and saving
    %%% purposes.
    ColoredSubregionObjectImage = label2rgb(SubregionObjectImage,'jet', 'k', 'shuffle');

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
    %%% A subplot of the figure window is set to display the original
    %%% primary object image.
    subplot(2,2,1); imagesc(PrimaryObjectImage);colormap(gray);
    title([PrimaryObjectName, ' Image, Image Set # ',num2str(handles.setbeinganalyzed)]);
    %%% A subplot of the figure window is set to display the original
    %%% secondary object image.
    subplot(2,2,2); imagesc(SecondaryObjectImage);
    title([SecondaryObjectName, ' Image']); colormap(gray);
    %%% A subplot of the figure window is set to display the resulting
    %%% subregion image in gray.
    subplot(2,2,3); imagesc(SubregionObjectImage); colormap(gray);
    title([SubregionObjectName, ' Image']);
    %%% A subplot of the figure window is set to display the resulting
    %%% subregion image in color.
    subplot(2,2,4); imagesc(ColoredSubregionObjectImage);
    title([SubregionObjectName, ' Color Image']);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the final, segmented label matrix image of secondary objects to
%%% the handles structure so it can be used by subsequent algorithms.
fieldname = ['dOTSegmented', SubregionObjectName];
handles.(fieldname) = SubregionObjectImage;

%%% Determines the filename of the image to be analyzed.
fieldname = ['dOTFilename', PrimaryObjectName];
FileName = handles.(fieldname)(handles.setbeinganalyzed);
%%% Saves the filename of the image to be analyzed.
fieldname = ['dOTFilename', SubregionObjectName];
handles.(fieldname)(handles.setbeinganalyzed) = FileName;
%%% Arbitrarily, I have chosen the primary object's filename to be saved
%%% here rather than the secondary object's filename.  