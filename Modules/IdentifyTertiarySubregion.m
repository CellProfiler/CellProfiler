function handles = AlgIdentifyTertiarySubregion(handles)

% Help for the Identify Tertiary Subregion module: 
% Category: Object Identification
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
% SAVING IMAGES: The images of the objects produced by this module can
% be easily saved using the Save Images module using the name:
% Segmented + whatever you called the objects (e.g.
% SegmentedCytoplasm. This will be a grayscale image where each object
% is a different intensity.
% 
% An additional image is normally calculated for display only: the
% colored label matrix image (the objects displayed as arbitrary
% colors). This image, and any other intermediate images of interest,
% can be saved by altering the code for this module to save those
% images to the handles structure (see the SaveImages module help) and
% then using the Save Images module.  Important note: The calculations
% of display images are only performed if the figure window is open,
% so the figure window must be left open or the Save Images module
% will fail.  If you are running the job on a cluster, figure windows
% are not open, so the Save Images module will also fail, unless you
% go into the code for this module and remove the 'if/end' statement
% surrounding the DISPLAY RESULTS section.
%
% See also identify Primary and Identify Secondary modules.

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

% PROGRAMMING NOTE
% HELP:
% The first unbroken block of lines will be extracted as help by
% CellProfiler's 'Help for this analysis module' button as well as
% Matlab's built in 'help' and 'doc' functions at the command line. It
% will also be used to automatically generate a manual page for the
% module. An example image demonstrating the function of the module
% can also be saved in tif format, using the same name as the
% algorithm (minus Alg), and it will automatically be included in the
% manual page as well.  Follow the convention of: purpose of the
% module, description of the variables and acceptable range for each,
% how it works (technical description), info on which images can be 
% saved, and See also CAPITALLETTEROTHERALGORITHMS. The license/author
% information should be separated from the help lines with a blank
% line so that it does not show up in the help displays.  Do not
% change the programming notes in any modules! These are standard
% across all modules for maintenance purposes, so anything
% module-specific should be kept separate.

% PROGRAMMING NOTE
% DRAWNOW:
% The 'drawnow' function allows figure windows to be updated and
% buttons to be pushed (like the pause, cancel, help, and view
% buttons).  The 'drawnow' function is sprinkled throughout the code
% so there are plenty of breaks where the figure windows/buttons can
% be interacted with.  This does theoretically slow the computation
% somewhat, so it might be reasonable to remove most of these lines
% when running jobs on a cluster where speed is important.
drawnow

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%

% PROGRAMMING NOTE
% VARIABLE BOXES AND TEXT: 
% The '%textVAR' lines contain the text which is displayed in the GUI
% next to each variable box. The '%defaultVAR' lines contain the
% default values which are displayed in the variable boxes when the
% user loads the algorithm. The line of code after the textVAR and
% defaultVAR extracts the value that the user has entered from the
% handles structure and saves it as a variable in the workspace of
% this algorithm with a descriptive name. The syntax is important for
% the %textVAR and %defaultVAR lines: be sure there is a space before
% and after the equals sign and also that the capitalization is as
% shown.  Don't allow the text to wrap around to another line; the
% second line will not be displayed.  If you need more space to
% describe a variable, you can refer the user to the help file, or you
% can put text in the %textVAR line above or below the one of
% interest, and do not include a %defaultVAR line so that the variable
% edit box for that variable will not be displayed; the text will
% still be displayed. CellProfiler is currently being restructured to
% handle more than 11 variable boxes. Keep in mind that you can have
% several inputs into the same box: for example, a box could be
% designed to receive two numbers separated by a comma, as long as you
% write a little extraction algorithm that separates the input into
% two distinct variables.  Any extraction algorithms like this should
% be within the VARIABLES section of the code, at the end.

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

% PROGRAMMING NOTE
% TO TEMPORARILY SHOW IMAGES DURING DEBUGGING: 
% figure, imshow(BlurredImage, []), title('BlurredImage') 
% TO TEMPORARILY SAVE IMAGES DURING DEBUGGING: 
% imwrite(BlurredImage, FileName, FileFormat);
% Note that you may have to alter the format of the image before
% saving.  If the image is not saved correctly, for example, try
% adding the uint8 command:
% imwrite(uint8(BlurredImage), FileName, FileFormat);
% To routinely save images produced by this module, see the help in
% the SaveImages module.

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

% PROGRAMMING NOTE
% DISPLAYING RESULTS:
% Each module checks whether its figure is open before calculating
% images that are for display only. This is done by examining all the
% figure handles for one whose handle is equal to the assigned figure
% number for this algorithm. If the figure is not open, everything
% between the "if" and "end" is ignored (to speed execution), so do
% not do any important calculations here. Otherwise an error message
% will be produced if the user has closed the window but you have
% attempted to access data that was supposed to be produced by this
% part of the code. If you plan to save images which are normally
% produced for display only, the corresponding lines should be moved
% outside this if statement.

fieldname = ['figurealgorithm',CurrentAlgorithm];
ThisAlgFigureNumber = handles.(fieldname);
if any(findobj == ThisAlgFigureNumber) == 1;
    %%% Converts the label matrix to a colored label matrix for display and saving
    %%% purposes.
    ColoredSubregionObjectImage = label2rgb(SubregionObjectImage,'jet', 'k', 'shuffle');
% PROGRAMMING NOTE
% DRAWNOW BEFORE FIGURE COMMAND:
% The "drawnow" function executes any pending figure window-related
% commands.  In general, Matlab does not update figure windows until
% breaks between image analysis modules, or when a few select commands
% are used. "figure" and "drawnow" are two of the commands that allow
% Matlab to pause and carry out any pending figure window- related
% commands (like zooming, or pressing timer pause or cancel buttons or
% pressing a help button.)  If the drawnow command is not used
% immediately prior to the figure(ThisAlgFigureNumber) line, then
% immediately after the figure line executes, the other commands that
% have been waiting are executed in the other windows.  Then, when
% Matlab returns to this module and goes to the subplot line, the
% figure which is active is not necessarily the correct one. This
% results in strange things like the subplots appearing in the timer
% window or in the wrong figure window, or in help dialog boxes.
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

% PROGRAMMING NOTE
% HANDLES STRUCTURE:
%       In CellProfiler (and Matlab in general), each independent
% function (module) has its own workspace and is not able to 'see'
% variables produced by other modules. For data or images to be shared
% from one module to the next, they must be saved to what is called
% the 'handles structure'. This is a variable, whose class is
% 'structure', and whose name is handles. Data which should be saved
% to the handles structure within each module includes: any images,
% data or measurements which are to be eventually saved to the hard
% drive (either in an output file, or using the SaveImages module) or
% which are to be used by a later module in the analysis pipeline. Any
% module which produces or passes on an image needs to also pass along
% the original filename of the image, named after the new image name,
% so that if the SaveImages module attempts to save the resulting
% image, it can be named by appending text to the original file name.
%       It is important to think about which of these data should be
% deleted at the end of an analysis run because of the way Matlab
% saves variables: For example, a user might process 12 image sets of
% nuclei which results in a set of 12 measurements ("TotalNucArea")
% stored in the handles structure. In addition, a processed image of
% nuclei from the last image set is left in the handles structure
% ("SegmNucImg"). Now, if the user uses a different algorithm which
% happens to have the same measurement output name "TotalNucArea" to
% analyze 4 image sets, the 4 measurements will overwrite the first 4
% measurements of the previous analysis, but the remaining 8
% measurements will still be present. So, the user will end up with 12
% measurements from the 4 sets. Another potential problem is that if,
% in the second analysis run, the user runs only an algorithm which
% depends on the output "SegmNucImg" but does not run an algorithm
% that produces an image by that name, the algorithm will run just
% fine: it will just repeatedly use the processed image of nuclei
% leftover from the last image set, which was left in the handles
% structure ("SegmNucImg").
%
% INCLUDE FURTHER DESCRIPTION OF MEASUREMENTS PER CELL AND PER IMAGE
% HERE>>>
%
%       Saving measurements: The data extraction functions of
% CellProfiler are designed to deal with only one "column" of data per
% named measurement field. So, for example, instead of creating a
% field of XY locations stored in pairs, it is better to store a field
% of X locations and a field of Y locations. Measurements must be
% stored in double format, because the extraction part of the program
% is designed to deal with that type of array only, not cell or
% structure arrays. It is wise to include the user's input for
% 'ObjectName' as part of the fieldname in the handles structure so
% that multiple modules can be run and their data will not overwrite
% each other.
%
%       Extracting measurements: handles.dMCCenterXNuclei{1}(2) gives
% the X position for the second object in the first image.
% handles.dMCAreaNuclei{2}(1) gives the area of the first object in
% the second image.

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