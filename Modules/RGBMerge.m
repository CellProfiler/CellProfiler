function handles = AlgRGBMerge(handles)

% Help for the RGB Merge module: 
% Category: Pre-processing
%
% Takes 1 to 3 images and assigns them to colors in a final, RGB
% image.  Each color's brightness can be adjusted independently.
%
% Settings:
%
% Adjustment factors: Leaving the adjustment factors set to 1 will
% balance all three colors equally in the final image, and they will
% use the same range of intensities as each individual incoming image.
% Using factors less than 1 will decrease the intensity of that
% color in the final image, and values greater than 1 will increase
% it.  Setting the adjustment factor to zero will cause that color to
% be entirely blank.
%
% SAVING IMAGES: The RGB image produced by this module can be easily
% saved using the Save Images module, using the name you assign. If
% you want to save other intermediate images, alter the code for this
% module to save those images to the handles structure (see the
% SaveImages module help) and then use the Save Images module.
%
% See also ALGRGBSPLIT, ALGRGBTOGRAY.

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
    fieldname = ['', BlueImageName];
    %%% Checks whether the image to be analyzed exists in the handles structure.
if isfield(handles.Pipeline, fieldname)==0,
        %%% If the image is not there, an error message is produced.  The error
        %%% is not displayed: The error function halts the current function and
        %%% returns control to the calling function (the analyze all images
        %%% button callback.)  That callback recognizes that an error was
        %%% produced because of its try/catch loop and breaks out of the image
        %%% analysis loop without attempting further modules.
        error(['Image processing was canceled because the RGB Merge module could not find the input image.  It was supposed to be named ', BlueImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
    end
    %%% Reads the image.
    BlueImage = handles.Pipeline.(fieldname);

    BlueImageExists = 1;
else BlueImageExists = 0;
end

drawnow
%%% Repeat for Green and Red.
if strcmp(upper(GreenImageName), 'N') == 0
    if isfield(handles, GreenImageName) == 0
        error(['Image processing was canceled because the RGB Merge module could not find the input image.  It was supposed to be named ', GreenImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
    end
    GreenImage = handles.Pipeline.(GreenImageName);
    GreenImageExists = 1;
else GreenImageExists = 0;
end
if strcmp(upper(RedImageName), 'N') == 0
    if isfield(handles, RedImageName) == 0
        error(['Image processing was canceled because the RGB Merge module could not find the input image.  It was supposed to be named ', RedImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
    end
    RedImage = handles.Pipeline.(RedImageName);
    RedImageExists = 1;
else RedImageExists = 0;
end
drawnow

%%% If any of the colors are to be left black, creates the appropriate
%%% image.
if BlueImageExists == 0 && RedImageExists == 0 && GreenImageExists == 0
    error('Image processing was canceled because you have not selected any images to be merged in the RGB Merge module.')
end
if BlueImageExists == 0 && RedImageExists == 0 && GreenImageExists == 1
    BlueImage = zeros(size(GreenImage));
    RedImage = zeros(size(GreenImage));
end
if BlueImageExists == 0 && RedImageExists == 1 && GreenImageExists == 0
    BlueImage = zeros(size(RedImage));
    GreenImage = zeros(size(RedImage));
end
if BlueImageExists == 1 && RedImageExists == 0 && GreenImageExists == 0
    RedImage = zeros(size(BlueImage));
    GreenImage = zeros(size(BlueImage));
end
if BlueImageExists == 1 && RedImageExists == 1 && GreenImageExists == 0
    GreenImage = zeros(size(BlueImage));
end
if BlueImageExists == 0 && RedImageExists == 1 && GreenImageExists == 1
    BlueImage = zeros(size(GreenImage));
end
if BlueImageExists == 1 && RedImageExists == 0 && GreenImageExists == 1
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

RGBImage(:,:,1) = immultiply(RedImage,str2double(RedAdjustmentFactor));
RGBImage(:,:,2) = immultiply(GreenImage,str2double(GreenAdjustmentFactor));
RGBImage(:,:,3) = immultiply(BlueImage,str2double(BlueAdjustmentFactor));

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

%%% Saves the adjusted image to the handles structure so it can be used by
%%% subsequent algorithms.
handles.Pipeline.(RGBImageName) = RGBImage;

%%% Determines the filename of the image to be analyzed. Only one of the
%%% original file names is chosen to name this field.
if BlueImageExists == 1
    fieldname = ['Filename', BlueImageName];
elseif GreenImageExists == 1
    fieldname = ['Filename', GreenImageName];
elseif RedImageExists == 1
    fieldname = ['Filename', RedImageName];
end
FileName = handles.Pipeline.(fieldname)(handles.setbeinganalyzed);
%%% Saves the original file name to the handles structure in a
%%% field named after the adjusted image name.
fieldname = ['Filename', RGBImageName];
handles.Pipeline.(fieldname)(handles.setbeinganalyzed) = FileName;