function handles = AlgSubtractImages(handles)

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
% The Original Code is the Subtract Image Module.
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

%%% The "drawnow" function allows figure windows to be updated and buttons
%%% to be pushed (like the pause, cancel, help, and view buttons).  The
%%% "drawnow" function is sprinkled throughout the algorithm so there are
%%% plenty of breaks where the figure windows/buttons can be interacted
%%% with.
drawnow 

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%

%%% The green "%textVAR" lines contain the text which is displayed in the
%%% GUI when the user is entering the variable values.  The green
%%% "%defaultVAR" lines contain the default values which are displayed in
%%% the variable boxes when the user loads the algorithm.  The spaces are
%%% important for these lines: be sure there is a space before and after
%%% the equals sign and also that the capitalization is as shown.  Don't
%%% allow the text to wrap around to another line; the second line will not be
%%% displayed.  If you need more space to describe a variable, you can
%%% refer the user to the help file, or you can put text in the %textVAR
%%% line below the one of interest.  Then, remove the %defaultVAR line, and
%%% the variable edit box for that variable will not be displayed, but the
%%% text will be displayed.
%%% If some variables are not needed, the textVAR and
%%% defaultVAR lines should be deleted so that these variable windows are
%%% not displayed.  There are 11 variable boxes.  If you need more user
%%% inputs than this, you can use the eleventh variable box to request
%%% several inputs in some structured format; it works just as well but is
%%% not as user-friendly since it
%%% requires that the user not mess up the syntax.  For example, you could
%%% ask they the user enter 5 different values separated by commas and then
%%% write a little extraction algorithm that separates the input into five
%%% distinct variables.

%%% The two lines of code after the textVAR and defaultVAR extract the value
%%% that the user has entered from the handles structure and saves it as a
%%% variable in the workspace of this algorithm with a descriptive name.

%textVAR01 = Subtract this image (enter the name here)
%defaultVAR01 = tubulinw3
fieldname = ['Vvariable',CurrentAlgorithm,'_01'];
SubtractImageName = handles.(fieldname);
%textVAR02 = From this image (enter the name here)
%defaultVAR02 = NHSw1
fieldname = ['Vvariable',CurrentAlgorithm,'_02'];
BasicImageName = handles.(fieldname);
%textVAR03 = What do you want to call the resulting image?
%defaultVAR03 = SubtractedCellStain
fieldname = ['Vvariable',CurrentAlgorithm,'_03'];
ResultingImageName = handles.(fieldname);
%textVAR04 = Enter the factor to multiply the subtracted image by:
%defaultVAR04 = 1
fieldname = ['Vvariable',CurrentAlgorithm,'_04'];
MultiplyFactor = str2num(handles.(fieldname));
%textVAR05 = Contrast stretch the resulting image?
%defaultVAR05 = N
fieldname = ['Vvariable',CurrentAlgorithm,'_05'];
Stretch = handles.(fieldname);
%textVAR06 = Blur radius for the basic image
%defaultVAR06 = 3
fieldname = ['Vvariable',CurrentAlgorithm,'_06'];
BlurRadius = str2num(handles.(fieldname));

%textVAR08 = To save the resulting image, enter text to append to the name 
%defaultVAR08 = N
fieldname = ['Vvariable',CurrentAlgorithm,'_08'];
SaveImage = handles.(fieldname);
%textVAR10 = Otherwise, leave as "N". To save or display other images, press Help button
%textVAR11 = If saving images, what file format do you want to use? Do not include a period.
%defaultVAR11 = tif
fieldname = ['Vvariable',CurrentAlgorithm,'_11'];
FileFormat = handles.(fieldname);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Note to programmers: *Any* data that you write to the handles structure
%%% must be preceded by 'dMT', 'dMC', or 'dOT' so that the data is deleted
%%% at the end of the processing batch.  dMT stands for deletable
%%% Measurements - Total image (i.e. any case where there is one
%%% measurement for the entire image).  dMC stands for deletable
%%% Measurements - Cell by cell (i.e. any case where there is more than one
%%% measurement for the entire image). dOT stands for deletable - OTher,
%%% which would typically include images that are stored in the handles
%%% structure for use by other algorithms.  If the data is not deleted, it
%%% is still available for use if the user runs a completely separate
%%% analysis.  This could be disastrous: For example, a user might process
%%% 12 image sets of nuclei which results in a set of 12 measurements
%%% ("TotalNucArea") stored in the handles structure.  In addition, a
%%% processed image of nuclei from the last image set is left in the
%%% handles structure ("SegmNucImg").  Now, if the user uses a different
%%% algorithm which happens to have the same measurement output name
%%% "TotalNucArea" to analyze 4 image sets, the 4 measurements will
%%% overwrite the first 4 measurements of the previous analysis, but the
%%% remaining 8 measurements will still be present.  So, the user will end
%%% up with 12 measurements from the 4 sets.  Another potential problem is
%%% that if, in the second analysis run, the user runs only an algorithm
%%% which depends on the output "SegmNucImg" but does not run an algorithm
%%% that produces an image by that name, the algorithm will run just fine: it will  
%%% just repeatedly use the processed image of nuclei leftover from the last
%%% image set, which was left in the handles structure ("SegmNucImg").

%%% As a policy, I think it is probably wise to disallow more than one
%%% "column" of data per object, to allow for uniform extraction of data
%%% later. So, for example, instead of creating a field of XY locations
%%% stored in pairs, it is better to store a field of X locations and a
%%% field of Y locations.

%%% Checks whether the file format the user entered is readable by Matlab.
IsFormat = imformats(FileFormat);
if isempty(IsFormat) == 1
    error('The image file type entered in the Subtract Images module is not recognized by Matlab. Or, you may have entered a period in the box. For a list of recognizable image file formats, type "imformats" (no quotes) at the command line in Matlab.','Error')
end
%%% Read (open) the image you want to analyze and assign it to a variable.
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

%%% Check whether the appendages to be added to the file names of images
%%% will result in overwriting the original file, or in a file name that
%%% contains spaces.
        
%%% Determine the filename of the image to be analyzed.
fieldname = ['dOTFilename', BasicImageName];
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
NewImageNameSaveImage = [BareFileName,SaveImage,'.',FileFormat];
%%% Check whether the new image name is going to result in a name with
%%% spaces.
A = isspace(SaveImage);
if any(A) == 1
    error('Image processing was canceled because you have entered one or more spaces in the box of text to append to the object outlines image name in the Subtract Images module.  If you do not want to save the object outlines image to the hard drive, type "N" into the appropriate box.')
end
%%% Check whether the new image name is going to result in overwriting the
%%% original file.
B = strcmp(upper(CharFileName), upper(NewImageNameSaveImage));
if B == 1
    error('Image processing was canceled because you have not entered text to append to the object outlines image name in the Subtract Images module.  If you do not want to save the object outlines image to the hard drive, type "N" into the appropriate box.')
end

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%

%%% Checks that the original image is two-dimensional (i.e. not a color
%%% image), which would disrupt several of the image functions.
if ndims(BasicImage) ~= 2
    error('Image processing was canceled because the Subtract Images module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
end
if ndims(SubtractImage) ~= 2
    error('Image processing was canceled because the Subtract Images module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
end

AdjustedSubtractImage = MultiplyFactor*SubtractImage;
figure, imagesc(AdjustedSubtractImage), colormap(gray)
if BlurRadius ~= 0
    %%% Blurs the image.
%%% Note: using filter2 is much faster than imfilter (e.g. 14.5 sec vs. 99.1 sec).
FiltSize = max(3,ceil(4*BlurRadius));
BasicImage = filter2(fspecial('gaussian',FiltSize, BlurRadius), BasicImage);
end
figure, imagesc(BasicImage), colormap(gray)
ResultingImage = imsubtract(BasicImage,AdjustedSubtractImage);
if strcmp(upper(Stretch),'Y') == 1
ResultingImage = imadjust(ResultingImage,stretchlim(ResultingImage,[.01 .99]));
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
%%% Executes pending figure-related commands so that the results are
%%% displayed.
drawnow

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Saves the processed image to the handles structure.
fieldname = ['dOT',ResultingImageName];
handles.(fieldname) = ResultingImage;
%figure, imagesc(ObjectsIdentifiedImage), colormap(gray), title('ObjectsIdentifiedImage')

%%% The original file name is saved to the handles structure in a
%%% field named after the adjusted image name.
fieldname = ['dOTFilename', ResultingImageName];
handles.(fieldname)(handles.setbeinganalyzed) = FileName;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE PROCESSED IMAGE TO HARD DRIVE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Determine whether the user wanted to save the image of object outlines
%%% by comparing their entry "SaveImage" with "N" (after
%%% converting SaveImage to uppercase).  The appropriate names
%%% were determined towards the beginning of the module during error
%%% checking.
if strcmp(upper(SaveImage),'N') ~= 1
%%% Save the image to the hard drive.    
imwrite(ResultingImage, NewImageNameSaveImage, FileFormat);
end

drawnow 

%%%%%%%%%%%%%%
%%%%%% HELP %%%%%
%%%%%%%%%%%%%%

%%% Help will be automatically extracted if a line is preceded by 5 % signs
%%% and a space.  The text will be displayed with the line breaks intact.
%%% To make a break between paragraphs, put 5% signs, a space, and then one
%%% character (for example a period) on a line.

%%%%% Help for Subtract Images module: 
%%%%% .
%%%%% SPEED OPTIMIZATION: Note that increasing the blur radius increases
%%%%% the processing time exponentially.
%%%%% .
%%%%% DISPLAYING AND SAVING PROCESSED IMAGES 
%%%%% PRODUCED BY THIS IMAGE ANALYSIS MODULE:
%%%%% Note: Images saved using the boxes in the main CellProfiler window
%%%%% will be saved in the default directory specified in STEP 1.
%%%%% If you want to save other processed images, open the m-file for this 
%%%%% image analysis module, go to the line in the
%%%%% m-file where the image is generated, and there should be 2 lines
%%%%% which have been inactivated.  These are green comment lines that are
%%%%% indented. To display an image, remove the percent sign before
%%%%% the line that says "figure, imshow...". This will cause the image to
%%%%% appear in a fresh display window for every image set. To save an
%%%%% image to the hard drive, remove the percent sign before the line
%%%%% that says "imwrite..." and adjust the file type and appendage to the
%%%%% file name as desired.  When you have finished removing the percent
%%%%% signs, go to File > Save As and save the m file with a new name.
%%%%% Then load the new image analysis module into the CellProfiler as
%%%%% usual.
%%%%% Please note that not all of these imwrite lines have been checked for
%%%%% functionality: it may be that you will have to alter the format of
%%%%% the image before saving.  Try, for example, adding the uint8 command:
%%%%% uint8(Image) surrounding the image prior to using the imwrite command
%%%%% if the image is not saved correctly.
