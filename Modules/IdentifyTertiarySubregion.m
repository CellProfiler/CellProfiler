function handles = AlgIdentifyTertiarySubregion1(handles)

%%% Reads the current algorithm number, since this is needed to find 
%%% the variable values that the user entered.
CurrentAlgorithm = handles.currentalgorithm;

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow

%textVAR01 = What did you call the larger identified objects?
%defaultVAR01 = Cells
fieldname = ['Vvariable',CurrentAlgorithm,'_01'];
PrimaryObjectName = handles.(fieldname);
%textVAR02 = What did you call the smaller identified objects?
%defaultVAR02 = Nuclei
fieldname = ['Vvariable',CurrentAlgorithm,'_02'];
SecondaryObjectName = handles.(fieldname);
%textVAR03 = What do you want to call the new subregions?
%defaultVAR03 = Cytoplasm
fieldname = ['Vvariable',CurrentAlgorithm,'_03'];
SubregionObjectName = handles.(fieldname);
%textVAR05 = To save grayscale objects as an image, enter text to append to the image name 
%defaultVAR05 = N
fieldname = ['Vvariable',CurrentAlgorithm,'_05'];
SaveGrayObjects = handles.(fieldname);
%textVAR06 = To save colored object blocks as an image, enter text to append to the name 
%defaultVAR06 = N
fieldname = ['Vvariable',CurrentAlgorithm,'_06'];
SaveColoredObjects = handles.(fieldname);
%textVAR07 =  Otherwise, leave as "N". To save or display other images, press Help button
%textVAR08 = In what file format do you want to save images? Do not include a period
%defaultVAR08 = tif
fieldname = ['Vvariable',CurrentAlgorithm,'_08'];
FileFormat = handles.(fieldname);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Checks whether the file format the user entered is readable by Matlab.
IsFormat = imformats(FileFormat);
if isempty(IsFormat) == 1
    error('The image file type entered in the Identify Tertiary Subregion module is not recognized by Matlab. Or, you may have entered a period in the box. For a list of recognizable image file formats, type "imformats" (no quotes) at the command line in Matlab.','Error')
end
%%% Read (open) the images you want to analyze and assign them to
%%% variables.
fieldname = ['dOTSegmented',PrimaryObjectName];
%%% Check whether the image to be analyzed exists in the handles structure.
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
% figure, imshow(PrimaryObjectImage), title('Primary Object Image')

%%% Similarly, retrieve the Secondary object segmented image.
fieldname = ['dOTSegmented', SecondaryObjectName];
if isfield(handles, fieldname) == 0
    error(['Image processing was canceled because the Identify Tertiary Subregion module could not find the input image.  It was supposed to be named ', SecondaryObjectName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
SecondaryObjectImage = handles.(fieldname);
        % figure, imshow(SecondaryObjectImage), title('Secondary Object Image')
       
%%% Check whether the appendages to be added to the file names of images
%%% will result in overwriting the original file, or in a file name that
%%% contains spaces.
%%% Determine the filename of the image to be analyzed.
fieldname = ['dOTFilename', PrimaryObjectName];
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
NewImageNameSaveGrayObjects = [BareFileName,SaveGrayObjects,'.',FileFormat];
%%% Check whether the new image name is going to result in a name with
%%% spaces.
A = isspace(SaveGrayObjects);
if any(A) == 1
    error('Image processing was canceled because you have entered one or more spaces in the box of text to append to the gray objects image name in the Identify Tertiary Subregion module.  If you do not want to save the object outlines image to the hard drive, type "N" into the appropriate box.')
end
%%% Check whether the new image name is going to result in overwriting the
%%% original file.
B = strcmp(upper(CharFileName), upper(NewImageNameSaveGrayObjects));
if B == 1
    error('Image processing was canceled because you have not entered text to append to the gray objects image name in the Identify Tertiary Subregion module.  If you do not want to save the object outlines image to the hard drive, type "N" into the appropriate box.')
end

%%% Repeat the above for the other image to be saved: 
NewImageNameSaveColoredObjects = [BareFileName,SaveColoredObjects,'.',FileFormat];
A = isspace(SaveColoredObjects);
if any(A) == 1
    error('Image processing was canceled because you have entered one or more spaces in the box of text to append to the colored objects image name in the Identify Tertiary Subregion module.  If you do not want to save the colored objects image to the hard drive, type "N" into the appropriate box.')
end
B = strcmp(upper(CharFileName), upper(NewImageNameSaveColoredObjects));
if B == 1
    error('Image processing was canceled because you have not entered text to append to the colored objects image name in the Identify Tertiary Subregion module.  If you do not want to save the colored objects image to the hard drive, type "N" into the appropriate box.')
end

%%% Checks that the original image is two-dimensional (i.e. not a color
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

%%% The secondary object image is eroded slightly and then subtracted
%%% from the primary object image.  This prevents
%%% the subregion from having zero pixels (which cannot be measured in
%%% subsequent measuremodules) in the cases where the secondary object is
%%% exactly the same size as the primary object.
ErodedSecondaryObjectImage = imerode(SecondaryObjectImage, ones(3));
    % figure, imshow(ErodedSecondaryObjectImage), title('ErodedSecondaryObjectImage')
SubregionObjectImage = PrimaryObjectImage - ErodedSecondaryObjectImage;
    % figure, imshow(SubregionObjectImage), title('Subregion Object Image')
%%% Converts the label matrix to a colored label matrix for display and saving
%%% purposes.
ColoredSubregionObjectImage = label2rgb(SubregionObjectImage,'jet', 'k', 'shuffle');

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
%%% Executes pending figure-related commands so that the results are
%%% displayed.
drawnow

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% The final, segmented label matrix image of secondary objects is saved to the
%%% handles structure so it can be used by subsequent algorithms.
fieldname = ['dOTSegmented', SubregionObjectName];
handles.(fieldname) = SubregionObjectImage;
%%% Removed for parallel: guidata(gcbo, handles);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE PROCESSED IMAGE TO HARD DRIVE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Determine whether the user wanted to save the image of object outlines
%%% by comparing their entry "SaveObjectOutlines" with "N" (after
%%% converting SaveObjectOutlines to uppercase).
if strcmp(upper(SaveGrayObjects),'N') ~= 1
%%% Save the image to the hard drive.    
imwrite(SubregionObjectImage, NewImageNameSaveGrayObjects, FileFormat);
end
%%% Same for the SaveColoredObjects image.
if strcmp(upper(SaveColoredObjects),'N') ~= 1
%%% Save the image to the hard drive.    
imwrite(ColoredSubregionObjectImage, NewImageNameSaveColoredObjects, FileFormat);
end

drawnow

%%%%%%%%%%%
%%% HELP %%%
%%%%%%%%%%%

%%%%% Help for the Identify Tertiary Subregion module: 
%%%%% . 
%%%%% This module will take the identified objects specified in the first
%%%%% box and remove from them the identified objects specified in the
%%%%% second box. For example, "subtracting" the nuclei from the cells will
%%%%% leave just the cytoplasm, the properties of which can then be
%%%%% measured by the MeasureIntensityTexture module. The first objects
%%%%% should therefore be equal in size or larger than the second objects
%%%%% and must completely contain the second objects.  Both images
%%%%% should be the result of a segmentation process, not grayscale images.
%%%%% Note that creating subregions using this module can result in objects
%%%%% that are not contiguous, which does not cause problems when running
%%%%% the Measure Intensity and Texture module, but does cause problems
%%%%% when running the Measure Area Shape Intensity Texture module because
%%%%% calculations of the perimeter, aspect ratio, solidity, etc. cannot be
%%%%% made for noncontiguous objects.
%%%%% .
%%%%% DISPLAYING AND SAVING PROCESSED IMAGES 
%%%%% PRODUCED BY THIS IMAGE ANALYSIS MODULE:
%%%%% Note: Images saved using the boxes in the main CellProfiler window
%%%%% will be saved in the default directory specified in STEP 1.
%%%%% .
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