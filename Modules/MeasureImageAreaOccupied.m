function handles = AlgMeasureAreaOccupied4(handles)

%%% Reads the current algorithm number, since this is needed to find 
%%% the variable values that the user entered.
CurrentAlgorithm = handles.currentalgorithm;

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow

%textVAR01 = What did you call the images you want to process? 
%defaultVAR01 = OrigGreen
fieldname = ['Vvariable',CurrentAlgorithm,'_01'];
ImageName = handles.(fieldname);
%textVAR02 = What do you want to call the objects measured by this algorithm?
%defaultVAR02 = Cells
fieldname = ['Vvariable',CurrentAlgorithm,'_02'];
ObjectName = handles.(fieldname);

%textVAR04 = Enter the threshold [0 = automatically calculate] (Positive number, Max = 1):
%defaultVAR04 = 0
fieldname = ['Vvariable',CurrentAlgorithm,'_04'];
Threshold = str2num(handles.(fieldname));
%textVAR05 = If auto threshold, enter an adjustment factor (Positive number, 1 = no adjustment):
%defaultVAR05 = 0.75
fieldname = ['Vvariable',CurrentAlgorithm,'_05'];
ThresholdAdjustmentFactor = str2num(handles.(fieldname));

%textVAR07 = To save the adjusted image, enter text to append to the image name 
%defaultVAR07 = N
fieldname = ['Vvariable',CurrentAlgorithm,'_07'];
SaveImage = handles.(fieldname);
%textVAR08 =  Otherwise, leave as "N". To save or display other images, press Help button
%textVAR09 = In what file format do you want to save images? Do not include a period
%defaultVAR09 = tif
fieldname = ['Vvariable',CurrentAlgorithm,'_09'];
FileFormat = handles.(fieldname);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Checks whether the file format the user entered is readable by Matlab.
IsFormat = imformats(FileFormat);
if isempty(IsFormat) == 1
    error('The image file type entered in the Area Occupied module is not recognized by Matlab. Or, you may have entered a period in the box. For a list of recognizable image file formats, type "imformats" (no quotes) at the command line in Matlab.','Error')
end
%%% Retrieves the pixel size that the user entered (micrometers per pixel).
PixelSize = str2num(handles.Vpixelsize{1});
%%% Read (open) the image you want to analyze and assign it to a variable,
%%% "OrigImageToBeAnalyzed".
fieldname = ['dOT', ImageName];
%%% Checks whether image has been loaded.
if isfield(handles, fieldname) == 0
    %%% If the image is not there, an error message is produced.  The error
    %%% is not displayed: The error function halts the current function and
    %%% returns control to the calling function (the analyze all images
    %%% button callback.)  That callback recognizes that an error was
    %%% produced because of its try/catch loop and breaks out of the image
    %%% analysis loop without attempting further modules.
    error(['Image processing was canceled because the Area Occupied module could not find the input image.  It was supposed to be named ', ImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
OrigImageToBeAnalyzed = handles.(fieldname);
        % figure, imshow(OrigImageToBeAnalyzed), title('OrigImageToBeAnalyzed')
%%% Update the handles structure.
%%% Removed for parallel: guidata(gcbo, handles);

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
    error('Image processing was canceled because you have entered one or more spaces in the box of text to append to the object outlines image name in the CellSegment17 module.  If you do not want to save the object outlines image to the hard drive, type "N" into the appropriate box.')
end
%%% Check whether the new image name is going to result in overwriting the
%%% original file.
B = strcmp(upper(CharFileName), upper(NewImageName));
if B == 1
    error('Image processing was canceled because you have not entered text to append to the object outlines image name in the CellSegment17 module.  If you do not want to save the object outlines image to the hard drive, type "N" into the appropriate box.')
end

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%

%%% Determines the threshold to be used, if the user has left the Threshold
%%% variable set to 0.
if Threshold == 0
    Threshold = graythresh(OrigImageToBeAnalyzed);
    Threshold = Threshold*ThresholdAdjustmentFactor;
end
%%% Threshold the original image.
ThresholdedOrigImage = im2bw(OrigImageToBeAnalyzed, Threshold);
         % figure, imshow(ThresholdedOrigImage, []), title('ThresholdedOrigImage')
          % imwrite(ThresholdedOrigImage, [NewImageName,'TOI','.',FileFormat], FileFormat);
AreaOccupiedPixels = sum(ThresholdedOrigImage(:));
AreaOccupied = AreaOccupiedPixels*PixelSize*PixelSize;

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
    %%% Sets the width of the figure window to be appropriate (half width).
    if handles.setbeinganalyzed == 1
        originalsize = get(ThisAlgFigureNumber, 'position');
        newsize = originalsize;
        newsize(3) = 0.5*originalsize(3);
        set(ThisAlgFigureNumber, 'position', newsize);
    end
    %%% Activates the appropriate figure window.
    figure(ThisAlgFigureNumber);
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,1,1); imagesc(OrigImageToBeAnalyzed);colormap(gray);
    title(['Input Image, Image Set # ',num2str(handles.setbeinganalyzed)]);
    %%% A subplot of the figure window is set to display the colored label
    %%% matrix image.
    subplot(2,1,2); imagesc(ThresholdedOrigImage); title('Thresholded Image');
    
    displaytexthandle = uicontrol(ThisAlgFigureNumber,'style','text', 'position', [0 0 235 30],'fontname','fixedwidth','backgroundcolor',[0.7,0.7,0.7]);
    displaytext = strvcat(['      Image Set # ',num2str(handles.setbeinganalyzed)],...
        ['Area occupied by ', ObjectName ,':      ', num2str(AreaOccupied, '%2.1E')]);
    set(displaytexthandle,'string',displaytext)
    set(ThisAlgFigureNumber,'toolbar','figure')
   end
%%% Executes pending figure-related commands so that the results are
%%% displayed.
drawnow

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fieldname = ['dMTAreaOccupied', ObjectName];
handles.(fieldname)(handles.setbeinganalyzed) = {AreaOccupied};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE PROCESSED IMAGE TO HARD DRIVE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Determine whether the user wanted to save the adjusted image
%%% by comparing their entry "SaveImage" with "N" (after
%%% converting SaveImage to uppercase).
if strcmp(upper(SaveImage),'N') ~= 1
%%% Save the image to the hard drive.    
imwrite(ThresholdedOrigImage, NewImageName, FileFormat);
end

%%%%%%%%%%%
%%% HELP %%%
%%%%%%%%%%%

%%%%% Help for the AreaOccupied module: 
%%%%% .
%%%%% Note: Images saved using the boxes in the main CellProfiler window
%%%%% will be saved in the default directory specified in STEP 1.
%%%%% .
%%%%% This module simply applies a threshold to the incoming image so that
%%%%% any pixels brighter than the specified value are assigned the value 1
%%%%% (white) and the remaining pixels are assigned the value zero (black),
%%%%% producing a binary image.  The number of white pixels are then
%%%%% counted.  This provides a measurement of the area occupied by
%%%%% fluorescence.  The threshold is calculated automatically and then
%%%%% adjusted by a user-specified factor.