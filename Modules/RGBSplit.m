function handles = AlgRGBSplit1(handles)
%%% Reads the current algorithm number, since this is needed to find 
%%% the variable values that the user entered.
CurrentAlgorithm = handles.currentalgorithm;

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow

%textVAR01 = What did you call the image to be split into black and white images?
%defaultVAR01 = OrigRGB
fieldname = ['Vvariable',CurrentAlgorithm,'_01'];
RGBImageName = handles.(fieldname);
%textVAR02 = What do you want to call the image that was red?
%defaultVAR02 = OrigRed
fieldname = ['Vvariable',CurrentAlgorithm,'_02'];
RedImageName = handles.(fieldname);
%textVAR03 = What do you want to call the image that was green?
%defaultVAR03 = OrigGreen
fieldname = ['Vvariable',CurrentAlgorithm,'_03'];
GreenImageName = handles.(fieldname);
%textVAR04 = What do you want to call the image that was blue?
%defaultVAR04 = OrigBlue
fieldname = ['Vvariable',CurrentAlgorithm,'_04'];
BlueImageName = handles.(fieldname);
%textVAR05 = Type "N" in any slots above to ignore that color.
%textVAR06 = To save the red image, enter text to append to the image name
%defaultVAR06 = N
fieldname = ['Vvariable',CurrentAlgorithm,'_06'];
RedTextAppend = handles.(fieldname);
%textVAR07 = To save the green image, enter text to append to the image name
%defaultVAR07 = N
fieldname = ['Vvariable',CurrentAlgorithm,'_07'];
GreenTextAppend = handles.(fieldname);
%textVAR08 = To save the blue image, enter text to append to the image name
%defaultVAR08 = N
fieldname = ['Vvariable',CurrentAlgorithm,'_08'];
BlueTextAppend = handles.(fieldname);
%textVAR09 =  Otherwise, leave as "N".
%textVAR10 = In what file format do you want to save images? Do not include a period
%defaultVAR10 = tif
fieldname = ['Vvariable',CurrentAlgorithm,'_10'];
FileFormat = handles.(fieldname);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Checks whether the file format the user entered is readable by Matlab.
IsFormat = imformats(FileFormat);
if isempty(IsFormat) == 1
    error('The image file type entered in the SplitRGB module is not recognized by Matlab. Or, you may have entered a period in the box. For a list of recognizable image file formats, type "imformats" (no quotes) at the command line in Matlab.','Error')
end

%%% Retrieves the RGB image from the handles structure.
fieldname = ['dOT', RGBImageName];
%%% Check whether the image to be analyzed exists in the handles structure.
if isfield(handles, fieldname) == 0
    %%% If the image is not there, an error message is produced.  The error
    %%% is not displayed: The error function halts the current function and
    %%% returns control to the calling function (the analyze all images
    %%% button callback.)  That callback recognizes that an error was
    %%% produced because of its try/catch loop and breaks out of the image
    %%% analysis loop without attempting further modules.
    error(['Image processing was canceled because the SplitRGB module could not find the input image.  It was supposed to be named ', RGBImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
%%% Read the image.
RGBImage = handles.(fieldname);
Size = size(RGBImage);
if length(Size) ~= 3
    error(['Image processing was canceled because the RGB image you specified in the SplitRGB module could not be separated into three layers of image data.  Is it a color image?  This module was only tested with TIF and BMP images.'])
end
if Size(3) ~= 3
    error(['Image processing was canceled because the RGB image you specified in the SplitRGB module could not be separated into three layers of image data.  This module was only tested with TIF and BMP images.'])
end
%%% Determine the filename of the image to be analyzed.
fieldname = ['dOTFilename', RGBImageName];
FileName = handles.(fieldname)(handles.setbeinganalyzed);
%%% Check whether the appendages to be added to the file names of images
%%% will result in overwriting the original file, or in a file name that
%%% contains spaces.
if strcmp(upper(GreenTextAppend),'N') ~= 1 | strcmp(upper(BlueTextAppend),'N') ~= 1 | strcmp(upper(RedTextAppend),'N') ~= 1
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
end
if strcmp(upper(RedTextAppend),'N') ~= 1
    %%% Assemble the new image name.
    NewImageNameRed = [BareFileName,RedTextAppend,'.',FileFormat];
    %%% Check whether the new image name is going to result in a name with
    %%% spaces.
    A = isspace(NewImageNameRed);
    if any(A) == 1
        NewImageNameRed = strrep(NewImageNameRed,'','');
        RedHandle = errordlg('The file name for the Red Image would contain one or more spaces, either because the original name had spaces or you entered a space in the box of text to append to the Red image name in the SplitRGB module. Spaces have been removed from the name and image processing will continue.','Red file name will be changed','on');
    end
    %%% Check whether the new image name is going to result in overwriting the
    %%% original file.
    B = strcmp(upper(CharFileName), upper(NewImageNameRed));
    if B == 1
        error('Image processing was canceled because you have not entered text to append to the Red image name in the SplitRGB module.  If you do not want to save that image to the hard drive, type "N" into the appropriate box.')
        return
    end
end
%%% Repeat for Green and Blue.
if strcmp(upper(GreenTextAppend),'N') ~= 1
    NewImageNameGreen = [BareFileName,GreenTextAppend,'.',FileFormat];
    A = isspace(NewImageNameGreen);
    if any(A) == 1
        NewImageNameGreen = strrep(NewImageNameGreen,'','');
        GreenHandle = errordlg('The file name for the Green Image would contain one or more spaces, either because the original name had spaces or you entered a space in the box of text to append to the Green image name in the SplitRGB module. Spaces have been removed from the name and image processing will continue.','Green file name will be changed','on');
    end
    B = strcmp(upper(CharFileName), upper(NewImageNameGreen));
    if B == 1
        error('Image processing was canceled because you have not entered text to append to the Green image name in the SplitRGB module.  If you do not want to save that image to the hard drive, type "N" into the appropriate box.')
        return
    end
end

if strcmp(upper(BlueTextAppend),'N') ~= 1
    NewImageNameBlue = [BareFileName,BlueTextAppend,'.',FileFormat];
    A = isspace(NewImageNameBlue);
    if any(A) == 1
        NewImageNameBlue = strrep(NewImageNameBlue,'','');
        BlueHandle = errordlg('The file name for the Blue Image would contain one or more spaces, either because the original name had spaces or you entered a space in the box of text to append to the Blue image name in the SplitRGB module. Spaces have been removed from the name and image processing will continue.','Blue file name will be changed','on');
    end
    B = strcmp(upper(CharFileName), upper(NewImageNameBlue));
    if B == 1
        error('Image processing was canceled because you have not entered text to append to the Blue image name in the SplitRGB module.  If you do not want to save that image to the hard drive, type "N" into the appropriate box.')
        return
    end
end
drawnow

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%

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
%%% Executes pending figure-related commands so that the results are
%%% displayed.
drawnow

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% The adjusted image is saved to the
%%% handles structure so it can be used by subsequent algorithms.
fieldname = ['dOT', RedImageName];
handles.(fieldname) = RedImage;
fieldname = ['dOT', GreenImageName];
handles.(fieldname) = GreenImage;
fieldname = ['dOT', BlueImageName];
handles.(fieldname) = BlueImage;
%%% Removed for parallel: guidata(gcbo, handles);

%%% The original file name is saved to the handles structure in a
%%% field named after the adjusted image name.
fieldname = ['dOTFilename', RGBImageName];
handles.(fieldname)(handles.setbeinganalyzed) = FileName;
fieldname = ['dOTFilename', RedImageName];
handles.(fieldname)(handles.setbeinganalyzed) = FileName;
fieldname = ['dOTFilename', GreenImageName];
handles.(fieldname)(handles.setbeinganalyzed) = FileName;
fieldname = ['dOTFilename', BlueImageName];
handles.(fieldname)(handles.setbeinganalyzed) = FileName;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE PROCESSED IMAGE TO HARD DRIVE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Determine whether the user wanted to save the adjusted image
%%% by comparing their entry "RedTextAppend" with "N" (after
%%% converting RedTextAppend to uppercase).
if strcmp(upper(RedTextAppend),'N') ~= 1
    %%% Save the image to the hard drive.    
    imwrite(RedImage, NewImageNameRed, FileFormat);
end
if strcmp(upper(GreenTextAppend),'N') ~= 1
    %%% Save the image to the hard drive.    
    imwrite(GreenImage, NewImageNameGreen, FileFormat);
end
if strcmp(upper(BlueTextAppend),'N') ~= 1
    %%% Save the image to the hard drive.    
    imwrite(BlueImage, NewImageNameBlue, FileFormat);
end
drawnow

%%%%%%%%%%%
%%% HELP %%%
%%%%%%%%%%%

%%%%% Help for the SplitRGB module: 
%%%%% .