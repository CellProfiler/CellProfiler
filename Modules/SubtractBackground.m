function handles = AlgSubtractBackground1(handles)

%%% Reads the current algorithm number, since this is needed to find 
%%% the variable values that the user entered.
CurrentAlgorithm = handles.currentalgorithm;

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow

%textVAR01 = What did you call the image to be corrected?
%defaultVAR01 = OrigBlue
fieldname = ['Vvariable',CurrentAlgorithm,'_01'];
ImageName = handles.(fieldname);
%textVAR02 = What do you want to call the corrected image?
%defaultVAR02 = CorrBlue
fieldname = ['Vvariable',CurrentAlgorithm,'_02'];
CorrectedImageName = handles.(fieldname);
%textVAR03 = To save each corrected image, enter text to append to the image name 
%defaultVAR03 = N
fieldname = ['Vvariable',CurrentAlgorithm,'_03'];
SaveImage = handles.(fieldname);
%textVAR04 =  Otherwise, leave as "N". To save or display other images, press Help button
%textVAR05 = In what file format do you want to save images? Do not include a period
%defaultVAR05 = tif
fieldname = ['Vvariable',CurrentAlgorithm,'_05'];
FileFormat = handles.(fieldname);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow
%%% Checks whether the file format the user entered is readable by Matlab.
IsFormat = imformats(FileFormat);
if isempty(IsFormat) == 1
    error('The image file type entered in the Subtract Background module is not recognized by Matlab. Or, you may have entered a period in the box. For a list of recognizable image file formats, type "imformats" (no quotes) at the command line in Matlab.','Error')
end
%%% Read (open) the image you want to analyze and assign it to a variable,
%%% "OrigImage".
fieldname = ['dOT', ImageName];
%%% Check whether the image to be analyzed exists in the handles structure.
if isfield(handles, fieldname) == 0
    %%% If the image is not there, an error message is produced.  The error
    %%% is not displayed: The error function halts the current function and
    %%% returns control to the calling function (the analyze all images
    %%% button callback.)  That callback recognizes that an error was
    %%% produced because of its try/catch loop and breaks out of the image
    %%% analysis loop without attempting further modules.
    error(['Image processing was canceled because the Subtract Background module could not find the input image.  It was supposed to be named ', ImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
%%% Read the image.
OrigImage = handles.(fieldname);
        % figure, imshow(OrigImage), title('OrigImage')

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
    error('Image processing was canceled because you have entered one or more spaces in the box of text to append to the object outlines image name in the Subtract Background module.  If you do not want to save the object outlines image to the hard drive, type "N" into the appropriate box.')
    return
end
%%% Check whether the new image name is going to result in overwriting the
%%% original file.
B = strcmp(upper(CharFileName), upper(NewImageName));
if B == 1
    error('Image processing was canceled because you have not entered text to append to the object outlines image name in the Subtract Background module.  If you do not want to save the object outlines image to the hard drive, type "N" into the appropriate box.')
    return
end

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%

%%% Checks that the original image is two-dimensional (i.e. not a color
%%% image), which would disrupt several of the image functions.
if ndims(OrigImage) ~= 2
    error('Image processing was canceled because the Subtract Background module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
end
%%% Make note of the current directory so the module can return to it
%%% at the end of this module.
CurrentDirectory = cd;

%%% The first time the module is run, the threshold shifting value must be
%%% calculated.
if handles.setbeinganalyzed == 1
    try
        drawnow
        %%% Retrieve the path where the images are stored from the handles
        %%% structure.
        fieldname = ['dOTPathName', ImageName];
        try PathName = handles.(fieldname);
        catch error('Image processing was canceled because the Subtract Background module must be run using images straight from a load images module (i.e. the images cannot have been altered by other image processing modules). This is because the Subtract Background module calculates an illumination correction image based on all of the images before correcting each individual image as CellProfiler cycles through them. One solution is to process the entire batch of images using the image analysis modules preceding this module and save the resulting images to the hard drive, then start a new stage of processing from this Subtract Background module onward.')
        end
        %%% Change to that directory.
        cd(PathName)
        %%% Retrieve the list of filenames where the images are stored from the
        %%% handles structure.
        fieldname = ['dOTFileList', ImageName];
        FileList = handles.(fieldname);
        %%% Calculates the pixel intensity of the pixel that is 10th dimmest in
        %%% each image, then finds the Minimum of that value across all
        %%% images. Our typical images have a million pixels. We are not
        %%% choosing the lowest pixel value, because it might be zero if
        %%% it’s a stuck pixel.  We are pretty sure there won’t be 10 stuck
        %%% pixels so this should be safe.
        %%% Starts with a high value for MinimumTenthMinimumPixelValue;
        MinimumTenthMinimumPixelValue = 1;
        %%% Waitbar shows the percentage of image sets remaining.
        WaitbarHandle = waitbar(0,'Preliminary calculations are under way for the Subtract Background module.  Subsequent image sets will be processed more quickly than the first image set.');
        %%% Obtains the screen size.
        ScreenSize = get(0,'ScreenSize');
        ScreenHeight = ScreenSize(4);
        PotentialBottom = [0, (ScreenHeight-720)];
        BottomOfMsgBox = max(PotentialBottom);
        PositionMsgBox = [500 BottomOfMsgBox 350 100];
        set(WaitbarHandle, 'Position', [PositionMsgBox])
        TimeStart = clock;
        NumberOfImages = length(FileList);
        for i=1:NumberOfImages
            Image = im2double(imread(char(FileList(i))));
            SortedColumnImage = sort(reshape(Image, [],1));
            TenthMinimumPixelValue = SortedColumnImage(10);
            if TenthMinimumPixelValue == 0
                msgbox(['Image number ', num2str(i), ', and possibly others in the set, has the 10th dimmest pixel equal to zero, which means there is no camera background to subtract, either because the exposure time was very short, or the camera has 10 or more pixels stuck at zero, or that images have been rescaled such that at least 10 pixels are zero, or that for some other reason you have more than 10 pixels of value zero in the image.  This means that the Subtract Background module will not alter the images in any way, although image processing has not been aborted.'], 'Warning', 'warn','replace')    
            end
            if TenthMinimumPixelValue < MinimumTenthMinimumPixelValue
                MinimumTenthMinimumPixelValue = TenthMinimumPixelValue;
            end
            CurrentTime = clock;
            TimeSoFar = etime(CurrentTime,TimeStart);
            TimePerSet = TimeSoFar/i;
            ImagesRemaining = NumberOfImages - i;
            TimeRemaining = round(TimePerSet*ImagesRemaining);
            WaitbarText = ['Extracting measurements... ', num2str(TimeRemaining), ' seconds remaining.'];
            waitbar(i/NumberOfImages, WaitbarHandle, WaitbarText)
            drawnow
        end
        close(WaitbarHandle) 
    catch [ErrorMessage, ErrorMessage2] = lasterr;
        error(['An error occurred in the Subtract Background module. Matlab says the problem is: ', ErrorMessage, ErrorMessage2])
    end
    %%% Stores the minimum tenth minimum pixel value in the handles structure for
    %%% later use.
    fieldname = ['dOTIntensityToShift', ImageName];
    handles.(fieldname) = MinimumTenthMinimumPixelValue;        
    %%% Updates the handles structure.
    %%% Removed for parallel: guidata(gcbo, handles);
end

%%% The following is run for every image set. Retrieves the minimum tenth
%%% minimum pixel value from the handles structure.
    fieldname = ['dOTIntensityToShift', ImageName];
    MinimumTenthMinimumPixelValue = handles.(fieldname);        
%%% The MinimumTenthMinimumPixelValue is subtracted from every pixel in the
%%% original image.  This strategy is similar to that used for the "Apply
%%% Threshold and Shift" module.
CorrectedImage = OrigImage - MinimumTenthMinimumPixelValue;
%%% Values below zero are set to zero.
CorrectedImage(CorrectedImage < 0) = 0;

%%% Returns to the original directory.
cd(CurrentDirectory)

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
    %%% Sets the figure window to half width the first time through.
    originalsize = get(ThisAlgFigureNumber, 'position');
    newsize = originalsize;
    if handles.setbeinganalyzed == 1
        newsize(3) = originalsize(3)*.5;
        set(ThisAlgFigureNumber, 'position', newsize);
    end
    newsize(1) = 0;
    newsize(2) = 0;
    newsize(4) = 20;
    displaytexthandle = uicontrol(ThisAlgFigureNumber,'style','text', 'position', newsize,'fontname','fixedwidth','backgroundcolor',[0.7,0.7,0.7]);
    %%% A subplot of the figure window is set to display the original
    %%% image, some intermediate images, and the final corrected image.
    subplot(2,1,1); imagesc(OrigImage);
    title(['Input Image, Image Set # ',num2str(handles.setbeinganalyzed)]);
    colormap(gray)
    %%% The mean image does not absolutely have to be present in order to
    %%% carry out the calculations if the illumination image is provided,
    %%% so the following subplot is only shown if MeanImage exists in the
    %%% workspace.
    subplot(2,1,2); imagesc(CorrectedImage); 
    title('Corrected Image'); colormap(gray)
    %%% Displays the text.
    displaytext = ['Background threshold used: ', num2str(MinimumTenthMinimumPixelValue)];
    set(displaytexthandle,'string',displaytext)
    set(ThisAlgFigureNumber,'toolbar','figure')
end
%%% Executes pending figure-related commands so that the results are
%%% displayed.
drawnow

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% The corrected image is saved to the
%%% handles structure so it can be used by subsequent algorithms.
fieldname = ['dOT', CorrectedImageName];
handles.(fieldname) = CorrectedImage;
%%% Removed for parallel: guidata(gcbo, handles);

%%% The original file name is saved to the handles structure in a
%%% field named after the corrected image name.
fieldname = ['dOTFilename', CorrectedImageName];
handles.(fieldname)(handles.setbeinganalyzed) = FileName;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE PROCESSED IMAGE TO HARD DRIVE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Determine whether the user wanted to save the corrected image
%%% by comparing their entry "SaveImage" with "N" (after
%%% converting SaveImage to uppercase).
if strcmp(upper(SaveImage),'N') ~= 1
%%% Save the image to the hard drive.    
imwrite(uint8(CorrectedImage), NewImageName, FileFormat);
end

drawnow

%%%%%%%%%%%
%%% HELP %%%
%%%%%%%%%%%

%%%%% Help for the Subtract Background module:  
%%%%% .
%%%%% The intensity due to camera or illumination or antibody background
%%%%% (intensity where no cells are sitting) can in good conscience be
%%%%% subtracted from the images, but it must be subtracted from every
%%%%% pixel, not just the pixels where cells actually are sitting.  This is
%%%%% because we assume that this staining is additive with real staining.
%%%%% This module calculates the camera background and subtracts this
%%%%% background value from each pixel. This module is identical to the
%%%%% Apply Threshold and Shift module, except in the Subtract Background
%%%%% module, the threshold is automatically calculated the first time
%%%%% through the module. This will not push any values below zero
%%%%% (therefore, we aren’t losing any information).  It moves the baseline
%%%%% up and looks prettier (improves signal to noise) without any
%%%%% 'ethical' concerns. 
%%%%% . 
%%%%% How it actually works: Sort each image’s pixel values and pick the
%%%%% 10th lowest pixel value as the minimum.  Our typical images have a
%%%%% million pixels. We are not choosing the lowest pixel value, because
%%%%% it might be zero if it’s a stuck pixel.  We are pretty sure there
%%%%% won’t be 10 stuck pixels so this should be safe.  Then, take the
%%%%% minimum of these values from all the images.  This scalar value
%%%%% should be subtracted from every pixel in the image.  We are not
%%%%% calculating a different value for each pixel position in the image
%%%%% because in a small image set, that position may always be occupied by
%%%%% real staining. 
%%%%% . 
%%%%% If images have already been quantified, then multiply the scalar by
%%%%% the number of pixels in the image to get the number that should be
%%%%% subtracted from the intensity measurements. 
%%%%% . 
%%%%% If you want to run this module only to calculate the proper threshold
%%%%% to use, simply run the module as usual and use the button on the
%%%%% Timer to stop processing after the first image set.
%%%%% . 
%%%%% DISPLAYING AND SAVING PROCESSED IMAGES 
%%%%% PRODUCED BY THIS IMAGE ANALYSIS MODULE:
%%%%% If you want to save processed images, open the m-file for this image
%%%%% analysis module, go to the line in the
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