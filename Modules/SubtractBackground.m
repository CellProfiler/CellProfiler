function handles = AlgSubtractBackground(handles)

% Help for the Subtract Background module:
% Category: Pre-processing
%
% Note that this is not an illumination correction module.  It
% subtracts a single value from every pixel across the image.
%
% The intensity due to camera or illumination or antibody background
% (intensity where no cells are sitting) can in good conscience be
% subtracted from the images, but it must be subtracted from every
% pixel, not just the pixels where cells actually are sitting.  This
% is because we assume that this staining is additive with real
% staining. This module calculates the camera background and subtracts
% this background value from each pixel. This module is identical to
% the Apply Threshold and Shift module, except in the Subtract
% Background module, the threshold is automatically calculated the
% first time through the module. This will not push any values below
% zero (therefore, we aren't losing any information).  It moves the
% baseline up and looks prettier (improves signal to noise) without
% any 'ethical' concerns.
%
% If images have already been quantified, then multiply the scalar by
% the number of pixels in the image to get the number that should be
% subtracted from the intensity measurements.
%
% If you want to run this module only to calculate the proper
% threshold to use, simply run the module as usual and use the button
% on the Timer to stop processing after the first image set.
%
% How it works: 
% Sort each image's pixel values and pick the 10th lowest pixel value
% as the minimum.  Our typical images have a million pixels. We are
% not choosing the lowest pixel value, because it might be zero if
% it's a stuck pixel.  We are pretty sure there won't be 10 stuck
% pixels so this should be safe.  Then, take the minimum of these
% values from all the images.  This scalar value should be subtracted
% from every pixel in the image.  We are not calculating a different
% value for each pixel position in the image because in a small image
% set, that position may always be occupied by real staining.
%
% SAVING IMAGES: The corrected image produced by this module can be
% easily saved using the Save Images module, using the name you
% assign. If you want to save other intermediate images, alter the
% code for this module to save those images to the handles structure
% (see the SaveImages module help) and then use the Save Images
% module.
%
% See also ALGAPPLYTHRESHOLDANDSHIFT.

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
% The Original Code is the Subtract Background module.
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

%textVAR01 = What did you call the image to be corrected?
%defaultVAR01 = OrigBlue
ImageName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,1});

%textVAR02 = What do you want to call the corrected image?
%defaultVAR02 = CorrBlue
CorrectedImageName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,2});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a variable,
%%% "OrigImage".
%%% Checks whether the image to be analyzed exists in the handles structure.
if isfield(handles.Pipeline, ImageName) == 0
    %%% If the image is not there, an error message is produced.  The error
    %%% is not displayed: The error function halts the current function and
    %%% returns control to the calling function (the analyze all images
    %%% button callback.)  That callback recognizes that an error was
    %%% produced because of its try/catch loop and breaks out of the image
    %%% analysis loop without attempting further modules.
    error(['Image processing was canceled because the Subtract Background module could not find the input image.  It was supposed to be named ', ImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
%%% Reads the image.
OrigImage = handles.Pipeline.(ImageName);

%%% Checks that the original image is two-dimensional (i.e. not a color
%%% image), which would disrupt several of the image functions.
if ndims(OrigImage) ~= 2
    error('Image processing was canceled because the Subtract Background module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
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

%%% Makes note of the current directory so the module can return to it
%%% at the end of this module.
CurrentDirectory = cd;

%%% The first time the module is run, the threshold shifting value must be
%%% calculated.
if handles.setbeinganalyzed == 1
    try
        drawnow
        %%% Retrieves the path where the images are stored from the handles
        %%% structure.
        fieldname = ['Pathname', ImageName];
        try Pathname = handles.Pipeline.(fieldname);
        catch error('Image processing was canceled because the Subtract Background module must be run using images straight from a load images module (i.e. the images cannot have been altered by other image processing modules). This is because the Subtract Background module calculates an illumination correction image based on all of the images before correcting each individual image as CellProfiler cycles through them. One solution is to process the entire batch of images using the image analysis modules preceding this module and save the resulting images to the hard drive, then start a new stage of processing from this Subtract Background module onward.')
        end
        %%% Changes to that directory.
        cd(Pathname)
        %%% Retrieves the list of filenames where the images are stored from the
        %%% handles structure.
        fieldname = ['FileList', ImageName];
        FileList = handles.Pipeline.(fieldname);
        %%% Calculates the pixel intensity of the pixel that is 10th dimmest in
        %%% each image, then finds the Minimum of that value across all
        %%% images. Our typical images have a million pixels. We are not
        %%% choosing the lowest pixel value, because it might be zero if
        %%% it’s a stuck pixel.  We are pretty sure there won’t be 10 stuck
        %%% pixels so this should be safe.
        %%% Starts with a high value for MinimumTenthMinimumPixelValue;
        MinimumTenthMinimumPixelValue = 1;
        %%% Obtains the screen size.
        ScreenSize = get(0,'ScreenSize');
        ScreenHeight = ScreenSize(4);
        PotentialBottom = [0, (ScreenHeight-720)];
        BottomOfMsgBox = max(PotentialBottom);
        PositionMsgBox = [500 BottomOfMsgBox 350 100];
        TimeStart = clock;
        NumberOfImages = length(FileList);
        WaitbarText = 'Preliminary background calculations underway... ';
        WaitbarHandle = waitbar(1/NumberOfImages, WaitbarText);
        set(WaitbarHandle, 'Position', PositionMsgBox)
        for i=1:NumberOfImages
            Image = im2double(imread(char(FileList(i))));
            SortedColumnImage = sort(reshape(Image, [],1));
            TenthMinimumPixelValue = SortedColumnImage(10);
            if TenthMinimumPixelValue == 0
                msgbox([ImageName , ' image number ', num2str(i), ', and possibly others in the set, has the 10th dimmest pixel equal to zero, which means there is no camera background to subtract, either because the exposure time was very short, or the camera has 10 or more pixels stuck at zero, or that images have been rescaled such that at least 10 pixels are zero, or that for some other reason you have more than 10 pixels of value zero in the image.  This means that the Subtract Background module will not alter the images in any way, although image processing has not been aborted.'], 'Warning', 'warn','replace')
                %%% Stores the minimum tenth minimum pixel value in the handles structure for
                %%% later use.
                fieldname = ['IntensityToShift', ImageName];
                MinimumTenthMinimumPixelValue = 0;
                handles.Pipeline.(fieldname) = 0;
                %%% Determines the figure number to close, because no
                %%% processing will be performed.
                fieldname = ['figurealgorithm',CurrentAlgorithm];
                ThisAlgFigureNumber = handles.(fieldname);
                close(ThisAlgFigureNumber)
                break
            end
            if TenthMinimumPixelValue < MinimumTenthMinimumPixelValue
                MinimumTenthMinimumPixelValue = TenthMinimumPixelValue;
            end
            CurrentTime = clock;
            TimeSoFar = etime(CurrentTime,TimeStart);
            TimePerSet = TimeSoFar/i;
            ImagesRemaining = NumberOfImages - i;
            TimeRemaining = round(TimePerSet*ImagesRemaining);
            WaitbarText = ['Preliminary background calculations underway... ', num2str(TimeRemaining), ' seconds remaining.'];
            waitbar(i/NumberOfImages, WaitbarHandle, WaitbarText);
            drawnow
        end
        close(WaitbarHandle)
    catch [ErrorMessage, ErrorMessage2] = lasterr;
        error(['An error occurred in the Subtract Background module. Matlab says the problem is: ', ErrorMessage, ErrorMessage2])
    end
    %%% Stores the minimum tenth minimum pixel value in the handles structure for
    %%% later use.
    fieldname = ['IntensityToShift', ImageName];
    handles.Pipeline.(fieldname) = MinimumTenthMinimumPixelValue;
end

%%% The following is run for every image set. Retrieves the minimum tenth
%%% minimum pixel value from the handles structure.
fieldname = ['IntensityToShift', ImageName];
MinimumTenthMinimumPixelValue = handles.Pipeline.(fieldname);
if MinimumTenthMinimumPixelValue ~= 0
    %%% Subtracts the MinimumTenthMinimumPixelValue from every pixel in the
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
else CorrectedImage = OrigImage;
end % This end goes with the if MinimumTenthMinimumPixelValue ~= 0 line above.

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
% handles.Pipeline is for storing data which must be retrieved by other modules.
% This data can be overwritten as each image set is processed, or it
% can be generated once and then retrieved during every subsequent image
% set's processing, or it can be saved for each image set by
% saving it according to which image set is being analyzed.
%       Anything stored in handles.Measurements or handles.Pipeline
% will be deleted at the end of the analysis run, whereas anything
% stored in handles.Settings will be retained from one analysis to the
% next. It is important to think about which of these data should be
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
%       Note that two types of measurements are typically made: Object
% and Image measurements.  Object measurements have one number for
% every object in the image (e.g. ObjectArea) and image measurements
% have one number for the entire image, which could come from one
% measurement from the entire image (e.g. ImageTotalIntensity), or
% which could be an aggregate measurement based on individual object
% measurements (e.g. ImageMeanArea).  Use the appropriate prefix to
% ensure that your data will be extracted properly.
%       Saving measurements: The data extraction functions of
% CellProfiler are designed to deal with only one "column" of data per
% named measurement field. So, for example, instead of creating a
% field of XY locations stored in pairs, they should be split into a field
% of X locations and a field of Y locations. Measurements must be
% stored in double format, because the extraction part of the program
% is designed to deal with that type of array only, not cell or
% structure arrays. It is wise to include the user's input for
% 'ObjectName' as part of the fieldname in the handles structure so
% that multiple modules can be run and their data will not overwrite
% each other.
%       Extracting measurements: handles.Measurements.CenterXNuclei{1}(2) gives
% the X position for the second object in the first image.
% handles.Measurements.AreaNuclei{2}(1) gives the area of the first object in
% the second image.

%%% Saves the corrected image to the handles structure so it can be used by
%%% subsequent algorithms.
handles.Pipeline.(CorrectedImageName) = CorrectedImage;

%%% Determines the filename of the image to be analyzed.
fieldname = ['Filename', ImageName];
FileName = handles.Pipeline.(fieldname)(handles.setbeinganalyzed);
%%% Saves the original file name to the handles structure in a
%%% field named after the corrected image name.
fieldname = ['Filename', CorrectedImageName];
handles.Pipeline.(fieldname)(handles.setbeinganalyzed) = FileName;