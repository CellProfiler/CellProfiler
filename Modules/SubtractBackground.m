function handles = SubtractBackground(handles)

% Help for the Subtract Background module:
% Category: Image Processing
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
% See also APPLYTHRESHOLDANDSHIFT.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne Carpenter <carpenter@wi.mit.edu>
%   Thouis Jones   <thouis@csail.mit.edu>
%   In Han Kang    <inthek@mit.edu>
%   Ola Friman     <friman@bwh.harvard.edu>
%   Steve Lowe     <stevelowe@alum.mit.edu>
%   Joo Han Chang  <joohan.chang@gmail.com>
%   Colin Clarke   <colinc@mit.edu>
%   Mike Lamprecht <mrl@wi.mit.edu>
%   Susan Ma       <xuefang_ma@wi.mit.edu>
%
% $Revision$




drawnow

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%



%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

%textVAR01 = What did you call the image to be corrected?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the corrected image?
%defaultVAR02 = CorrBlue
%infotypeVAR02 = imagegroup indep
CorrectedImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a
%%% variable.
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



%%% The first time the module is run, the threshold shifting value must be
%%% calculated.
if handles.Current.SetBeingAnalyzed == 1
    try
        drawnow
        %%% Retrieves the path where the images are stored from the handles
        %%% structure.
        fieldname = ['Pathname', ImageName];
        try Pathname = handles.Pipeline.(fieldname);
        catch error('Image processing was canceled because the Subtract Background module must be run using images straight from a load images module (i.e. the images cannot have been altered by other image processing modules). This is because the Subtract Background module calculates an illumination correction image based on all of the images before correcting each individual image as CellProfiler cycles through them. One solution is to process the entire batch of images using the image analysis modules preceding this module and save the resulting images to the hard drive, then start a new stage of processing from this Subtract Background module onward.')
        end
        %%% Retrieves the list of filenames where the images are stored from the
        %%% handles structure.
        fieldname = ['FileList', ImageName];
        FileList = handles.Pipeline.(fieldname);
        %%% Calculates the pixel intensity of the pixel that is 10th dimmest in
        %%% each image, then finds the Minimum of that value across all
        %%% images. Our typical images have a million pixels. We are not
        %%% choosing the lowest pixel value, because it might be zero if
        %%% it?s a stuck pixel.  We are pretty sure there won?t be 10 stuck
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
            [Image, handles] = CPimread(fullfile(Pathname,char(FileList(i))), handles);
            SortedColumnImage = sort(reshape(Image, [],1));
            TenthMinimumPixelValue = SortedColumnImage(10);
            if TenthMinimumPixelValue == 0
                CPmsgbox([ImageName , ' image number ', num2str(i), ', and possibly others in the set, has the 10th dimmest pixel equal to zero, which means there is no camera background to subtract, either because the exposure time was very short, or the camera has 10 or more pixels stuck at zero, or that images have been rescaled such that at least 10 pixels are zero, or that for some other reason you have more than 10 pixels of value zero in the image.  This means that the Subtract Background module will not alter the images in any way, although image processing has not been aborted.'], 'Warning', 'warn','replace')
                %%% Stores the minimum tenth minimum pixel value in the handles structure for
                %%% later use.
                fieldname = ['IntensityToShift', ImageName];
                MinimumTenthMinimumPixelValue = 0;
                handles.Pipeline.(fieldname) = 0;
                %%% Determines the figure number to close, because no
                %%% processing will be performed.
                fieldname = ['FigureNumberForModule',CurrentModule];
                ThisModuleFigureNumber = handles.Current.(fieldname);
                close(ThisModuleFigureNumber)
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


    %%%%%%%%%%%%%%%%%%%%%%
    %%% DISPLAY RESULTS %%%
    %%%%%%%%%%%%%%%%%%%%%%
    drawnow


fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
    if any(findobj == ThisModuleFigureNumber) == 1;

        drawnow
        %%% Activates the appropriate figure window.
        CPfigure(handles,ThisModuleFigureNumber);
        %%% Sets the figure window to half width the first time through.
        originalsize = get(ThisModuleFigureNumber, 'position');
        newsize = originalsize;
        if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
            newsize(3) = originalsize(3)*.5;
            set(ThisModuleFigureNumber, 'position', newsize);
        end
        newsize(1) = 0;
        newsize(2) = 0;
        newsize(4) = 20;
        displaytexthandle = uicontrol(ThisModuleFigureNumber,'style','text', 'position', newsize,'fontname','fixedwidth','backgroundcolor',[0.7,0.7,0.7], 'FontSize',handles.Current.FontSize);
        %%% A subplot of the figure window is set to display the original
        %%% image, some intermediate images, and the final corrected image.
        subplot(2,1,1); imagesc(OrigImage);
        title(['Input Image, Image Set # ',num2str(handles.Current.SetBeingAnalyzed)]);
        %%% The mean image does not absolutely have to be present in order to
        %%% carry out the calculations if the illumination image is provided,
        %%% so the following subplot is only shown if MeanImage exists in the
        %%% workspace.
        subplot(2,1,2); imagesc(CorrectedImage);
        title('Corrected Image'); 
        %%% Displays the text.
        displaytext = ['Background threshold used: ', num2str(MinimumTenthMinimumPixelValue)];
        set(displaytexthandle,'string',displaytext)
        set(ThisModuleFigureNumber,'toolbar','figure')
    end
else CorrectedImage = OrigImage;
end % This end goes with the if MinimumTenthMinimumPixelValue ~= 0 line above.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow



%%% Saves the corrected image to the handles structure so it can be used by
%%% subsequent modules.
handles.Pipeline.(CorrectedImageName) = CorrectedImage;
