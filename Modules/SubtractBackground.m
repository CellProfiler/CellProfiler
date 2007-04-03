function handles = SubtractBackground(handles)

% Help for the Subtract Background module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Calculates the minimum pixel intensity value for the entire set of images
% and subtracts this value from every pixel in every image.
% *************************************************************************
%
% Note that this is not an illumination correction module. It subtracts a
% single value from every pixel across the image.
%
% The intensity due to camera or illumination or antibody background
% (intensity where no cells are sitting) can in good conscience be
% subtracted from the images, but it must be subtracted from every pixel,
% not just the pixels where cells actually are sitting.  This is because we
% assume that this staining is additive with real staining. This module
% calculates the lowest possible pixel intensity across the entire image
% set and subtracts this background value from every pixel in every image.
% This module is identical to the Apply Threshold module (in shift mode),
% except in the SubtractBackground module, the threshold is automatically
% calculated as the 10th lowest pixel value. This will not push any values
% below zero (therefore, we aren't losing any information). It moves the
% baseline up and looks prettier (improves signal to noise) without any
% 'ethical' concerns.
%
% If images have already been quantified and you want to apply the concept
% of this module without reprocessing your images, then multiply the
% background threshold calculated by this module during the first image 
% cycle by the number of pixels in the image to get the number that should 
% be subtracted from the intensity measurements.
%
% If you want to run this module only to calculate the proper threshold to
% use, simply run the module as usual and use the button on the Status
% window to stop processing after the first image cycle.
%
% How it works:
% Sort each image's pixel values and pick the 10th lowest pixel value as
% the minimum. Typical images have a million pixels. The lowest pixel value
% is chosen because it might be zero if it is a stuck pixel. It is quite
% certain that there will not be 10 stuck pixels so this should be safe. 
% Then, take the minimum of these values from all the images. This scalar 
% value should be subtracted from every pixel in the image. CellProfiler is
% not calculating a different value for each pixel position in the image 
% because in a small image set, that position may always be occupied by 
% real staining.
%
% Features measured:    Feature Number:
% IntensityToShift    |   1
%
%
% See also ApplyThreshold.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne E. Carpenter
%   Thouis Ray Jones
%   In Han Kang
%   Ola Friman
%   Steve Lowe
%   Joo Han Chang
%   Colin Clarke
%   Mike Lamprecht
%   Peter Swire
%   Rodrigo Ipince
%   Vicky Lay
%   Jun Liu
%   Chris Gang
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the image to be corrected?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the corrected image?
%defaultVAR02 = CorrBlue
%infotypeVAR02 = imagegroup indep
CorrectedImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a
%%% variable.
OrigImage = CPretrieveimage(handles,ImageName,ModuleName,'MustBeGray','CheckScale');

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
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
        catch error(['Image processing was canceled in the ', ModuleName, ' module because it must be run using images straight from a load images module (i.e. the images cannot have been altered by other image processing modules). This is because the Subtract Background module calculates an illumination correction image based on all of the images before correcting each individual image as CellProfiler cycles through them. One solution is to process the entire batch of images using the image analysis modules preceding this module and save the resulting images to the hard drive, then start a new stage of processing from the ', ModuleName,' module onward.'])
        end
        %%% Retrieves the list of filenames where the images are stored from the
        %%% handles structure.
        fieldname = ['FileList', ImageName];
        FileList = handles.Pipeline.(fieldname);
        if size(FileList,1) == 2
            error(['Image processing was canceled in the ', ModuleName, ' module because it cannot function on movies.']);
        end
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
        LeftPos = ScreenSize(3)/4;
        PositionMsgBox = [LeftPos BottomOfMsgBox 280 60];
        TimeStart = clock;
        NumberOfImages = length(FileList);
        WaitbarText = 'Preliminary background calculations underway... ';
        WaitbarHandle = waitbar(1/NumberOfImages, WaitbarText);
        set(WaitbarHandle,'Position', PositionMsgBox,'color',[.7 .7 .9])
        for i=1:NumberOfImages
            [Image, handles] = CPimread(fullfile(Pathname,char(FileList(i))), handles);
            SortedColumnImage = sort(reshape(Image, [],1));
            TenthMinimumPixelValue = SortedColumnImage(10);
            if TenthMinimumPixelValue == 0
                CPmsgbox([ImageName , ' image number ', num2str(i), ', and possibly others in the set, has the 10th dimmest pixel equal to zero, which means there is no camera background to subtract, either because the exposure time was very short, or the camera has 10 or more pixels stuck at zero, or that images have been rescaled such that at least 10 pixels are zero, or that for some other reason you have more than 10 pixels of value zero in the image.  This means that the ', ModuleName, ' module will not alter the images in any way, although image processing has not been aborted.'], 'Warning', 'warn','replace')
                %%% Stores the minimum tenth minimum pixel value in the handles structure for
                %%% later use.
                fieldname = ['IntensityToShift', ImageName];
                MinimumTenthMinimumPixelValue = 0;
                handles.Pipeline.(fieldname) = 0;
                %%% Determines the figure number to close, because no
                %%% processing will be performed.
                ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
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
        error(['Image processing was canceled in the ', ModuleName, '. Matlab says the problem is: ', ErrorMessage, ErrorMessage2])
    end
    %%% Stores the minimum tenth minimum pixel value in the handles structure for
    %%% later use.
    if isfield(handles.Measurements.Image,'IntensityToShiftFeatures')
        NewPos = length(handles.Measurements.Image.IntensityToShiftFeatures)+1;
        handles.Measurements.Image.IntensityToShiftFeatures(1,NewPos) = {ImageName};
        handles.Measurements.Image.IntensityToShift(:,NewPos) = MinimumTenthMinimumPixelValue;
    else
        handles.Measurements.Image.IntensityToShiftFeatures = {ImageName};
        handles.Measurements.Image.IntensityToShift = MinimumTenthMinimumPixelValue;
    end
    fieldname = ['IntensityToShift',ImageName];
    handles.Pipeline.(fieldname) = MinimumTenthMinimumPixelValue;
end

%%% The following is run for every cycle. Retrieves the minimum tenth
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

    %%%%%%%%%%%%%%%%%%%%%%%
    %%% DISPLAY RESULTS %%%
    %%%%%%%%%%%%%%%%%%%%%%%
    drawnow

    ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
    if any(findobj == ThisModuleFigureNumber)
        %%% Activates the appropriate figure window.
        CPfigure(handles,'Image',ThisModuleFigureNumber);
        if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
            CPresizefigure(OrigImage,'TwoByOne',ThisModuleFigureNumber)
        end
        %%% A subplot of the figure window is set to display the original
        %%% image, some intermediate images, and the final corrected image.
        subplot(2,1,1); 
        CPimagesc(OrigImage,handles);
        title(['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
        %%% The mean image does not absolutely have to be present in order to
        %%% carry out the calculations if the illumination image is provided,
        %%% so the following subplot is only shown if MeanImage exists in the
        %%% workspace.
        subplot(2,1,2); 
        CPimagesc(CorrectedImage,handles);
        title('Corrected Image');
        %%% Displays the text.
        if isempty(findobj('Parent',ThisModuleFigureNumber,'tag','DisplayText'))
            displaytexthandle = uicontrol(ThisModuleFigureNumber,'tag','DisplayText','style','text', 'position', [0 0 200 20],'fontname','helvetica','backgroundcolor',[0.7 0.7 0.9],'FontSize',handles.Preferences.FontSize);
        else
            displaytexthandle = findobj('Parent',ThisModuleFigureNumber,'tag','DisplayText');
        end
        displaytext = ['Background threshold used: ', num2str(MinimumTenthMinimumPixelValue)];
        set(displaytexthandle,'string',displaytext)
    end
else CorrectedImage = OrigImage;
end % This end goes with the if MinimumTenthMinimumPixelValue ~= 0 line above.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the corrected image to the handles structure so it can be used by
%%% subsequent modules.
handles.Pipeline.(CorrectedImageName) = CorrectedImage;