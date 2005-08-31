function handles = CorrectIllumination_CalculateUsingBackgroundIntensities(handles)

% Help for the Correct Illumination_Calculate Using Background Intensities module:
% Category: Image Processing
%
% This module calculates an illumination function based on the
% background intensities of images.  The illumination function can
% then be saved to the hard drive for later use (see SAVING IMAGES),
% or it can be immediately applied to images later in the pipeline
% (using the CorrectIllumination_Apply module). This will correct for
% uneven illumination of each image.
%
% How it works:
% An image is produced where the value of every pixel is equal to the
% minimum value of any pixel within a "block" of pixels centered
% around that pixel. Theoretically, the intensity values of these
% background pixels should be the same across the image. In reality,
% with uneven illumination, the background pixels will vary across the
% image, and this yields a function that presumably affects the
% intensity of the "real" pixels, e.g. those that comprise cells.
% Therefore, once the average minimums are determined across the
% image(s), the minimums are smoothed out (optional). This produces an
% image that represents the variation in illumination across the field
% of view.
%
% Settings:
%
% Block Size:
% The minimum pixel value is determined within each "block" of the
% image(s). The block dimensions should be large enough that every
% block is likely to contain some "background" pixels, where no cells
% are located.
%
% Enter E or A:
% Enter E to calculate an illumination function for Each image
% individually, or enter A to average together the minimums in All
% images at each pixel location (this processing is done at the time
% you specify by choosing L or P in the next box - see 'Enter L or P'
% for more details). Note that applying illumination correction on
% each image individually may make intensity measures not directly
% comparable across different images. Using illumination correction
% based on all images makes the assumption that the illumination
% anomalies are consistent across all the images in the set.
%
% Enter L or P:
% If you choose L, the module will calculate the illumination
% correction function the first time through the pipeline by loading
% every image of the type specified in the Load Images module. It is
% then acceptable to use the resulting image later in the pipeline. If
% you choose P, the module will allow the pipeline to cycle through
% all of the image sets.  With this option, the module does not need
% to follow a Load Images module; it is acceptable to make the single,
% averaged projection from images resulting from other image
% processing steps in the pipeline. However, the resulting projection
% image will not be available until the last image set has been
% processed, so it cannot be used in subsequent modules unless they
% are instructed to wait until the last image set.
%
% Smoothing Method:
% If requested, the resulting image is smoothed. See the help for the
% Smooth module for more details.
%
% Rescaling:
% The illumination function can be rescaled so that the pixel
% intensities are all equal to or greater than one. This is
% recommended if you plan to use the division option in
% CorrectIllumination_Apply so that the corrected images are in the
% range 0 to 1. Note that as a result of the illumination function
% being rescaled from 1 to infinity, if there is substantial variation
% across the field of view, the rescaling of each image might be
% dramatic, causing the corrected images to be very dark.
%
% SAVING IMAGES:
% The illumination correction function produced by this module can be
% easily saved using the Save Images module, using the name you
% assign. The raw illumination function (before smoothing) can be
% saved in a similar manner using the name you assign. If you want to
% save the illumination image to use it in a later analysis, it is
% very important to save the illumination image in '.mat' format or
% else the quality of the illumination function values will be
% degraded.
%
% This module is loosely based on the Matlab demo "Correction of
% non-uniform illumination" in the Image Processing Toolbox demos
% "Enhancement" category.
% MATLAB6p5/toolbox/images/imdemos/examples/enhance/ipss003.html
%
% See also CORRECTILLUMINATION_APPLY, SMOOTHIMAGE,
% CORRECTILLUMINATION_CALCULATEUSINGINTENSITIES.

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
%
% $Revision$




drawnow

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow



%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

%textVAR01 = What did you call the images to be used to calculate the illumination function?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the illumination function?
%defaultVAR02 = IllumBlue
%infotypeVAR02 = imagegroup indep
IlluminationImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = (Optional) What do you want to call the raw image of average minimums prior to smoothing? (This is an image produced during the calculations - it is typically not needed for downstream modules)
%defaultVAR03 = AverageMinimumsBlue
%infotypeVAR03 = imagegroup indep
AverageMinimumsImageName = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = Block size. This should be set large enough that every square block of pixels is likely to contain some background pixels, where no cells are located.
%defaultVAR04 = 60
BlockSize = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,4}));

%textVAR05 = Enter Each to calculate an illumination function for Each image individually (in which case, choose P in the next box) or All to calculate an illumination function based on All the specified images to be corrected. See the help for details.
%choiceVAR05 = Each
%choiceVAR05 = All
EachOrAll = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu

%textVAR06 = Are the images you want to use to calculate the illumination function to be loaded straight from a Load Images module (L), or are they being produced by the pipeline (P)? See the help for details.
%choiceVAR06 = Pipeline
%choiceVAR06 = Load Images module
SourceIsLoadedOrPipeline = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%inputtypeVAR06 = popupmenu

%textVAR07 = Smoothing method: Enter the width of the artifacts (choose an even number) that are to be smoothed out by median filtering, or type P to fit a low order polynomial instead. For no smoothing, enter N. Note that smoothing is a time-consuming process.
%choiceVAR07 = No smoothing
%choiceVAR07 = Fit polynomial
SmoothingMethod = char(handles.Settings.VariableValues{CurrentModuleNum,7});
%inputtypeVAR07 = popupmenu custom

%textVAR08 = Do you want to rescale the illumination function so that the pixel intensities are all equal to or greater than one (Y or N)? This is recommended if you plan to use the division option in CorrectIllumination_Apply so that the resulting images will be in the range 0 to 1.
%choiceVAR08 = No
%choiceVAR08 = Yes
RescaleOption = char(handles.Settings.VariableValues{CurrentModuleNum,8});
%inputtypeVAR08 = popupmenu

%%%VariableRevisionNumber = 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% If the illumination correction function was to be calculated using
%%% all of the incoming images from a LoadImages module, it will already have been calculated
%%% the first time through the image set. No further calculations are
%%% necessary.
if (strcmp(EachOrAll,'All') == 1 && strcmp(SourceIsLoadedOrPipeline,'Load Images module') == 1) && handles.Current.SetBeingAnalyzed ~= 1
    return
end

%%% Reads (opens) the image you want to analyze and assigns it to a
%%% variable.
fieldname = ['', ImageName];
%%% Checks whether the image to be analyzed exists in the handles structure.
if isfield(handles.Pipeline, fieldname)==0,
    %%% If the image is not there, an error message is produced.  The error
    %%% is not displayed: The error function halts the current function and
    %%% returns control to the calling function (the analyze all images
    %%% button callback.)  That callback recognizes that an error was
    %%% produced because of its try/catch loop and breaks out of the image
    %%% analysis loop without attempting further modules.
    error(['Image processing was canceled because the Correct Illumination module could not find the input image.  It was supposed to be named ', ImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
%%% Reads the image.
OrigImage = handles.Pipeline.(fieldname);

%%% Checks whether the chosen block size is larger than the image itself.
[m,n] = size(OrigImage);
MinLengthWidth = min(m,n);
if BlockSize >= MinLengthWidth
        error('Image processing was canceled because in the Correct Illumination module the selected block size is greater than or equal to the image size itself.')
end

%%% Checks that the original image is two-dimensional (i.e. not a color
%%% image), which would disrupt several of the image functions.
if ndims(OrigImage) ~= 2
    error('Image processing was canceled because the Correct Illumination module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
end

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%



ReadyFlag = 'Not Ready';
if strcmp(EachOrAll,'All') == 1
    try
        if strcmp(SourceIsLoadedOrPipeline, 'Load Images module') == 1 && handles.Current.SetBeingAnalyzed == 1
            %%% The first time the module is run, the average minimums image is
            %%% calculated.
            %%% Notifies the user that the first image set will take much longer than
            %%% subsequent sets.
            %%% Obtains the screen size.
            ScreenSize = get(0,'ScreenSize');
            ScreenHeight = ScreenSize(4);
            PotentialBottom = [0, (ScreenHeight-720)];
            BottomOfMsgBox = max(PotentialBottom);
            PositionMsgBox = [500 BottomOfMsgBox 350 100];
            h = CPmsgbox('Preliminary calculations are under way for the Correct Illumination_Calculate Using Background Intensities module.  Subsequent image sets will be processed more quickly than the first image set.');
            set(h, 'Position', PositionMsgBox)
            drawnow
            %%% Retrieves the path where the images are stored from the handles
            %%% structure.
            fieldname = ['Pathname', ImageName];
            try Pathname = handles.Pipeline.(fieldname);
            catch error('Image processing was canceled because the Correct Illumination_Calculate Using Background Intensities module uses all the images in a set to calculate the illumination correction. Therefore, the entire image set to be illumination corrected must exist prior to processing the first image set through the pipeline. In other words, the Correct Illumination_Calculate Using Background Intensities module must be run straight from a LoadImages module rather than following an image analysis module. One solution is to process the entire batch of images using the image analysis modules preceding this module and save the resulting images to the hard drive, then start a new stage of processing from this Correct Illumination_Calculate Using Background Intensities module onward.')
            end
            %%% Retrieves the list of filenames where the images are stored from the
            %%% handles structure.
            fieldname = ['FileList', ImageName];
            FileList = handles.Pipeline.(fieldname);
            [BestBlockSize, RowsToAdd, ColumnsToAdd] = CalculateBlockSize(m,n,BlockSize);
            %%% Calculates a coarse estimate of the background
            %%% illumination by determining the minimum of each block
            %%% in the image.  If the minimum is zero, it is recorded
            %%% as the minimum non-zero number to prevent divide by
            %%% zero errors later.
            [LoadedImage, handles] = CPimread(fullfile(Pathname,char(FileList(1))),handles);
            SumMiniIlluminationImage = blkproc(padarray(LoadedImage,[RowsToAdd ColumnsToAdd],'replicate','post'),[BestBlockSize(1) BestBlockSize(2)],'min(x(x>0))');
            for i=2:length(FileList)
                [LoadedImage, handles] = CPimread(fullfile(Pathname,char(FileList(i))),handles);
                SumMiniIlluminationImage = SumMiniIlluminationImage + blkproc(padarray(LoadedImage,[RowsToAdd ColumnsToAdd],'replicate','post'),[BestBlockSize(1) BestBlockSize(2)],'min(x(x>0))');
            end
            MiniIlluminationImage = SumMiniIlluminationImage / length(FileList);
            %%% The coarse estimate is then expanded in size so that it is the same
            %%% size as the original image. Bilinear interpolation is used to ensure the
            %%% values do not dip below zero.
            [LoadedImage, handles] = CPimread(fullfile(Pathname,char(FileList(1))),handles);
            IlluminationImage = imresize(MiniIlluminationImage, size(LoadedImage), 'bilinear');
            ReadyFlag = 'Ready';
        elseif strcmp(SourceIsLoadedOrPipeline, 'Pipeline') == 1
            %%% In Pipeline (cycling) mode, each time through the image sets,
            %%% the minimums from the image are added to the existing cumulative image.
            [BestBlockSize, RowsToAdd, ColumnsToAdd] = CalculateBlockSize(m,n,BlockSize);
            if handles.Current.SetBeingAnalyzed == 1
                %%% Creates the empty variable so it can be retrieved later
                %%% without causing an error on the first image set.
                handles.Pipeline.(IlluminationImageName) = zeros(size(blkproc(padarray(OrigImage,[RowsToAdd ColumnsToAdd],'replicate','post'),[BestBlockSize(1) BestBlockSize(2)],'min(x(x>0))')));
            end
            %%% Retrieves the existing illumination image, as accumulated so
            %%% far.
            SumMiniIlluminationImage = handles.Pipeline.(IlluminationImageName);
            %%% Adds the current image to it.
            SumMiniIlluminationImage = SumMiniIlluminationImage + blkproc(padarray(OrigImage,[RowsToAdd ColumnsToAdd],'replicate','post'),[BestBlockSize(1) BestBlockSize(2)],'min(x(x>0))');
            %%% If the last image set has just been processed, indicate that
            %%% the projection image is ready.
            if handles.Current.SetBeingAnalyzed == handles.Current.NumberOfImageSets
                %%% Divides by the total number of images in order to average.
                MiniIlluminationImage = SumMiniIlluminationImage / handles.Current.NumberOfImageSets;
                %%% The coarse estimate is then expanded in size so that it is the same
                %%% size as the original image. Bilinear interpolation is used to ensure the
                %%% values do not dip below zero.
                IlluminationImage = imresize(MiniIlluminationImage, size(OrigImage), 'bilinear');
                ReadyFlag = 'Ready';
            end
        else
            error('Image processing was canceled because you must choose either "L" or "P" in answer to the question "Are the images you want to use to calculate the illumination correction function to be loaded straight from a Load Images module (L), or are they being produced by the pipeline (P)" in the Correct Illumination_Calculate Using Intensities module.');
        end
    catch [ErrorMessage, ErrorMessage2] = lasterr;
        error(['An error occurred in the Correct Illumination_Calculate Using Intensities module. Matlab says the problem is: ', ErrorMessage, ErrorMessage2])
    end
elseif strcmp(EachOrAll,'Each') == 1
    [BestBlockSize, RowsToAdd, ColumnsToAdd] = CalculateBlockSize(m,n,BlockSize);
            %%% Calculates a coarse estimate of the background
            %%% illumination by determining the minimum of each block
            %%% in the image.  If the minimum is zero, it is recorded
            %%% as the minimum non-zero number to prevent divide by
            %%% zero errors later.
    %%% Not sure why this line differed from the one above for 'A'
    %%% mode, so I changed it to use the padarray version.
    % MiniIlluminationImage = blkproc(OrigImage,[BlockSize BlockSize],'min(x(x>0))');
    MiniIlluminationImage = blkproc(padarray(OrigImage,[RowsToAdd ColumnsToAdd],'replicate','post'),[BestBlockSize(1) BestBlockSize(2)],'min(min(x))');
    drawnow
    %%% The coarse estimate is then expanded in size so that it is the same
    %%% size as the original image. Bilinear interpolation is used to ensure the
    %%% values do not dip below zero.
    IlluminationImage = imresize(MiniIlluminationImage, size(OrigImage), 'bilinear');
    ReadyFlag = 'Ready';
else error('Image processing was canceled because you must enter E or A in answer to the question "Enter E to calculate an illumination function for each image individually or A to calculate an illumination function based on all the specified images to be corrected."')
end

if strcmp(SmoothingMethod,'No smoothing') ~= 1
    %%% Smooths the Illumination image, if requested, but saves a raw copy
    %%% first.
    AverageMinimumsImage = IlluminationImage;
    if strcmp(SmoothingMethod,'Fit polynomial')
        SmoothingMethod = 'P';
    end
    IlluminationImage = CPsmooth(IlluminationImage,SmoothingMethod);
end

%%% The resulting illumination image is rescaled to be in the range 1
%%% to infinity, if requested.
if strcmp(RescaleOption,'Yes') == 1
    %%% To save time, the handles argument is not fed to this
    %%% subfunction because it is not needed.
    [ignore,IlluminationImage] = CPrescale('',IlluminationImage,'G',[]);
end

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
    %%% A subplot of the figure window is set to display the original
    %%% image, some intermediate images, and the final corrected image.
    subplot(2,2,1); imagesc(OrigImage);
    title(['Input Image, Image Set # ',num2str(handles.Current.SetBeingAnalyzed)]);
    if exist('IlluminationImage','var') == 1
        subplot(2,2,4);
        imagesc(IlluminationImage);
        CPcolormap(handles)
        text(1,50,['Min Value: ' num2str(min(min(IlluminationImage)))],'Color','red');
        text(1,150,['Max Value: ' num2str(max(max(IlluminationImage)))],'Color','red');
        title('Final illumination correction function');
    else subplot(2,2,4);
        title('Illumination correction function is not yet calculated');
    end
    %%% Whether these images exist depends on whether the images have
    %%% been calculated yet (if running in pipeline mode, this won't occur
    %%% until the last image set is processed).  It also depends on
    %%% whether the user has chosen to smooth the average minimums
    %%% image.
    if exist('AverageMinimumsImage','var') == 1
        subplot(2,2,3); imagesc(AverageMinimumsImage);
        title(['Average minimums image']);
    end
    CPFixAspectRatio(OrigImage);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow



%%% If running in non-cycling mode (straight from the hard drive using
%%% a LoadImages module), the illumination image and its flag need only
%%% be saved to the handles structure after the first image set is
%%% processed. If running in cycling mode (Pipeline mode), the
%%% illumination image and its flag are saved to the handles structure
%%% after every image set is processed.
if strcmp(SourceIsLoadedOrPipeline, 'Pipeline') == 1 | (strcmp(SourceIsLoadedOrPipeline, 'Load Images module') == 1 && handles.Current.SetBeingAnalyzed == 1)
    fieldname = [IlluminationImageName];
    handles.Pipeline.(fieldname) = IlluminationImage;
    %%% Whether these images exist depends on whether the user has chosen
    %%% to smooth the averaged minimums image.
    if exist('AverageMinimumsImage','var') == 1
        fieldname = [AverageMinimumsImageName];
        handles.Pipeline.(fieldname) = AverageMinimumsImage;
    end
    %%% Saves the ready flag to the handles structure so it can be used by
    %%% subsequent modules.
    fieldname = [IlluminationImageName,'ReadyFlag'];
    handles.Pipeline.(fieldname) = ReadyFlag;
end

%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTIONS %%%
%%%%%%%%%%%%%%%%%%%%

function [BestBlockSize, RowsToAdd, ColumnsToAdd] = CalculateBlockSize(m,n,BlockSize)
%%% Calculates the best block size that minimizes padding with
%%% zeros, so that the illumination function will not have dim
%%% artifacts at the right and bottom edges. (Based on Matlab's
%%% bestblk function, but changing the minimum of the range
%%% searched to be 75% of the suggested block size rather than
%%% 50%.
%%% Defines acceptable block sizes.  m and n were
%%% calculated above as the size of the original image.
MM = floor(BlockSize):-1:floor(min(ceil(m/10),ceil(BlockSize*3/4)));
NN = floor(BlockSize):-1:floor(min(ceil(n/10),ceil(BlockSize*3/4)));
%%% Chooses the acceptable block that has the minimum padding.
[dum,ndx] = min(ceil(m./MM).*MM-m); %#ok We want to ignore MLint error checking for this line.
BestBlockSize(1) = MM(ndx);
[dum,ndx] = min(ceil(n./NN).*NN-n); %#ok We want to ignore MLint error checking for this line.
BestBlockSize(2) = NN(ndx);
BestRows = BestBlockSize(1)*ceil(m/BestBlockSize(1));
BestColumns = BestBlockSize(2)*ceil(n/BestBlockSize(2));
RowsToAdd = BestRows - m;
ColumnsToAdd = BestColumns - n;
