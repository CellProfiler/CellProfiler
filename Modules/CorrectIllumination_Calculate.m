function handles = CorrectIllumination_Calculate(handles)

% Help for the Correct Illumination_Calculate Using Intensities module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Calculates an illuminatoiin function, used to correct errors in lighting
% on images. Can also be used to reduce uneven background in images.
% *************************************************************************
%
% This module calculates an illumination function based on the
% intensities of images. The illumination function can then be saved
% to the hard drive for later use (see SAVING IMAGES), or it can be
% immediately applied to images later in the pipeline (using the
% CorrectIllumination_Apply module). This will correct for uneven
% illumination of each image.
%
% How it works:
% This module is most often used to calculate an illumination function
% based on information from a set of images collected at the same
% time. This module works by averaging together all of the images
% (making a projection).  This image is then smoothed (optional). This
% produces an image that represents the variation in illumination
% across the field of view, as long as the cells are spatially
% distributed uniformly across each image. Note that if you are using
% a small image set, there will be spaces in the average image that
% contain no objects and smoothing by median filtering is unlikely to
% work well.
%
% Settings:
%
% Enter E or A:
% Enter E to calculate an illumination function for Each image
% individually, or enter A to average together All images at each
% pixel location (this processing is done at the time you specify by
% choosing L or P in the next box - see 'Enter L or P' for more
% details). Note that applying illumination correction on each image
% individually may make intensity measures not directly comparable
% across different images. Using illumination correction based on all
% images makes the assumption that the illumination anomalies are
% consistent across all the images in the set.
%
% Enter L or P:
% If you choose L, the module will calculate the illumination
% correction function the first time through the pipeline by loading
% every image of the type specified in the Load Images module. It is
% then acceptable to use the resulting image later in the pipeline. If
% you choose P, the module will allow the pipeline to cycle through
% all of the image sets.  With this option, the module does not need
% to follow a Load Images module; it is acceptable to make the single,
% averaged image from images resulting from other image
% processing steps in the pipeline. However, the resulting average
% image will not be available until the last cycle has been
% processed, so it cannot be used in subsequent modules unless they
% are instructed to wait until the last cycle.
%
% Dilation:
% For some applications, the incoming images are binary and each
% object should be dilated with a gaussian filter in the final
% averaged (projection) image.
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
% See also CORRECTILLUMINATION_APPLY, SMOOTH

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne Carpenter
%   Thouis Jones
%   In Han Kang
%   Ola Friman
%   Steve Lowe
%   Joo Han Chang
%   Colin Clarke
%   Mike Lamprecht
%   Susan Ma
%   Wyman Li
%
% Website: http://www.cellprofiler.org
%
% $Revision: 1750 $

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);
ModuleName = char(handles.Settings.ModuleNames(CurrentModuleNum));

%textVAR01 = What did you call the images to be used to calculate the illumination function?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the illumination function?
%defaultVAR02 = IllumBlue
%infotypeVAR02 = imagegroup indep
IlluminationImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Do you want to calculate using regular intensities or background intensities?
%choiceVAR03 = Regular
%choiceVAR03 = Background
IntensityChoice = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 = Enter Each to calculate an illumination function for Each image individually (in which case, choose Pipeline mode in the next box) or All to calculate an illumination function based on All the specified images to be corrected. See the help for details.
%choiceVAR04 = Each
%choiceVAR04 = All
EachOrAll = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%textVAR05 = Are the images you want to use to calculate the illumination function to be loaded straight from a Load Images module (L), or are they being produced by the pipeline (P)? See the help for details.
%choiceVAR05 = Pipeline
%choiceVAR05 = Load Images module
SourceIsLoadedOrPipeline = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu

%textVAR06 = Smoothing method: Enter the width of the artifacts (choose an even number) that are to be smoothed out by median filtering, or type P to fit a low order polynomial instead. For no smoothing, enter N. Note that smoothing is a time-consuming process.
%choiceVAR06 = No smoothing
%choiceVAR06 = Fit polynomial
SmoothingMethod = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%inputtypeVAR06 = popupmenu custom

%textVAR07 = Do you want to rescale the illumination function so that the pixel intensities are all equal to or greater than one (Y or N)? This is recommended if you plan to use the division option in CorrectIllumination_Apply so that the resulting images will be in the range 0 to 1.
%choiceVAR07 = Yes
%choiceVAR07 = No
RescaleOption = char(handles.Settings.VariableValues{CurrentModuleNum,7});
%inputtypeVAR07 = popupmenu

%textVAR08 = (For 'All' mode only) What do you want to call the averaged image (prior to dilation or smoothing)? (This is an image produced during the calculations - it is typically not needed for downstream modules)
%choiceVAR08 = Do not save
%infotypeVAR08 = imagegroup indep
AverageImageName = char(handles.Settings.VariableValues{CurrentModuleNum,8});
%inputtypeVAR08 = popupmenu custom

%textVAR09 = What do you want to call the image after dilation but prior to smoothing?  (This is an image produced during the calculations - it is typically not needed for downstream modules)
%choiceVAR09 = Do not save
%infotypeVAR09 = imagegroup indep
DilatedImageName = char(handles.Settings.VariableValues{CurrentModuleNum,9});
%inputtypeVAR09 = popupmenu custom

%textVAR10 = REGULAR INTENSITY OPTIONS

%textVAR11 = If the incoming images are binary and you want to dilate each object in the final averaged image, enter the radius (roughly equal to the original radius of the objects). Otherwise, enter 0.
%defaultVAR11 = 0
ObjectDilationRadius = char(handles.Settings.VariableValues{CurrentModuleNum,11});

%textVAR12 = BACKGROUND INTENSITY OPTIONS

%textVAR13 = Block size. This should be set large enough that every square block of pixels is likely to contain some background pixels, where no cells are located.
%defaultVAR13 = 60
BlockSize = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,13}));

%%%VariableRevisionNumber = 4

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

if strcmp(EachOrAll,'Each') && strcmp(SourceIsLoadedOrPipeline,'Load Images module')
    error(['Image processing was canceled in the ', ModuleName, ' module because you must choose Pipeline mode if you are using Each mode.'])
end

%%% If the illumination correction function was to be calculated using
%%% all of the incoming images from a LoadImages module, it will already have been calculated
%%% the first time through the cycle. No further calculations are
%%% necessary.
if strcmp(EachOrAll,'All') && handles.Current.SetBeingAnalyzed ~= 1 && strcmp(SourceIsLoadedOrPipeline,'Load Images module')
    return
end

try NumericalObjectDilationRadius = str2num(ObjectDilationRadius);
catch
    error(['Image processingwas canceled in the ', ModuleName, ' module because you must enter a number for the radius to use to dilate objects. If you do not want to dilate objects enter 0 (zero).'])
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
    error(['Image processing was canceled in the ', ModuleName, ' module because it could not find the input image.  It was supposed to be named ', ImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
%%% Reads the image.
OrigImage = handles.Pipeline.(fieldname);

if strcmp(IntensityChoice,'Background')
    %%% Checks whether the chosen block size is larger than the image itself.
    [m,n] = size(OrigImage);
    MinLengthWidth = min(m,n);
    if BlockSize >= MinLengthWidth
        error(['Image processing was canceled in the ', ModuleName, ' module because the selected block size is greater than or equal to the image size itself.'])
    end
end

%%% Checks that the original image is two-dimensional (i.e. not a color
%%% image), which would disrupt several of the image functions.
if ndims(OrigImage) ~= 2
    error(['Image processing was canceled in the ', ModuleName, ' module because it requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.'])
end

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

if strcmp(AverageImageName,'Do not save')
    AverageImageSaveFlag = 0;
    AverageImageName = ['Averaged',ImageName];
else AverageImageSaveFlag = 1;
end

ReadyFlag = 'Not Ready';
if strcmp(EachOrAll,'All')
    try
        if strcmp(SourceIsLoadedOrPipeline, 'Load Images module') == 1 && handles.Current.SetBeingAnalyzed == 1
            %%% The first time the module is run, the averaged image is
            %%% calculated.
            %%% Notifies the user that the first image set will take much longer than
            %%% subsequent sets.
            %%% Obtains the screen size.
            ScreenSize = get(0,'ScreenSize');
            ScreenHeight = ScreenSize(4);
            PotentialBottom = [0, (ScreenHeight-720)];
            BottomOfMsgBox = max(PotentialBottom);
            h = CPmsgbox('Preliminary calculations are under way for the Correct Illumination_Calculate Using Background Intensities module.  Subsequent image sets will be processed more quickly than the first image set.');
            OldPos = get(h,'position');
            set(h, 'Position',[250 BottomOfMsgBox OldPos(3) OldPos(4)]);
            drawnow

            if strcmp(IntensityChoice,'Regular')
                [handles, RawImage, ReadyFlag] = CPaverageimages(handles, 'DoNow', ImageName, 'ignore');
            elseif strcmp(IntensityChoice,'Background')
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
            end
        elseif strcmp(SourceIsLoadedOrPipeline,'Pipeline')
            if strcmp(IntensityChoice,'Regular')
                [handles, RawImage, ReadyFlag] = CPaverageimages(handles, 'Accumulate', ImageName, AverageImageName);
            elseif strcmp(IntensityChoice,'Background')
                %%% In Pipeline mode, each time through the cycle,
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
            end
        else
            error('Image processing was canceled because you must choose either "L" or "P" in answer to the question "Are the images you want to use to calculate the illumination correction function to be loaded straight from a Load Images module (L), or are they being produced by the pipeline (P)" in the Correct Illumination_Calculate Using Intensities module.');
        end
    catch [ErrorMessage, ErrorMessage2] = lasterr;
        error(['An error occurred in the Correct Illumination_Calculate Using Intensities module. Matlab says the problem is: ', ErrorMessage, ErrorMessage2])
    end
elseif strcmp(EachOrAll,'Each')
    if strcmp(IntensityChoice,'Regular')
        RawImage = OrigImage;
    elseif strcmp(IntensityChoice,'Background')
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
    end
    ReadyFlag = 'Ready';
else error('Image processing was canceled because you must choose either "E" or "A" in answer to the question "Enter E to calculate an illumination function for each image individually (in which case, choose P in the next box) or A to calculate an illumination function based on all the specified images to be corrected" in the Correct Illumination_Calculate Using Intensities module.');
end

%%% Dilates the objects, and/or smooths the RawImage if the user requested.
if strcmp(ReadyFlag, 'Ready')
    if strcmp(IntensityChoice,'Regular')
        if NumericalObjectDilationRadius ~= 0
            DilatedImage = CPdilatebinaryobjects(RawImage, NumericalObjectDilationRadius);
        end

        if ~strcmp(SmoothingMethod,'No smoothing')
            %%% Smooths the averaged image, if requested, but saves a raw copy
            %%% first.
            if strcmp(SmoothingMethod,'Fit polynomial')
                SmoothingMethod = 'P';
            end

            if exist('DilatedImage','var')
                SmoothedImage = CPsmooth(DilatedImage,SmoothingMethod,handles.Current.SetBeingAnalyzed);
            elseif exist('RawImage','var')
                SmoothedImage = CPsmooth(RawImage,SmoothingMethod,handles.Current.SetBeingAnalyzed);
            else error('something is wrong; this should never happen.')
            end
        end

        drawnow
        %%% Which image is the final function depends on whether we chose to
        %%% dilate or smooth.
        if exist('SmoothedImage','var')
            FinalIlluminationFunction = SmoothedImage;
        elseif exist('DilatedImage','var')
            FinalIlluminationFunction = DilatedImage;
        else FinalIlluminationFunction = RawImage;
        end

    elseif strcmp(IntensityChoice,'Background')
        if ~strcmp(SmoothingMethod,'No smoothing')
            %%% Smooths the Illumination image, if requested, but saves a raw copy
            %%% first.
            AverageMinimumsImage = IlluminationImage;
            if strcmp(SmoothingMethod,'Fit polynomial')
                SmoothingMethod = 'P';
            end
            FinalIlluminationFunction = CPsmooth(IlluminationImage,SmoothingMethod,handles.Current.SetBeingAnalyzed);
        end
    end

    %%% The resulting image is rescaled to be in the range 1
    %%% to infinity, if requested.
    if strcmp(RescaleOption,'Yes') == 1
        %%% To save time, the handles argument is not fed to this
        %%% subfunction because it is not needed.
        [ignore,FinalIlluminationFunction] = CPrescale('',FinalIlluminationFunction,'G',[]); %#ok
    end
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
if any(findobj == ThisModuleFigureNumber) == 1;
    drawnow
    %%% Activates the appropriate figure window.
    CPfigure(handles,ThisModuleFigureNumber);
    if strcmp(IntensityChoice,'Regular')
        %%% Whether these images exist depends on whether the images have
        %%% been calculated yet (if running in pipeline mode, this won't occur
        %%% until the last cycle is processed).  It also depends on
        %%% whether the user has chosen to dilate or smooth the averaged
        %%% image.

        %%% If we are in Each mode, the Raw image will be identical to the
        %%% input image so there is no need to display it again.  If we
        %%% are in All mode, there is no OrigImage, so we can plot both to
        %%% the 2,2,1 location.
        if strcmp(EachOrAll,'All')
            subplot(2,2,1); imagesc(RawImage);
            if strcmp(ReadyFlag, 'Ready')
                title('Averaged image');
            else
                title('Averaged image calculated so far');
            end
        else subplot(2,2,1); imagesc(OrigImage);
            title(['Input Image, Cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
            CPFixAspectRatio(OrigImage);
        end
        if strcmp(ReadyFlag, 'Ready')
            if exist('DilatedImage','var')
                subplot(2,2,3); imagesc(DilatedImage);
                title('Dilated image');
            end
            if exist('SmoothedImage','var')
                subplot(2,2,4); imagesc(SmoothedImage);
                title('Smoothed image');
            end
            subplot(2,2,2);
            imagesc(FinalIlluminationFunction);
            text(1,50,['Min Value: ' num2str(min(min(FinalIlluminationFunction)))],'Color','red');
            text(1,150,['Max Value: ' num2str(max(max(FinalIlluminationFunction)))],'Color','red');
            title('Final illumination function');
        end
    elseif strcmp(IntensityChoice,'Background')
        %%% A subplot of the figure window is set to display the original
        %%% image, some intermediate images, and the final corrected image.
        subplot(2,2,1); imagesc(OrigImage);
        title(['Input Image, Image Set # ',num2str(handles.Current.SetBeingAnalyzed)]);
        if exist('FinalIlluminationFunction','var') == 1
            subplot(2,2,4);
            imagesc(FinalIlluminationFunction);

            text(1,50,['Min Value: ' num2str(min(min(FinalIlluminationFunction)))],'Color','red');
            text(1,150,['Max Value: ' num2str(max(max(FinalIlluminationFunction)))],'Color','red');
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
            title('Average minimums image');
        end
        CPFixAspectRatio(OrigImage);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves images to the handles structure.
%%% If running in non-cycling mode (straight from the hard drive using
%%% a LoadImages module), the average image and its flag need only
%%% be saved to the handles structure after the first cycle is
%%% processed. If running in cycling mode (Pipeline mode), the
%%% average image and its flag are saved to the handles structure
%%% after every cycle is processed.
if strcmp(SourceIsLoadedOrPipeline, 'Pipeline') || (strcmp(SourceIsLoadedOrPipeline, 'Load Images module') && handles.Current.SetBeingAnalyzed == 1)
    if strcmp(ReadyFlag, 'Ready') == 1
        handles.Pipeline.(IlluminationImageName) = FinalIlluminationFunction;
    end
    if strcmp(IntensityChoice,'Regular')
        %%% Whether these images exist depends on whether the user has chosen
        %%% to dilate or smooth the average image.
        if AverageImageSaveFlag == 1
            if strcmp(EachOrAll,'Each')
                error('Image processing was canceled because in the Correct Illumination module you attempted to pass along the averaged image, but because you are in Each mode, an averaged image has not been calculated.')
            end
            try handles.Pipeline.(AverageImageName) = RawImage;
            catch error('There was a problem passing along the average image in the Correct Illumination module. This image can only be passed along if you choose to dilate.')
            end
            %%% Saves the ready flag to the handles structure so it can be used by
            %%% subsequent modules.
            fieldname = [AverageImageName,'ReadyFlag'];
            handles.Pipeline.(fieldname) = ReadyFlag;
        end
        if ~strcmpi(DilatedImageName,'Do not save')
            try handles.Pipeline.(DilatedImageName) = DilatedImage;
            catch error('There was a problem passing along the dilated image in the Correct Illumination module. This image can only be passed along if you choose to dilate.')
            end
        end
    elseif strcmp(IntensityChoice,'Background')
        %%% Whether these images exist depends on whether the user has chosen
        %%% to smooth the averaged minimums image.
        %if exist('AverageMinimumsImage','var') == 1
        %    fieldname = [AverageMinimumsImageName];
        %    handles.Pipeline.(fieldname) = AverageMinimumsImage;
        %end
        %%% Saves the ready flag to the handles structure so it can be used by
        %%% subsequent modules.
        fieldname = [IlluminationImageName,'ReadyFlag'];
        handles.Pipeline.(fieldname) = ReadyFlag;
    end
end

%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTIONS %%%
%%%%%%%%%%%%%%%%%%%%
drawnow

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