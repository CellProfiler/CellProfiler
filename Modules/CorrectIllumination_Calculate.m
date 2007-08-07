function handles = CorrectIllumination_Calculate(handles)

% Help for the Correct Illumination Calculate module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Calculates an illumination function, used to correct uneven
% illumination/lighting/shading or to reduce uneven background in images.
% *************************************************************************
%
% This module calculates an illumination function which can be saved to the
% hard drive for later use (you should save in .mat format using the Save
% Images module), or it can be immediately applied to images later in the
% pipeline (using the CorrectIllumination_Apply module). This will correct
% for uneven illumination of each image.
%
% Illumination correction is challenging and we are writing a paper on it
% which should help clarify it (TR Jones, AE Carpenter, P Golland, in
% preparation). In the meantime, please be patient in trying to understand
% this module.
%
% Settings:
%
% * Regular or Background intensities?
%
% Regular intensities:
% If you have objects that are evenly dispersed across your image(s) and
% cover most of the image, then you can choose Regular intensities. Regular
% intensities makes the illumination function based on the intensity at
% each pixel of the image (or group of images if you are in All mode) and
% is most often rescaled (see below) and applied by division using
% CorrectIllumination_Apply. Note that if you are in Each mode or using a
% small set of images with few objects, there will be regions in the
% average image that contain no objects and smoothing by median filtering
% is unlikely to work well.
% Note: it does not make sense to choose (Regular + no smoothing + Each)
% because the illumination function would be identical to the original
% image and applying it will yield a blank image. You either need to smooth
% each image or you need to use All images.
%
% Background intensities:
% If you think that the background (dim points) between objects show the
% same pattern of illumination as your objects of interest, you can choose
% Background intensities. Background intensities finds the minimum pixel
% intensities in blocks across the image (or group of images if you are in
% All mode) and is most often applied by subtraction using the
% CorrectIllumination_Apply module.
% Note: if you will be using the Subtract option in the
% CorrectIllumination_Apply module, you almost certainly do NOT want to
% Rescale! See below!!
%
% * Each or All?
% Enter Each to calculate an illumination function for each image
% individually, or enter All to calculate the illumination function from
% all images at each pixel location. All is more robust, but depends on the
% assumption that the illumination patterns are consistent across all the
% images in the set and that the objects of interest are randomly
% positioned within each image. Applying illumination correction on each
% image individually may make intensity measures not directly comparable
% across different images.
%
% * Pipeline or Load Images?
% If you choose Load Images, the module will calculate the illumination
% correction function the first time through the pipeline by loading every
% image of the type specified in the Load Images module. It is then
% acceptable to use the resulting image later in the pipeline. If you
% choose Pipeline, the module will allow the pipeline to cycle through all
% of the cycles. With this option, the module does not need to follow a
% Load Images module; it is acceptable to make the single, averaged image
% from images resulting from other image processing steps in the pipeline.
% However, the resulting average image will not be available until the last
% cycle has been processed, so it cannot be used in subsequent modules
% unless they are instructed to wait until the last cycle.
%
% * Dilation:
% For some applications, the incoming images are binary and each object
% should be dilated with a gaussian filter in the final averaged
% (projection) image. This is for a sophisticated method of illumination
% correction where model objects are produced.
%
% * Smoothing Method:
% If requested, the resulting image is smoothed. See the help for the
% Smooth module for more details. If you are using Each mode, this is
% almost certainly necessary. If you have few objects in each image or a
% small image set, you may want to smooth. The goal is to smooth to the
% point where the illumination function resembles a believable pattern.
% That is, if it is a lamp illumination problem you are trying to correct,
% you would apply smoothing until you obtain a fairly smooth pattern
% without sharp bright or dim regions.  Note that smoothing is a
% time-consuming process, and fitting a polynomial is fastest but does not
% allow a very tight fit as compared to the slower median filtering method.
% Another option is to *completely* smooth the entire image by choosing
% "Smooth to average", which will create a flat, smooth image where every
% pixel of the image is the average of what the illumination function would
% otherwise have been.
%
% * Approximate width of objects:
% For certain smoothing methods, this will be used to calculate an adequate
% filter size. If you don't know the width of your objects, you can use the
% ShowOrHidePixelData image tool to find out or leave the word 'Automatic'
% to calculate a smoothing filter simply based on the size of the image.
%
%
% Rescaling:
% The illumination function can be rescaled so that the pixel intensities
% are all equal to or greater than one. This is recommended if you plan to
% use the division option in CorrectIllumination_Apply so that the
% corrected images are in the range 0 to 1. It is NOT recommended if you
% plan to use the Subtract option in CorrectIllumination_Apply! Note that
% as a result of the illumination function being rescaled from 1 to
% infinity, if there is substantial variation across the field of view, the
% rescaling of each image might be dramatic, causing the corrected images
% to be very dark.
%
% See also Average, CorrectIllumination_Apply, and Smooth modules.

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
%   Kyungnam Kim
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

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

%textVAR04 = For REGULAR INTENSITY: If the incoming images are binary and you want to dilate each object in the final averaged image, enter the radius (roughly equal to the original radius of the objects). Otherwise, enter 0.
%defaultVAR04 = 0
ObjectDilationRadius = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = For BACKGROUND INTENSITY: Enter the block size, which should be large enough that every square block of pixels is likely to contain some background pixels, where no objects are located.
%defaultVAR05 = 60
BlockSize = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,5}));

%textVAR06 = Do you want to rescale the illumination function so that the pixel intensities are all equal to or greater than one (Y or N)? This is recommended if you plan to use the division option in CorrectIllumination_Apply so that the resulting images will be in the range 0 to 1.
%choiceVAR06 = Yes
%choiceVAR06 = No
RescaleOption = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%inputtypeVAR06 = popupmenu

%textVAR07 = Enter Each to calculate an illumination function for Each image individually (in which case, choose Pipeline mode in the next box) or All to calculate an illumination function based on All the specified images to be corrected. See the help for details.
%choiceVAR07 = Each
%choiceVAR07 = All
EachOrAll = char(handles.Settings.VariableValues{CurrentModuleNum,7});
%inputtypeVAR07 = popupmenu

%textVAR08 = Are the images you want to use to calculate the illumination function to be loaded straight from a Load Images module, or are they being produced by the pipeline? See the help for details.
%choiceVAR08 = Pipeline
%choiceVAR08 = Load Images module
SourceIsLoadedOrPipeline = char(handles.Settings.VariableValues{CurrentModuleNum,8});
%inputtypeVAR08 = popupmenu

%textVAR09 = Enter the smoothing method you would like to use, if any.
%choiceVAR09 = No smoothing
%choiceVAR09 = Fit Polynomial
%choiceVAR09 = Median Filter
%choiceVAR09 = Gaussian Filter
%choiceVAR09 = Sum of Squares
%choiceVAR09 = Square of Sum
%choiceVAR09 = Smooth to Average
SmoothingMethod = char(handles.Settings.VariableValues{CurrentModuleNum,9});
%inputtypeVAR09 = popupmenu

%textVAR10 = For MEDIAN FILTER, GAUSSIAN FILTER, SUM OF SQUARES, or SQUARE OF SUM, specify the approximate width of the artifacts to be smoothed (in pixels), or leave the word 'Automatic'.
%defaultVAR10 = Automatic
ObjectWidth = handles.Settings.VariableValues{CurrentModuleNum,10};

%%% TODO: it is unclear why we ask for width of objects and then allow
%%% overriding, since one is calculated from the other. I asked Rodrigo
%%% about it 8-31-06  Most likely we will remove the following variable and
%%% instead provide instructions in the help to tell you how the
%%% SizeOfSmoothingFilter is calculate from Artifact width, in case someone
%%% wants to enter a precise vaule. We should then also check the
%%% Average/Smooth module as well -Anne.

%textVAR11 = If you want override the above width of artifacts and set your own filter size (in pixels), please specify it here. Otherwise leave '/'.
%defaultVAR11 = /
SizeOfSmoothingFilter = char(handles.Settings.VariableValues{CurrentModuleNum,11});

%textVAR12 = (For 'All' mode only) What do you want to call the averaged image (prior to dilation or smoothing)? (This is an image produced during the calculations - it is typically not needed for downstream modules)
%choiceVAR12 = Do not save
%infotypeVAR12 = imagegroup indep
AverageImageName = char(handles.Settings.VariableValues{CurrentModuleNum,12});
%inputtypeVAR12 = popupmenu custom

%textVAR13 = What do you want to call the image after dilation but prior to smoothing?  (This is an image produced during the calculations - it is typically not needed for downstream modules)
%choiceVAR13 = Do not save
%infotypeVAR13 = imagegroup indep
DilatedImageName = char(handles.Settings.VariableValues{CurrentModuleNum,13});
%inputtypeVAR13 = popupmenu custom

%%%VariableRevisionNumber = 7

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

if strcmp(EachOrAll,'Each') && strcmp(SourceIsLoadedOrPipeline,'Load Images module')
    error(['Image processing was canceled in the ', ModuleName, ' module because you must choose Pipeline mode if you are using Each mode.'])
end

%%% If the illumination correction function was to be calculated using
%%% all of the incoming images from a Load Images module, it will already have been calculated
%%% the first time through the cycle. No further calculations are
%%% necessary.
if strcmp(EachOrAll,'All') && handles.Current.SetBeingAnalyzed ~= 1 && strcmp(SourceIsLoadedOrPipeline,'Load Images module')
    return
end

try NumericalObjectDilationRadius = str2double(ObjectDilationRadius);
catch
    error(['Image processing was canceled in the ', ModuleName, ' module because you must enter a number for the radius to use to dilate objects. If you do not want to dilate objects enter 0 (zero).'])
end

%%% Checks smooth method variables
if ~strcmp(SizeOfSmoothingFilter,'/')
    SizeOfSmoothingFilter = str2double(SizeOfSmoothingFilter);
    if isnan(SizeOfSmoothingFilter) || (SizeOfSmoothingFilter < 0)
        if isempty(findobj('Tag',['Msgbox_' ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Smoothing filter invalid']))
            CPwarndlg(['The size of smoothing filter you specified in the ' ModuleName ' module was invalid. It is being reset to automatically calculated.'],[ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Smoothing filter invalid'],'replace');
        end
        SizeOfSmoothingFilter = '/';
    else
        SizeOfSmoothingFilter = floor(SizeOfSmoothingFilter);
        WidthFlg = 0;
    end
end
if strcmp(SizeOfSmoothingFilter,'/')
    if ~strcmpi(ObjectWidth,'Automatic')
        ObjectWidth = str2double(ObjectWidth);
        if isnan(ObjectWidth) || ObjectWidth<0
            if isempty(findobj('Tag',['Msgbox_' ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Object width invalid']))
                CPwarndlg(['The object width you specified in the ', ModuleName, ' module was invalid. It is being reset to automatically calculated.'],[ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Object width invalid'],'replace');
            end
            SizeOfSmoothingFilter = 'A';
            WidthFlg = 0;
        else
            SizeOfSmoothingFilter = 2*floor(ObjectWidth/2);
            WidthFlg = 1;
        end
    else
        SizeOfSmoothingFilter = 'A';
        WidthFlg = 0;
    end
end

%%% Reads (opens) the image you want to analyze and assigns it to a
%%% variable.
OrigImage = CPretrieveimage(handles,ImageName,ModuleName,'MustBeGray','CheckScale');

if strcmp(IntensityChoice,'Background')
    %%% Checks whether the chosen block size is larger than the image itself.
    [m,n] = size(OrigImage);
    MinLengthWidth = min(m,n);
    if BlockSize <= 0
        if isempty(findobj('Tag',['Msgbox_' ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Block size below zero']))
            CPwarndlg(['The selected block size in the ' ModuleName ' module is less than or equal to zero. The block size is being reset to the default value of 60.'],[ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Block size changed'],'replace');
        end
        BlockSize = 60;
    end
    if BlockSize > MinLengthWidth
        if isempty(findobj('Tag',['Msgbox_' ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Block size too large']))
            CPwarndlg(['The selected block size in the ' ModuleName ' module is either not a positive number, or it is larger than the image size itself. The block size is being set to ',num2str(MinLengthWidth),'.'],[ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Block size changed'],'replace');
        end
        BlockSize = MinLengthWidth;
    end
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
            %%% Notifies the user that the first cycle will take much longer than
            %%% subsequent sets.
            %%% Obtains the screen size.
            ScreenSize = get(0,'ScreenSize');
            ScreenHeight = ScreenSize(4);
            PotentialBottom = [0, (ScreenHeight-720)];
            BottomOfMsgBox = max(PotentialBottom);
            h = CPmsgbox(['Preliminary calculations are under way for the ', ModuleName, ' module.  Subsequent cycles will be processed more quickly than the first cycle.']);
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
                catch error(['Image processing was canceled in the ', ModuleName, ' module because it uses all the images of one type to calculate the illumination correction. Therefore, the entire set of images to be illumination corrected must exist prior to processing the first cycle through the pipeline. In other words, the ',ModuleName, ' module must be run straight after a Load Images module rather than following an image analysis module. One solution is to process the entire batch of images using the image analysis modules preceding this module and save the resulting images to the hard drive, then start a new stage of processing from this ', ModuleName, ' module onward.'])
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
                    %%% without causing an error on the first cycle.
                    handles.Pipeline.(IlluminationImageName) = zeros(size(blkproc(padarray(OrigImage,[RowsToAdd ColumnsToAdd],'replicate','post'),[BestBlockSize(1) BestBlockSize(2)],'min(x(x>0))')));
                end
                %%% Retrieves the existing illumination image, as accumulated so
                %%% far.
                SumMiniIlluminationImage = handles.Pipeline.(IlluminationImageName);
                %%% Adds the current image to it.
                SumMiniIlluminationImage = SumMiniIlluminationImage + blkproc(padarray(OrigImage,[RowsToAdd ColumnsToAdd],'replicate','post'),[BestBlockSize(1) BestBlockSize(2)],'min(x(x>0))');
                %%% If the last cycle has just been processed, indicate that
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
            error(['Image processing was canceled in the ', ModuleName, ' module because you must choose either Load Images or Pipeline in answer to the question "Are the images you want to use to calculate the illumination correction function to be loaded straight from a Load Images module, or are they being produced by the pipeline".']);
        end
    catch [ErrorMessage, ErrorMessage2] = lasterr;
        error(['Image processing was canceled in the ', ModuleName, ' module. Matlab says the problem is: ', ErrorMessage, ErrorMessage2])
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
else error(['Image processing was canceled in the ', ModuleName, ' module because you must choose either Each or All.']);
end

%%% Dilates the objects, and/or smooths the RawImage if the user requested.
if strcmp(ReadyFlag, 'Ready')
    if strcmp(IntensityChoice,'Regular')
        if (NumericalObjectDilationRadius > 0)
            DilatedImage = CPdilatebinaryobjects(RawImage, NumericalObjectDilationRadius);
        elseif (NumericalObjectDilationRadius < 0)
            if isempty(findobj('Tag',['Msgbox_' ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Dilation factor too small']))
                CPwarndlg(['The dilation factor you have entered in the ', ModuleName, ' module is below the minimum value of 0, it is being reset to 0.'],[ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Dilation factor too small'],'replace');
            end
        end

        if ~strcmp(SmoothingMethod,'No smoothing')
            %%% Smooths the averaged image, if requested, but saves a raw copy
            %%% first.

            if exist('DilatedImage','var')
                SmoothedImage = CPsmooth(DilatedImage,SmoothingMethod,SizeOfSmoothingFilter,WidthFlg);
            elseif exist('RawImage','var')
                SmoothedImage = CPsmooth(RawImage,SmoothingMethod,SizeOfSmoothingFilter,WidthFlg);
            else error(['Image processing was canceled in the ', ModuleName, ' due to some sort of programming error.']);
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
            FinalIlluminationFunction = CPsmooth(IlluminationImage,SmoothingMethod,SizeOfSmoothingFilter,WidthFlg);
        else
            FinalIlluminationFunction = IlluminationImage;
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

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    drawnow
    %%% Activates the appropriate figure window.
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure(OrigImage,'TwoByTwo',ThisModuleFigureNumber);
    end
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
            subplot(2,2,1);
            CPimagesc(RawImage,handles);
            if strcmp(ReadyFlag, 'Ready')
                title('Averaged image');
            else
                title('Averaged image calculated so far');
            end
        else subplot(2,2,1);
            CPimagesc(OrigImage,handles);
            title(['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
        end
        if strcmp(ReadyFlag, 'Ready')
            if exist('DilatedImage','var')
                subplot(2,2,3);
                CPimagesc(DilatedImage,handles);
                title('Dilated image');
            end
            if exist('SmoothedImage','var')
                subplot(2,2,4);
                CPimagesc(SmoothedImage,handles);
                title('Smoothed image');
            end
            subplot(2,2,2);
            CPimagesc(FinalIlluminationFunction,handles);
            text(1,50,['Min Value: ' num2str(min(min(FinalIlluminationFunction)))],'Color','red','fontsize',handles.Preferences.FontSize);
            text(1,150,['Max Value: ' num2str(max(max(FinalIlluminationFunction)))],'Color','red','fontsize',handles.Preferences.FontSize);
            title('Final illumination function');
        end
    elseif strcmp(IntensityChoice,'Background')
        %%% A subplot of the figure window is set to display the original
        %%% image, some intermediate images, and the final corrected image.
        subplot(2,2,1);
        CPimagesc(OrigImage,handles);
        title(['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
        if exist('FinalIlluminationFunction','var') == 1
            subplot(2,2,4);
            CPimagesc(FinalIlluminationFunction,handles);
            text(1,50,['Min Value: ' num2str(min(min(FinalIlluminationFunction)))],'Color','red','fontsize',handles.Preferences.FontSize);
            text(1,150,['Max Value: ' num2str(max(max(FinalIlluminationFunction)))],'Color','red','fontsize',handles.Preferences.FontSize);
            title('Final illumination correction function');
        else subplot(2,2,4);
            title('Illumination correction function is not yet calculated');
        end
        %%% Whether these images exist depends on whether the images have
        %%% been calculated yet (if running in pipeline mode, this won't occur
        %%% until the last cycle is processed).  It also depends on
        %%% whether the user has chosen to smooth the average minimums
        %%% image.
        if exist('AverageMinimumsImage','var') == 1
            subplot(2,2,3);
            CPimagesc(AverageMinimumsImage,handles);
            title('Average minimums image');
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves images to the handles structure.
%%% If running in non-cycling mode (straight from the hard drive using
%%% a Load Images module), the average image and its flag need only
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
                error(['Image processing was canceled in the ', ModuleName, ' module because you attempted to pass along the averaged image, but because you are in Each mode, an averaged image has not been calculated.'])
            end
            try handles.Pipeline.(AverageImageName) = RawImage;
            catch error(['Image processing was canceled in the ', ModuleName, ' module. There was a problem passing along the averaged image. This image can only be passed along if you choose to dilate.'])
            end
            %%% Saves the ready flag to the handles structure so it can be used by
            %%% subsequent modules.
            fieldname = [AverageImageName,'ReadyFlag'];
            handles.Pipeline.(fieldname) = ReadyFlag;
        end
        if ~strcmpi(DilatedImageName,'Do not save')
            try handles.Pipeline.(DilatedImageName) = DilatedImage;
            catch error(['Image processing was canceled in the ', ModuleName, ' module. There was a problem passing along the dilated image. This image can only be passed along if you choose to dilate.'])
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
