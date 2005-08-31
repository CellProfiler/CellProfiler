function handles = CorrectIllumination_CalculateUsingIntensities(handles)

% Help for the Correct Illumination_Calculate Using Intensities module:
% Category: Image Processing
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
% a small image set, there will be spaces in the projection image that
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
% averaged projection from images resulting from other image
% processing steps in the pipeline. However, the resulting projection
% image will not be available until the last image set has been
% processed, so it cannot be used in subsequent modules unless they
% are instructed to wait until the last image set.
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
% SAVING IMAGES:
% The illumination correction function produced by this module can be
% easily saved using the Save Images module, using the name you
% assign. Intermediate images - prior to dilation and smoothing, or
% after dilation but prior to smoothing - can be saved in a similar
% manner using the name you assign. If you want to save the
% illumination image to use it in a later analysis, it is very
% important to save the illumination image in '.mat' format or else
% the quality of the illumination function values will be degraded.
%
% See also CORRECTILLUMINATION_APPLY, SMOOTH
% CORRECTILLUMINATION_CALCULATEUSINGBACKGROUNDINTENSITIES.

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

%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

%textVAR01 = What did you call the images to be used to calculate the illumination function?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the illumination function?
%infotypeVAR02 = imagegroup indep
%defaultVAR02 = IllumBlue
IlluminationImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = (Optional) What do you want to call the raw projection image prior to dilation or smoothing? (This is an image produced during the calculations - it is typically not needed for downstream modules)
%infotypeVAR03 = imagegroup indep
%defaultVAR03 = ProjectedBlue
ProjectionImageName = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = (Optional) What do you want to call the projection image after dilation but prior to smoothing?  (This is an image produced during the calculations - it is typically not needed for downstream modules)
%infotypeVAR04 = imagegroup indep
%defaultVAR04 = DilatedProjectedBlue
DilatedProjectionImageName = char(handles.Settings.VariableValues{CurrentModuleNum,4});

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

%textVAR07 = If the incoming images are binary and you want to dilate each object in the final projection image, enter the radius (roughly equal to the original radius of the objects). Otherwise, enter 0.
%defaultVAR07 = 0
ObjectDilationRadius = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = Smoothing method: Enter the width of the artifacts (choose an even number) that are to be smoothed out by median filtering, or type P to fit a low order polynomial instead. For no smoothing, enter N. Note that smoothing is a time-consuming process.
%choiceVAR08 = No smoothing
%choiceVAR08 = Fit polynomial
SmoothingMethod = char(handles.Settings.VariableValues{CurrentModuleNum,8});
%inputtypeVAR08 = popupmenu custom

%textVAR09 = Do you want to rescale the illumination function so that the pixel intensities are all equal to or greater than one (Y or N)? This is recommended if you plan to use the division option in CorrectIllumination_Apply so that the resulting images will be in the range 0 to 1.
%choiceVAR09 = Yes
%choiceVAR09 = No
RescaleOption = char(handles.Settings.VariableValues{CurrentModuleNum,9});
%inputtypeVAR09 = popupmenu

%%%VariableRevisionNumber = 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% If the illumination correction function was to be calculated using
%%% all of the incoming images from a LoadImages module, it will already have been calculated
%%% the first time through the image set. No further calculations are
%%% necessary.
if (strcmp(EachOrAll,'All') == 1 && handles.Current.SetBeingAnalyzed ~= 1 && strcmp(SourceIsLoadedOrPipeline,'Load Images module') == 1)
    return
end

try
    NumericalObjectDilationRadius = str2num(ObjectDilationRadius);
catch
    error('In the Correct Illumination_Calculate Using Intensities module, you must enter a number for the radius to use to dilate objects. If you do not want to dilate objects enter 0 (zero).')
end

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow


ReadyFlag = 'Not Ready';
if strcmp(EachOrAll,'All')
    try
        if strcmp(SourceIsLoadedOrPipeline, 'Load Images module') == 1 && handles.Current.SetBeingAnalyzed == 1
            %%% The first time the module is run, the projection image is
            %%% calculated.
            [handles, IlluminationImage, ReadyFlag] = CPaverageimages(handles, 'DoNow', ImageName, 'ignore');
        elseif strncmpi(SourceIsLoadedOrPipeline, 'P',1) == 1
            [handles, IlluminationImage, ReadyFlag] = CPaverageimages(handles, 'Accumulate', ImageName, ProjectionImageName);
        else
            error('Image processing was canceled because you must choose either "L" or "P" in answer to the question "Are the images you want to use to calculate the illumination correction function to be loaded straight from a Load Images module (L), or are they being produced by the pipeline (P)" in the Correct Illumination_Calculate Using Intensities module.');
        end
    catch [ErrorMessage, ErrorMessage2] = lasterr;
        error(['An error occurred in the Correct Illumination_Calculate Using Intensities module. Matlab says the problem is: ', ErrorMessage, ErrorMessage2])
    end
elseif strcmp(EachOrAll,'Each')
    %%% Retrieves the current image.
    OrigImage = handles.Pipeline.(ImageName);
    %%% Checks that the original image is two-dimensional (i.e. not a
    %%% color image), which would disrupt several of the image
    %%% functions.
    if ndims(OrigImage) ~= 2
        error('Image processing was canceled because the Correct Illumination_Calculate Using Intensities module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
    end
    IlluminationImage = OrigImage;
    ReadyFlag = 'Ready';
else error('Image processing was canceled because you must choose either "E" or "A" in answer to the question "Enter E to calculate an illumination function for each image individually (in which case, choose P in the next box) or A to calculate an illumination function based on all the specified images to be corrected" in the Correct Illumination_Calculate Using Intensities module.');
end

%%% Dilates the objects, and/or smooths the ProjectedImage if the user requested.
if strcmp(ReadyFlag, 'Ready') == 1
    if NumericalObjectDilationRadius ~= 0
        ProjectionImage = IlluminationImage;
        IlluminationImage = CPdilatebinaryobjects(IlluminationImage, NumericalObjectDilationRadius);
    end
    if ~strcmp(SmoothingMethod,'No smoothing')
        %%% Smooths the projection image, if requested, but saves a raw copy
        %%% first.
        DilatedProjectionImage = IlluminationImage;
        if strcmp(SmoothingMethod,'Fit polynomial')
            SmoothingMethod = 'P';
        end
        IlluminationImage = CPsmooth(IlluminationImage,SmoothingMethod);
    end
    drawnow
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
    if exist('OrigImage','var')
        subplot(2,2,1); imagesc(OrigImage); CPcolormap(handles)
        title(['Input Image, Image Set # ',num2str(handles.Current.SetBeingAnalyzed)]);
        CPFixAspectRatio(OrigImage);
    end
    %%% Whether these images exist depends on whether the images have
    %%% been calculated yet (if running in pipeline mode, this won't occur
    %%% until the last image set is processed).  It also depends on
    %%% whether the user has chosen to dilate or smooth the projection
    %%% image.
    if exist('ProjectionImage','var')
        subplot(2,2,2); imagesc(ProjectionImage); CPcolormap(handles)
        title('Raw projection image prior to dilation');
    end
    if exist('DilatedProjectionImage','var')
        subplot(2,2,3); imagesc(DilatedProjectionImage); CPcolormap(handles)
        title('Projection image prior to smoothing');
    end
    subplot(2,2,4);
    imagesc(IlluminationImage);
    CPcolormap(handles);
    text(1,50,['Min Value: ' num2str(min(min(IlluminationImage)))],'Color','red');
    text(1,150,['Max Value: ' num2str(max(max(IlluminationImage)))],'Color','red');
    if strcmp(ReadyFlag, 'Ready')
        title('Final illumination correction function');
    else
        title('Projection calculated so far');
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow



%%% Saves images to the handles structure.
%%% If running in non-cycling mode (straight from the hard drive using
%%% a LoadImages module), the projection image and its flag need only
%%% be saved to the handles structure after the first image set is
%%% processed. If running in cycling mode (Pipeline mode), the
%%% projection image and its flag are saved to the handles structure
%%% after every image set is processed.
if strcmp(SourceIsLoadedOrPipeline, 'Pipeline') == 1 | (strcmp(SourceIsLoadedOrPipeline, 'Load Images module') == 1 && handles.Current.SetBeingAnalyzed == 1)
    fieldname = [IlluminationImageName];
    handles.Pipeline.(fieldname) = IlluminationImage;
    %%% Whether these images exist depends on whether the user has chosen
    %%% to dilate or smooth the projection image.
    if exist('ProjectionImage','var') == 1
        fieldname = [ProjectionImageName];
        handles.Pipeline.(fieldname) = ProjectionImage;
    end
    if exist('DilatedProjectionImage','var') == 1
        fieldname = [DilatedProjectionImageName];
        handles.Pipeline.(fieldname) = DilatedProjectionImage;
    end
    %%% Saves the ready flag to the handles structure so it can be used by
    %%% subsequent modules.
    fieldname = [ProjectionImageName,'ReadyFlag'];
    handles.Pipeline.(fieldname) = ReadyFlag;
end
