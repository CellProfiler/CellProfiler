function handles = CorrectIllumination_CalculateUsingIntensities(handles)

% Help for the Correct Illumination_Calculate Using Intensities module: 
% Category: Pre-processing
%
% This module corrects for uneven illumination of each image, based on
% information from a set of images collected at the same time.   If
% cells are distributed uniformly in the images, the mean of all the
% images should be a good estimate of the illumination.
%
% How it works:
% This module works by averaging together all of the images (making a
% projection), then smoothing this image and rescaling it.  This
% produces an image that represents the variation in illumination
% across the field of view.  This process is carried out before the
% first image set is processed; subsequent image sets use the already
% calculated image. Each image is divided by this illumination image
% to produce the corrected image.
%
% The smoothing can be done by fitting a low-order polynomial to the
% mean (projection) image (option = P), or by applying a filter to the
% image. The user enters an even number for the artifact width, and
% this number is divided by two to obtain the radius of a disk shaped
% structuring element which is used for filtering. Note that with
% either mode of calculation, the illumination function is scaled from
% 1 to infinity, so that if there is substantial variation across the
% field of view, the rescaling of each image might be dramatic,
% causing the image to appear darker.
%
% If you want to run this module only to calculate the mean and
% illumination images and not to correct every image in the directory,
% simply run the module as usual and use the button on the Timer to
% stop processing after the first image set.
%
% SAVING IMAGES: The illumination corrected images produced by this
% module can be easily saved using the Save Images module, using the
% name you assign. The mean image can be saved using the name
% ProjectionImageAD plus whatever you called the corrected image (e.g.
% ProjectionImageADCorrBlue). The Illumination correction image can be saved
% using the name IllumImageAD plus whatever you called the corrected
% image (e.g. IllumImageADCorrBlue).  Note that using the Save Images
% module saves a copy of the image in an image file format, which has
% lost some of the detail that a matlab file format would contain.  In
% other words, if you want to save the illumination image to use it in
% a later analysis, you should use the settings boxes within this
% module to save the illumination image in '.mat' format. If you want
% to save other intermediate images, alter the code for this module to
% save those images to the handles structure (see the SaveImages
% module help) and then use the Save Images module.
%
% See also CORRECTILLUMDIVIDEALLMEANRETRIEVEIMG,
% CORRECTILLUMSUBTRACTALLMIN,
% CORRECTILLUMDIVIDEEACHMIN_9, CORRECTILLUMDIVIDEEACHMIN_10,
% CORRECTILLUMSUBTRACTEACHMIN.

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

% PROGRAMMING NOTE
% HELP:
% The first unbroken block of lines will be extracted as help by
% CellProfiler's 'Help for this analysis module' button as well as Matlab's
% built in 'help' and 'doc' functions at the command line. It will also be
% used to automatically generate a manual page for the module. An example
% image demonstrating the function of the module can also be saved in tif
% format, using the same name as the module, and it will automatically be
% included in the manual page as well.  Follow the convention of: purpose
% of the module, description of the variables and acceptable range for
% each, how it works (technical description), info on which images can be 
% saved, and See also CAPITALLETTEROTHERMODULES. The license/author
% information should be separated from the help lines with a blank line so
% that it does not show up in the help displays.  Do not change the
% programming notes in any modules! These are standard across all modules
% for maintenance purposes, so anything module-specific should be kept
% separate.

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
drawnow

% PROGRAMMING NOTE
% VARIABLE BOXES AND TEXT: 
% The '%textVAR' lines contain the variable descriptions which are
% displayed in the CellProfiler main window next to each variable box.
% This text will wrap appropriately so it can be as long as desired.
% The '%defaultVAR' lines contain the default values which are
% displayed in the variable boxes when the user loads the module.
% The line of code after the textVAR and defaultVAR extracts the value
% that the user has entered from the handles structure and saves it as
% a variable in the workspace of this module with a descriptive
% name. The syntax is important for the %textVAR and %defaultVAR
% lines: be sure there is a space before and after the equals sign and
% also that the capitalization is as shown. 
% CellProfiler uses VariableRevisionNumbers to help programmers notify
% users when something significant has changed about the variables.
% For example, if you have switched the position of two variables,
% loading a pipeline made with the old version of the module will not
% behave as expected when using the new version of the module, because
% the settings (variables) will be mixed up. The line should use this
% syntax, with a two digit number for the VariableRevisionNumber:
% '%%%VariableRevisionNumber = 01'  If the module does not have this
% line, the VariableRevisionNumber is assumed to be 00.  This number
% need only be incremented when a change made to the modules will affect
% a user's previously saved settings. There is a revision number at
% the end of the license info at the top of the m-file for revisions
% that do not affect the user's previously saved settings files.

%%% Reads the current module number, because this is needed to find 
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

%textVAR01 = What did you call the image to be used to calculate the illumination correction function?
%defaultVAR01 = OrigBlue
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = What do you want to call the final illumination correction function?
%defaultVAR02 = IllumBlue
IlluminationImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = (Optional) What do you want to call the raw projection image prior to dilation or smoothing? (This is an image produced during the calculations - it is typically not needed for downstream modules)
%defaultVAR03 = ProjectedBlue
ProjectionImageName = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = (Optional) What do you want to call the projection image after dilation but prior to smoothing?  (This is an image produced during the calculations - it is typically not needed for downstream modules)
%defaultVAR04 = DilatedProjectedBlue
DilatedProjectionImageName = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = Enter E to calculate an illumination function for each image individually (in which case, choose P in the next box) or A to calculate an illumination function based on all the specified images to be corrected. Note that applying illumination correction on each image individually may make intensity measures not directly comparable across different images. Using illumination correction based on all images makes the assumption that the illumination anomalies are consistent across all the images in the set.
%defaultVAR05 = A
EachOrAll = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = Are the images you want to use to calculate the illumination correction function to be loaded straight from a Load Images module (L), or are they being produced by the pipeline (P)? If you choose L, the module will calculate the single, averaged projection image the first time through the pipeline by loading every image of the type specified in the Load Images module. It is then acceptable to use the resulting image later in the pipeline. If you choose P, the module will allow the pipeline to cycle through all of the image sets.  With this option, the module does not need to follow a Load Images module; it is acceptable to make the single, averaged projection from images resulting from other image processing steps in the pipeline. However, the resulting projection image will not be available until the last image set has been processed, so it cannot be used in subsequent modules.
%defaultVAR06 = L
SourceIsLoadedOrPipeline = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = If the incoming images are binary and you want to dilate each object in the final projection image, enter the radius (roughly equal to the original radius of the objects). Otherwise, enter 0. Note that if you are using a small image set, there will be spaces in the projection image that contain no objects and median filtering is unlikely to work well. 
%defaultVAR07 = 0
ObjectDilationRadius = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = Smoothing method: Enter the width of the artifacts (choose an even number) that are to be smoothed out by median filtering, or type P to fit a low order polynomial instead. For no smoothing, enter N. Note that smoothing is a time-consuming process.
%defaultVAR08 = N
SmoothingMethod = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% If the illumination correction function was to be calculated using
%%% all of the incoming images from a LoadImages module, it will already have been calculated
%%% the first time through the image set. No further calculations are
%%% necessary.
if (strcmpi(EachOrAll,'A') == 1 && handles.Current.SetBeingAnalyzed ~= 1 && strcmpi(SourceIsLoadedOrPipeline,'L') == 1)
    return
end

try NumericalObjectDilationRadius = str2num(ObjectDilationRadius);
catch error('In the Correct Illumination_Calculate Using Intensities module, you must enter a number for the radius to use to dilate objects. If you do not want to dilate objects enter 0 (zero).')
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

ReadyFlag = 'Not Ready';
if strcmpi(EachOrAll,'A') == 1
    try
        if strncmpi(SourceIsLoadedOrPipeline, 'L',1) == 1 && handles.Current.SetBeingAnalyzed == 1
            %%% The first time the module is run, the projection image is
            %%% calculated.
            [IlluminationImage, ReadyFlag] = CPaverageimages(handles, 'DoNow', ImageName);
        elseif strncmpi(SourceIsLoadedOrPipeline, 'P',1) == 1
            [IlluminationImage, ReadyFlag] = CPaverageimages(handles, 'Accumulate', ImageName);
        else
            error('Image processing was canceled because you must choose either "L" or "P" in answer to the question "Are the images you want to use to calculate the illumination correction function to be loaded straight from a Load Images module (L), or are they being produced by the pipeline (P)" in the Correct Illumination_Calculate Using Intensities module.');
        end
    catch [ErrorMessage, ErrorMessage2] = lasterr;
        error(['An error occurred in the Correct Illumination_Calculate Using Intensities module. Matlab says the problem is: ', ErrorMessage, ErrorMessage2])
    end
elseif strcmpi(EachOrAll,'E') == 1
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
    if strcmpi(SmoothingMethod,'N') ~= 1
        %%% Smooths the projection image, if requested, but saves a raw copy
        %%% first.
        DilatedProjectionImage = IlluminationImage;
        IlluminationImage = CPsmooth(IlluminationImage,SmoothingMethod);
    end
    drawnow
end

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

% PROGRAMMING NOTE
% DISPLAYING RESULTS:
% Some calculations produce images that are used only for display or
% for saving to the hard drive, and are not used by downstream
% modules. To speed processing, these calculations are omitted if the
% figure window is closed and the user does not want to save the
% images.

fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
if any(findobj == ThisModuleFigureNumber) == 1;
% PROGRAMMING NOTE
% DRAWNOW BEFORE FIGURE COMMAND:
% The "drawnow" function executes any pending figure window-related
% commands.  In general, Matlab does not update figure windows until
% breaks between image analysis modules, or when a few select commands
% are used. "figure" and "drawnow" are two of the commands that allow
% Matlab to psause and carry out any pending figure window- related
% commands (like zooming, or pressing timer pause or cancel buttons or
% pressing a help button.)  If the drawnow command is not used
% immediately prior to the figure(ThisModuleFigureNumber) line, then
% immediately after the figure line executes, the other commands that
% have been waiting are executed in the other windows.  Then, when
% Matlab returns to this module and goes to the subplot line, the
% figure which is active is not necessarily the correct one. This
% results in strange things like the subplots appearing in the timer
% window or in the wrong figure window, or in help dialog boxes.
    drawnow
    %%% Activates the appropriate figure window.
    figure(ThisModuleFigureNumber);
    %%% A subplot of the figure window is set to display the original
    %%% image, some intermediate images, and the final corrected image.
    if exist('OrigImage','var') == 1
    subplot(2,2,1); imagesc(OrigImage); colormap(gray)
    title(['Input Image, Image Set # ',num2str(handles.Current.SetBeingAnalyzed)]);
    end
    %%% Whether these images exist depends on whether the images have
    %%% been calculated yet (if running in pipeline mode, this won't occur
    %%% until the last image set is processed).  It also depends on
    %%% whether the user has chosen to dilate or smooth the projection
    %%% image.
    if exist('ProjectionImage','var') == 1
        subplot(2,2,2); imagesc(ProjectionImage); colormap(gray)
        title('Raw projection image prior to dilation or smoothing');
    end
    if exist('DilatedProjectionImage','var') == 1
        subplot(2,2,3); imagesc(DilatedProjectionImage); colormap(gray)
        title('Projection image after dilation but prior to smoothing');
    end
    if exist('IlluminationImage','var') == 1
        subplot(2,2,4); imagesc(IlluminationImage); colormap(gray)
        title('Final illumination correction function');
    else subplot(2,2,4);
        title('Illumination correction function is not yet calculated');
    end
end

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
% 'structure', and whose name is handles. The contents of the handles
% structure are printed out at the command line of Matlab using the
% Tech Diagnosis button. The only variables present in the main
% handles structure are handles to figures and gui elements.
% Everything else should be saved in one of the following
% substructures:
%
% handles.Settings:
%       Everything in handles.Settings is stored when the user uses
% the Save pipeline button, and these data are loaded into
% CellProfiler when the user uses the Load pipeline button. This
% substructure contains all necessary information to re-create a
% pipeline, including which modules were used (including variable
% revision numbers), their setting (variables), and the pixel size.
%   Fields currently in handles.Settings: PixelSize, ModuleNames,
% VariableValues, NumbersOfVariables, VariableRevisionNumbers.
%
% handles.Pipeline:
%       This substructure is deleted at the beginning of the
% analysis run (see 'Which substructures are deleted prior to an
% analysis run?' below). handles.Pipeline is for storing data which
% must be retrieved by other modules. This data can be overwritten as
% each image set is processed, or it can be generated once and then
% retrieved during every subsequent image set's processing, or it can
% be saved for each image set by saving it according to which image
% set is being analyzed, depending on how it will be used by other
% modules. Any module which produces or passes on an image needs to
% also pass along the original filename of the image, named after the
% new image name, so that if the SaveImages module attempts to save
% the resulting image, it can be named by appending text to the
% original file name.
%   Example fields in handles.Pipeline: FileListOrigBlue,
% PathnameOrigBlue, FilenameOrigBlue, OrigBlue (which contains the actual image).
%
% handles.Current:
%       This substructure contains information needed for the main
% CellProfiler window display and for the various modules to
% function. It does not contain any module-specific data (which is in
% handles.Pipeline).
%   Example fields in handles.Current: NumberOfModules,
% StartupDirectory, DefaultOutputDirectory, DefaultImageDirectory,
% FilenamesInImageDir, CellProfilerPathname, ImageToolHelp,
% DataToolHelp, FigureNumberForModule01, NumberOfImageSets,
% SetBeingAnalyzed, TimeStarted, CurrentModuleNumber.
%
% handles.Preferences: 
%       Everything in handles.Preferences is stored in the file
% CellProfilerPreferences.mat when the user uses the Set Preferences
% button. These preferences are loaded upon launching CellProfiler.
% The PixelSize, DefaultImageDirectory, and DefaultOutputDirectory
% fields can be changed for the current session by the user using edit
% boxes in the main CellProfiler window, which changes their values in
% handles.Current. Therefore, handles.Current is most likely where you
% should retrieve this information if needed within a module.
%   Fields currently in handles.Preferences: PixelSize, FontSize,
% DefaultModuleDirectory, DefaultOutputDirectory,
% DefaultImageDirectory.
%
% handles.Measurements:
%       Everything in handles.Measurements contains data specific to each
% image set analyzed for exporting. It is used by the ExportProjectionImage
% and ExportCellByCell data tools. This substructure is deleted at the
% beginning of the analysis run (see 'Which substructures are deleted
% prior to an analysis run?' below).
%    Note that two types of measurements are typically made: Object
% and Image measurements.  Object measurements have one number for
% every object in the image (e.g. ObjectArea) and image measurements
% have one number for the entire image, which could come from one
% measurement from the entire image (e.g. ImageTotalIntensity), or
% which could be an aggregate measurement based on individual object
% measurements (e.g. ImageMeanArea).  Use the appropriate prefix to
% ensure that your data will be extracted properly. It is likely that
% Subobject will become a new prefix, when measurements will be
% collected for objects contained within other objects. 
%       Saving measurements: The data extraction functions of
% CellProfiler are designed to deal with only one "column" of data per
% named measurement field. So, for example, instead of creating a
% field of XY locations stored in pairs, they should be split into a
% field of X locations and a field of Y locations. It is wise to
% include the user's input for 'ObjectName' or 'ImageName' as part of
% the fieldname in the handles structure so that multiple modules can
% be run and their data will not overwrite each other.
%   Example fields in handles.Measurements: ImageCountNuclei,
% ObjectAreaCytoplasm, FilenameOrigBlue, PathnameOrigBlue,
% TimeElapsed.
%
% Which substructures are deleted prior to an analysis run?
%       Anything stored in handles.Measurements or handles.Pipeline
% will be deleted at the beginning of the analysis run, whereas
% anything stored in handles.Settings, handles.Preferences, and
% handles.Current will be retained from one analysis to the next. It
% is important to think about which of these data should be deleted at
% the end of an analysis run because of the way Matlab saves
% variables: For example, a user might process 12 image sets of nuclei
% which results in a set of 12 measurements ("ImageTotalNucArea")
% stored in handles.Measurements. In addition, a processed image of
% nuclei from the last image set is left in the handles structure
% ("SegmNucImg"). Now, if the user uses a different module which
% happens to have the same measurement output name "ImageTotalNucArea"
% to analyze 4 image sets, the 4 measurements will overwrite the first
% 4 measurements of the previous analysis, but the remaining 8
% measurements will still be present. So, the user will end up with 12
% measurements from the 4 sets. Another potential problem is that if,
% in the second analysis run, the user runs only a module which
% depends on the output "SegmNucImg" but does not run a module that
% produces an image by that name, the module will run just fine: it
% will just repeatedly use the processed image of nuclei leftover from
% the last image set, which was left in handles.Pipeline.

%%% Saves images to the handles structure.
%%% If running in non-cycling mode (straight from the hard drive using
%%% a LoadImages module), the projection image and its flag need only
%%% be saved to the handles structure after the first image set is
%%% processed. If running in cycling mode (Pipeline mode), the
%%% projection image and its flag are saved to the handles structure
%%% after every image set is processed.
if strncmpi(SourceIsLoadedOrPipeline, 'P',1) == 1 | (strncmpi(SourceIsLoadedOrPipeline, 'L',1) == 1 && handles.Current.SetBeingAnalyzed == 1)
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