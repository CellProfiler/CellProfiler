function handles = SmoothImageForIllumCorrection(handles)

% Help for the Smooth Image For Illum Correction module:
% Category: Pre-processing
%
% This module applies a smoothing function to the incoming image. The most
% common use for the module is to smooth a projection image prior to using
% it to correct for uneven illumination of each image.
%
% How it works:
% This module works by smoothing an image and rescaling it to an
% appropriate range to be used for illumination correction.  This
% produces an image that represents the variation in illumination
% across the field of view. If the user specifies that the module is to be
% run on a projection image produced by another module, the module waits
% for the make projection module to produce a flag telling this module that
% the projection image is ready. This is because projection image modules
% can be run in two modes: one where the projection image is produced
% during the first image set's processing (non-cycling, LoadImages mode)
% and the other where the projection image is produced only at the end of
% the last image set's processing (cycling, Pipeline mode).
%
% The smoothing can be done by fitting a low-order polynomial to the mean
% (projection) image (option = P), or by applying a median filter to the
% image (option = a number). In filtering mode, the user enters an even
% number for the artifact width, and this number is divided by two to
% obtain the radius of a disk shaped structuring element which is used for
% filtering. Note that with either mode of calculation, the illumination
% function is scaled from 1 to infinity, so that if there is substantial
% variation across the field of view, the rescaling of each image might be
% dramatic, causing the image to appear darker.
%
% THIS PART MAY BE OUTDATED >>> SAVING IMAGES: The illumination corrected
% images produced by this module can be easily saved using the Save Images
% module, using the name you assign. The mean image can be saved using the
% name MeanImageAD plus whatever you called the corrected image (e.g.
% MeanImageADCorrBlue). The Illumination correction image can be saved
% using the name IllumImageAD plus whatever you called the corrected image
% (e.g. IllumImageADCorrBlue).  Note that using the Save Images module
% saves a copy of the image in an image file format, which has lost some of
% the detail that a matlab file format would contain.  In other words, if
% you want to save the smoothed image to use it in a later analysis,
% you should use the settings boxes within this module to save the
% smoothed image in '.mat' format. If you want to save other
% intermediate images, alter the code for this module to save those images
% to the handles structure (see the SaveImages module help) and then use
% the Save Images module.
%
% See also MAKEPROJECTION, CORRECTILLUMDIVIDEALLMEAN,
% CORRECTILLUMDIVIDEALLMEANRETRIEVEIMG,
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

%textVAR01 = What did you call the image to be smoothed?
%defaultVAR01 = OrigBlue
OrigImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = What do you want to call the smoothed image?
%defaultVAR02 = CorrBlue
SmoothedImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Are you using this module to smooth a projection image?
%defaultVAR03 = Y
ProjectionImageOrNot = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = Smoothing method: Enter the width of the artifacts (choose an even number) that are to be smoothed out by median filtering, or type P to fit a low order polynomial instead.
%defaultVAR04 = 50
SmoothingMethod = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%%%VariableRevisionNumber = 0

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%%%%%%%%%%%%%%%%%%
%%% I am pretty sure that this Filename section should be done each time
%%% through the set. The return function is used at several points below,
%%% so I could not leave this section at the end of the file, where items
%%% are normally stored to the handles structure.

% %%% Determines the filename of the image to be analyzed.
% fieldname = ['Filename', OrigImageName];
% FileName = handles.Pipeline.(fieldname)(handles.Current.SetBeingAnalyzed);
% %%% Saves the original file name to the handles structure in a
% %%% field named after the corrected image name.
% fieldname = ['Filename', SmoothedImageName];
% handles.Pipeline.(fieldname)(handles.Current.SetBeingAnalyzed) = FileName;
%%%%%%%%%%%%%%%%%%

%%% The following checks to see whether it is appropriate to calculate the
%%% smooth image at this time or not.  If not, the return function abandons
%%% the remainder of the code, which will otherwise calculate the smooth
%%% image, save it to the handles, and display the results.
if strncmpi(ProjectionImageOrNot,'Y',1) == 1
    fieldname = [OrigImageName,'ReadyFlag'];
    ReadyFlag = handles.Pipeline.(fieldname);
    if strcmp(ReadyFlag, 'ProjectedImageNotReady') == 1
        %%% If the projection image is not ready, the module aborts until
        %%% the next cycle.
        return
    elseif strcmp(ReadyFlag, 'ProjectedImageReady') == 1
        %%% If the smoothed image has already been calculated, the module
        %%% aborts until the next cycle.
        if isfield(handles.Pipeline, SmoothedImageName) == 1
            return
        end
        %%% If we make it to this point, it is OK to proceed to calculating the smooth
        %%% image, etc.
    else error(['There is a programming error of some kind. The Smooth Image For Illum Correction module was expecting to find the text ProjectedImageReady or ProjectedImageNotReady in the field called ', fieldname, ' but that text was not matched for some reason.'])
    end
elseif strncmpi(ProjectionImageOrNot,'N',1) == 1
    %%% If we make it to this point, it is OK to proceed to calculating the smooth
    %%% image, etc.
else
    error(['Your response to the question "Are you using this module to smooth a projection image?" was not recognized. Please enter Y or N.'])
end


%%% If we make it to this point, it is OK to proceed to calculating the smooth
%%% image, etc.

%%% Reads (opens) the image you want to analyze and assigns it to a
%%% variable.
%%% Checks whether the image to be analyzed exists in the handles structure.
if isfield(handles.Pipeline, OrigImageName)==0,
    %%% If the image is not there, an error message is produced.  The error
    %%% is not displayed: The error function halts the current function and
    %%% returns control to the calling function (the analyze all images
    %%% button callback.)  That callback recognizes that an error was
    %%% produced because of its try/catch loop and breaks out of the image
    %%% analysis loop without attempting further modules.
    error(['Image processing was canceled because the Smooth Image For Illum Correction module could not find the input image.  It was supposed to be named ', OrigImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
%%% Reads the image.
OrigImage = handles.Pipeline.(OrigImageName);

%%% Checks that the original image is two-dimensional (i.e. not a color
%%% image), which would disrupt several of the image functions.
if ndims(OrigImage) ~= 2
    error('Image processing was canceled because the Smooth Image For Illum Correction module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
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

%%% Smooths the OrigImage according to the user's specifications.
if strcmpi(SmoothingMethod,'P') == 1
    %%% The following is used to fit a low-dimensional polynomial to the original image.
    [x,y] = meshgrid(1:size(OrigImage,2), 1:size(OrigImage,1));
    x2 = x.*x;
    y2 = y.*y;
    xy = x.*y;
    o = ones(size(OrigImage));
    Ind = find(OrigImage > 0);
    Coeffs = [x2(Ind) y2(Ind) xy(Ind) x(Ind) y(Ind) o(Ind)] \ double(OrigImage(Ind));
    drawnow
    SmoothedImage1 = reshape([x2(:) y2(:) xy(:) x(:) y(:) o(:)] * Coeffs, size(OrigImage));
    %%% The final SmoothedImage is produced by dividing each
    %%% pixel of the smoothed image by a scalar: the minimum
    %%% pixel value anywhere in the smoothed image. (If the
    %%% minimum value is zero, .00000001 is substituted instead.)
    %%% This rescales the SmoothedImage from 1 to some number.
    %%% This ensures that the final, corrected image will be in a
    %%% reasonable range, from zero to 1.
    drawnow
    SmoothedImage = SmoothedImage1 ./ max([min(min(SmoothedImage1)); .00000001]);
    %%% Note: the following "imwrite" saves the illumination
    %%% correction image in TIF format, but the image is compressed
    %%% so it is not as smooth as the image that is saved using the
    %%% "save" function below, which is stored in matlab ".mat"
    %%% format.
    % imwrite(SmoothedImage, 'SmoothedImage.tif', 'tif')
else try ArtifactWidth = str2num(SmoothingMethod);
        ArtifactRadius = 0.5*ArtifactWidth;
        StructuringElementLogical = getnhood(strel('disk', ArtifactRadius));
 %       MsgBoxHandle = msgbox('Now calculating the illumination function, which may take a long time.');
        SmoothedImage1 = ordfilt2(OrigImage, floor(sum(sum(StructuringElementLogical))/2), StructuringElementLogical, 'symmetric');
        SmoothedImage = SmoothedImage1 ./ max([min(min(SmoothedImage1)); .00000001]);
 %       MsgBox = 'Calculations for the illumination function are complete.';
    catch
        error(['The text you entered for the smoothing method in the Smooth Image For Illum Correction module is unrecognizable for some reason. You must enter a positive, even number or the letter P.  Your entry was ',SmoothingMethod])
    end
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
    %%% Sets the width of the figure window to be appropriate (half width),
    %%% the first time through the set.
    if handles.Current.SetBeingAnalyzed == 1 | strncmpi(ProjectionImageOrNot,'Y',1) == 1
        originalsize = get(ThisModuleFigureNumber, 'position');
        newsize = originalsize;
        newsize(3) = originalsize(3)/2;
        set(ThisModuleFigureNumber, 'position', newsize);
        drawnow
    end
    drawnow
    %%% Activates the appropriate figure window.
    figure(ThisModuleFigureNumber);
    %%% A subplot of the figure window is set to display the original
    %%% image and the smoothed image.
    subplot(2,1,1); imagesc(OrigImage);
    colormap(gray)
    title('Input Image');
    subplot(2,1,2); imagesc(SmoothedImage);
    colormap(gray)
    title('Smoothed Image');
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
% image set analyzed for exporting. It is used by the ExportMeanImage
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

%%% Saves the corrected image to the
%%% handles structure so it can be used by subsequent modules.
handles.Pipeline.(SmoothedImageName) = SmoothedImage;