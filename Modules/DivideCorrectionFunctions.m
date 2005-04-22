function handles = DivideCorrectionFunctions(handles)

% Help for the Divide Correction Functions module:
% Category: Pre-processing
%
% Sorry, this module has not yet been documented.
%
% SAVING IMAGES: The illumination correction function image produced by
% this module can be easily saved using the Save Images module, using the
% name you assign. If you want to save other intermediate images, alter the
% code for this module to save those images to the handles structure (see
% the SaveImages module help) and then use the Save Images module.
%
% See also MAKEPROJECTION and SMOOTHIMAGEFORILLUMCORRECTION.

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

%textVAR01 = What did you call the projection of intensity images?
%defaultVAR01 = OrigBlue
IntensityProjectionImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = What did you call the projection of the images in which objects were identified?
%defaultVAR02 = ThreshBlue
MaskedProjectionImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = What do you want to call the resulting illumination function image?
%defaultVAR03 = IllumCorrImgBlue
IlluminationFunctionImageName = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% None of the following code is performed until the last set of images
%%% has been analyzed, since that is when the projection images will
%%% typically be ready.
if handles.Current.SetBeingAnalyzed == handles.Current.NumberOfImageSets
    %%% POSSIBLE IMPROVEMENT: We are always going to use this module when
    %%% the last image set is being analyzed, because that's when the projection images are
    %%% ready to be run. It is possible someone else might want to run it after
    %%% the first image set, if both projection images are being calculated in
    %%% LoadImages mode (see the MakeProjection module for an explanation), but
    %%% currently we do not have support for that because I don't know whether
    %%% anyone will ever need it.

    %%% Reads (opens) the image to be analyzed and assigns it to a variable.
    %%% Checks whether the image to be analyzed exists in the handles structure.
    if isfield(handles.Pipeline, IntensityProjectionImageName)==0,
        %%% If the image is not there, an error message is produced.  The error
        %%% is not displayed: The error function halts the current function and
        %%% returns control to the calling function (the analyze all images
        %%% button callback.)  That callback recognizes that an error was
        %%% produced because of its try/catch loop and breaks out of the image
        %%% analysis loop without attempting further modules.
        error(['Image processing was canceled because the Divide Correction Functions module could not find the input image.  It was supposed to be named ', IntensityProjectionImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
    end
    %%% Reads the image.
    IntensityProjectionImage = handles.Pipeline.(IntensityProjectionImageName);

    %%% Checks whether the image to be analyzed exists in the handles structure.
    if isfield(handles.Pipeline, MaskedProjectionImageName)==0,
        error(['Image processing was canceled because the Divide Correction Functions module could not find the input image.  It was supposed to be named ', MaskedProjectionImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
    end
    %%% Reads the image.
    MaskedProjectionImage = handles.Pipeline.(MaskedProjectionImageName);

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

    %%% Checks to be sure the two images are the same size.
    if size(IntensityProjectionImage) ~= size(MaskedProjectionImage)
        error('Image processing was canceled because the two input images into the Divide Correction Functions module are not the same size')
    end
    %%% Makes sure neither projection image has zeros to prevent
    %%% errors when dividing.
    PixelIntensities = unique(IntensityProjectionImage(:,:));
    if PixelIntensities(1) == 0
        %%% The minimum acceptable value is set to 0.01, or the lowest
        %%% non-zero pixel intensity in the image.
        MinimumAcceptableValue = min(.01, PixelIntensities(2));
        IntensityProjectionImage(IntensityProjectionImage == 0) = MinimumAcceptableValue;
    end
    %%% Makes sure neither projection image has zeros to prevent
    %%% errors when dividing.
    PixelIntensities2 = unique(MaskedProjectionImage(:,:));
    if PixelIntensities2(1) == 0
        %%% The minimum acceptable value is set to 0.01, or the lowest
        %%% non-zero pixel intensity in the image.
        MinimumAcceptableValue2 = min(.01, PixelIntensities2(2));
        MaskedProjectionImage(MaskedProjectionImage == 0) = MinimumAcceptableValue2;
    end
    %%% Divides the Intensity projection image by the masked projection image.
    IlluminationImage = IntensityProjectionImage./MaskedProjectionImage;
    drawnow
    %%% The final IlluminationImage is produced by dividing each
    %%% pixel of the illumination image by a scalar: the minimum
    %%% pixel value anywhere in the illumination image. (If the
    %%% minimum value is zero, .00000001 is substituted instead.)
    %%% This rescales the IlluminationImage from 1 to some number.
    %%% This ensures that the final, corrected image will be in a
    %%% reasonable range, from zero to 1.
    IlluminationFunctionImage = IlluminationImage ./ max([min(min(IlluminationImage)); .00000001]);
    ReadyFlag = 1;
else
    ReadyFlag = 0;
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
    % Matlab to pause and carry out any pending figure window- related
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
    if ReadyFlag == 1
        %%% Activates the appropriate figure window.
        figure(ThisModuleFigureNumber);
        %%% A subplot of the figure window is set to display the original image.
        subplot(2,2,1); imagesc(IntensityProjectionImage);colormap(gray);
        title('Input Intensity Projection Image');
        subplot(2,2,2); imagesc(MaskedProjectionImage);
        title('Input Identified Object Projection Image');
        subplot(2,2,3); imagesc(IlluminationFunctionImage);
        title('Resulting Illumination Function Image');
    elseif handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        %%% Activates the appropriate figure window.
        figure(ThisModuleFigureNumber);
        title({'Waiting for projection images to be calculated.';'The results of this module will be shown when the last image set is processed.'});
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

if ReadyFlag == 1
    %%% The IlluminationFunctionImage is saved to the handles structure so
    %%% it can be used by subsequent modules.
    handles.Pipeline.(IlluminationFunctionImageName) = IlluminationFunctionImage;
end