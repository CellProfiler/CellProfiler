function handles = RGBMerge(handles)

% Help for the RGB Merge module:
% Category: Pre-processing
%
% Takes 1 to 3 images and assigns them to colors in a final, RGB
% image.  Each color's brightness can be adjusted independently.
%
% Settings:
%
% Adjustment factors: Leaving the adjustment factors set to 1 will
% balance all three colors equally in the final image, and they will
% use the same range of intensities as each individual incoming image.
% Using factors less than 1 will decrease the intensity of that
% color in the final image, and values greater than 1 will increase
% it.  Setting the adjustment factor to zero will cause that color to
% be entirely blank.
%
% SAVING IMAGES: The RGB image produced by this module can be easily
% saved using the Save Images module, using the name you assign. If
% you want to save other intermediate images, alter the code for this
% module to save those images to the handles structure (see the
% SaveImages module help) and then use the Save Images module.
%
% See also RGBSPLIT, RGBTOGRAY.

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

%textVAR01 = What did you call the image to be colored blue?
%infotypeVAR01 = imagegroup
%choiceVAR01 = Leave this black
BlueImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What did you call the image to be colored green?
%infotypeVAR02 = imagegroup
%choiceVAR02 = Leave this black
GreenImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 = What did you call the image to be colored red?
%infotypeVAR03 = imagegroup
%choiceVAR03 = Leave this black
RedImageName = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 = Type "N" in any slots above to leave that color black.

%textVAR05 = What do you want to call the resulting image?
%infotypeVAR05 = imagegroup indep
%defaultVAR05 = RGBImage
RGBImageName = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = Enter the adjustment factor for the blue image
%defaultVAR06 = 1
BlueAdjustmentFactor = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = Enter the adjustment factor for the green image
%defaultVAR07 = 1
GreenAdjustmentFactor = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = Enter the adjustment factor for the red image
%defaultVAR08 = 1
RedAdjustmentFactor = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines whether the user has specified an image to be loaded in
%%% blue.
if strcmp(BlueImageName, 'Leave this black') == 0
    %%% Read (open) the images and assign them to variables.
    fieldname = ['', BlueImageName];
    %%% Checks whether the image to be analyzed exists in the handles structure.
if isfield(handles.Pipeline, fieldname)==0,
        %%% If the image is not there, an error message is produced.  The error
        %%% is not displayed: The error function halts the current function and
        %%% returns control to the calling function (the analyze all images
        %%% button callback.)  That callback recognizes that an error was
        %%% produced because of its try/catch loop and breaks out of the image
        %%% analysis loop without attempting further modules.
        error(['Image processing was canceled because the RGB Merge module could not find the input image.  It was supposed to be named ', BlueImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
    end
    %%% Reads the image.
    BlueImage = handles.Pipeline.(fieldname);
    BlueImageExists = 1;
else BlueImageExists = 0;
end

drawnow
%%% Repeat for Green and Red.
if strcmp(GreenImageName, 'Leave this black') == 0
    if isfield(handles.Pipeline, GreenImageName) == 0
        error(['Image processing was canceled because the RGB Merge module could not find the input image.  It was supposed to be named ', GreenImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
    end
    GreenImage = handles.Pipeline.(GreenImageName);
    GreenImageExists = 1;
else GreenImageExists = 0;
end
if strcmp(RedImageName, 'Leave this black') == 0
    if isfield(handles.Pipeline, RedImageName) == 0
        error(['Image processing was canceled because the RGB Merge module could not find the input image.  It was supposed to be named ', RedImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
    end
    RedImage = handles.Pipeline.(RedImageName);
    RedImageExists = 1;
else RedImageExists = 0;
end
drawnow

%%% If any of the colors are to be left black, creates the appropriate
%%% image.
if BlueImageExists == 0 && RedImageExists == 0 && GreenImageExists == 0
    error('Image processing was canceled because you have not selected any images to be merged in the RGB Merge module.')
end
if BlueImageExists == 0 && RedImageExists == 0 && GreenImageExists == 1
    BlueImage = zeros(size(GreenImage));
    RedImage = zeros(size(GreenImage));
end
if BlueImageExists == 0 && RedImageExists == 1 && GreenImageExists == 0
    BlueImage = zeros(size(RedImage));
    GreenImage = zeros(size(RedImage));
end
if BlueImageExists == 1 && RedImageExists == 0 && GreenImageExists == 0
    RedImage = zeros(size(BlueImage));
    GreenImage = zeros(size(BlueImage));
end
if BlueImageExists == 1 && RedImageExists == 1 && GreenImageExists == 0
    GreenImage = zeros(size(BlueImage));
end
if BlueImageExists == 0 && RedImageExists == 1 && GreenImageExists == 1
    BlueImage = zeros(size(GreenImage));
end
if BlueImageExists == 1 && RedImageExists == 0 && GreenImageExists == 1
    RedImage = zeros(size(BlueImage));
end

%%% Checks whether the three images are the same size.
try
    if size(BlueImage) ~= size(GreenImage)
        error('Image processing was canceled because the three images selected for the RGB Merge module are not the same size.  The pixel dimensions must be identical.')
    end
    if size(RedImage) ~= size(GreenImage)
        error('Image processing was canceled because the three images selected for the RGB Merge module are not the same size.  The pixel dimensions must be identical.')
    end
catch error('Image processing was canceled because there was a problem with one of three images selected for the RGB Merge module. Most likely one of the images is not in the same format as the others - for example, one of the images might already be in RGB format.')
end

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%

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

%%% If any of the images are binary/logical format, they must be
%%% converted to a double first before immultiply.
RGBImage(:,:,1) = immultiply(double(RedImage),str2double(RedAdjustmentFactor));
RGBImage(:,:,2) = immultiply(double(GreenImage),str2double(GreenAdjustmentFactor));
RGBImage(:,:,3) = immultiply(double(BlueImage),str2double(BlueAdjustmentFactor));

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
    %%% Activates the appropriate figure window.
    CPfigure(handles,ThisModuleFigureNumber);
    %%% A subplot of the figure window is set to display the Merged RGB
    %%% image.  Using imagesc or image instead of imshow doesn't work when
    %%% some of the pixels are saturated.
    subplot(2,2,1); imagesc(RGBImage);
    title(['Merged RGB Image, Image Set # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the blue image.
    subplot(2,2,2); imagesc(BlueImage); colormap(gray), title('Blue Image');
    %%% A subplot of the figure window is set to display the green image.
    subplot(2,2,3); imagesc(GreenImage); colormap(gray), title('Green Image');
    %%% A subplot of the figure window is set to display the red image.
    subplot(2,2,4); imagesc(RedImage); colormap(gray), title('Red Image');
    CPFixAspectRatio(RGBImage);
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
% handles.Measurements
%      Data extracted from input images are stored in the
% handles.Measurements substructure for exporting or further analysis.
% This substructure is deleted at the beginning of the analysis run
% (see 'Which substructures are deleted prior to an analysis run?'
% below). The Measurements structure is organized in two levels. At
% the first level, directly under handles.Measurements, there are
% substructures (fields) containing measurements of different objects.
% The names of the objects are specified by the user in the Identify
% modules (e.g. 'Cells', 'Nuclei', 'Colonies').  In addition to these
% object fields is a field called 'Image' which contains information
% relating to entire images, such as filenames, thresholds and
% measurements derived from an entire image. That is, the Image field
% contains any features where there is one value for the entire image.
% As an example, the first level might contain the fields
% handles.Measurements.Image, handles.Measurements.Cells and
% handles.Measurements.Nuclei.
%      In the second level, the measurements are stored in matrices
% with dimension [#objects x #features]. Each measurement module
% writes its own block; for example, the MeasureAreaShape module
% writes shape measurements of 'Cells' in
% handles.Measurements.Cells.AreaShape. An associated cell array of
% dimension [1 x #features] with suffix 'Features' contains the names
% or descriptions of the measurements. The export data tools, e.g.
% ExportData, triggers on this 'Features' suffix. Measurements or data
% that do not follow the convention described above, or that should
% not be exported via the conventional export tools, can thereby be
% stored in the handles.Measurements structure by leaving out the
% '....Features' field. This data will then be invisible to the
% existing export tools.
%      Following is an example where we have measured the area and
% perimeter of 3 cells in the first image and 4 cells in the second
% image. The first column contains the Area measurements and the
% second column contains the Perimeter measurements.  Each row
% contains measurements for a different cell:
% handles.Measurements.Cells.AreaShapeFeatures = {'Area' 'Perimeter'}
% handles.Measurements.Cells.AreaShape{1} = 	40		20
%                                               100		55
%                                              	200		87
% handles.Measurements.Cells.AreaShape{2} = 	130		100
%                                               90		45
%                                               100		67
%                                               45		22
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
% ("SegmNucImg"). Now, if the user uses a different algorithm which
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

%%% Saves the adjusted image to the handles structure so it can be used by
%%% subsequent modules.
handles.Pipeline.(RGBImageName) = RGBImage;
