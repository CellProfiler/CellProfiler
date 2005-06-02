function handles = CorrectIllumination_CalculateUsingBackgroundIntensities(handles)

% Help for the Correct Illumination_Calculate Using Background Intensities module: 
% Category: Pre-processing
% 
% This module corrects for uneven illumination of each image, based on
% an image calculated by another module in the pipeline, or loaded
% from a .mat file using the LoadSingleImage module.
%
% How it works:
% The minimum pixel value is determined within each "block" of
% each image. If requested, the values are averaged together for all
% images (this processing is done the first time through the image
% analysis pipeline). If requested, these values are then smoothed.
% The block dimensions are entered by the user, and should be large
% enough that every block is likely to contain some "background"
% pixels, where no cells are located. Theoretically, the intensity
% values of these background pixels should always be the same number.
% With uneven illumination, the background pixels will vary across the
% image, and this yields a function that presumably affects the
% intensity of the "real" pixels, those that comprise cells.
% Therefore, once the average minimums are determined across the
% images, the minimums are smoothed out. This produces an image that
% represents the variation in illumination across the field of view.
%
% This module is loosely based on the Matlab demo "Correction of
% non-uniform illumination" in the Image Processing Toolbox demos
% "Enhancement" category.
% MATLAB6p5/toolbox/images/imdemos/examples/enhance/ipss003.html
%
% SAVING IMAGES: The illumination correction function produced by this
% module can be easily saved using the Save Images module, using the
% name you assign. The raw illumination function (before smoothing)
% can be saved in a similar manner by prepending the word 'Raw' to the
% name you assigned.
% If you want to save the illumination image to use it in a later
% analysis, it is very important to save the illumination image in
% '.mat' format or else the quality of the illumination function
% values will be degraded.
%
% See also CORRECTILLUMDIVIDEALLMEANRETRIEVEIMG,
% CORRECTILLUMDIVIDEALLMEAN,
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

%textVAR02 = What do you want to call the illumination correction function?
%defaultVAR02 = IllumBlue
IlluminationImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = (Optional) What do you want to call the raw image of average minimums prior to smoothing? (This is an image produced during the calculations - it is typically not needed for downstream modules)
%defaultVAR03 = AverageMinimumsBlue
AverageMinimumsImageName = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = Block size. This should be set large enough that every square block of pixels is likely to contain some background.
%defaultVAR04 = 60
BlockSize = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,4}));

%textVAR05 = Enter E to calculate an illumination function for each image individually (in which case, choose P in the next box) or A to calculate an illumination function based on all the specified images to be corrected. Note that applying illumination correction on each image individually may make intensity measures not directly comparable across different images. Using illumination correction based on all images makes the assumption that the illumination anomalies are consistent across all the images in the set.
%defaultVAR05 = E
EachOrAll = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = Are the images you want to use to calculate the illumination correction function to be loaded straight from a Load Images module (L), or are they being produced by the pipeline (P)? If you choose L, the module will calculate the illumination correction function the first time through the pipeline by loading every image of the type specified in the Load Images module. It is then acceptable to use the resulting image later in the pipeline. If you choose P, the module will allow the pipeline to cycle through all of the image sets.  With this option, the module does not need to follow a Load Images module; it is acceptable to make the single, averaged projection from images resulting from other image processing steps in the pipeline. However, the resulting projection image will not be available until the last image set has been processed, so it cannot be used in subsequent modules.
%defaultVAR06 = L
SourceIsLoadedOrPipeline = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = Smoothing method: Enter the width of the artifacts (choose an even number) that are to be smoothed out by median filtering, or type P to fit a low order polynomial instead. For no smoothing, enter N. Note that smoothing is a time-consuming process.
%defaultVAR07 = N
SmoothingMethod = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% If the illumination correction function was to be calculated using
%%% all of the incoming images from a LoadImages module, it will already have been calculated
%%% the first time through the image set. No further calculations are
%%% necessary.
if (strcmpi(EachOrAll,'A') == 1 && strcmpi(SourceIsLoadedOrPipeline,'L') == 1) && handles.Current.SetBeingAnalyzed ~= 1
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
            %%% Calculates a coarse estimate of the background illumination by
            %%% determining the minimum of each block in the image.  If the minimum is
            %%% zero, it is recorded as .0001 to prevent divide by zero errors later.
            [LoadedImage, handles] = CPimread(fullfile(Pathname,char(FileList(1))),handles);
            SumMiniIlluminationImage = blkproc(padarray(LoadedImage,[RowsToAdd ColumnsToAdd],'replicate','post'),[BestBlockSize(1) BestBlockSize(2)],'max([min(x(:)); .0001])');
            for i=2:length(FileList)
                [LoadedImage, handles] = CPimread(fullfile(Pathname,char(FileList(i))),handles);
                SumMiniIlluminationImage = SumMiniIlluminationImage + blkproc(padarray(LoadedImage,[RowsToAdd ColumnsToAdd],'replicate','post'),[BestBlockSize(1) BestBlockSize(2)],'max([min(x(:)); .0001])');
            end
            MiniIlluminationImage = SumMiniIlluminationImage / length(FileList);
%%% The coarse estimate is then expanded in size so that it is the same
%%% size as the original image. Bilinear interpolation is used to ensure the
%%% values do not dip below zero.
            [LoadedImage, handles] = CPimread(fullfile(Pathname,char(FileList(1))),handles);
            IlluminationImage = imresize(MiniIlluminationImage, size(LoadedImage), 'bilinear');
            ReadyFlag = 'Ready';
        elseif strncmpi(SourceIsLoadedOrPipeline, 'P',1) == 1
            %%% In Pipeline (cycling) mode, each time through the image sets,
            %%% the minimums from the image are added to the existing cumulative image.
            [BestBlockSize, RowsToAdd, ColumnsToAdd] = CalculateBlockSize(m,n,BlockSize);
            if handles.Current.SetBeingAnalyzed == 1
                %%% Creates the empty variable so it can be retrieved later
                %%% without causing an error on the first image set.
                handles.Pipeline.(IlluminationImageName) = zeros(size(blkproc(padarray(OrigImage,[RowsToAdd ColumnsToAdd],'replicate','post'),[BestBlockSize(1) BestBlockSize(2)],'max([min(x(:)); .0001])')));
            end
            %%% Retrieves the existing illumination image, as accumulated so
            %%% far.
            SumMiniIlluminationImage = handles.Pipeline.(IlluminationImageName);
            %%% Adds the current image to it.
            SumMiniIlluminationImage = SumMiniIlluminationImage + blkproc(padarray(OrigImage,[RowsToAdd ColumnsToAdd],'replicate','post'),[BestBlockSize(1) BestBlockSize(2)],'max([min(x(:)); .0001])');
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
elseif strcmpi(EachOrAll,'E') == 1
    [BestBlockSize, RowsToAdd, ColumnsToAdd] = CalculateBlockSize(m,n,BlockSize);
    %%% Calculates a coarse estimate of the background illumination by
    %%% determining the minimum of each block in the image.  If the minimum is
    %%% zero, it is recorded as .0001 to prevent divide by zero errors later.
    
    %%% Not sure why this line differed from the one above for 'A'
    %%% mode, so I changed it to use the padarray version.
    % MiniIlluminationImage = blkproc(OrigImage,[BlockSize BlockSize],'max([min(x(:)); .0001])');
    MiniIlluminationImage = blkproc(padarray(OrigImage,[RowsToAdd ColumnsToAdd],'replicate','post'),[BestBlockSize(1) BestBlockSize(2)],'max([min(x(:)); .0001])');
    drawnow
%%% The coarse estimate is then expanded in size so that it is the same
%%% size as the original image. Bilinear interpolation is used to ensure the
%%% values do not dip below zero.
    IlluminationImage = imresize(MiniIlluminationImage, size(OrigImage), 'bilinear');
    ReadyFlag = 'Ready';
else error('Image processing was canceled because you must enter E or A in answer to the question "Enter E to calculate an illumination function for each image individually or A to calculate an illumination function based on all the specified images to be corrected."')
end

if strcmpi(SmoothingMethod,'N') ~= 1
    %%% Smooths the Illumination image, if requested, but saves a raw copy
    %%% first.
    AverageMinimumsImage = IlluminationImage;
    IlluminationImage = CPsmooth(IlluminationImage,SmoothingMethod);
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
    %%% Activates the appropriate figure window.
    CPfigure(ThisModuleFigureNumber);
    %%% A subplot of the figure window is set to display the original
    %%% image, some intermediate images, and the final corrected image.
    subplot(2,2,1); imagesc(OrigImage);
    title(['Input Image, Image Set # ',num2str(handles.Current.SetBeingAnalyzed)]);
    if exist('IlluminationImage','var') == 1
        subplot(2,2,4); imagesc(IlluminationImage); colormap(gray)
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

%%% If running in non-cycling mode (straight from the hard drive using
%%% a LoadImages module), the illumination image and its flag need only
%%% be saved to the handles structure after the first image set is
%%% processed. If running in cycling mode (Pipeline mode), the
%%% illumination image and its flag are saved to the handles structure
%%% after every image set is processed.
if strncmpi(SourceIsLoadedOrPipeline, 'P',1) == 1 | (strncmpi(SourceIsLoadedOrPipeline, 'L',1) == 1 && handles.Current.SetBeingAnalyzed == 1)
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