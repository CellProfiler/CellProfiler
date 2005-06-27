function handles = LoadImagesText(handles)

% Help for the Load Images Text module:
% Category: File Handling
%
% Tells CellProfiler where to retrieve images and gives each image a
% meaningful name for the other modules to access.
%
% If more than four images per set must be loaded, more than one Load
% Images Text module can be run sequentially. Running more than one of
% these modules also allows images to be retrieved from different
% folders. If you want to load all images in a directory, you can
% enter the file extension as the text for which to search.
%
% This module is different from the Load Images Order module because
% Load Images Text can be used to load images that are not in a
% defined order.  That is, Load Images Order is useful when images are
% present in a repeating order, like DAPI, FITC, Red, DAPI, FITC, Red,
% and so on, where images are selected based on how many images are in
% each set and what position within each set a particular color is
% located (e.g. three images per set, DAPI is always first).  Load
% Images Text is used instead to load images that have a particular
% piece of text in the name.
%
% You have the option of matching text exactly, or using regular
% expressions to match text. For example, typing image[12]dapi in the box
% asking for text in common and typing R in the Exact/Regular expression
% box will select any file containing the digit 1 or 2 immediately in
% between the text 'image' and 'dapi'.
%
% You may have subfolders within the folder that is being searched, but the
% names of the folders themselves must not contain the text you are
% searching for or an error will result.
%
% SAVING IMAGES: The images loaded by this module can be easily saved
% using the Save Images module, using the name you assign (e.g.
% OrigBlue).  In the Save Images module, the images can be saved in a
% different format, allowing this module to function as a file format
% converter.
%
% See also LOADIMAGESORDER.

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

%textVAR01 = Type the text that this set of images has in common
%defaultVAR01 = DAPI
TextToFind{1} = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%infotypeVAR02 = imagegroup indep
%textVAR02 = What do you want to call these images?
%defaultVAR02 = OrigBlue
ImageName{1} = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Type the text that this set of images has in common
%defaultVAR03 = /
TextToFind{2} = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%infotypeVAR04 = imagegroup indep
%textVAR04 = What do you want to call these images?
%defaultVAR04 = /
ImageName{2} = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = Type the text that this set of images has in common
%defaultVAR05 = /
TextToFind{3} = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%infotypeVAR06 = imagegroup indep
%textVAR06 = What do you want to call these images?
%defaultVAR06 = /
ImageName{3} = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = Type the text that this set of images has in common
%defaultVAR07 = /
TextToFind{4} = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%infotypeVAR08 = imagegroup indep
%textVAR08 = What do you want to call these images?
%defaultVAR08 = /
ImageName{4} = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%textVAR09 = If an image slot above is not being used, type a slash  /  in the box. Do you want to match the text exactly or use regular expressions?
%choiceVAR09 = Exact
%choiceVAR09 = Regular expressions
ExactOrRegExp = char(handles.Settings.VariableValues{CurrentModuleNum,9});
ExactOrRegExp = ExactOrRegExp(1);
%inputtypeVAR09 = popupmenu

%textVAR10 = Type the file format of the images
%choiceVAR10 = tif
%choiceVAR10 = bmp
%choiceVAR10 = gif
%choiceVAR10 = jpg
%choiceVAR10 = mat
%choiceVAR10 = DIB
FileFormat = char(handles.Settings.VariableValues{CurrentModuleNum,10});
%inputtypeVAR10 = popupmenu

%textVAR11 = Analyze all subdirectories within the selected directory (Y or N)?
%choiceVAR11 = No
%choiceVAR11 = Yes
AnalyzeSubDir = char(handles.Settings.VariableValues{CurrentModuleNum,11});
%inputtypeVAR11 = popupmenu

%textVAR12 = Enter the path name to the folder where the images to be loaded are located. Leave a period (.) to retrieve images from the default image directory #LongBox#
%defaultVAR12 = .
Pathname = char(handles.Settings.VariableValues{CurrentModuleNum,12});

%%%VariableRevisionNumber = 3

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines which image set is being analyzed.
SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;

%%% Remove slashes '/' from the input
tmp1 = {};
tmp2 = {};
for n = 1:4
    if ~strcmp(TextToFind{n}, '/') && ~strcmp(ImageName{n}, '/')
        tmp1{end+1} = TextToFind{n};
        tmp2{end+1} = ImageName{n};
    end
end
TextToFind = tmp1;
ImageName = tmp2;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% FIRST IMAGE SET FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

%%% Extracting the list of files to be analyzed occurs only the first time
%%% through this module.
if SetBeingAnalyzed == 1

    %%% Get the pathname and check that it exists
    if strcmp(Pathname, '.')
        Pathname = handles.Current.DefaultImageDirectory;
    end
    SpecifiedPathname = Pathname;
    if ~exist(SpecifiedPathname,'dir')
        error(['Image processing was canceled because the directory "',SpecifiedPathname,'" does not exist. Be sure that no spaces or unusual characters exist in your typed entry and that the pathname of the directory begins with /.'])
    end

    %%% Extract the file names
    for n = 1:length(ImageName)
        FileList = CPretrieveMediaFileNames(SpecifiedPathname,char(TextToFind(n)),AnalyzeSubDir(1), ExactOrRegExp,'Image');
        %%% Checks whether any files are left.
        if isempty(FileList)
            error(['Image processing was canceled because there are no image files with the text "', TextToFind{n}, '" in the chosen directory (or subdirectories, if you requested them to be analyzed as well), according to the LoadImagesText module.'])
        end
        %%% Saves the File Lists and Path Names to the handles structure.
        fieldname = ['FileList', ImageName{n}];
        handles.Pipeline.(fieldname) = FileList;
        fieldname = ['Pathname', ImageName{n}];
        handles.Pipeline.(fieldname) = SpecifiedPathname;

        NumberOfFiles{n} = num2str(length(FileList)); %#ok We want to ignore MLint error checking for this line.
        clear FileList % Prevents confusion when loading this value later, for each image set.
    end

    %%% Determines which slots are empty.  None should be zero, because there is
    %%% an error check for that when looping through n = 1:5.
    for g = 1: length(NumberOfFiles)
        LogicalSlotsToBeDeleted(g) =  isempty(NumberOfFiles{g});
    end
    %%% Removes the empty slots from both the Number of Files array and the
    %%% Image Name array.
    NumberOfFiles = NumberOfFiles(~LogicalSlotsToBeDeleted);
    ImageName2 = ImageName(~LogicalSlotsToBeDeleted);
    %%% Determines how many unique numbers of files there are.  If all the image
    %%% types have loaded the same number of images, there should only be one
    %%% unique number, which is the number of image sets.
    UniqueNumbers = unique(NumberOfFiles);
    %%% If NumberOfFiles is not all the same number at each position, generate an error.
    if length(UniqueNumbers) ~= 1
        CharImageName = char(ImageName2);
        CharNumberOfFiles = char(NumberOfFiles);
        Number = length(CharNumberOfFiles);
        for f = 1:Number
            SpacesArray(f,:) = ':     ';
        end
        PreErrorText = cat(2, CharImageName, SpacesArray);
        ErrorText = cat(2, PreErrorText, CharNumberOfFiles);
        CPmsgbox(ErrorText)
        error('In the Load Images Text module, the number of images identified for each image type is not equal.  In the window under this box you will see how many images have been found for each image type.')
    end
    NumberOfImageSets = str2double(UniqueNumbers{1});
    %%% Checks whether another load images module has already recorded a
    %%% number of image sets.  If it has, it will not be set at the default
    %%% of 1.  Then, it checks whether the number already stored as the
    %%% number of image sets is equal to the number of image sets that this
    %%% module has found.  If not, an error message is generated. Note:
    %%% this will not catch the case where the number of image sets
    %%% detected by this module is more than 1 and another module has
    %%% detected only one image set, since there is no way to tell whether
    %%% the 1 stored in handles.Current.NumberOfImageSets is the default value or a
    %%% value determined by another image-loading module.
    if handles.Current.NumberOfImageSets ~= 1;
        if handles.Current.NumberOfImageSets ~= NumberOfImageSets
            error(['The number of image sets loaded by the Load Images Text module (', num2str(NumberOfImageSets),') does not equal the number of image sets loaded by another image-loading module (', num2str(handles.Current.NumberOfImageSets), '). Please check the settings.'])
        end
    end
    handles.Current.NumberOfImageSets = NumberOfImageSets;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% LOADING IMAGES EACH TIME %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for n = 1:length(ImageName)
    %%% This try/catch will catch any problems in the load images module.
    try
        %%% The following runs every time through this module (i.e. for every image set).
        %%% Determines which image to analyze.
        fieldname = ['FileList', ImageName{n}];
        FileList = handles.Pipeline.(fieldname);
        %%% Determines the file name of the image you want to analyze.
        CurrentFileName = FileList(SetBeingAnalyzed);
        %%% Determines the directory to switch to.
        fieldname = ['Pathname', ImageName{n}];
        Pathname = handles.Pipeline.(fieldname);
        [LoadedImage, handles] = CPimread(fullfile(Pathname,CurrentFileName{1}), handles);

        %%% Saves the original image file name to the handles
        %%% structure.  The field is named appropriately based on
        %%% the user's input, in the Pipeline substructure so that
        %%% this field will be deleted at the end of the analysis
        %%% batch.
        fieldname = ['Filename', ImageName{n}];
        handles.Pipeline.(fieldname)(SetBeingAnalyzed) = CurrentFileName;
        handles.Pipeline.(ImageName{n}) = LoadedImage;
    catch ErrorMessage = lasterr;
        ErrorNumber = {'first','second','third','fourth'};
        error(['An error occurred when trying to load the ', ErrorNumber{n}, ' set of images using the Load Images Text module. Please check the settings. A common problem is that there are non-image files in the directory you are trying to analyze, or that the image file is not in the format you specified: ', FileFormat, '. Matlab says the problem is: ', ErrorMessage])
    end % Goes with: catch

    % Create a cell array with the filenames
    FileNames(n) = CurrentFileName(1);

end

%%% -- Save to the handles.Measurements structure for reference in output files --------------- %%%
%%% NOTE: The structure for filenames and pathnames will be a cell array of cell arrays

%%% First, fix feature names and the pathname
PathNames = cell(1,length(ImageName));
FileNamesFeatures = cell(1,length(ImageName));
PathNamesFeatures = cell(1,length(ImageName));
for n = 1:length(ImageName)
    PathNames{n} = Pathname;
    FileNamesFeatures{n} = ['Filename ', ImageName{n}];
    PathNamesFeatures{n} = ['Path ', ImageName{n}];
end

%%% Since there may be several load modules in the pipeline which all write to the
%%% handles.Measurements.Image.FileName field, we have store filenames in an "appending" style.
%%% Here we check if any of the modules above the current module in the pipline has written to
%%% handles.Measurements.Image.Filenames. Then we should append the current filenames and path
%%% names to the already written ones.
if  isfield(handles,'Measurements') && isfield(handles.Measurements,'Image') &&...
        length(handles.Measurements.Image.FileNames) == SetBeingAnalyzed
    % Get existing file/path names. Returns a cell array of names
    ExistingFileNamesFeatures = handles.Measurements.Image.FileNamesFeatures;
    ExistingFileNames         = handles.Measurements.Image.FileNames{SetBeingAnalyzed};
    ExistingPathNamesFeatures = handles.Measurements.Image.PathNamesFeatures;
    ExistingPathNames         = handles.Measurements.Image.PathNames{SetBeingAnalyzed};

    % Append current file names to existing file names
    FileNamesFeatures = cat(2,ExistingFileNamesFeatures,FileNamesFeatures);
    FileNames         = cat(2,ExistingFileNames,FileNames);
    PathNamesFeatures = cat(2,ExistingPathNamesFeatures,PathNamesFeatures);
    PathNames         = cat(2,ExistingPathNames,PathNames);
end

%%% Write to the handles.Measurements.Image structure
handles.Measurements.Image.FileNamesFeatures                   = FileNamesFeatures;
handles.Measurements.Image.FileNames(SetBeingAnalyzed)         = {FileNames};
handles.Measurements.Image.PathNamesFeatures                   = PathNamesFeatures;
handles.Measurements.Image.PathNames(SetBeingAnalyzed)         = {PathNames};
%%% ------------------------------------------------------------------------------------------------ %%%

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

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%

if SetBeingAnalyzed == handles.Current.StartingImageSet
    %%% The figure window display is unnecessary for this module, so the figure
    %%% window is closed the first time through the module.
    %%% Determines the figure number.
    fieldname = ['FigureNumberForModule',CurrentModule];
    ThisModuleFigureNumber = handles.Current.(fieldname);
    %%% Closes the window if it is open.
    if any(findobj == ThisModuleFigureNumber) == 1;
        close(ThisModuleFigureNumber)
    end
    drawnow
end

% PROGRAM NOTES THAT ARE UNNECESSARY FOR THIS MODULE:
% PROGRAMMING NOTE
% DISPLAYING RESULTS:
% Some calculations produce images that are used only for display or
% for saving to the hard drive, and are not used by downstream
% modules. To speed processing, these calculations are omitted if the
% figure window is closed and the user does not want to save the
% images.

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