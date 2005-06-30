function handles = LoadImagesOrder(handles)

% Help for the Load Images Order module:
% Category: File Handling
%
% Tells CellProfiler where to retrieve images and gives each image a
% meaningful name for the other modules to access.
%
% If more than four images per set must be loaded, more than one Load
% Images Order module can be run sequentially. Running more than one
% of these modules also allows images to be retrieved from different
% folders.  If you want to load all images in a directory, the number
% of images per set can be set to 1.
%
% Load Images Order is useful when images are present in a repeating
% order, like DAPI, FITC, Red, DAPI, FITC, Red, and so on, where
% images are selected based on how many images are in each set and
% what position within each set a particular color is located (e.g.
% three images per set, DAPI is always first).  By contrast, Load
% Images Text is used to load images that have a particular piece of
% text in the name.
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
% See also LOADIMAGESTEXT.

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

%textVAR01 = The images to be loaded are located in what position in each set? (1,2,3,...)
%defaultVAR01 = 1
NumberInSet1 = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = What do you want to call these images?
%infotypeVAR02 = imagegroup indep
%defaultVAR02 = OrigBlue
ImageName1 = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = The images to be loaded are located in what position in each set? (1,2,3,...)
%defaultVAR03 = 0
NumberInSet2 = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = What do you want to call these images?
%infotypeVAR04 = imagegroup indep
%defaultVAR04 = OrigGreen
ImageName2 = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = The images to be loaded are located in what position in each set? (1,2,3,...)
%defaultVAR05 = 0
NumberInSet3 = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = What do you want to call these images?
%infotypeVAR06 = imagegroup indep
%defaultVAR06 = OrigRed
ImageName3 = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = The images to be loaded are located in what position in each set? (1,2,3,...)
%defaultVAR07 = 0
NumberInSet4 = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = What do you want to call these images?
%infotypeVAR08 = imagegroup indep
%defaultVAR08 = OrigOther1
ImageName4 = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%textVAR09 = How many images are there in each set (i.e. each field of view)?
%defaultVAR09 = 3
ImagesPerSet = char(handles.Settings.VariableValues{CurrentModuleNum,9});

%textVAR10 = Type the file format of the images
%defaultVAR10 = tif
FileFormat = char(handles.Settings.VariableValues{CurrentModuleNum,10});

%textVAR11 = Analyze all subdirectories within the selected directory?
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
ImagesPerSet = str2double(ImagesPerSet);
%%% If the user left boxes blank, sets the values to 0.
if isempty(NumberInSet1) == 1
    NumberInSet1 = '0';
end
if isempty(NumberInSet2) == 1
    NumberInSet2 = '0';
end
if isempty(NumberInSet3) == 1
    NumberInSet3 = '0';
end
if isempty(NumberInSet4) == 1
    NumberInSet4 = '0';
end
%%% Stores the text the user entered into cell arrays.
NumberInSet{1} = str2double(NumberInSet1);
NumberInSet{2} = str2double(NumberInSet2);
NumberInSet{3} = str2double(NumberInSet3);
NumberInSet{4} = str2double(NumberInSet4);
%%% Checks whether the position in set exceeds the number per set.
Max12 = max(NumberInSet{1}, NumberInSet{2});
Max34 = max(NumberInSet{3}, NumberInSet{4});
Max1234 = max(Max12, Max34);
if ImagesPerSet < Max1234
    error(['Image processing was canceled during the Load Images Order module because the position of one of the image types within each image set exceeds the number of images per set that you entered (', num2str(ImagesPerSet), ').'])
end
ImageName{1} = ImageName1;
ImageName{2} = ImageName2;
ImageName{3} = ImageName3;
ImageName{4} = ImageName4;

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
    %%% Checks whether the file format the user entered is readable by Matlab.
    IsFormat = imformats(FileFormat);
    if isempty(IsFormat) == 1
        %%% Checks if the image is a DIB image file.
        if (strcmpi(FileFormat,'DIB') == 0)&(strcmpi(FileFormat,'mat') == 0)
            error(['The image file type "', FileFormat , '" entered in the Load Images Order module is not recognized by Matlab. Or, you may have entered a period in the box. For a list of recognizable image file formats, type "imformats" (no quotes) at the command line in Matlab.'])
        end
    end
    if strcmp(Pathname, '.') == 1
        Pathname = handles.Current.DefaultImageDirectory;
    end
    SpecifiedPathname = Pathname;
    %%% For all 4 image slots, extracts the file names.
    for n = 1:4
        %%% Checks whether the two variables required have been entered by
        %%% the user.
        if NumberInSet{n} ~= 0 && isempty(ImageName{n}) == 0
            %%% If a directory was typed in, retrieves the filenames
            %%% from the chosen directory.
            if exist(SpecifiedPathname) ~= 7
                error(['Image processing was canceled because the directory "',SpecifiedPathname,'" does not exist. Be sure that no spaces or unusual characters exist in your typed entry and that the pathname of the directory begins with /.'])
            end
            FileNames = CPretrieveMediaFileNames(SpecifiedPathname,'',AnalyzeSubDir,'Regular','Image');
            %%% Checks whether any files have been specified.
            if isempty(FileNames) == 1
                error(['Image processing was canceled because there are no image files of type "', ImageName{n}, '" in the chosen directory (or subdirectories, if you requested them to be analyzed as well), according to the Load Images Order module.'])
            end
            %%% Determines the number of image sets to be analyzed.
            NumberOfImageSets = fix(length(FileNames)/ImagesPerSet);
            if length(FileNames) < ImagesPerSet
                error(['Image processing was canceled because there are fewer files of type "', ImageName{n}, '" in the chosen directory (or subdirectories, if you requested them to be analyzed as well), than the number of files per set, according to the Load Images Order module.'])
            end
            handles.Current.NumberOfImageSets = NumberOfImageSets;
            %%% Loops through the names in the FileNames listing,
            %%% creating a new list of files.
            for i = 1:NumberOfImageSets
                Number = (i - 1) .* ImagesPerSet + NumberInSet{n};
                FileList(i) = FileNames(Number);
            end
            %%% Saves the File Lists and Path Names to the handles structure.
            fieldname = ['FileList', ImageName{n}];
            handles.Pipeline.(fieldname) = FileList;
            fieldname = ['Pathname', ImageName{n}];
            handles.Pipeline.(fieldname) = SpecifiedPathname;
            clear FileList % Prevents confusion when loading this value later, for each image set.
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% LOADING IMAGES EACH TIME %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for n = 1:4
    %%% This try/catch will catch any problems in the load images module.
    try
        if NumberInSet{n} ~= 0 && isempty(ImageName{n}) == 0
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
            %%% Also saved to the handles.Measurements.Image structure for reference in output files.
            handles.Measurements.Image.(fieldname)(SetBeingAnalyzed) = CurrentFileName;
            %%% Saves the loaded image to the handles structure.  The field is named
            %%% appropriately based on the user's input, and put into the Pipeline
            %%% substructure so it will be deleted at the end of the analysis batch.
            handles.Pipeline.(ImageName{n}) = LoadedImage;
        end
    catch ErrorMessage = lasterr;
        ErrorNumber(1) = {'first'};
        ErrorNumber(2) = {'second'};
        ErrorNumber(3) = {'third'};
        ErrorNumber(4) = {'fourth'};
        error(['An error occurred when trying to load the ', ErrorNumber{n}, ' set of images using the Load Images Order module. Please check the settings. A common problem is that there are non-image files in the directory you are trying to analyze. Matlab says the problem is: ', ErrorMessage])
    end % Goes with: catch
end

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
