function handles = LoadMoviesText(handles)

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
TextToFind1 = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = What do you want to call these images?
%defaultVAR02 = OrigBlue
MovieName1 = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Type the text that this set of images has in common
%defaultVAR03 = /
TextToFind2 = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = What do you want to call these images?
%defaultVAR04 = /
MovieName2 = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = Type the text that this set of images has in common
%defaultVAR05 = /
TextToFind3 = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = What do you want to call these images?
%defaultVAR06 = /
MovieName3 = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = Type the text that this set of images has in common
%defaultVAR07 = /
TextToFind4 = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = What do you want to call these images?
%defaultVAR08 = /
MovieName4 = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%textVAR09 = If an image slot above is not being used, type a slash  /  in the box. 

%textVAR10 = Do you want to match the text exactly (E), or use regular expressions (R)?
%defaultVAR10 = E
ExactOrRegExp = char(handles.Settings.VariableValues{CurrentModuleNum,10});

%textVAR11 = Analyze all subdirectories within the selected directory (Y or N)?
%defaultVAR11 = N
AnalyzeSubDir = char(handles.Settings.VariableValues{CurrentModuleNum,11});

%textVAR12 = Enter the path name to the folder where the images to be loaded are located. Leave a period (.) to retrieve images from the default image directory #LongBox#
%defaultVAR12 = .
Pathname = char(handles.Settings.VariableValues{CurrentModuleNum,12});

%textVAR13 = CellProfiler can currently read only uncompressed avi files. For more details, see the help for this module.

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines which image set is being analyzed.
SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;
%%% Stores the text the user entered into cell arrays.
TextToFind{1} = TextToFind1;
TextToFind{2} = TextToFind2;
TextToFind{3} = TextToFind3;
TextToFind{4} = TextToFind4;
MovieName{1} = MovieName1;
MovieName{2} = MovieName2;
MovieName{3} = MovieName3;
MovieName{4} = MovieName4;

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
    %%% Makes sure this entry is appropriate.
    if strncmpi(ExactOrRegExp,'E',1) == 1 | strncmpi(ExactOrRegExp,'R',1) == 1
    else error('You must enter E or R in the Load Images Text module to look for exact text matches or regular expression text matches.')
    end
    %%% If the user did not enter any data in the first slot (they put
    %%% a slash in either box), no images are retrieved.
    if strcmp(TextToFind{1}, '/') == 1 || strcmp(MovieName{1}, '/') == 1
        error('Image processing was canceled because the first movie slot in the Load Movies Text module was left blank.')
    end
    if strcmp(Pathname, '.') == 1
        Pathname = handles.Current.DefaultImageDirectory;
    end
    SpecifiedPathname = Pathname;
    %%% For all 4 image slots, extracts the file names.
    for n = 1:4
        %%% Checks whether the two variables required have been entered by
        %%% the user.
        if strcmp(TextToFind{n}, '/') == 0 && strcmp(MovieName{n}, '/') == 0
            %%% If a directory was typed in, retrieves the filenames
            %%% from the chosen directory.
            if exist(SpecifiedPathname) ~= 7
                error(['Image processing was canceled because the directory "',SpecifiedPathname,'" does not exist. Be sure that no spaces or unusual characters exist in your typed entry and that the pathname of the directory begins with /.'])
            end
            FileList{n} = RetrieveImageFileNames(SpecifiedPathname,char(TextToFind(n)),AnalyzeSubDir, ExactOrRegExp);
            %%% Checks whether any files are left.
            if isempty(FileList{n})
                error(['Image processing was canceled because there are no movie files with the text "', TextToFind{n}, '" in the chosen directory (or subdirectories, if you requested them to be analyzed as well), according to the LoadMoviesText module.'])
            end
            StartingPositionForThisMovie = 0;
            for MovieFileNumber = 1:length(FileList{:})
                CurrentMovieFileName = fullfile(SpecifiedPathname, char(FileList{n}(1, MovieFileNumber)));
                MovieAttributes = aviinfo(char(CurrentMovieFileName));
                for FrameNumber = 1:MovieAttributes.NumFrames
                    %%% Puts the file name into the FrameByFrameFileList in the first row.
                    FrameByFrameFileList{n}(1,StartingPositionForThisMovie + FrameNumber) = FileList{n}(1, MovieFileNumber);
                    %%% Puts the frame number into the FrameByFrameFileList in the second row.
                    FrameByFrameFileList{n}(2,StartingPositionForThisMovie + FrameNumber) = {FrameNumber};
                end
                StartingPositionForThisMovie = StartingPositionForThisMovie + MovieAttributes.NumFrames;
            end
            %%% Saves the File Lists and Path Names to the handles structure.
            fieldname = ['FileList', MovieName{n}];
            handles.Pipeline.(fieldname) = FrameByFrameFileList{n};
            fieldname = ['Pathname', MovieName{n}];
            handles.Pipeline.(fieldname) = SpecifiedPathname;
            %% for reference in saved files
            handles.Measurements.(fieldname) = SpecifiedPathname;
            NumberOfFiles{n} = num2str(length(FrameByFrameFileList{n})); %#ok We want to ignore MLint error checking for this line.
        end
    end
    %%% Determines which slots are empty.  None should be zero, because there is
    %%% an error check for that when looping through n = 1:5.
    for g = 1: length(NumberOfFiles)
        LogicalSlotsToBeDeleted(g) =  isempty(NumberOfFiles{g});
    end
    %%% Removes the empty slots from both the Number of Files array and the
    %%% Image Name array.
    NumberOfFiles = NumberOfFiles(~LogicalSlotsToBeDeleted);
    MovieName2 = MovieName(~LogicalSlotsToBeDeleted);
    %%% Determines how many unique numbers of files there are.  If all the image
    %%% types have loaded the same number of images, there should only be one
    %%% unique number, which is the number of image sets.
    UniqueNumbers = unique(NumberOfFiles);
    %%% If NumberOfFiles is not all the same number at each position, generate an error.
    if length(UniqueNumbers) ~= 1
        CharMovieName = char(MovieName2);
        CharNumberOfFiles = char(NumberOfFiles);
        Number = length(CharNumberOfFiles);
        for f = 1:Number
            SpacesArray(f,:) = ':     ';
        end
        PreErrorText = cat(2, CharMovieName, SpacesArray);
        ErrorText = cat(2, PreErrorText, CharNumberOfFiles);
        msgbox(ErrorText)
        error('In the Load Movies Text module, the number of movies identified for each movie type is not equal.  In the window under this box you will see how many movie have been found for each movie type.')
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
end % Goes with: if SetBeingAnalyzed == 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% LOADING IMAGES EACH TIME %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for n = 1:4
    %%% This try/catch will catch any problems in the load images module.
    try
        if strcmp(TextToFind{n}, '/') == 0 && strcmp(MovieName{n}, '/') == 0
            %%% The following runs every time through this module (i.e. for
            %%% every image set).
            %%% Determines which image to analyze.
            fieldname = ['FileList', MovieName{n}];
            FileList = handles.Pipeline.(fieldname);
            %%% Determines the file name of the image you want to analyze.
            CurrentFileName = FileList(:,SetBeingAnalyzed);
            %%% Determines the directory to switch to.
            fieldname = ['Pathname', MovieName{n}];
            Pathname = handles.Pipeline.(fieldname);
            LoadedRawImage = aviread(fullfile(Pathname, char(CurrentFileName(1))), cell2mat(CurrentFileName(2)));
            LoadedImage = im2double(LoadedRawImage.cdata);
            %%% Saves the original image file name to the handles
            %%% structure.  The field is named appropriately based on
            %%% the user's input, in the Pipeline substructure so that
            %%% this field will be deleted at the end of the analysis
            %%% batch.
            fieldname = ['Filename', MovieName{n}];
            [SubdirectoryPathName,BareFileName,ext,versn] = fileparts(char(CurrentFileName(1)));
            CurrentFileNameWithFrame = [BareFileName, '_', num2str(cell2mat(CurrentFileName(2))),ext];
            handles.Pipeline.(fieldname)(SetBeingAnalyzed) = {CurrentFileNameWithFrame};
            %%% Also saved to the handles.Measurements structure for reference in output files.
            handles.Measurements.(fieldname)(SetBeingAnalyzed) = {CurrentFileNameWithFrame};
            %%% Saves the loaded image to the handles structure.  The field is named
            %%% appropriately based on the user's input, and put into the Pipeline
            %%% substructure so it will be deleted at the end of the analysis batch.
            handles.Pipeline.(MovieName{n}) = LoadedImage;
        end
    catch ErrorMessage = lasterr;
        ErrorNumber(1) = {'first'};
        ErrorNumber(2) = {'second'};
        ErrorNumber(3) = {'third'};
        ErrorNumber(4) = {'fourth'};
        error(['An error occurred when trying to load the ', ErrorNumber{n}, ' set of images using the Load Images Text module. Please check the settings. A common problem is that there are non-image files in the directory you are trying to analyze, or that the image file is not in uncompressed avi format. Matlab says the problem is: ', ErrorMessage])
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

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%

if SetBeingAnalyzed == 1
    %%% The figure window display is unnecessary for this module, so the figure
    %%% window is closed the first time through the module.
    %%% Determines the figure number.
    fieldname = ['FigureNumberForModule',CurrentModule];
    ThisModuleFigureNumber = handles.Current.(fieldname);
    %%% Closes the window if it is open.
    if any(findobj == ThisModuleFigureNumber) == 1;
        close(ThisModuleFigureNumber)
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTION TO RETRIEVE FILE NAMES %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function FileNames = RetrieveImageFileNames(Pathname, TextToFind, recurse, ExactOrRegExp)
%%% Lists all the contents of that path into a structure which includes the
%%% name of each object as well as whether the object is a file or
%%% directory.
FilesAndDirsStructure = dir(Pathname);
%%% Puts the names of each object into a list.
FileAndDirNames = sortrows({FilesAndDirsStructure.name}');
%%% Puts the logical value of whether each object is a directory into a list.
LogicalIsDirectory = [FilesAndDirsStructure.isdir];
%%% Eliminates directories from the list of file names.
FileNamesNoDir = FileAndDirNames(~LogicalIsDirectory);
if isempty(FileNamesNoDir) == 1
    FileNames = [];
else
    %%% Makes a logical array that marks with a "1" all file names that start
    %%% with a period (hidden files):
    DiscardLogical1 = strncmp(FileNamesNoDir,'.',1);
    %%% Makes logical arrays that mark with a "1" all file names that have
    %%% particular suffixes (mat, m, m~, and frk). The dollar sign indicates
    %%% that the pattern must be at the end of the string in order to count as
    %%% matching.  The first line of each set finds the suffix and marks its
    %%% location in a cell array with the index of where that suffix begins;
    %%% the third line converts this cell array of numbers into a logical
    %%% array of 1's and 0's.   cellfun only works on arrays of class 'cell',
    %%% so there is a check to make sure the class is appropriate.  When there
    %%% are very few files in the directory (I think just one), the class is
    %%% not cell for some reason.
    DiscardsByExtension = regexpi(FileNamesNoDir, '\.(m|mat|m~|frk~|xls|doc|rtf|txt|csv)$', 'once');
    if strcmp(class(DiscardsByExtension), 'cell')
        DiscardsByExtension = cellfun('prodofsize',DiscardsByExtension);
    else
        DiscardsByExtension = [];
    end
    %%% Combines all of the DiscardLogical arrays into one.
    DiscardLogical = DiscardLogical1 | DiscardsByExtension;
    %%% Eliminates filenames to be discarded.
    if isempty(DiscardLogical) == 1
        NotYetTextMatchedFileNames = FileNamesNoDir;
    else NotYetTextMatchedFileNames = FileNamesNoDir(~DiscardLogical);
    end

    %%% Loops through the names in the Directory listing, looking for the text
    %%% of interest.  Creates the array Match which contains the numbers of the
    %%% file names that match.
    FileNames = cell(0);
    Count = 1;
    for i=1:length(NotYetTextMatchedFileNames),
        if strncmpi(ExactOrRegExp,'E',1) == 1
            if findstr(char(NotYetTextMatchedFileNames(i)), TextToFind),
                FileNames{Count} = char(NotYetTextMatchedFileNames(i));
                Count = Count + 1;
            end
        elseif strncmpi(ExactOrRegExp,'R',1) == 1
            if regexp(char(NotYetTextMatchedFileNames(i)), TextToFind),
                FileNames{Count} = char(NotYetTextMatchedFileNames(i));
                Count = Count + 1;
            end
        else error('You must enter E or R in the Load Images Text module to look for exact text matches or regular expression text matches.')
        end
    end
end
if(strcmp(upper(recurse),'Y'))
    DirNamesNoFiles = FileAndDirNames(LogicalIsDirectory);
    DiscardLogical1Dir = strncmp(DirNamesNoFiles,'.',1);
    DirNames = DirNamesNoFiles(~DiscardLogical1Dir);
    if (length(DirNames) > 0)
        for i=1:length(DirNames),
            MoreFileNames = RetrieveImageFileNames(fullfile(Pathname, char(DirNames(i))), TextToFind, recurse, ExactOrRegExp);
            for j = 1:length(MoreFileNames)
                MoreFileNames{j} = fullfile(char(DirNames(i)), char(MoreFileNames(j)));
            end
            if isempty(FileNames) == 1
                FileNames = MoreFileNames;
            else
                FileNames(end+1:end+length(MoreFileNames)) = MoreFileNames(1:end);
            end
        end
    end
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