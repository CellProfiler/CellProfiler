function handles = LoadMoviesOrder(handles)

% Help for the Load Movies Order module:
% Category: File Handling
%
% Tells CellProfiler where to retrieve avi-formatted movies (avi
% movies must be in uncompressed avi format on UNIX and Mac
% platforms) or stk-format movies (stacks of tif images produced by
% MetaMorph or NIH ImageJ). Once the files are identified, this module
% extracts each frame of each movie as a separate image,
% and gives these images a meaningful name for the other modules to
% access.
%
% If more than four movies per set must be loaded, more than one Load
% Movies Order module can be run sequentially. Running more than one
% of these modules also allows movies to be retrieved from different
% folders.  If you want to load all movies in a directory, the number
% of movies per set can be set to 1.
%
% Load Movies Order is useful when movies are present in a repeating
% order, like DAPI, FITC, Red, DAPI, FITC, Red, and so on, where
% movies are selected based on how many movies are in each set and
% what position within each set a particular color is located (e.g.
% three movies per set, DAPI is always the first movie in each set).
% By contrast, Load Movies Text is used to load movies that have a
% particular piece of text in the name.
%
% You may have subfolders within the folder that is being searched,
% but the names of the folders themselves must not contain the text
% you are searching for or an error will result.
%
% The ability to read stk files is due to code by:
% Francois Nedelec, EMBL, Copyright 1999-2003.
%
% ------------------------------------------------------------------
% NOTE:  MATLAB only reads AVI files, and only UNCOMPRESSED AVI files
% on UNIX and MAC platforms.  As a result, you may need to use 3rd party
% software to uncompress AVI files and convert MOV files.  
% 
% WINDOWS...
% To convert movies to uncompressed avi format, you can use a free
% software product called RAD Video Tools, which is available from:
%   http://www.radgametools.com/down/Bink/RADTools.exe
% 
% To convert a compressed AVI file or a MOV file into an uncompressed AVI:
%   1. Open RAD Video Tools
%   2. Select the file you want to convert
%   3. Click the "Convert a file" button
%   4. On the next screen, type the desired output file name, and 
% click the "Convert" button.  Everything else can be left as default.
%   5. A window will pop up that asks you for the Video Compression to
% use.  Choose "Full Frames (Uncompressed)", and click OK.
%
% MAC OSX...
% The iMovie program which comes with Mac OSX can be used to convert
% movies to uncompressed avi format as follows: 
% 
% 1. File > New Project
% 2. File > Import (select the movie)
% 3. File > Share
% 	Choose the QuickTime tab
% 	Compress movie for Expert Settings, click Share
% 	Name the file, choose Export: Movie to Avi
% 	Click Options...
% 	Click Settings...
% 		Compression = None
% 		Depth = Millions of Colors (NOT "+")
% 		Quality = best
% 		Frames per second = doesn't matter. 
% 	OK, OK, Save
% 
% 4. To check/troubleshoot the conversion, you can use the following commands in Matlab:  
% >> MovieInfo = aviinfo('My Great Movie3.avi')
% 
% MovieInfo = 
%               Filename: 'My Great Movie3.avi'
%               FileSize: 481292920
%            FileModDate: '25-Mar-2005 09:59:56'
%              NumFrames: 422
%        FramesPerSecond: 20
%                  Width: 720
%                 Height: 528
%              ImageType: 'truecolor'
%       VideoCompression: 'none'
%                Quality: 4.2950e+07
%     NumColormapEntries: 0
% 
% The following error means that the Depth was improper (either you
% tried to save in grayscale or the wrong bit depth color):
% >> movie = aviread('My Great Movie2.avi');
% ??? Error using ==> aviread
% Bitmap data must be 8-bit Index images or 24-bit TrueColor images
% ------------------------------------------------------------------
%
% SAVING IMAGES: The frames of the movies loaded by this module can be
% easily saved using the Save Images module, using the name you assign
% (e.g. OrigBlue).  In the Save Images module, the images can be saved
% in a different format, allowing this module to function as a file
% format converter.
%
% See also LOADMOVIESTEXT.

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

%textVAR01 = The movies to be loaded are located in what position in each set? (1,2,3,...)
%defaultVAR01 = 1
NumberInSet1 = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = What do you want to call the images loaded from these movies?
%defaultVAR02 = OrigBlue
MovieName1 = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = The movies to be loaded are located in what position in each set? (1,2,3,...)
%defaultVAR03 = 0
NumberInSet2 = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = What do you want to call the images loaded from these movies?
%defaultVAR04 = OrigGreen
MovieName2 = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = The movies to be loaded are located in what position in each set? (1,2,3,...)
%defaultVAR05 = 0
NumberInSet3 = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = What do you want to call the images loaded from these movies?
%defaultVAR06 = OrigRed
MovieName3 = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = The movies to be loaded are located in what position in each set? (1,2,3,...)
%defaultVAR07 = 0
NumberInSet4 = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = What do you want to call the images loaded from these movies?
%defaultVAR08 = OrigOther1
MovieName4 = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%textVAR09 = If an image slot above is not being used, type a zero in the box.

%textVAR10 = Enter the movie file format (avi or stk)
%defaultVAR10 = avi
FileFormat = char(handles.Settings.VariableValues{CurrentModuleNum,10});

%textVAR11 = How many movies are there in each set (i.e. each field of view)?
%defaultVAR11 = 3
MoviesPerSet = char(handles.Settings.VariableValues{CurrentModuleNum,11});

%textVAR12 = Analyze all subdirectories within the selected directory (Y or N)?
%defaultVAR12 = N
AnalyzeSubDir = char(handles.Settings.VariableValues{CurrentModuleNum,12});

%textVAR13 = Enter the path name to the folder where the movies to be loaded are located. Leave a period (.) to retrieve movies from the default image directory #LongBox#
%defaultVAR13 = .
Pathname = char(handles.Settings.VariableValues{CurrentModuleNum,13});

%textVAR14 = CellProfiler can currently read only certain types of avi and stk movie files. For more details, see the help for this module.

%%%VariableRevisionNumber = 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines which image set is being analyzed.
SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;
MoviesPerSet = str2double(MoviesPerSet);
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
if MoviesPerSet < Max1234
    error(['Image processing was canceled during the Load Movies Order module because the position of one of the movie types within each movie set exceeds the number of movies per set that you entered (', num2str(MoviesPerSet), ').'])
end
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
    if strcmp(Pathname, '.') == 1
        Pathname = handles.Current.DefaultImageDirectory;
    end
    SpecifiedPathname = Pathname;
    %%% For all 4 movie slots, extracts the file names.
    for n = 1:4
        %%% Checks whether the two variables required have been entered by
        %%% the user.
        if NumberInSet{n} ~= 0 && isempty(MovieName{n}) == 0
            %%% If a directory was typed in, retrieves the filenames
            %%% from the chosen directory.
            if exist(SpecifiedPathname) ~= 7
                error(['Image processing was canceled because the directory "',SpecifiedPathname,'" does not exist. Be sure that no spaces or unusual characters exist in your typed entry and that the pathname of the directory begins with /.'])
            end
            FileNames = RetrieveImageFileNames(SpecifiedPathname,AnalyzeSubDir);
            %%% Checks whether any files have been specified.
            if isempty(FileNames) == 1
                error(['Image processing was canceled because there are no movie files of type "', MovieName{n}, '" in the chosen directory (or subdirectories, if you requested them to be analyzed as well), according to the Load Movies Order module.'])
            end
            %%% Determines the number of movie sets to be analyzed.
            NumberOfMovieSets = fix(length(FileNames)/MoviesPerSet);
            if length(FileNames) < MoviesPerSet
                error(['Image processing was canceled because there are fewer files of type "', MovieName{n}, '" in the chosen directory (or subdirectories, if you requested them to be analyzed as well), than the number of files per set, according to the Load Movies Order module.'])
            end
            handles.Current.NumberOfMovieSets = NumberOfMovieSets;
            %%% Loops through the names in the FileNames listing,
            %%% creating a new list of files.
            for i = 1:NumberOfMovieSets
                Number = (i - 1) .* MoviesPerSet + NumberInSet{n};
                FileList(i) = FileNames(Number);
            end
            StartingPositionForThisMovie = 0;
            for MovieFileNumber = 1:length(FileList)
                CurrentMovieFileName = char(FileList(MovieFileNumber));
                if strcmpi(FileFormat,'avi') == 1
                    try MovieAttributes = aviinfo(fullfile(SpecifiedPathname, CurrentMovieFileName));
                    catch error(['Image processing was canceled because the file ',fullfile(SpecifiedPathname, CurrentMovieFileName),' was not readable as an uncompressed avi file.'])
                    end
                    NumFrames = MovieAttributes.NumFrames;
                    for FrameNumber = 1:NumFrames
                        %%% Puts the file name into the FrameByFrameFileList in the first row.
                        FrameByFrameFileList{n}(1,StartingPositionForThisMovie + FrameNumber) = {CurrentMovieFileName};
                        %%% Puts the frame number into the FrameByFrameFileList in the second row.
                        FrameByFrameFileList{n}(2,StartingPositionForThisMovie + FrameNumber) = {FrameNumber};
                    end
                elseif strcmpi(FileFormat,'stk') == 1
                    try
                        %%% Reads metamorph or NIH ImageJ movie stacks of tiffs.
                        [S, NumFrames] = tiffread(fullfile(SpecifiedPathname, CurrentMovieFileName),1);
                        for FrameNumber = 1:NumFrames
                            %%% Puts the file name into the FrameByFrameFileList in the first row.
                            FrameByFrameFileList{n}(1,StartingPositionForThisMovie + FrameNumber) = {CurrentMovieFileName};
                            %%% Puts the frame number into the FrameByFrameFileList in the second row.
                            FrameByFrameFileList{n}(2,StartingPositionForThisMovie + FrameNumber) = {FrameNumber};
                        end
                    catch error(['Image processing was canceled because the file ',fullfile(SpecifiedPathname, CurrentMovieFileName),' was not readable as a stk file.'])
                    end
                else
                    error('CellProfiler can currently read only avi or stk movie files.')
                end
                StartingPositionForThisMovie = StartingPositionForThisMovie + NumFrames;
            end
            %%% Saves the File Lists and Path Names to the handles structure.
            fieldname = ['FileList', MovieName{n}];
            handles.Pipeline.(fieldname) = FrameByFrameFileList{n};
            fieldname = ['Pathname', MovieName{n}];
            handles.Pipeline.(fieldname) = SpecifiedPathname;
            %% for reference in saved files
            handles.Measurements.Image.(fieldname) = SpecifiedPathname;
            NumberOfFiles{n} = num2str(length(FrameByFrameFileList{n})); %#ok We want to ignore MLint error checking for this line.
            clear FileList % Prevents confusion when loading this value later, for each movie set.
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
    %%% Determines how many unique numbers of files there are.  If all
    %%% the movie types have loaded the same number of images, there
    %%% should only be one unique number, which is the number of image
    %%% sets.
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
        error('In the Load Movies Order module, the number of movies identified for each movie type is not equal.  In the window under this box you will see how many movie have been found for each movie type.')
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
            error(['The number of image sets loaded by the Load Movies Order module (', num2str(NumberOfImageSets),') does not equal the number of image sets loaded by another image-loading module (', num2str(handles.Current.NumberOfImageSets), '). Please check the settings.'])
        end
    end
    handles.Current.NumberOfImageSets = NumberOfImageSets;
end % Goes with: if SetBeingAnalyzed == 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% LOADING IMAGES EACH TIME %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for n = 1:4
    %%% This try/catch will catch any problems in the load movies module.
    try
        if NumberInSet{n} ~= 0 && isempty(MovieName{n}) == 0
            %%% Determines which movie to analyze.
            fieldname = ['FileList', MovieName{n}];
            FileList = handles.Pipeline.(fieldname);
            %%% Determines the file name of the movie you want to analyze.
            CurrentFileName = FileList(:,SetBeingAnalyzed);
            %%% Determines the directory to switch to.
            fieldname = ['Pathname', MovieName{n}];
            Pathname = handles.Pipeline.(fieldname);
            if strcmpi(FileFormat,'avi') == 1
                LoadedRawImage = aviread(fullfile(Pathname, char(CurrentFileName(1))), cell2mat(CurrentFileName(2)));
                LoadedImage = im2double(LoadedRawImage.cdata);
            elseif strcmpi(FileFormat,'stk') == 1
                LoadedRawImage = tiffread(fullfile(Pathname, char(CurrentFileName(1))), cell2mat(CurrentFileName(2)));
                LoadedImage = im2double(LoadedRawImage.data);
            end
            %%% Saves the original movie file name to the handles
            %%% structure.  The field is named appropriately based on
            %%% the user's input, in the Pipeline substructure so that
            %%% this field will be deleted at the end of the analysis
            %%% batch.
            fieldname = ['Filename', MovieName{n}];
            [SubdirectoryPathName,BareFileName,ext,versn] = fileparts(char(CurrentFileName(1)));
            CurrentFileNameWithFrame = [BareFileName, '_', num2str(cell2mat(CurrentFileName(2))),ext];
            handles.Pipeline.(fieldname)(SetBeingAnalyzed) = {CurrentFileNameWithFrame};
            %%% Also saved to the handles.Measurements structure for reference in output files.
            handles.Measurements.Image.(fieldname)(SetBeingAnalyzed) = {CurrentFileNameWithFrame};
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
        error(['An error occurred when trying to load the ', ErrorNumber{n}, ' set of movies using the Load Movies Order module. Please check the settings. A common problem is that there are non-movie files in the directory you are trying to analyze. Matlab says the problem is: ', ErrorMessage])
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTION TO RETRIEVE FILE NAMES %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function FileNames = RetrieveImageFileNames(Pathname,recurse)
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
        FileNames = FileNamesNoDir;
    else FileNames = FileNamesNoDir(~DiscardLogical);
    end
end
if(strcmp(upper(recurse),'Y'))
    DirNamesNoFiles = FileAndDirNames(LogicalIsDirectory);
    DiscardLogical1Dir = strncmp(DirNamesNoFiles,'.',1);
    DirNames = DirNamesNoFiles(~DiscardLogical1Dir);
    if (length(DirNames) > 0)
        for i=1:length(DirNames),
            MoreFileNames = RetrieveImageFileNames(fullfile(Pathname,char(DirNames(i))), recurse);
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTIONS FOR READING STK FILES %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [S, stack_cnt] = tiffread(filename, img_first, img_last)
% [S, nbimages] = tiffread;
% [S, nbimages] = tiffread(filename);
% [S, nbimages] = tiffread(filename, image);
% [S, nbimages] = tiffread(filename, first_image, last_image);
%
% Reads 8,16,32 bits uncompressed grayscale tiff and stacks of tiff images, 
% for example those produced by metamorph or NIH-image. The function can be
% called with a file name in the current directory, or without argument, in
% which case it pop up a file openning dialog to allow selection of file.
% the pictures read in a stack can be restricted by specifying the first
% and last images to read, or just one image to read.
% 
% at return, nbimages contains the number of images read, and S is a vector 
% containing the different images with their tiff tags informations. The image 
% pixels values are stored in the field .data, in the native format (integer),
% and must be converted to be used in most matlab functions.
%
% EX. to show image 5 read from a 16 bit stack, call image( double(S(5).data) ); 
%
% Francois Nedelec, EMBL, Copyright 1999-2003.
% last modified April 4, 2003.
% Please, feedback/bugs/improvements to  nedelec (at) embl.de

if (nargin == 0)
    [filename, pathname] = uigetfile('*.tif;*.stk', 'select image file');
    filename = [ pathname, filename ];
end

if (nargin<=1)  img_first = 1; img_last = 10000; end
if (nargin==2)  img_last = img_first;            end

img_skip  = 0;
img_read  = 0;
stack_cnt = 1;

% not all valid tiff tags have been included, but they can be easily added
% to this code (see the official list of tags at
% http://partners.adobe.com/asn/developer/pdfs/tn/TIFF6.pdf
%
% the structure TIFIM is returned to the user, while TIF is not.
% so tags usefull to the user should be stored in TIFIM, while 
% those used only internally can be stored in TIF.


% set defaults values :
TIF.sample_format     = 1;
TIF.samples_per_pixel = 1;
TIF.BOS               = 'l';          %byte order string

if  isempty(findstr(filename,'.'))
   filename=[filename,'.tif'];
end

[TIF.file, message] = fopen(filename,'r','l');
if TIF.file == -1
   filename = strrep(filename, '.tif', '.stk');
   [TIF.file, message] = fopen(filename,'r','l');
   if TIF.file == -1
      error(['file <',filename,'> not found.']);
   end
end


% read header
% read byte order: II = little endian, MM = big endian
byte_order = setstr(fread(TIF.file, 2, 'uchar'));
if ( strcmp(byte_order', 'II') )
   TIF.BOS = 'l';                                %normal PC format
elseif ( strcmp(byte_order','MM') )
   TIF.BOS = 'b';
else
   error('This is not a TIFF file (no MM or II).');
end

%----- read in a number which identifies file as TIFF format
tiff_id = fread(TIF.file,1,'uint16', TIF.BOS);
if (tiff_id ~= 42)  error('This is not a TIFF file (missing 42).'); end

%----- read the byte offset for the first image file directory (IFD)
ifd_pos = fread(TIF.file,1,'uint32', TIF.BOS);

while (ifd_pos ~= 0)
    
   clear TIFIM;
   TIFIM.filename = [ pwd, '\', filename ];
   % move in the file to the first IFD
   fseek(TIF.file, ifd_pos,-1);
   %disp(strcat('reading img at pos :',num2str(ifd_pos)));
   
   %read in the number of IFD entries
   num_entries = fread(TIF.file,1,'uint16', TIF.BOS);
   %disp(strcat('num_entries =', num2str(num_entries)));
   
   %read and process each IFD entry
   for i = 1:num_entries
      file_pos  = ftell(TIF.file);                     % save the current position in the file
      TIF.entry_tag = fread(TIF.file, 1, 'uint16', TIF.BOS);      % read entry tag
      entry = readIFDentry(TIF);
      %disp(strcat('reading entry <',num2str(entry_tag),'>:'));
      
      switch TIF.entry_tag
      case 254
          TIFIM.NewSubfiletype = entry.val;         
      case 256         % image width - number of column
          TIFIM.width          = entry.val;       
      case 257         % image height - number of row
          TIFIM.height         = entry.val;       
      case 258         % bits per sample
          TIFIM.bits           = entry.val;
          TIF.bytes_per_pixel  = entry.val / 8;
          %disp(sprintf('%i bits per pixels', entry.val));
      case 259         % compression
          if (entry.val ~= 1) error('Compression format not supported.'); end
      case 262         % photometric interpretatio
          TIFIM.photo_type     = entry.val;
      case 269
          TIFIM.document_name  = entry.val;
      case 270         % comment:
          TIFIM.info           = entry.val;
      case 271
          TIFIM.make           = entry.val;
      case 273         % strip offset
          TIF.strip_offsets    = entry.val;
          TIF.num_strips       = entry.cnt;
          %disp(strcat('num_strips =', num2str(TIF.num_strips)));
      case 277         % sample_per pixel
          TIF.samples_per_pixel = entry.val;
          if (TIF.samples_per_pixel ~= 1) error('color not supported'); end
      case 278         % rows per strip
          TIF.rows_per_strip   = entry.val;
      case 279         % strip byte counts - number of bytes in each strip after any compressio
          TIF.strip_bytes      = entry.val;
      case 282        % X resolution
          TIFIM.x_resolution   = entry.val;
      case 283         % Y resolution         
          TIFIM.y_resolution   = entry.val;
      case 296        % resolution unit
          TIFIM.resolution_unit= entry.val;
      case 305         % software 
          TIFIM.software       = entry.val;
      case 306         % datetime
          TIFIM.datetime       = entry.val; 
      case 315
          TIFIM.artist         = entry.val;
      case 317        %predictor for compression
          if (entry.val ~= 1) error('unsuported predictor value'); end
      case 320         % color map
          TIFIM.cmap          = entry.val;
          TIFIM.colors        = entry.cnt/3;
      case 339
          TIF.sample_format   = entry.val;
          if ( TIF.sample_format > 2 ) 
              error(sprintf('unsuported sample format = %i', TIF.sample_format));
          end
      case 33628       %metamorph specific data
          TIFIM.MM_private1   = entry.val;
      case 33629       %metamorph stack data?
          TIFIM.MM_stack      = entry.val;
          stack_cnt           = entry.cnt;
  %        disp([num2str(stack_cnt), ' frames, read:      ']);
      case 33630       %metamorph stack data: wavelength
          TIFIM.MM_wavelength = entry.val;
      case 33631       %metamorph stack data: gain/background?
          TIFIM.MM_private2   = entry.val;
      otherwise
          disp(sprintf('ignored tiff entry with tag %i cnt %i', TIF.entry_tag, entry.cnt));      
      end
      % move to next IFD entry in the file
      fseek(TIF.file, file_pos+12,-1);
  end
  
   %read the next IFD address:
   ifd_pos = fread(TIF.file, 1, 'uint32', TIF.BOS);
   %if (ifd_pos) disp(['next ifd at', num2str(ifd_pos)]); end
   
   if ( img_last > stack_cnt ) img_last = stack_cnt; end
   
   
   stack_pos = 0;
   
   for i=1:stack_cnt
       
      if ( img_skip + 1 >= img_first )
         img_read = img_read + 1;
         %disp(sprintf('reading MM frame %i at %i',num2str(img_read),num2str(TIF.strip_offsets(1)+stack_pos)));
         if ( stack_cnt > 1 ) disp(sprintf('\b\b\b\b\b%4i', img_read)); end
         TIFIM.data = read_strips(TIF, TIF.strip_offsets + stack_pos, TIFIM.width, TIFIM.height);
         S( img_read ) = TIFIM;

         %==============distribute the metamorph infos to each frame:
         if  isfield( TIFIM, 'MM_stack' )
             x = length(TIFIM.MM_stack) / stack_cnt;
             if  rem(x, 1) == 0
                 S( img_read ).MM_stack = TIFIM.MM_stack( 1+x*(img_read-1) : x*img_read );
                 if  isfield( TIFIM, 'info' )
                     x = length(TIFIM.info) / stack_cnt;
                     if rem(x, 1) == 0
                         S( img_read ).info = TIFIM.info( 1+x*(img_read-1) : x*img_read );
                     end
                 end
             end
         end
         if  isfield( TIFIM, 'MM_wavelength' )
             x = length(TIFIM.MM_wavelength) / stack_cnt;
             if rem(x, 1) == 0
                 S( img_read ).MM_wavelength = TIFIM.MM_wavelength( 1+x*(img_read-1) : x*img_read );
             end
         end         
         
         if ( img_skip + img_read >= img_last ) 
              fclose(TIF.file);
              return;
         end
     else
         %disp('skiping strips');
         img_skip = img_skip + 1;
         skip_strips(TIF, TIF.strip_offsets + stack_pos);
     end
      stack_pos = ftell(TIF.file) - TIF.strip_offsets(1);
   end
   
end

fclose(TIF.file);

return;


%============================================================================
%============================================================================



function data = read_strips(TIF, strip_offsets, width, height)
   
   % compute the width of each row in bytes:
   numRows     = width * TIF.samples_per_pixel;
   width_bytes = numRows * TIF.bytes_per_pixel;   
   numCols     = sum( TIF.strip_bytes / width_bytes );
   
   typecode = sprintf('int%i', 8 * TIF.bytes_per_pixel / TIF.samples_per_pixel );
   if TIF.sample_format == 1 
       typecode = [ 'u', typecode ];
   end
   
   % Preallocate strip matrix:
   data = eval( [ typecode, '(zeros(numRows, numCols));'] ); 
   
   colIndx = 1;
   for i = 1:TIF.num_strips
       fseek(TIF.file, strip_offsets(i), -1);
       strip = fread( TIF.file, TIF.strip_bytes(i) ./ TIF.bytes_per_pixel, typecode, TIF.BOS );
       if TIF.sample_format == 2
           %strip == bitcmp( strip );
       end
       
       if length(strip) ~= TIF.strip_bytes(i) / TIF.bytes_per_pixel
           error('End of file reached unexpectedly.');
       end
       stripCols = TIF.strip_bytes(i) ./ width_bytes;
       data(:, colIndx:(colIndx+stripCols-1)) = reshape(strip, numRows, stripCols);
       colIndx = colIndx + stripCols;   
   end
   % Extract valid part of data
   if ~all(size(data) == [width height]),
      data = data(1:width, 1:height);
      disp('extracting data');
   end
   % transpose the image
   data = data';
return;


function skip_strips(TIF, strip_offsets)
   fseek(TIF.file, strip_offsets(TIF.num_strips) + TIF.strip_bytes(TIF.num_strips),-1);
return;

%===================sub-functions that reads an IFD entry:===================


function [nbbytes, typechar] = matlabtype(tifftypecode)
   switch (tifftypecode)
   case 1
      nbbytes=1;
      typechar='uint8';
   case 2
      nbbytes=1;
      typechar='uchar';
   case 3
      nbbytes=2;
      typechar='uint16';
   case 4
      nbbytes=4;
      typechar='uint32';
   case 5
      nbbytes=8;
      typechar='uint32';
   otherwise
      error('tiff type not supported')
   end
return;


function  entry = readIFDentry(TIF)
   
   entry.typecode = fread(TIF.file, 1, 'uint16', TIF.BOS);
   entry.cnt      = fread(TIF.file, 1, 'uint32', TIF.BOS);
   %disp(strcat('typecode =', num2str(entry.typecode),', cnt = ',num2str(entry.cnt)));
   [ entry.nbbytes, entry.typechar ] = matlabtype(entry.typecode);
   if entry.nbbytes * entry.cnt > 4
      % next field contains an offset:
      offset = fread(TIF.file, 1, 'uint32', TIF.BOS);
      %disp(strcat('offset = ', num2str(offset)));
      fseek(TIF.file, offset, -1);
   end
   
   if TIF.entry_tag == 33629   %special metamorph 'rationals'
       entry.val = fread(TIF.file, 6*entry.cnt, entry.typechar, TIF.BOS);
   else
       if entry.typecode == 5
           entry.val = fread(TIF.file, 2*entry.cnt, entry.typechar, TIF.BOS);
       else
           entry.val = fread(TIF.file, entry.cnt, entry.typechar, TIF.BOS);
       end
   end
   if ( entry.typecode == 2 ) entry.val = char(entry.val'); end
   
return;

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