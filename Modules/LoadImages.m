function handles = LoadImages(handles)

% Help for the Load Images module:
% Category: File Processing
%
% Tells CellProfiler where to retrieve images and gives each image a
% meaningful name for the other modules to access.
%
% If more than four images per set must be loaded, more than one Load
% Images module can be run sequentially. Running more than one of
% these modules also allows images to be retrieved from different
% folders. If you want to load all images in a directory, you can
% enter the file extension as the text for which to search.
%
% LOADING IMAGES:
%
% ORDER:
% Load Images Order is useful when images are present in a repeating
% order, like DAPI, FITC, Red, DAPI, FITC, Red, and so on, where
% images are selected based on how many images are in each set and
% what position within each set a particular color is located (e.g.
% three images per set, DAPI is always first).  By contrast, Load
% Images Text is used to load images that have a particular piece of
% text in the name.
%
% TEXT:
% This method is different from the Load Images Order method because
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
% LOADING MOVIES:
% Tells CellProfiler where to retrieve avi-formatted movies (avi
% movies must be in uncompressed avi format on UNIX and Mac
% platforms) or stk-format movies (stacks of tif images produced by
% MetaMorph or NIH ImageJ). Once the files are identified, this module
% extracts each frame of each movie as a separate image,
% and gives these images a meaningful name for the other modules to
% access.
%
% The ability to load by order or by text are the same for both loading
% movies and loading images. Please refer to the Loading Images section for
% more information.
%
% You may have subfolders within the folder that is being searched, but the
% names of the folders themselves must not contain the text you are
% searching for or an error will result.
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
% SAVING IMAGES: The images loaded by this module can be easily saved
% using the Save Images module, using the name you assign (e.g.
% OrigBlue).  In the Save Images module, the images can be saved in a
% different format, allowing this module to function as a file format
% converter.
%
% See also <nothing relevant>.

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
% $Revision: 1725 $

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow

%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

%textVAR01 = How do you want to load these files?
%choiceVAR01 = Text-Exact match
%choiceVAR01 = Text-Regular expressions
%choiceVAR01 = Order
LoadChoice = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

if strcmp(LoadChoice(1),'T')
    ExactOrRegExp = LoadChoice(6);
end

%textVAR02 = Type the text that one type of image has in common (for TEXT options), or their position in each group (for ORDER option):
%defaultVAR02 = DAPI
TextToFind{1} = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = What do you want to call these images within CellProfiler?
%defaultVAR03 = OrigBlue
%infotypeVAR03 = imagegroup indep
ImageName{1} = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = Type the text that one type of image has in common (for TEXT options), or their position in each group (for ORDER option):
%defaultVAR04 = /
TextToFind{2} = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = What do you want to call these images within CellProfiler?
%defaultVAR05 = /
%infotypeVAR05 = imagegroup indep
ImageName{2} = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = Type the text that one type of image has in common (for TEXT options), or their position in each group (for ORDER option):
%defaultVAR06 = /
TextToFind{3} = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = What do you want to call these images within CellProfiler?
%defaultVAR07 = /
%infotypeVAR07 = imagegroup indep
ImageName{3} = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = Type the text that one type of image has in common (for TEXT options), or their position in each group (for ORDER option):
%defaultVAR08 = /
TextToFind{4} = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%textVAR09 = What do you want to call these images within CellProfiler?
%defaultVAR09 = /
%infotypeVAR09 = imagegroup indep
ImageName{4} = char(handles.Settings.VariableValues{CurrentModuleNum,9});

%textVAR10 = If using ORDER, how many images are there in each group (i.e. each field of view)?
%defaultVAR10 = 3
ImagesPerSet = str2num(char(handles.Settings.VariableValues{CurrentModuleNum,10}));

%textVAR11 = Are you loading image or movie files?
%choiceVAR11 = Image
%choiceVAR11 = Movie
ImageOrMovie = char(handles.Settings.VariableValues{CurrentModuleNum,11});
%inputtypeVAR11 = popupmenu

%textVAR12 = If your file names do not have extensions, choose the file format of the images (Note, this doesn't work currently)
%choiceVAR12 = tif
%choiceVAR12 = bmp
%choiceVAR12 = gif
%choiceVAR12 = jpg
%choiceVAR12 = mat
%choiceVAR12 = DIB
%choiceVAR12 = avi
%choiceVAR12 = stk
FileFormat = char(handles.Settings.VariableValues{CurrentModuleNum,12});
%inputtypeVAR12 = popupmenu

%textVAR13 = Analyze all subdirectories within the selected directory (Y or N)?
%choiceVAR13 = No
%choiceVAR13 = Yes
AnalyzeSubDir = char(handles.Settings.VariableValues{CurrentModuleNum,13});
%inputtypeVAR13 = popupmenu

%pathnametextVAR14 = Enter the path name to the folder where the images to be loaded are located. Type period (.) for default directory.
Pathname = char(handles.Settings.VariableValues{CurrentModuleNum,14});

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines which image set is being analyzed.
SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;

%%% Remove slashes entries with N/A or no filename from the input,
%%% i.e., store only valid entries
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

if strcmp(LoadChoice,'Order')
    TextToFind = str2num(char(TextToFind));
    %%% Checks whether the position in set exceeds the number per set.
    if ImagesPerSet < max(TextToFind)
        error(['Image processing was canceled during the Load Images Order module because the position of one of the image types within each image set exceeds the number of images per set that you entered (', num2str(ImagesPerSet), ').'])
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% FIRST IMAGE SET FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Extracting the list of files to be analyzed occurs only the first time
%%% through this module.
if SetBeingAnalyzed == 1

    %%% Get the pathname and check that it exists
    if strncmp(Pathname,'.',1)
        if length(Pathname) == 1
            Pathname = handles.Current.DefaultImageDirectory;
        else
            Pathname = fullfile(handles.Current.DefaultImageDirectory,Pathname(2:end));
        end
    end
    SpecifiedPathname = Pathname;
    if ~exist(SpecifiedPathname,'dir')
        error(['Image processing was canceled because the directory "',SpecifiedPathname,'" does not exist. Be sure that no spaces or unusual characters exist in your typed entry and that the pathname of the directory begins with /.'])
    end

    if strcmp(LoadChoice,'Order')

        if strcmp(ImageOrMovie,'Image')
            % Get all filenames in the specified directory wich contains the specified extension (e.g., .tif, .jpg, .DIB).
            % Note that there is no check that the extensions actually is the last part of the filename.
            FileNames = CPretrieveMediaFileNames(SpecifiedPathname,['.',FileFormat],AnalyzeSubDir,'Regular','Image');

            %%% Checks whether any files have been specified.
            if isempty(FileNames)
                errordlg(['Image processing was canceled because there are no image files in the chosen directory (or subdirectories, if you requested them to be analyzed as well), according to the Load Images Order module.'])
            end

            %%% Determines the number of image sets to be analyzed.
            NumberOfImageSets = length(FileNames)/ImagesPerSet;
            if rem(NumberOfImageSets,1) ~= 0
                errordlg(sprintf('Image processing was canceled because because the number of image files (%d) found in the specified directory is not a multiple of the number of images per set (%d), according to the Load Images Order module.',length(FileNames),ImagesPerSet));
            end
            handles.Current.NumberOfImageSets = NumberOfImageSets;

            %%% For all valid entries, write list of image files to handles structure
            for n = 1:length(ImageName)
                % Get the list of filenames
                FileList = FileNames(TextToFind(n):ImagesPerSet:end);

                %%% Saves the File Lists and Path Names to the handles structure.
                fieldname = ['FileList', ImageName{n}];
                handles.Pipeline.(fieldname) = FileList;
                fieldname = ['Pathname', ImageName{n}];
                handles.Pipeline.(fieldname) = SpecifiedPathname;
            end
            clear FileNames

        else
            % Get all filenames in the specified directory wich contains the specified extension (e.g., .avi or .stk).
            % Note that there is no check that the extensions actually is the last part of the filename.
            FileNames = CPretrieveMediaFileNames(SpecifiedPathname,['.',FileFormat],AnalyzeSubDir,'Regular','Movie');

            %%% Checks whether any files have been found
            if isempty(FileNames)
                errordlg(['Image processing was canceled because there are no movie files in the chosen directory (or subdirectories, if you requested them to be analyzed as well), according to the LoadMoviesOrder module.'])
            end

            %%% Determines the number of movie sets to be analyzed.
            NumberOfMovieSets = fix(length(FileNames)/ImagesPerSet);
            if rem(NumberOfMovieSets,1) ~= 0
                errordlg(sprintf('Image processing was canceled because because the number of movie files (%d) found in the specified directory is not a multiple of the number of movies per set (%d), according to the LoadMoviesOrder module.',length(FileNames),MoviesPerSet));
            end
            handles.Current.NumberOfMovieSets = NumberOfMovieSets;

            %%% For all valid movie slots, extracts the file names.
            for n = 1:length(ImageName)
                % Get the list of filenames
                FileList = FileNames(TextToFind{n}:ImagesPerSet:end);

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
                    end
                    StartingPositionForThisMovie = StartingPositionForThisMovie + NumFrames;
                end

                %%% Saves the File Lists and Path Names to the handles structure.
                fieldname = ['FileList', ImageName{n}];
                handles.Pipeline.(fieldname) = FrameByFrameFileList{n};
                fieldname = ['Pathname', ImageName{n}];
                handles.Pipeline.(fieldname) = SpecifiedPathname;
                NumberOfFiles{n} = num2str(length(FrameByFrameFileList{n})); %#ok We want to ignore MLint error checking for this line.
            end
            clear FileNames


            %%% Determines how many unique numbers of files there are.  If all
            %%% the movie types have loaded the same number of images, there
            %%% should only be one unique number, which is the number of image
            %%% sets.
            UniqueNumbers = unique(NumberOfFiles);
            %%% If NumberOfFiles is not all the same number at each position, generate an error.
            if length(UniqueNumbers) ~= 1
                CharMovieName = char(ImageName);
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
        end
    else
        if strcmp(ImageOrMovie,'Image')
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

        else
            %%% For all non-empty slots, extracts the file names.
            for n = 1:length(ImageName)
                FileList = CPretrieveMediaFileNames(SpecifiedPathname,char(TextToFind(n)),AnalyzeSubDir, ExactOrRegExp,'Movie');
                %%% Checks whether any files are left.
                if isempty(FileList)
                    error(['Image processing was canceled because there are no movie files with the text "', TextToFind{n}, '" in the chosen directory (or subdirectories, if you requested them to be analyzed as well), according to the LoadMoviesText module.'])
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
                        %%% Reads metamorph or NIH ImageJ movie stacks of tiffs.
                        [S, NumFrames] = tiffread(fullfile(SpecifiedPathname, CurrentMovieFileName),1);
                        for FrameNumber = 1:NumFrames
                            %%% Puts the file name into the FrameByFrameFileList in the first row.
                            FrameByFrameFileList{n}(1,StartingPositionForThisMovie + FrameNumber) = {CurrentMovieFileName};
                            %%% Puts the frame number into the FrameByFrameFileList in the second row.
                            FrameByFrameFileList{n}(2,StartingPositionForThisMovie + FrameNumber) = {FrameNumber};
                        end
                    else
                        error('CellProfiler can currently read only avi or stk movie files.')
                    end
                    StartingPositionForThisMovie = StartingPositionForThisMovie + NumFrames;

                    %%% Saves the File Lists and Path Names to the handles structure.
                    fieldname = ['FileList', ImageName{n}];
                    handles.Pipeline.(fieldname) = FrameByFrameFileList{n};
                    fieldname = ['Pathname', ImageName{n}];
                    handles.Pipeline.(fieldname) = SpecifiedPathname;
                    %% for reference in saved files
                    handles.Measurements.Image.(fieldname) = SpecifiedPathname;
                    NumberOfFiles{n} = num2str(length(FrameByFrameFileList{n})); %#ok We want to ignore MLint error checking for this line.
                end
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
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% LOADING IMAGES EACH TIME %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
FileName = {};
for n = 1:length(ImageName)
    if strcmp(ImageOrMovie,'Image')
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

            if (max(LoadedImage(:)) <= .0625) && (handles.Current.SetBeingAnalyzed == 1)
                CPwarndlg('Warning: your images are very dim (they are using 1/16th or less of the dynamic range of the image file format). This often happens when a 12-bit camera saves in 16-bit image format. You might consider using the RescaleIntensity module to rescale the images using a factor of 0.0625.');
            end
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
    else
        %%% This try/catch will catch any problems in the load movies module.
        try

            %%% Determines which movie to analyze.
            fieldname = ['FileList', ImageName{n}];
            FileList = handles.Pipeline.(fieldname);
            %%% Determines the file name of the movie you want to analyze.
            CurrentFileName = FileList(:,SetBeingAnalyzed);
            %%% Determines the directory to switch to.
            fieldname = ['Pathname', ImageName{n}];
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
            fieldname = ['Filename', ImageName{n}];
            [SubdirectoryPathName,BareFileName,ext,versn] = fileparts(char(CurrentFileName(1)));
            CurrentFileNameWithFrame = [BareFileName, '_', num2str(cell2mat(CurrentFileName(2))),ext];
            %%% Saves the loaded image to the handles structure.  The field is named
            %%% appropriately based on the user's input, and put into the Pipeline
            %%% substructure so it will be deleted at the end of the analysis batch.
            handles.Pipeline.(fieldname)(SetBeingAnalyzed) = {CurrentFileNameWithFrame};
            handles.Pipeline.(ImageName{n}) = LoadedImage;
        catch ErrorMessage = lasterr;
            ErrorNumber = {'first','second','third','fourth'};
            error(['An error occurred when trying to load the ', ErrorNumber{n}, ' set of movies using the Load Movies Text module. Please check the settings. A common problem is that there are non-image files in the directory you are trying to analyze, or that the image file is not in uncompressed avi format. Matlab says the problem is: ', ErrorMessage])
        end % Goes with: catch
        FileNames(n) = {CurrentFileNameWithFrame};
    end
end
%%% -- Save to the handles.Measurements structure for reference in output files --------------- %%%
%%% NOTE: The structure for filenames and pathnames will be a cell array of cell arrays

%%% First, fix feature names and the pathname
PathNames = cell(1,length(ImageName));
FileNamesText = cell(1,length(ImageName));
PathNamesText = cell(1,length(ImageName));
for n = 1:length(ImageName)
    PathNames{n} = Pathname;
    FileNamesText{n} = ['Filename ', ImageName{n}];
    PathNamesText{n} = ['Path ', ImageName{n}];
end

%%% Since there may be several load modules in the pipeline which all write to the
%%% handles.Measurements.Image.FileName field, we have store filenames in an "appending" style.
%%% Here we check if any of the modules above the current module in the pipline has written to
%%% handles.Measurements.Image.Filenames. Then we should append the current filenames and path
%%% names to the already written ones.
if  isfield(handles,'Measurements') && isfield(handles.Measurements,'Image') &&...
        isfield(handles.Measurements.Image,'FileNames') && length(handles.Measurements.Image.FileNames) == SetBeingAnalyzed
    % Get existing file/path names. Returns a cell array of names
    ExistingFileNamesText = handles.Measurements.Image.FileNamesText;
    ExistingFileNames     = handles.Measurements.Image.FileNames{SetBeingAnalyzed};
    ExistingPathNamesText = handles.Measurements.Image.PathNamesText;
    ExistingPathNames     = handles.Measurements.Image.PathNames{SetBeingAnalyzed};

    % Append current file names to existing file names
    FileNamesText = cat(2,ExistingFileNamesText,FileNamesText);
    FileNames     = cat(2,ExistingFileNames,FileNames);
    PathNamesText = cat(2,ExistingPathNamesText,PathNamesText);
    PathNames     = cat(2,ExistingPathNames,PathNames);
end

%%% Write to the handles.Measurements.Image structure
handles.Measurements.Image.FileNamesText                   = FileNamesText;
handles.Measurements.Image.FileNames(SetBeingAnalyzed)         = {FileNames};
handles.Measurements.Image.PathNamesText                   = PathNamesText;
handles.Measurements.Image.PathNames(SetBeingAnalyzed)         = {PathNames};
%%% ------------------------------------------------------------------------------------------------ %%%

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
                %         disp([num2str(stack_cnt), ' frames, read:      ']);
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