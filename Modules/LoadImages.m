function handles = LoadImages(handles)

% Help for the Load Images module:
% Category: File Processing
%
% SHORT DESCRIPTION:
% Allows you to specify which images or movies are to be loaded and in
% which order. Groups of images will be loaded per cycle of CellProfiler
% processing.
% *************************************************************************
%
% Tells CellProfiler where to retrieve images and gives each image a
% meaningful name for the other modules to access. When used in combination
% with a SaveImages module, you can load images in one file format and
% save in another file format, making CellProfiler work as a file format
% converter.
%
% If more than four images per cycle must be loaded, more than one
% LoadImages module can be run sequentially. Running more than one of these
% modules also allows images to be retrieved from different folders. Hint:
% if you want to load all images in a directory, you can enter the file
% extension as the text for which to search.
%
% Relative pathnames can be used. For example, in regular expressions text
% mode, on the Mac platform you could leave the folder where images are to
% be loaded as '.' to choose the default image folder, and then enter
% ../DAPI[123456789].tif as the name of the files you would like to load in
% order to load images from the directory one above the default image
% directory. Or, you could type .../AnotherSubfolder (note the three
% periods: the first is interpreted as a standin for the default image
% folder) as the folder from which images are to be loaded and enter the
% filename as .tif to load an image from a different subfolder of the
% parent of the default image folder.
%
% Note: You can test a pipeline's settings on a single image cycle by
% setting the Load Images module appropriately. For example, if loading by
% order, you can set the number of images per set to equal the total number
% of images in the folder (even if it is thousands) so that only the first
% cycle will be analyzed. Or, if loading by text, you can make the
% identifying text specific enough that it will recognize only one group of
% images in the folder. Once the settings look good for a few test images,
% you can change the Load Images module to recognize all images in your
% folder.
%
% Settings:
%
% How do you want to load these files?
% - Order is used when images (or movies) are present in a repeating order,
% like DAPI, FITC, Red, DAPI, FITC, Red, and so on, where images are
% selected based on how many images are in each group and what position
% within each group a particular color is located (e.g. three images per
% group, DAPI is always first).
%
% - Text is used to load images (or movies) that have a particular piece of
% text in the name. You have the option of matching text exactly, or using
% regular expressions to match text. The files containing the text that are
% in image format will be loaded.
%
% - When regular expressions is selected, patterns are specified using
% combinations of metacharacters and literal characters. There are a few
% classes of metacharacters, partially listed below. More extensive
% explanation can be found at:
%   http://www.mathworks.com/access/helpdesk/help/techdoc/matlab_prog/
%   f0-42649.html
%
% The following metacharacters match exactly one character from its
% respective set of characters:
%
%   Metacharacter   Meaning
%  ---------------  --------------------------------
%              .    Any character
%             []    Any character contained within the brackets
%            [^]    Any character not contained within the brackets
%             \w    A word character [a-z_A-Z0-9]
%             \W    Not a word character [^a-z_A-Z0-9]
%             \d    A digit [0-9]
%             \D    Not a digit [^0-9]
%             \s    Whitespace [ \t\r\n\f\v]
%             \S    Not whitespace [^ \t\r\n\f\v]
%
% The following metacharacters are used to logically group subexpressions
% or to specify context for a position in the match. These metacharacters
% do not match any characters in the string:
%
%   Metacharacter   Meaning
%  ---------------  --------------------------------
%            ()     Group subexpression
%             |     Match subexpression before or after the |
%             ^     Match expression at the start of string
%             $     Match expression at the end of string
%            \<     Match expression at the start of a word
%            \>     Match expression at the end of a word
%
% The following metacharacters specify the number of times the previous
% metacharacter or grouped subexpression may be matched:
%
%   Metacharacter   Meaning
%  ---------------  --------------------------------
%             *     Match zero or more occurrences
%             +     Match one or more occurrences
%             ?     Match zero or one occurrence
%          {n,m}    Match between n and m occurrences
%
% Characters that are not special metacharacters are all treated literally
% in a match. To match a character that is a special metacharacter, escape
% that character with a '\'. For example '.' matches any character, so to
% match a '.' specifically, use '\.' in your pattern.
%
% Examples:
%
%     * [trm]ail matches 'tail' or 'rail' or 'mail'
%     * [0-9] matches any digit between 0 to 9
%     * [^Q-S] matches any character other than 'Q' or 'R' or 'S'
%     * [[]A-Z] matches any upper case alphabet along with square brackets
%     * [ag-i-9] matches characters 'a' or 'g' or 'h' or 'i' or '-' or '9'
%     * [a-p]* matches '' or 'a' or 'aab' or 'p' etc.
%     * [a-p]+ matches  'a' or 'abc' or 'p' etc.
%     * [^0-9] matches any string that is not a number
%     * ^[0-9]*$ matches any string that is a natural number or ''
%     * ^-[0-9]+$|^\+?[0-9]+$ matches any integer
%
% If you want to exclude files, type in the text that the excluded files
% have in common.
% The image/movie files specified with the TEXT option may also include
% files that you want to exclude from analysis (such as thumbnails created
% by an imaging system). Here you can specify text that mark files for
% exclusion. This text is treated as a exact match and not as a regular
% expression. Note: This choice is ignored with the ORDER option.
% 
% What do you want to call these images within CellProfiler?
% Give your images a meaningful name that you will use when referring to
% these images in later modules.  To avoid errors, image names should
% follow Matlab naming conventions:
% 
% 1. Field names must begin with a letter, which may be followed by any 
% combination of letters, digits, and underscores. The following statements are all invalid:
% w = setfield(w, 'My.Score', 3);
% w = setfield(w, '1stScore', 3);
% w = setfield(w, '1+1=3', 3);
% w = setfield(w, '@MyScore', 3);
% 3. Although field names can be of any length, MATLAB uses only the first N 
% characters of the field name, (where N is the number returned by the function 
% namelengthmax), and ignores the rest.
%
% **NOTE:** When CellProfiler saves image and object measurements, it
% appends the text of your ImageName with meaningful text about the
% measurement (ie, Intensity_MeanIntensity_DAPI) which can quickly reach
% this 63-char limit and are truncated by CellProfiler to avoid errors.  
% Take care to name your images and measurements conservatively to avoid 
% truncating measurements to the same name.
% 3. MATLAB distinguishes between uppercase and lowercase characters. 
% Field name length is not the same as field name Length.
% 4. In most cases, you should refrain from using the names of functions and 
% variables as field names.
% 
% Analyze all subfolders within the selected folder?
% You may have subfolders within the folder that is being searched, but if
% you are in TEXT mode, the names of the folders themselves must not
% contain the text you are searching for or an error will result.
%
% If the images you are loading are binary (black/white only), in what 
% format do you want to store them?
% CellProfiler will save your image in binary format if your image has 
% only two distinct values and you have selected "binary" instead of
% "grayscale".
%
% Do you want to select the subfolders to process?
% If you answered "Yes" to both this question and "Analyze subfolders,"
% CellProfiler will provide a dialog box on the first cycle which will
% allow you to select which folders under the Image directory you want to
% process.
%
% Do you want to check image sets for missing or duplicate files? 
% Tokens must be defined for the unique parts of the string. (REGULAR EXPRESSIONS ONLY)
% Selecting this option with REGULAR mode will examine the filenames for 
% unmatched or duplicate files based on the filename prefix (such as those 
% generated by HCS systems).  This setting is only functional if tokens are
% used in the expression string (either named or unnamed), and does not 
% check for files missing from a particular plate layout.  A dialog box 
% will report the results. 
% Unnamed tokens are defined by enclosing the string in parentheses.
% For example, if you have 2 channels defined by '..._w1...' and '..._w2...'
% then call these _w(1) and _w(2)
% Named tokens are defined by this syntax: (?<name>expr), so that in the same 
% example as above, the named tokens would be _w(?<wavelength1>1) and 
% _w(?<wavelength2>2).
%
% Notes about loading images:
%
% CellProfiler can open and read .ZVI files. .ZVI files are Zeiss files
% that are generated by the microscope imaging software Axiovision. These
% images are stored in 12-bit depth. Currently, CellProfiler cannot read
% stacked or color ZVI images.
%
% CellProfiler can open and read .DIB files. These files are stored with
% 12-bit depth using a 16-bit file format.
%
% Notes about loading movies:
%
% (Update 10-11-2007) CellProfiler can read tif,tiff,flex multi-page
% tif file in addition to those formats specified below.
%
% Movies can be avi-formatted movies (must be uncompressed avi format on
% UNIX and Mac platforms) or stk-format movies (stacks of tif images
% produced by MetaMorph or NIHImage/ImageJ; The ability to read stk files
% is thanks to code by: Francois Nedelec, EMBL, Copyright 1999-2003). Once
% the files are identified, this module extracts each frame of each movie
% as a separate image, and gives these images a meaningful name for the
% other modules to access.
%
% Suggestions for third party software to uncompress AVI files and convert
% MOV files:
%
% WINDOWS...
% To convert movies to uncompressed avi format, you can use a free software
% product called RAD Video Tools, which is available from:
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
% The iMovie program which comes with Mac OSX can be used to convert movies
% to uncompressed avi format as follows:
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
% 4. To check/troubleshoot the conversion, you can use the following
% commands in Matlab:
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
% The following error means that the Depth was improper (either you tried
% to save in grayscale or the wrong bit depth color):
% >> movie = aviread('My Great Movie2.avi');
% ??? Error using ==> aviread
% Bitmap data must be 8-bit Index images or 24-bit TrueColor images
% ------------------------------------------------------------------
%
% See also LoadSingleImage.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

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

%textVAR04 = Type the text that one type of image has in common (for TEXT options), or their position in each group (for ORDER option). Type "Do not use" to ignore:
%defaultVAR04 = Do not use
TextToFind{2} = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = What do you want to call these images within CellProfiler? (Type "Do not use" to ignore)
%defaultVAR05 = Do not use
%infotypeVAR05 = imagegroup indep
ImageName{2} = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = Type the text that one type of image has in common (for TEXT options), or their position in each group (for ORDER option):
%defaultVAR06 = Do not use
TextToFind{3} = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = What do you want to call these images within CellProfiler?
%defaultVAR07 = Do not use
%infotypeVAR07 = imagegroup indep
ImageName{3} = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = Type the text that one type of image has in common (for TEXT options), or their position in each group (for ORDER option):
%defaultVAR08 = Do not use
TextToFind{4} = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%textVAR09 = What do you want to call these images within CellProfiler?
%defaultVAR09 = Do not use
%infotypeVAR09 = imagegroup indep
ImageName{4} = char(handles.Settings.VariableValues{CurrentModuleNum,9});

%textVAR10 = If using ORDER, how many images are there in each group (i.e. each field of view)?
%defaultVAR10 = 3
ImagesPerSet = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,10}));

%textVAR11 = If you want to exclude files, type the text that the excluded images have in common (for TEXT options). Type "Do not use" to ignore.
%defaultVAR11 = Do not use
TextToExclude = char(handles.Settings.VariableValues{CurrentModuleNum,11});

%textVAR12 = What type of files are you loading?
%choiceVAR12 = individual images
%choiceVAR12 = stk movies
%choiceVAR12 = avi movies
%choiceVAR12 = tif,tiff,flex movies
FileFormat = char(handles.Settings.VariableValues{CurrentModuleNum,12});
%inputtypeVAR12 = popupmenu

%textVAR13 = Analyze all subfolders within the selected folder?
%choiceVAR13 = No
%choiceVAR13 = Yes
AnalyzeSubDir = char(handles.Settings.VariableValues{CurrentModuleNum,13});
%inputtypeVAR13 = popupmenu

%pathnametextVAR14 = Enter the path name to the folder where the images to be loaded are located. Type period (.) for default image folder or ampersand (&) for default output folder.
%defaultVAR14 = .
Pathname = char(handles.Settings.VariableValues{CurrentModuleNum,14});

%textVAR15 = If the images you are loading are binary (black/white only), in what format do you want to store them?
%defaultVAR15 = grayscale
%choiceVAR15 = grayscale
%choiceVAR15 = binary
%inputtypeVAR15 = popupmenu
SaveAsBinary = strcmp(char(handles.Settings.VariableValues(CurrentModuleNum,15)),'binary');

%textVAR16 = If Yes to "Analyze all subfolders", do you want to select the subfolders to process?
%choiceVAR16 = Yes
%choiceVAR16 = No
SelectSubfolders = char(handles.Settings.VariableValues{CurrentModuleNum,16});
%inputtypeVAR16 = popupmenu

%textVAR17 = Do you want to check image sets for missing or duplicate files? Tokens must be defined for the unique parts of the string. (REGULAR EXPRESSIONS ONLY)
%choiceVAR17 = No
%choiceVAR17 = Yes
CheckImageSets = char(handles.Settings.VariableValues{CurrentModuleNum,17});
%inputtypeVAR17 = popupmenu

%textVAR18 = Note - If the movies contain more than just one image type (e.g., brightfield, fluorescent, field-of-view), add the GroupMovieFrames module.

%%%VariableRevisionNumber = 5

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

if strcmpi(AnalyzeSubDir,'Yes') && (~ isfield(handles.Current, 'BatchInfo'))
    if strcmpi(SelectSubfolders,'Yes')
        AnalyzeSubDir = 'Select';    %%% Select subdirectories using the GUI
    end
end
if strcmp(FileFormat,'individual images')
    ImageOrMovie = 'Image';
else
    ImageOrMovie = 'Movie';
end

%%% Determines which cycle is being analyzed.
SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;

%%% Remove "Do not use" entries from the input,
%%% i.e., store only valid entries
idx = strcmp(TextToFind,'Do not use') | strcmp(ImageName,'Do not use');
TextToFind = TextToFind(~idx);
ImageName = ImageName(~idx);

if strcmp(LoadChoice,'Order')
    TextToFind = str2num(char(TextToFind)); %#ok Ignore MLint
    %%% Checks whether the position in set exceeds the number per set.
    if ImagesPerSet < max(TextToFind)
        error(['Image processing was canceled in the ', ModuleName, ' module because the position of one of the image types within each cycle exceeds the number of images per set that you entered (', num2str(ImagesPerSet), ').'])
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% FIRST CYCLE FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Extracting the list of files to be analyzed occurs only the first time
%%% through this module.
if SetBeingAnalyzed == 1

    for i = 1:length(ImageName)
        %%% Only complain about reloads if not running on the cluster (to allow rerunning batch 1)
        if isfield(handles.Pipeline,ImageName{i}) && (~ isfield(handles.Current, 'BatchInfo')),
            error(['Image processing was cancelled in the ', ModuleName, ' module because you are trying to load two sets of images with the same name (e.g. OrigBlue). The last set loaded will always overwrite the first set and make it obselete. Please remove one of these modules.']);
        end
        CPvalidfieldname(ImageName{i});
    end

    %%% Get the pathname and check that it exists
    if strncmp(Pathname,'.',1)
        if length(Pathname) == 1
            Pathname = handles.Current.DefaultImageDirectory;
        else
	    % If the pathname start with '.', interpret it relative to
        % the default image dir.
            Pathname = fullfile(handles.Current.DefaultImageDirectory,strrep(strrep(Pathname(2:end),'/',filesep),'\',filesep),'');
        end
    elseif strncmp(Pathname, '&', 1)
        if length(Pathname) == 1
            Pathname = handles.Current.DefaultOutputDirectory;
        else
	    % If the pathname start with '&', interpret it relative to
        % the default output dir.
            Pathname = fullfile(handles.Current.DefaultOutputDirectory,strrep(strrep(Pathname(2:end),'/',filesep),'\',filesep),'');
        end
    end
    if ~exist(Pathname,'dir')
        error(['Image processing was canceled in the ', ModuleName, ' module because the directory "',Pathname,'" does not exist. Be sure that no spaces or unusual characters exist in your typed entry and that the pathname of the directory begins with /.'])
    end

    if strcmp(LoadChoice,'Order')

        if strcmp(ImageOrMovie,'Image')
            % Get all filenames in the specified directory wich contains the specified extension (e.g., .tif, .jpg, .DIB).
            % Note that there is no check that the extensions actually is the last part of the filename.
            [handles,FileNames] = CPretrievemediafilenames(handles, Pathname,'',AnalyzeSubDir,'Regular','Image');

            %%% Checks whether any files have been specified.
            if isempty(FileNames)
                error(['Image processing was canceled in the ', ModuleName, ' module because there are no image files in the chosen directory (or subdirectories, if you requested them to be analyzed as well).'])
            end

            %%% Determines the number of cycles to be analyzed.
            NumberOfImageSets = length(FileNames)/ImagesPerSet;
            if rem(NumberOfImageSets,1) ~= 0
                error(['Image processing was canceled in the ', ModuleName,' module becauses the number of image files (',length(FileNames),') found in the specified directory is not a multiple of the number of images per set (',ImagesPerSet,').'])
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
                handles.Pipeline.(fieldname) = Pathname;
            end
            clear FileNames

        else
            % Get all filenames in the specified directory wich contains the specified extension (e.g., .avi or .stk).
            % Note that there is no check that the extensions actually is the last part of the filename.
            [handles,FileNames] = CPretrievemediafilenames(handles, Pathname,'',AnalyzeSubDir,'Regular','Movie');

            %%% Checks whether any files have been found
            if isempty(FileNames)
                error(['Image processing was canceled in the ', ModuleName, ' module because there are no movie files in the chosen directory (or subdirectories, if you requested them to be analyzed as well).'])
            end

            %%% Determines the number of movie sets to be analyzed.
            NumberOfMovieSets = fix(length(FileNames)/ImagesPerSet);
            if rem(NumberOfMovieSets,1) ~= 0
                error(['Image processing was canceled in the ', ModuleName, ' module because the number of movie files (',length(FileNames),') found in the specified directory is not a multiple of the number of movies per set (',ImagesPerSet,').'])
            end
            handles.Current.NumberOfMovieSets = NumberOfMovieSets;

            %%% For all valid movie slots, extracts the file names.
            for n = 1:length(ImageName)
                % Get the list of filenames
                FileList = FileNames(TextToFind{n}:ImagesPerSet:end);

                StartingPositionForThisMovie = 0;
                for MovieFileNumber = 1:length(FileList)
                    CurrentMovieFileName = char(FileList(MovieFileNumber));
                    if strcmpi(FileFormat,'avi movies') == 1
                        try MovieAttributes = aviinfo(fullfile(Pathname, CurrentMovieFileName));
                        catch error(['Image processing was canceled in the ', ModuleName, ' module because the file ',fullfile(Pathname, CurrentMovieFileName),' was not readable as an uncompressed avi file.'])
                        end
                        NumFrames = MovieAttributes.NumFrames;
                        for FrameNumber = 1:NumFrames
                            %%% Puts the file name into the FrameByFrameFileList in the first row.
                            FrameByFrameFileList{n}(1,StartingPositionForThisMovie + FrameNumber) = {CurrentMovieFileName};
                            %%% Puts the frame number into the FrameByFrameFileList in the second row.
                            FrameByFrameFileList{n}(2,StartingPositionForThisMovie + FrameNumber) = {FrameNumber};
                            %%% Puts the movie length into the FrameByFrameFileList in the third row.
                            FrameByFrameFileList{n}(3,StartingPositionForThisMovie + FrameNumber) = {NumFrames};
                        end
                    elseif strcmpi(FileFormat,'stk movies') == 1
                        try
                            %%% Reads metamorph or NIH ImageJ movie stacks of tiffs.
                            [S, NumFrames] = CPtiffread(fullfile(Pathname, CurrentMovieFileName),1);
                            for FrameNumber = 1:NumFrames
                                %%% Puts the file name into the FrameByFrameFileList in the first row.
                                FrameByFrameFileList{n}(1,StartingPositionForThisMovie + FrameNumber) = {CurrentMovieFileName};
                                %%% Puts the frame number into the FrameByFrameFileList in the second row.
                                FrameByFrameFileList{n}(2,StartingPositionForThisMovie + FrameNumber) = {FrameNumber};
                                %%% Puts the movie length into the FrameByFrameFileList in the third row.
                                FrameByFrameFileList{n}(3,StartingPositionForThisMovie + FrameNumber) = {NumFrames};
                            end
                        catch error(['Image processing was canceled in the ', ModuleName, ' module because the file ',fullfile(Pathname, CurrentMovieFileName),' was not readable as a stk file.'])
                        end
                    elseif (strcmpi(FileFormat,'tif,tiff,flex movies') == 1)
                        try MultiTifAttributes = imfinfo(fullfile(Pathname, CurrentMovieFileName));
                        catch error(['Image processing was canceled in the ', ModuleName, ' module because the file ',fullfile(Pathname, CurrentMovieFileName),' was not readable as a tif, tiff, or flex file.']);
                        end
                        NumFrames = length(MultiTifAttributes);
                        for FrameNumber = 1:NumFrames
                            %%% Puts the file name into the FrameByFrameFileList in the first row.
                            FrameByFrameFileList{n}(1,StartingPositionForThisMovie + FrameNumber) = {CurrentMovieFileName};
                            %%% Puts the frame number into the FrameByFrameFileList in the second row.
                            FrameByFrameFileList{n}(2,StartingPositionForThisMovie + FrameNumber) = {FrameNumber};
                            %%% Puts the movie length into the FrameByFrameFileList in the third row.
                            FrameByFrameFileList{n}(3,StartingPositionForThisMovie + FrameNumber) = {NumFrames};
                        end  
                    else
                        error(['Image processing was canceled in the ', ModuleName, ' module because CellProfiler can currently read only avi, stk, tif, tiff, or flex movie files.'])
                    end
                    StartingPositionForThisMovie = StartingPositionForThisMovie + NumFrames;
                end

                %%% Saves the File Lists and Path Names to the handles structure.
                fieldname = ['FileList', ImageName{n}];
                handles.Pipeline.(fieldname) = FrameByFrameFileList{n};
                fieldname = ['FileFormat', ImageName{n}];
                handles.Pipeline.(fieldname) = FileFormat;
                fieldname = ['Pathname', ImageName{n}];
                handles.Pipeline.(fieldname) = Pathname;
                NumberOfFiles{n} = num2str(length(FrameByFrameFileList{n})); %#ok We want to ignore MLint error checking for this line.
            end
            clear FileNames

            %%% Determines how many unique numbers of files there are.  If all
            %%% the movie types have loaded the same number of images, there
            %%% should only be one unique number, which is the number of
            %%% image
            %%% sets.
            UniqueNumbers = unique(NumberOfFiles);
            %%% If NumberOfFiles is not all the same number at each position, generate an error.
            if length(UniqueNumbers) ~= 1
                CharMovieName = char(ImageName);
                CharNumberOfFiles = char(NumberOfFiles);
                Number = length(NumberOfFiles);
                for f = 1:Number
                    SpacesArray(f,:) = ':     ';
                end
                PreErrorText = cat(2, CharMovieName, SpacesArray);
                ErrorText = cat(2, PreErrorText, CharNumberOfFiles);
                msgbox(ErrorText)
                error(['Image processing was canceled in the ', ModuleName, ' module because the number of movies identified for each movie type is not equal.  In the window under this box you will see how many movie have been found for each movie type.'])
            end
            NumberOfImageSets = str2double(UniqueNumbers{1});
            %%% Checks whether another load images module has already recorded a
            %%% number of cycles.  If it has, it will not be set at the default
            %%% of 1.  Then, it checks whether the number already stored as the
            %%% number of cycles is equal to the number of cycles that this
            %%% module has found.  If not, an error message is generated. Note:
            %%% this will not catch the case where the number of cycles
            %%% detected by this module is more than 1 and another module has
            %%% detected only one cycle, since there is no way to tell whether
            %%% the 1 stored in handles.Current.NumberOfImageSets is the default value or a
            %%% value determined by another image-loading module.
            if handles.Current.NumberOfImageSets ~= 1;
                if handles.Current.NumberOfImageSets ~= NumberOfImageSets
                    error(['Image processing was canceled in the ', ModuleName, ' module because the number of image cycles loaded (', num2str(NumberOfImageSets),') does not equal the number of image cycles loaded by another image-loading module (', num2str(handles.Current.NumberOfImageSets), '). Please check the settings.'])
                end
            end
            handles.Current.NumberOfImageSets = NumberOfImageSets;
        end
    else % Not Order
        if strcmp(ImageOrMovie,'Image')
            %%% Extract the file names
            for n = 1:length(ImageName)
                
                [handles,FileList] = CPretrievemediafilenames(handles, Pathname,strtrim(char(TextToFind(n))),AnalyzeSubDir, ExactOrRegExp,'Image');
                
                % Remove excluded images
                if ~isempty(FileList) && ~strcmp(TextToExclude,'Do not use'),
                    excluded_idx = regexp(FileList,TextToExclude);
                    if ~isempty(excluded_idx), FileList = FileList(cellfun(@isempty,excluded_idx)); end
                end
    
                %%% Checks whether any files are left.
                if isempty(FileList)
                    error(['Image processing was canceled in the ', ModuleName, ' module because there are no image files with the text "', TextToFind{n}, '" in the chosen directory (or subdirectories, if you requested them to be analyzed as well).'])
                end
                %%% Saves the File Lists and Path Names to the handles structure.
                fieldname = ['FileList', ImageName{n}];
                handles.Pipeline.(fieldname) = FileList;
                fieldname = ['Pathname', ImageName{n}];
                handles.Pipeline.(fieldname) = Pathname;

                clear FileList % Prevents confusion when loading this value later, for each cycle.
            end
            if strcmpi(CheckImageSets,'Yes')
                handles = CPconfirmallimagespresent(handles, strtrim(TextToFind), ImageName, ExactOrRegExp, 'Yes');
            end
            for n= 1:length(ImageName)
                fieldname = ['FileList', ImageName{n}];
                NumberOfFiles{n} = num2str(length(handles.Pipeline.(fieldname))); %#ok We want to ignore MLint error checking for this line.
            end
        else
            %%% For all non-empty slots, extracts the file names.
            for n = 1:length(ImageName)
                [handles,FileList] = CPretrievemediafilenames(handles, Pathname,strtrim(char(TextToFind(n))),AnalyzeSubDir, ExactOrRegExp,'Movie');
                
                % Remove excluded images
                if ~isempty(FileList) && ~strcmp(TextToExclude,'Do not use'),
                    excluded_idx = regexp(FileList,TextToExclude);
                    if ~isempty(excluded_idx), FileList = FileList(cellfun(@isempty,excluded_idx)); end
                end
                
                %%% Checks whether any files are left.
                if isempty(FileList)
                    error(['Image processing was canceled in the ', ModuleName, ' module because there are no movie files with the text "', TextToFind{n}, '" in the chosen directory (or subdirectories, if you requested them to be analyzed as well).'])
                end
                StartingPositionForThisMovie = 0;
                for MovieFileNumber = 1:length(FileList)
                    CurrentMovieFileName = char(FileList(MovieFileNumber));
                    if strcmpi(FileFormat,'avi movies') == 1
                        try MovieAttributes = aviinfo(fullfile(Pathname, CurrentMovieFileName));
                        catch error(['Image processing was canceled in the ', ModuleName, ' module because the file ',fullfile(Pathname, CurrentMovieFileName),' was not readable as an uncompressed avi file.'])
                        end
                        NumFrames = MovieAttributes.NumFrames;
                        for FrameNumber = 1:NumFrames
                            %%% Puts the file name into the FrameByFrameFileList in the first row.
                            FrameByFrameFileList{n}(1,StartingPositionForThisMovie + FrameNumber) = {CurrentMovieFileName};
                            %%% Puts the frame number into the FrameByFrameFileList in the second row.
                            FrameByFrameFileList{n}(2,StartingPositionForThisMovie + FrameNumber) = {FrameNumber};
                            %%% Puts the movie length into the FrameByFrameFileList in the third row.
                            FrameByFrameFileList{n}(3,StartingPositionForThisMovie + FrameNumber) = {NumFrames};
                        end
                    elseif strcmpi(FileFormat,'stk movies') == 1
                        %%% Reads metamorph or NIH ImageJ movie stacks of tiffs.
                        [S, NumFrames] = CPtiffread(fullfile(Pathname, CurrentMovieFileName),1);
                        for FrameNumber = 1:NumFrames
                            %%% Puts the file name into the FrameByFrameFileList in the first row.
                            FrameByFrameFileList{n}(1,StartingPositionForThisMovie + FrameNumber) = {CurrentMovieFileName};
                            %%% Puts the frame number into the FrameByFrameFileList in the second row.
                            FrameByFrameFileList{n}(2,StartingPositionForThisMovie + FrameNumber) = {FrameNumber};
                            %%% Puts the movie length into the FrameByFrameFileList in the third row.
                            FrameByFrameFileList{n}(3,StartingPositionForThisMovie + FrameNumber) = {NumFrames};
                        end
                    elseif (strcmpi(FileFormat,'tif,tiff,flex movies') == 1)
                        try MultiTifAttributes = imfinfo(fullfile(Pathname, CurrentMovieFileName));
                        catch error(['Image processing was canceled in the ', ModuleName, ' module because the file ',fullfile(Pathname, CurrentMovieFileName),' was not readable as a tif, tiff, or flex file.']);
                        end
                        NumFrames = length(MultiTifAttributes);
                        for FrameNumber = 1:NumFrames
                            %%% Puts the file name into the FrameByFrameFileList in the first row.
                            FrameByFrameFileList{n}(1,StartingPositionForThisMovie + FrameNumber) = {CurrentMovieFileName};
                            %%% Puts the frame number into the FrameByFrameFileList in the second row.
                            FrameByFrameFileList{n}(2,StartingPositionForThisMovie + FrameNumber) = {FrameNumber};
                            %%% Puts the movie length into the FrameByFrameFileList in the third row.
                            FrameByFrameFileList{n}(3,StartingPositionForThisMovie + FrameNumber) = {NumFrames};
                        end
                    else
                        error(['Image processing was canceled in the ', ModuleName, ' module because CellProfiler can currently read only avi, stk, tif, tiff, or flex movie files.'])
                    end
                    StartingPositionForThisMovie = StartingPositionForThisMovie + NumFrames;

                    %%% Saves the File Lists and Path Names to the handles structure.
                    fieldname = ['FileList', ImageName{n}];
                    handles.Pipeline.(fieldname) = FrameByFrameFileList{n};
                    fieldname = ['FileFormat', ImageName{n}];
                    handles.Pipeline.(fieldname) = FileFormat;                    
                    fieldname = ['Pathname', ImageName{n}];
                    handles.Pipeline.(fieldname) = Pathname;
                    NumberOfFiles{n} = num2str(length(FrameByFrameFileList{n})); %#ok We want to ignore MLint error checking for this line.
                end
                clear FileList % Prevents confusion when loading this value later, for each movie set.
            end
        end
        %%% Determines which slots are empty.  None should be zero, because there is
        %%% an error check for that when looping through n = 1:5.
        for g = 1: length(NumberOfFiles)
            LogicalSlotsToBeDeleted(g) = isempty(NumberOfFiles{g});
        end
        %%% Removes the empty slots from both the Number of Files array and the
        %%% Image Name array.
        NumberOfFiles = NumberOfFiles(~LogicalSlotsToBeDeleted);
        ImageName2 = ImageName(~LogicalSlotsToBeDeleted);
        %%% Determines how many unique numbers of files there are.  If all the image
        %%% types have loaded the same number of images, there should only be one
        %%% unique number, which is the number of cycles.
        UniqueNumbers = unique(NumberOfFiles);
        %%% If NumberOfFiles is not all the same number at each position, generate an error.
        if length(UniqueNumbers) ~= 1
            CharImageName = char(ImageName2);
            CharNumberOfFiles = char(NumberOfFiles);
            Number = length(NumberOfFiles);
            for f = 1:Number
                SpacesArray(f,:) = ':     ';
            end
            PreErrorText = cat(2, CharImageName, SpacesArray);
            ErrorText = cat(2, PreErrorText, CharNumberOfFiles);
            CPmsgbox(ErrorText)
            error(['Image processing was canceled in the ', ModuleName, ' module because the number of images identified for each image type is not equal.  In the window under this box you will see how many images have been found for each image type.'])
        end
        NumberOfImageSets = str2double(UniqueNumbers{1});
        %%% Checks whether another load images module has already recorded a
        %%% number of cycles.  If it has, it will not be set at the default
        %%% of 1.  Then, it checks whether the number already stored as the
        %%% number of cycles is equal to the number of cycles that this
        %%% module has found.  If not, an error message is generated. Note:
        %%% this will not catch the case where the number of cycles
        %%% detected by this module is more than 1 and another module has
        %%% detected only one cycle, since there is no way to tell whether
        %%% the 1 stored in handles.Current.NumberOfImageSets is the default value or a
        %%% value determined by another image-loading module.
        if handles.Current.NumberOfImageSets ~= 1;
            if handles.Current.NumberOfImageSets ~= NumberOfImageSets
                error(['Image processing was canceled in the ', ModuleName, ' module because the number of cycles loaded (', num2str(NumberOfImageSets),') does not equal the number of cycles loaded by another image-loading module (', num2str(handles.Current.NumberOfImageSets), '). Please check the settings.'])
            end
        end
        handles.Current.NumberOfImageSets = NumberOfImageSets;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% LOADING IMAGES EACH TIME %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

if strcmp(ImageOrMovie,'Image'),
    ImageSizes = cell(length(ImageName),1);
    MissingFileIdx = false(length(ImageName),1);
end

for n = 1:length(ImageName)
    if strcmp(ImageOrMovie,'Image')
        %%% This try/catch will catch any problems in the load images module.
        try
            %%% The following runs every time through this module (i.e. for every cycle).
            %%% Determines which image to analyze.
            fieldname = ['FileList', ImageName{n}];
            FileList = handles.Pipeline.(fieldname);
            %%% Determines the file name of the image you want to analyze.
            CurrentFileName = FileList(SetBeingAnalyzed);
            %%% Determines the directory to switch to.
            fieldname = ['Pathname', ImageName{n}];
            Pathname = handles.Pipeline.(fieldname);
            if isempty(CurrentFileName{1})
                LoadedImage = 0;    % Missing file: Use 0's to replace
                MissingFileIdx(n) = true;
            else
                LoadedImage = CPimread(fullfile(Pathname,CurrentFileName{1}));
                MissingFileIdx(n) = false;
            end
            if SaveAsBinary
                minval=min(LoadedImage(:));
                maxval=max(LoadedImage(:));
                %%% If any member of LoadedImage is not the min or max 
                %%% value, then the image is not binary
                if all(ismember(LoadedImage,[minval,maxval]))
                    LoadedImage = (LoadedImage==maxval);
                end
            end
            %%% Note, we are not using the CPretrieveimage subfunction because we are
            %%% here retrieving the image from the hard drive, not from the handles
            %%% structure.

            if (max(LoadedImage(:)) <= .0625) && (handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet)
                A = strmatch('RescaleIntensity', handles.Settings.ModuleNames);
                if length(A) < length(ImageName)
                    CPwarndlg(['Warning: the images loaded by ', ModuleName, ' are very dim (they are using 1/16th or less of the dynamic range of the image file format). This often happens when a 12-bit camera saves in 16-bit image format. If this is the case, use the Rescale Intensity module in "Enter max and min" mode to rescale the images using the values 0, 0.0625, 0, 1, 0, 1.'],'Outside 0-1 Range','replace');
                end
            end
            %%% Saves the original image file name to the handles
            %%% structure.  The field is named appropriately based on
            %%% the user's input, in the Pipeline substructure so that
            %%% this field will be deleted at the end of the analysis
            %%% batch.
            fieldname = ['Filename', ImageName{n}];
            handles.Pipeline.(fieldname)(SetBeingAnalyzed) = CurrentFileName;
            handles.Pipeline.(ImageName{n}) = LoadedImage;
        catch
            CPerrorImread(ModuleName, n);
        end % Goes with: catch

        % Create a cell array with the filenames
        FileNames(n) = CurrentFileName(1);
        
        % Record the image size
        ImageSizes{n} = size(handles.Pipeline.(ImageName{n}));
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
            if strcmpi(FileFormat,'avi movies') == 1
                %%%If you do not subtract 1 from the index, as specified 
                %%%in aviread.m, the  movie will fail to load.  However,
                %%%the first frame will fail if the index=0.  
            	IndexLocation=(cell2mat(CurrentFileName(2)));
                NumberOfImageSets = handles.Current.NumberOfImageSets;
                if (cell2mat(CurrentFileName(2)) ~= NumberOfImageSets)
                LoadedRawImage = aviread(fullfile(Pathname, char(CurrentFileName(1))), (IndexLocation));
                LoadedImage = im2double(LoadedRawImage.cdata);
                else
                LoadedRawImage = aviread(fullfile(Pathname, char(CurrentFileName(1))), (IndexLocation-1));
                LoadedImage = im2double(LoadedRawImage.cdata);
                end
            elseif strcmpi(FileFormat,'stk movies') == 1
                LoadedRawImage = CPtiffread(fullfile(Pathname, char(CurrentFileName(1))), cell2mat(CurrentFileName(2)));
                LoadedImage = im2double(LoadedRawImage.data);
            elseif (strcmpi(FileFormat,'tif,tiff,flex movies') == 1)
                LoadedRawImage = CPimread(fullfile(Pathname, char(CurrentFileName(1))), cell2mat(CurrentFileName(2)));
                LoadedImage = im2double(LoadedRawImage);                
            end
            %%% Saves the original movie file name to the handles
            %%% structure.  The field is named appropriately based on
            %%% the user's input, in the Pipeline substructure so that
            %%% this field will be deleted at the end of the analysis
            %%% batch.
            fieldname = ['Filename', ImageName{n}];
            [SubdirectoryPathName,BareFileName,ext] = fileparts(char(CurrentFileName(1))); %#ok Ignore MLint
            CurrentFileNameWithFrame = [BareFileName, '_', num2str(cell2mat(CurrentFileName(2))),ext];
            CurrentFileNameWithFrame = fullfile(SubdirectoryPathName, CurrentFileNameWithFrame);
            
            %%% Saves the loaded image to the handles structure.  The field is named
            %%% appropriately based on the user's input, and put into the Pipeline
            %%% substructure so it will be deleted at the end of the analysis batch.
            handles.Pipeline.(fieldname)(SetBeingAnalyzed) = {CurrentFileNameWithFrame};
            handles.Pipeline.(ImageName{n}) = LoadedImage;
        catch 
            CPerrorImread(ModuleName, n);
        end % Goes with: catch
        FileNames(n) = {CurrentFileNameWithFrame};
    end
end

if strcmp(ImageOrMovie,'Image')        
    % Check if any of the files are missing (i.e., zero)
    uniqueImageSize = ImageSizes(~MissingFileIdx);
    uniqueImageSize = unique(cat(1,uniqueImageSize{:}),'rows');
    if any(MissingFileIdx)
        if size(uniqueImageSize,1) ~= 1,
            CPerror('There are image files missing in the specified directory and the original size of the image cannot be inferred.');
        else
            % If there are siblings, create a zero matrix with the same size
            % in place of the missing file
            handles.Pipeline.(ImageName{MissingFileIdx}) = zeros(uniqueImageSize);
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber);
    % Remove uicontrols from last cycle
    delete(findobj(ThisModuleFigureNumber,'tag','TextUIControl'));
    
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure('','NarrowText',ThisModuleFigureNumber)
    end
    for n = 1:length(ImageName)
        %%% Activates the appropriate figure window.
        currentfig = CPfigure(handles,'Text',ThisModuleFigureNumber);
        if iscell(ImageName)
            TextString = [ImageName{n},': ',FileNames{n}];
        else
            TextString = [ImageName,': ',FileNames];
        end
        uicontrol(currentfig,'style','text','units','normalized','fontsize',handles.Preferences.FontSize,'HorizontalAlignment','left','string',TextString,'position',[.05 .85-(n-1)*.15 .95 .1],'BackgroundColor',[.7 .7 .9],'tag','TextUIControl');
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

handles = CPsaveFileNamesToHandles(handles, ImageName, Pathname, FileNames);
% If images checked, record QC measurements
if strcmpi(CheckImageSets,'Yes')
    if isfield(handles.Pipeline,'idxUnmatchedFiles') %% If no tokens are defined, this field won't exist, and the check is cancelled
        handles = CPaddmeasurements(handles, 'Image', CPjoinstrings('QualityControl','isImageUnmatched'), handles.Pipeline.idxUnmatchedFiles{SetBeingAnalyzed});
    end
    if isfield(handles.Pipeline,'idxDuplicateFiles') %% If no tokens are defined, this field won't exist, and the check is cancelled
        handles = CPaddmeasurements(handles, 'Image', CPjoinstrings('QualityControl','isImageDuplicated'),handles.Pipeline.idxDuplicateFiles{SetBeingAnalyzed});
    end
end