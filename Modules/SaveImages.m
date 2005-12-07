function handles = SaveImages(handles)

% Help for the Save Images module:
% Category: File Processing
%
% SHORT DESCRIPTION:
% Saves any image produced during the image analysis, in any image format.
% *************************************************************************
%
% Because CellProfiler usually performs many image analysis steps on many
% groups of images, it does *not* save any of the resulting images to the
% hard drive unless you use the Save Images module to do so. Any of the
% processed images created by CellProfiler during the analysis can be
% saved using Save Images.
%
% The Save Images module can also be used as a file format converter
% by loading files in their original format and then saving them in an
% alternate format. 
%
% Please note that this module works for the cases we have tried, but it
% has not been extensively tested, particularly for how it handles color
% images, non-8 bit images, images coming from subdirectories, multiple
% incoming movie files, or filenames made by numerical increments.
%
% Update file names within CellProfiler? 
% This allows downstream modules (e.g. Create Web Page) to look up the
% newly saved files on the hard drive. Normally, whatever files are present
% on the hard drive when CellProfiler processing begins (and when the
% LoadImages module processes its first cycle) are the only files that are
% accessible within CellProfiler. This setting allows the newly saved files
% to be accessible to downstream modules. This setting might yield unusual
% consequences if you are using the Save Images module to save an image
% directly as loaded (e.g. using the Save Images module to convert file
% formats), because it will, in some places in the output file, overwrite
% the file names of the loaded files with the file names of the the saved
% files. Because this function is rarely needed and may introduce
% complications, the default answer is "No".
%
% Special notes for saving in movie format (avi):
% The movie will be saved after the last cycle is processed. You have the
% option to also save the movie periodically during image processing, so
% that the partial movie will be available in case image processing is
% canceled partway through. Saving movies in avi format is quite slow, so
% you can enter a number to save the movie after every Nth cycle. For
% example, entering a 1 will save the movie after every cycle. When working
% with very large movies, you may also want to save the CellProfiler output
% file every Nth cycle to save time, because the entire movie is stored in
% the output file (this may only be the case if you are working in
% diagnostic mode, see Set Preferences). See the Speed Up CellProfiler
% module. If you are processing multiple movies, especially movies in
% subdirectories, you should save after every cycle (and also, be aware
% that this module has not been thoroughly tested under those conditions).
% Note also that the movie data is stored in the handles.Pipeline.Movie
% structure of the output file, so you can retrieve the movie data there in
% case image processing is aborted. At the time this module was written,
% Matlab was only capable of saving in uncompressed avi format (at least on
% the UNIX platform), which is time and space-consuming. You should convert
% the results to a compressed movie format, like .mov using third-party
% software. For suggested third-party software, see the help for the Load
% Images module.
%
% Developer's note:
% If you want to save images that are produced by other modules but that
% are not given an official name in the settings boxes for that module,
% alter the code for the module to save those images to the handles
% structure and then use the Save Images module.
% The code should look like this:
% fieldname = ['SomethingDescriptive(optional)',ImageorObjectNameFromSettingsBox];
% handles.Pipeline.(fieldname) = ImageProducedBytheModule;
% Example 1:
% fieldname = ['Segmented', ObjectName];
% handles.Pipeline.(fieldname) = SegmentedObjectImage;
% Example 2:
% fieldname = CroppedImageName;
% handles.Pipeline.(fieldname) = CroppedImage;

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne E. Carpenter
%   Thouis Ray Jones
%   In Han Kang
%   Ola Friman
%   Steve Lowe
%   Joo Han Chang
%   Colin Clarke
%   Mike Lamprecht
%   Susan Ma
%   Wyman Li
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

%%%%%%%%%%%%%%%%%%%%%%%%   WARNING   %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%If you change anything here, make sure the image tool SaveImageAs is
%consistent, in CPimagetool.
%%%%%%%%%%%%%%%%%%%%%%%%   WARNING   %%%%%%%%%%%%%%%%%%%%%%%%%%%%

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the images you want to save (If you would like to save a figure, enter the module number here)?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu custom

%textVAR02 = Which images' original filenames do you want use as a base for these new images' filenames? Your choice MUST be images loaded directly with a Load module. Type N to use sequential numbers or any word to use the same base name (for avi (movies) you need to have the same base name).
%infotypeVAR02 = imagegroup
ImageFileName = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu custom

%textVAR03 = Enter text to append to the image name, type N to use sequential numbers, or leave "\" to not append anything.
%defaultVAR03 = \
Appendage = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = In what file format do you want to save images (figures must be saved as fig, which is only openable in Matlab)?
%choiceVAR04 = bmp
%choiceVAR04 = gif
%choiceVAR04 = hdf
%choiceVAR04 = jpg
%choiceVAR04 = jpeg
%choiceVAR04 = pbm
%choiceVAR04 = pcx
%choiceVAR04 = pgm
%choiceVAR04 = png
%choiceVAR04 = pnm
%choiceVAR04 = ppm
%choiceVAR04 = ras
%choiceVAR04 = tif
%choiceVAR04 = tiff
%choiceVAR04 = xwd
%choiceVAR04 = avi
%choiceVAR04 = fig
%choiceVAR04 = mat
%inputtypeVAR04 = popupmenu
FileFormat = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%%%%%%%%%%%%%%%%%%%%%%% NOTE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% WE CANNOT PUT DIB OR STK HERE, BECAUSE WE CANNOT SAVE IN THOSE FORMATS
%%% These formats listed are the only possible ones, according to the
%%% imwrite function, plus avi, mat, and fig for which this module contains
%%% special code for handling.

%pathnametextVAR05 = Enter the pathname to the directory where you want to save the images.  Type period (.) for default output directory.
FileDirectory = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = Enter the bit depth at which to save the images (Note: some image formats do not support saving at a bit depth of 12 or 16; see Matlab's imwrite function for more details.)
%choiceVAR06 = 8
%choiceVAR06 = 12
%choiceVAR06 = 16
BitDepth = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%inputtypeVAR06 = popupmenu

%textVAR07 = Do you want to always check whether you will be overwriting a file when saving images?
%choiceVAR07 = Yes
%choiceVAR07 = No
CheckOverwrite = char(handles.Settings.VariableValues{CurrentModuleNum,7});
%inputtypeVAR07 = popupmenu

%textVAR08 = At what point in the pipeline do you want to save the image? When saving in avi (movie) format, choose Every cycle.
%choiceVAR08 = Every cycle
%choiceVAR08 = First cycle
%choiceVAR08 = Last cycle
SaveWhen = char(handles.Settings.VariableValues{CurrentModuleNum,8});
%inputtypeVAR08 = popupmenu

%textVAR09 = If you are saving in avi (movie) format, do you want to save the movie only after the last cycle is processed (enter 'L'), or after every Nth cycle (1,2,3...)? Saving movies is time-consuming. See the help for this module for more details.
%defaultVAR09 = L
SaveMovieWhen = char(handles.Settings.VariableValues{CurrentModuleNum,9});

%textVAR10 = Do you want to rescale the images to use a full 8 bit (256 graylevel) dynamic range (Y or N)? Use the RescaleIntensity module for other rescaling options.
%choiceVAR10 = No
%choiceVAR10 = Yes
RescaleImage = char(handles.Settings.VariableValues{CurrentModuleNum,10});
%inputtypeVAR10 = popupmenu

%textVAR11 = For grayscale images, specify the colormap to use (e.g. gray, jet, bone) if you are saving movie (avi) files.
%defaultVAR11 = gray
ColorMap = char(handles.Settings.VariableValues{CurrentModuleNum,11});

%textVAR12 = Enter any optional parameters here ('Quality',1 or 'Quality',100 etc.) or leave / for no optional parameters.
%defaultVAR12 = /
OptionalParameters = char(handles.Settings.VariableValues{CurrentModuleNum,12});

%textVAR13 = Update file names within CellProfiler? See help for details.
%choiceVAR13 = No
%choiceVAR13 = Yes
UpdateFileOrNot = char(handles.Settings.VariableValues{CurrentModuleNum,13});
%inputtypeVAR13 = popupmenu

%textVAR14 = Warning! It is possible to overwrite existing files using this module!

%%%VariableRevisionNumber = 12

%%%%%%%%%%%%%%%%%%%%%%%
%%% FILE PROCESSING %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;

if strcmp(SaveWhen,'Every cycle') || strcmp(SaveWhen,'First cycle') && SetBeingAnalyzed == 1 || strcmp(SaveWhen,'Last cycle') && SetBeingAnalyzed == handles.Current.NumberOfImageSets
    try
        if iscell(handles.Pipeline.(['Filename', ImageFileName]))
            FileName = handles.Pipeline.(['Filename', ImageFileName]){SetBeingAnalyzed};
        else
            FileName = handles.Pipeline.(['Filename', ImageFileName]);
        end
        [temp FileName] = fileparts(FileName); %#ok Ignore MLint
    catch
        if strcmp(ImageFileName,'N')
            FileName = TwoDigitString(SetBeingAnalyzed);
        else
            Spaces = isspace(FileName);
            if any(Spaces)
                error(['Image processing was canceled in the ', ModuleName, ' module because you have entered one or more spaces in the box of text for the filename of the image.'])
            end
            FileName = ImageFileName;
        end
    end

    if strcmp(Appendage,'N')
        FileName = [FileName TwoDigitString(SetBeingAnalyzed)];
    else
        if ~strcmp(Appendage,'\')
            Spaces = isspace(Appendage);
            if any(Spaces)
                error(['Image processing was canceled in the ', ModuleName, ' module because you have entered one or more spaces in the box of text for the filename of the image.'])
            end
            FileName = [FileName Appendage];
        end
    end

    FileName = [FileName '.' FileFormat];

    if strncmp(FileDirectory,'.',1)
        if strcmp(FileDirectory,'.')
            PathName = handles.Current.DefaultOutputDirectory;
        else
            PathName = fullfile(handles.Current.DefaultOutputDirectory,FileDirectory(2:end));
        end
    else
        PathName = FileDirectory;
    end

    %%% Makes sure that the File Directory specified by the user exists.
    if ~isdir(PathName)
        error(['Image processing was canceled in the ', ModuleName, ' module because the specified directory "', PathName, '" does not exist.']);
    end

    if ~strcmp(FileFormat,'fig')
        if ~isfield(handles.Pipeline, ImageName)
            error(['Image processing was canceled in the ', ModuleName, ' module because it could not find the input image.  It was supposed to be named ', ImageName, ' but that does not exist.'])
        end
        Image = handles.Pipeline.(ImageName);

        if max(Image(:)) > 1 || min(Image(:)) < 0
            CPwarndlg(['The images you have loaded in the ', ModuleName, ' module are outside the 0-1 range, and you may be losing data.'],'Outside 0-1 Range','replace');
        end

        if strcmp(RescaleImage,'Yes')
            LOW_HIGH = stretchlim(Image,0);
            Image = imadjust(Image,LOW_HIGH,[0 1]);
        end

        %%% Checks whether the file format the user entered is readable by Matlab.
        if ~any(strcmp(FileFormat,CPimread)) && ~strcmp(FileFormat,'avi')
            error(['Image processing was canceled in the ', ModuleName, ' module because the image file type entered is not recognized by Matlab. For a list of recognizable image file formats, type "CPimread" (no quotes) at the command line in Matlab, or see the help for this module.'])
        end
    end
    
    %%% Creates the fields that the LoadImages module normally creates when
    %%% loading images.
    if strcmp(UpdateFileOrNot,'Yes')
        %%% Stores file and path name data in handles.Pipeline.
        handles.Pipeline.(['FileList',ImageName])(SetBeingAnalyzed) = {FileName};
        handles.Pipeline.(['Pathname',ImageName]) = PathName;
        handles.Pipeline.(['Filename',ImageName])(SetBeingAnalyzed) = {FileName};

        %%% Sets the initial values, which may get added to if we need to
        %%% append in the next step.
        FileNames = FileName;
        PathNames = PathName;
        FileNamesText = ['Filename ', ImageName];
        PathNamesText = ['Path ', ImageName];
        
        %%% Stores file and path name data in handles.Measurements. Since
        %%% there may be several load/save modules in the pipeline which
        %%% all write to the handles.Measurements.Image.FileName field, we
        %%% store filenames in an "appending" style. Here we check if any
        %%% of the modules above the current module in the pipeline has
        %%% written to handles.Measurements.Image.Filenames. Then we should
        %%% append the current filenames and path names to the already
        %%% written ones. If this is the first module to put anything into
        %%% the handles.Measurements.Image structure (though I think this
        %%% is not really possible for SaveImages), then this section is
        %%% skipped and the FileNamesText fields are created with their
        %%% initial entry coming from this module.
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
    end

    FileAndPathName = fullfile(PathName, FileName);

    if strcmp(CheckOverwrite,'Yes') && ~strcmp(FileFormat,'avi')
        %%% Checks whether the new image name is going to overwrite the
        %%% original file. This check is not done here if this is an avi
        %%% (movie) file, because otherwise the check would be done on each
        %%% frame of the movie.
        if exist(FileAndPathName) == 2 %#ok Ignore MLint
            try
                Answer = CPquestdlg(['The settings in the ', ModuleName, ' module will cause the file "', FileAndPathName,'" to be overwritten. Do you want to continue or cancel?'], 'Warning', 'Continue','Skip Module','Cancel','Cancel');
            catch
                error(['Image processing was canceled in the ', ModuleName, ' module because the settings will cause the file "', FileAndPathName,'" to be overwritten and you have specified to not allow overwriting without confirming. When running on the cluster there is no way to confirm overwriting (no dialog boxes allowed), so image processing was canceled.'])
            end
            if strcmp(Answer,'Skip Module')
                return;
            end
            if strcmp(Answer,'Cancel')
                error(['Image processing was canceled in the ', ModuleName, ' module at your request.'])
            end
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% SAVE IMAGE TO HARD DRIVE %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    drawnow

    FileSavingParameters = [];
    if ~strcmp(BitDepth,'8') && (strcmp(FileFormat,'jpg') || strcmp(FileFormat,'jpeg') || strcmp(FileFormat,'png'))
        FileSavingParameters = [',''bitdepth'', ', BitDepth,''];
        %%% In jpeg format at 12 and 16 bits, the mode must be set to
        %%% lossless to avoid failure of the imwrite function.
        if strcmp(FileFormat,'jpg') || strcmp(FileFormat,'jpeg')
            FileSavingParameters = [FileSavingParameters, ',''mode'', ''lossless'''];
        end
    end

    if ~strcmp(OptionalParameters,'/')
        FileSavingParameters = [',',OptionalParameters,FileSavingParameters];
    end

    if strcmp(FileFormat,'mat')
        try
            eval(['save(''',FileAndPathName,''',''Image'')']);
        catch
            error(['Image processing was canceled in the ', ModuleName, ' module because the image could not be saved to the hard drive for some reason. Check your settings.  The error is: ', lasterr])
        end
    elseif strcmp(FileFormat,'fig')
        if length(ImageName) == 1
            fieldname = ['FigureNumberForModule0',ImageName];
        elseif length(ImageName) == 2
            fieldname = ['FigureNumberForModule',ImageName];
        else
            error(['Image processing was canceled in the ', ModuleName, ' module because the figure number was not in XX format.']);
        end
        FigureHandle = handles.Current.(fieldname); %#ok Ignore MLint
        try
            saveas(FigureHandle,FileAndPathName,'fig');
        catch
            error(['Image processing was canceled in the ', ModuleName, ' module because the figure could not be saved to the hard drive for some reason. Check your settings.  The error is: ', lasterr])
        end
    elseif strcmpi(FileFormat,'avi')
        if SetBeingAnalyzed == 1 &&  strcmp(CheckOverwrite,'Y')
            %%% Checks whether the new image name is going to overwrite
            %%% the original file, but only on the first cycle,
            %%% because otherwise the check would be done on each frame
            %%% of the movie.
            if exist(FileAndPathName) == 2 %#ok Ignore MLint
                try
                    Answer = CPquestdlg(['The settings in the ', ModuleName, ' module will cause the file "', FileAndPathName,'" to be overwritten. Do you want to continue or cancel?'], 'Warning', 'Continue','Cancel','Cancel');
                catch
                    error(['Image processing was canceled in the ', ModuleName, ' module because the settings will cause the file "', FileAndPathName,'" to be overwritten and you have specified to not allow overwriting without confirming. When running on the cluster there is no way to confirm overwriting (no dialog boxes allowed), so image processing was canceled.'])
                end
                if strcmp(Answer,'Cancel')
                    error(['Image processing was canceled in the ', ModuleName, ' module at your request.'])
                end
            end
        end
        fieldname = ['Movie', ImageName];
        if SetBeingAnalyzed == 1
            NumberExistingFrames = 0;
            %%% Preallocates the variable which signficantly speeds processing
            %%% time.
            handles.Pipeline.(fieldname)(handles.Current.NumberOfImageSets) = struct('colormap',[],'cdata',[]);
        else
            Movie = handles.Pipeline.(fieldname);
            NumberExistingFrames = size(Movie,2);
        end
        %%% Determines whether the image is RGB.
        if size(Image,3) == 3
            IsRGB = 1;
        else IsRGB = 0;
        end
        if IsRGB == 1
            Movie(NumberExistingFrames+1) = im2frame(Image);
        else
            %%% For non-RGB images, the colormap will be specified all
            %%% at once later, when the file is saved.
            Movie(NumberExistingFrames+1).colormap = [];
            %%% Adds the image as the last frame in the movie.
            Movie(NumberExistingFrames+1).cdata = Image*256;
        end
        %%% Saves the movie to the handles structure.
        handles.Pipeline.(fieldname) = Movie;

        %%% Saves the Movie under the appropriate file name after the
        %%% appropriate cycle.
        try MovieSavingIncrement = str2double(SaveMovieWhen);
            MovieIsNumber = 1;
        catch MovieIsNumber = 0;
        end
        %%% Initializes this value in order to determine whether it's
        %%% time to save the movie file.
        TimeToSave = 0;
        if MovieIsNumber == 1
            if rem(SetBeingAnalyzed,MovieSavingIncrement) == 0 || SetBeingAnalyzed == handles.Current.NumberOfImageSets
                TimeToSave = 1;
            end
        else
            if strncmpi(SaveMovieWhen,'L',1)
                if SetBeingAnalyzed == handles.Current.NumberOfImageSets
                    TimeToSave = 1;
                end
            end
        end

        if TimeToSave == 1
            %%% If the image is an RGB image (3-d), the colormaps
            %%% have been calculated for each frame and are
            %%% already stored in Movie.colormap.
            if IsRGB == 1
                try movie2avi(Movie,FileAndPathName)
                catch error(['Image processing was canceled in the ', ModuleName, ' module because there was an error saving the movie to the hard drive.'])
                end
            else
                %%% Specifying the size of the colormap is critical
                %%% to prevent a bunch of annoying weird errors. I assume
                %%% the avi format is always 8-bit (=256 levels).
                eval(['ChosenColormap = colormap(',ColorMap,'(256));']);
                %%% It's ok that this shows up as an error in the
                %%% dependency report, I think, because the variable
                %%% ChosenColormap will not exist until the eval function
                %%% is carried out.
                try movie2avi(Movie,FileAndPathName,'colormap',ChosenColormap)
                catch error(['Image processing was canceled in the ', ModuleName, ' module because there was an error saving the movie to the hard drive.'])
                end
            end
        end
    else
        try eval(['imwrite(Image, FileAndPathName, FileFormat', FileSavingParameters,')']);
        catch
            error(['Image processing was canceled in the ', ModuleName, ' module because the image could not be saved to the hard drive for some reason. Check your settings, and see the Matlab imwrite function for details about parameters for each file format.  The error is: ', lasterr])
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% The figure window display is unnecessary for this module, so it is
%%% closed during the starting image cycle.
if SetBeingAnalyzed == handles.Current.StartingImageSet
    ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
    if any(findobj == ThisModuleFigureNumber)
        close(ThisModuleFigureNumber)
    end
end

function twodigit = TwoDigitString(val)
%TwoDigitString is a function like num2str(int) but it returns a two digit
%representation of a string for our purposes.
if ((val > 99) || (val < 0)),
    error(['TwoDigitString: Can''t convert ' num2str(val) ' to a 2 digit number']);
end
twodigit = sprintf('%02d', val);