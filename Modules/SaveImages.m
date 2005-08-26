function handles = SaveImages(handles)

% Help for the Save Images module:
% Category: File Handling
%
% This module allows you to save images to the hard drive.  Any of the
% processed images created by CellProfiler during the analysis can be
% saved. SaveImages can also be used as a file format converter by
% loading files in their original format and then saving them in an
% alternate format.  Please note that this module works for the few
% cases we have tried, but you may run into difficulties when dealing
% with images that are not 8 bit.  For example, you may wish to alter
% the code to handle 16 bit images.  These features will hopefully be
% added soon.
%
% If you want to save images that are produced by other modules but
% that are not given an official name in the settings boxes for that
% module, alter the code for the module to save those images to the
% handles structure and then use the Save Images module.
% The code should look like this:
% fieldname = ['SomethingDescriptive(optional)',ImageorObjectNameFromSettingsBox];
% handles.Pipeline.(fieldname) = ImageProducedBytheModule;
% Example 1:
% fieldname = ['Segmented', ObjectName];
% handles.Pipeline.(fieldname) = SegmentedObjectImage;
% Example 2:
% fieldname = CroppedImageName;
% handles.Pipeline.(fieldname) = CroppedImage;
%
% Special notes for saving in movie format (avi):
% The movie will be saved after the last image set is processed. You
% have the option to also save the movie periodically during image
% processing, so that the partial movie will be available in case
% image processing is canceled partway through. Saving movies in avi
% format is quite slow, so you can enter a number to save the movie
% after every Nth image set. For example, entering a 1 will save the
% movie after every image set, so that if image analysis is aborted,
% the movie up to that point will be saved. Saving large movie files
% is time-consuming, so it may be better to save after every 10th
% image set, for example. If you are processing multiple movies,
% especially movies in subdirectories, you should save after every
% image set (and also, be aware that this module has not been
% thoroughly tested under those conditions). Note also that the movie
% data is stored in the handles.Pipeline.Movie structure of the output
% file, so you can retrieve the movie data there in case image
% processing is aborted. When working with very large movies, you may
% also want to save the CellProfiler output file every Nth image set
% to save time, because the entire movie is stored in the output file.
% See the SpeedUpCellProfiler module. This module has not been
% extensively tested, particularly for how it handles color images and
% how it handles images coming from subdirectories, multiple incoming
% movie files, or filenames made by numerical increments. At the time
% this module was written, Matlab was only capable of saving in
% uncompressed avi format (at least on the UNIX platform), which is
% time and space-consuming. You should convert the results to a
% compressed movie format, like .mov using third-party software. For
% suggested third-party software, see the help for the LoadMovies
% modules.
%
% See also <nothing relevant>

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

drawnow

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%

%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

%textVAR01 = What did you call the images you want to save?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu custom

%textVAR02 = Which image's original filename do you want to use as a base to create the new file name? Type N to use sequential numbers.
%infotypeVAR02 = imagegroup
ImageFileName = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu custom

%textVAR03 = Enter text to append to the image name, or leave "\" to keep the name the same except for the file extension.
%defaultVAR03 = \
Appendage = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = In what file format do you want to save images?
%defaultVAR04 = tif
FileFormat = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%pathnametextVAR05 = Enter the pathname to the directory where you want to save the images.
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

%textVAR08 = At what point in the pipeline do you want to save the image? When saving in movie format, choose Every cycle.
%choiceVAR08 = Every cycle
%choiceVAR08 = First cycle
%choiceVAR08 = Last cycle
SaveWhen = char(handles.Settings.VariableValues{CurrentModuleNum,8});
%inputtypeVAR08 = popupmenu

%textVAR09 = If you are only saving the image once (e.g. last or first option), enter the filename to use (with no extension). To use the automatically determined filename (derived from the source images), enter A.
%defaultVAR09 = A
OverrideFileName = char(handles.Settings.VariableValues{CurrentModuleNum,9});

%textVAR10 = If you are saving in movie format, do you want to save the movie only after the last image set is processed (enter 'L'), or after every Nth image set (1,2,3...)? Saving movies is time-consuming. See the help for this module for more details.
%defaultVAR10 = L
SaveMovieWhen = char(handles.Settings.VariableValues{CurrentModuleNum,10});

%textVAR11 = Do you want to rescale the images to use a full 8 bit (256 graylevel) dynamic range (Y or N)?
%choiceVAR11 = No
%choiceVAR11 = Yes
RescaleImage = char(handles.Settings.VariableValues{CurrentModuleNum,11});
%inputtypeVAR11 = popupmenu

%textVAR12 = For grayscale images, specify the colormap to use (e.g. gray, jet, bone) if you are saving movie (avi) files.
%defaultVAR12 = gray
ColorMap = char(handles.Settings.VariableValues{CurrentModuleNum,12});

%textVAR13 = Update file names within CellProfiler?
%choiceVAR13 = Yes
%choiceVAR13 = No
UpdateFileOrNot = char(handles.Settings.VariableValues{CurrentModuleNum,13});
%inputtypeVAR13 = popupmenu

%textVAR14 = Warning! It is possible to overwrite existing files using this module!

%%%VariableRevisionNumber = 10

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%

if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
    %%% Determines the figure number.
    fieldname = ['FigureNumberForModule',CurrentModule];
    ThisModuleFigureNumber = handles.Current.(fieldname);
    %%% The figure window is closed since there is nothing to display.
    try close(ThisModuleFigureNumber)
    end
end
drawnow

%%% The module is only carried out if this is the appropriate set being
%%% analyzed, or if the user wants it done every time.
if (strncmpi(SaveWhen,'E',1) == 1) | (strncmpi(SaveWhen,'F',1) == 1 && handles.Current.SetBeingAnalyzed == 1) | (strncmpi(SaveWhen,'L',1) == 1 && handles.Current.SetBeingAnalyzed == handles.Current.NumberOfImageSets)

    if strcmp(FileDirectory,'.') == 1
        FileDirectoryToSave = handles.Current.DefaultOutputDirectory;
    elseif strcmpi(FileDirectory,'I') == 1
        FileDirectoryToSave = handles.Current.DefaultImageDirectory;
    elseif strcmpi(FileDirectory,'S') == 1
        %%% If 'S', then the directory name will be determined below,
        %%% when the file name is retrieved.
    else FileDirectoryToSave = FileDirectory;
        %%% If none of the above, the user must have typed an actual
        %%% path, which will be used.
    end

    %%% Retrieves the image you want to analyze and assigns it to a variable,
    %%% "Image".
    %%% Checks whether image has been loaded.
    if isfield(handles.Pipeline, ImageName) == 0,
        %%% If the image is not there, the module tries in a field named
        %%% 'Segmented' which would be produced by an Identify module.
        if isfield(handles.Pipeline, ['Segmented',ImageName])==1,
            ImageName = ['Segmented',ImageName];
        else %%% If the image is not there, an error message is produced.  The error
            %%% is not displayed: The error function halts the current function and
            %%% returns control to the calling function (the analyze all images
            %%% button callback.)  That callback recognizes that an error was
            %%% produced because of its try/catch loop and breaks out of the image
            %%% analysis loop without attempting further modules.
            error(['Image processing was canceled because the Save Images module could not find the input image.  It was supposed to be named ', ImageName, ' but neither that nor an image with the name ', ['Segmented',ImageName] , ' exists.  Perhaps there is a typo in the name.'])
        end
    end
    Image = handles.Pipeline.(ImageName);

    if strncmpi(RescaleImage,'Y',1) == 1
        LOW_HIGH = stretchlim(Image,0);
        Image = imadjust(Image,LOW_HIGH,[0 1]);
    end

    %%% Checks whether the file format the user entered is readable by Matlab.
    if strcmp(FileFormat(1),'.')
        FileFormat = FileFormat(2:end);
    end
    IsFormat = imformats(FileFormat);
    if isempty(IsFormat) == 1
        if strcmpi(FileFormat,'mat') && strcmpi(FileFormat,'avi') &&strcmpi(FileFormat,'dib')
            error('The image file type entered in the Save Images module is not recognized by Matlab. For a list of recognizable image file formats, type "CPimread" (no quotes) at the command line in Matlab.')
        end
    end

    %%% Creates the file name automatically, if the user requested.
    if strcmpi(OverrideFileName,'A') == 1
        %%% Checks whether the appendage is going to result in a name with
        %%% spaces.
        Spaces = isspace(Appendage);
        if any(Spaces) == 1
            error('Image processing was canceled because you have entered one or more spaces in the box of text to append to the image name in the Save Images module.')
        end
        %%% Determines the file name.
        if strcmp(upper(ImageFileName), 'N') == 1
            %%% Sets the filename to be sequential numbers.
            FileName = num2str(handles.Current.SetBeingAnalyzed);
            CharFileName = char(FileName);
            BareFileName = CharFileName;
            if strcmpi(FileDirectory,'S') == 1
                %%% Determine the filename of the image to be analyzed, just in order to find the subdirectory.
                fieldname = ['Filename', ImageFileName];
                IGNOREFileName = handles.Pipeline.(fieldname)(handles.Current.SetBeingAnalyzed);
                %%% If subdirectories are being analyzed, the filename will
                %%% include subdirectory pathnames.
                [SubdirectoryPathName,IGNOREBAREFILENAME,ext,versn] = fileparts(IGNOREFileName{1});
                FileDirectoryToSave = fullfile(handles.Current.DefaultImageDirectory,SubdirectoryPathName);
            end
        elseif strcmpi(FileFormat,'avi') == 1
            %%% If it's a movie, we don't want to use different
            %%% filenames for each image set.

            %%% If it's coming from a LoadMovies module:
            try
                %%% Determine the filename of the image to be analyzed.
                fieldname = ['FileList', ImageFileName];
                FileName = handles.Pipeline.(fieldname){1}{handles.Current.SetBeingAnalyzed}
                %%% If subdirectories are being analyzed, the filename will
                %%% include subdirectory pathnames.
                [SubdirectoryPathName,BareFileName,ext,versn] = fileparts(FileName);
                if strcmpi(FileDirectory,'S') == 1
                    FileDirectoryToSave = fullfile(handles.Current.DefaultImageDirectory,SubdirectoryPathName);
                end
            catch
                %%% If it's coming from a LoadImages module:

                %%% NEED TO JUST PUT ALL IMAGES INTO ONE MOVIE>

                %%% Determine the filename of the image to be analyzed.
                fieldname = ['Filename', ImageFileName];
                %%% Here, note that the first filename is always used;
                %%% it does not increment by setbeinganalyzed.
                FileName = handles.Pipeline.(fieldname)(1);
                %%% If subdirectories are being analyzed, the filename will
                %%% include subdirectory pathnames.
                [SubdirectoryPathName,BareFileName,ext,versn] = fileparts(FileName{1});
                if strcmpi(FileDirectory,'S') == 1
                    FileDirectoryToSave = fullfile(handles.Current.DefaultImageDirectory,SubdirectoryPathName);
                end

            end

        else
            %%% Determine the filename of the image to be analyzed.
            fieldname = ['Filename', ImageFileName];
            FileName = handles.Pipeline.(fieldname)(handles.Current.SetBeingAnalyzed);
            %%% If subdirectories are being analyzed, the filename will
            %%% include subdirectory pathnames.
            [SubdirectoryPathName,BareFileName,ext,versn] = fileparts(FileName{1});
            if strcmpi(FileDirectory,'S') == 1
                FileDirectoryToSave = fullfile(handles.Current.DefaultImageDirectory,SubdirectoryPathName);
            end
        end
        %%% Assembles the new image name.
        if strcmp(Appendage, '\')
            NewImageName = [BareFileName '.' FileFormat];
        elseif strcmp(Appendage(1),'\')
            NewImageName = [Appendage(2:end) '.' FileFormat];
        else
            NewImageName = [BareFileName Appendage '.' FileFormat];
        end
    else
        %%% Otherwise, use the filename the user entered.
        NewImageName = [OverrideFileName,'.',FileFormat];
        Spaces = isspace(NewImageName);
        if any(Spaces) == 1
            error('Image processing was canceled because you have entered one or more spaces in the proposed filename in the Save Images module.')
        end
    end

    %%% Makes sure that the File Directory specified by the user exists.
    if isdir(FileDirectoryToSave) ~= 1
        error(['Image processing was canceled because the specified directory "', FileDirectoryToSave, '" in the Save Images module does not exist.']);
    end

    if strcmp(UpdateFileOrNot,'Yes')
        handles.Pipeline.(['FileList',ImageName])(handles.Current.SetBeingAnalyzed) = {NewImageName};
        handles.Pipeline.(['Pathname',ImageName]) = FileDirectoryToSave;
    end

    NewFileAndPathName = fullfile(FileDirectoryToSave, NewImageName);
    if strcmpi(CheckOverwrite,'Y') == 1 && strcmpi(FileFormat,'avi') ~= 1
        %%% Checks whether the new image name is going to overwrite the
        %%% original file. This check is not done here if this is an avi
        %%% (movie) file, because otherwise the check would be done on each
        %%% frame of the movie.
        if exist(NewFileAndPathName) == 2
            Answer = CPquestdlg(['The settings in the Save Images module will cause the file "', NewFileAndPathName,'" to be overwritten. Do you want to continue or cancel?'], 'Warning', 'Continue','Cancel','Cancel');
            if strcmp(Answer,'Cancel') == 1
                error('Image processing was canceled')
            end
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% SAVE IMAGE TO HARD DRIVE %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    FileSavingParameters = [];
    if strcmpi(BitDepth,'8') ~=1
        FileSavingParameters = [',''bitdepth'', ', BitDepth,''];
        %%% In jpeg format at 12 and 16 bits, the mode must be set to
        %%% lossless to avoid failure of the imwrite function.
        if strcmpi(FileFormat,'jpg') == 1 | strcmpi(FileFormat,'jpeg') == 1
            FileSavingParameters = [FileSavingParameters, ',''mode'', ''lossless'''];
        end
    end

    if strcmpi(FileFormat,'mat') == 1
        try eval(['save(''',NewFileAndPathName, ''',''Image'')']);
        catch
            error(['In the save images module, the image could not be saved to the hard drive for some reason. Check your settings.  The error is: ', lasterr])
        end
    elseif strcmpi(FileFormat,'avi') == 1
        if handles.Current.SetBeingAnalyzed == 1
            if strcmpi(CheckOverwrite,'Y') == 1
                %%% Checks whether the new image name is going to overwrite
                %%% the original file, but only on the first image set,
                %%% because otherwise the check would be done on each frame
                %%% of the movie.
                if exist(NewFileAndPathName) == 2
                    Answer = CPquestdlg(['The settings in the Save Images module will cause the file "', NewFileAndPathName,'" to be overwritten. Do you want to continue or cancel?'], 'Warning', 'Continue','Cancel','Cancel');
                    if strcmp(Answer,'Cancel') == 1
                        error('Image processing was canceled')
                    end
                end
            end
        end
        fieldname = ['Movie', ImageName];
        if handles.Current.SetBeingAnalyzed == 1
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
        %%% appropriate image set.
        try MovieSavingIncrement = str2double(SaveMovieWhen);
            MovieIsNumber = 1;
        catch MovieIsNumber = 0;
        end
        %%% Initializes this value in order to determine whether it's
        %%% time to save the movie file.
        TimeToSave = 0;
        if MovieIsNumber == 1
            if rem(handles.Current.SetBeingAnalyzed,MovieSavingIncrement) == 0 | handles.Current.SetBeingAnalyzed == handles.Current.NumberOfImageSets
                TimeToSave = 1;
            end
        else
            if strncmpi(SaveMovieWhen,'L',1) == 1
                if handles.Current.SetBeingAnalyzed == handles.Current.NumberOfImageSets
                    TimeToSave = 1;
                end
            end
        end

        if TimeToSave == 1
            %%% If the image is an RGB image (3-d), the colormaps
            %%% have been calculated for each frame and are
            %%% already stored in Movie.colormap.
            if IsRGB == 1
                try movie2avi(Movie,NewFileAndPathName)
                catch error('There was an error saving the movie to the hard drive in the SaveImages module.')
                end
            else
                %%% Specifying the size of the colormap is critical
                %%% to prevent a bunch of annoying weird errors. I assume
                %%% the avi format is always 8-bit (=256 levels).
                eval(['ChosenColormap = colormap(',ColorMap,'(256));']);
                try movie2avi(Movie,NewFileAndPathName,'colormap',ChosenColormap)
                catch error('There was an error saving the movie to the hard drive in the SaveImages module.')
                end
            end
        end


%%%%%%% THIS IS FUNCTIONAL, BUT SLOW >>>>>>>>>>>
%%% It opens the entire file from the hard drive and re-saves the whole
%%% thing.
%         %%% If this movie file already exists, open it.
%         try
%
%             Movie = aviread(NewFileAndPathName);
%             NumberExistingFrames = size(Movie,2);
%          %%% If the movie does not yet exist, create the colormap
%          %%% field as empty to prevent errors when trying to save as a
%          %%% movie.
%
%
%         catch   Movie.colormap = [];
%             NumberExistingFrames = 0;
%         end
%         %%% Adds the image as the last frame in the movie.
%         Movie(1,NumberExistingFrames+1).cdata = Image*256;
%         % Movie(1,NumberExistingFrames+1).colormap = colormap(gray(256));
%         %%% Saves the Movie under the appropriate file name.
%         movie2avi(Movie,NewFileAndPathName,'colormap',colormap(gray(256)))

%%% TRYING TO FIGURE OUT HOW TO ADD COLORMAP INFO TO RGB IMAGES>>>
            %%% If the image is an RGB image (3-d), convert it to an
            %%% indexed image plus a colormap to allow saving as a
            %%% movie.
            %%% I THINK ONLY ONE COLORMAP IS ALLOWED FOR THE WHOLE
            %%% MOVIE.>>>>>>>>>>>>> MAYBE NEED TO SPECIFY A SINGLE COLORMAP
            %%% HERE RATHER THAN HAVING IT AUTO CALCULATED.
%             [Image,map] = rgb2ind(Image,256,'nodither');
%             Movie(NumberExistingFrames+1).colormap = map;
%             %%% Adds the image as the last frame in the movie.
%             %%% MAYBE I SHOULD BE USING im2frame??>>>>>>>>>>>
%             %%%
%             Movie(NumberExistingFrames+1).cdata = Image;
          %   [Image,map] = rgb2ind(Image,256,'nodither');
            %%% Adds the image as the last frame in the movie.
            %%% MAYBE I SHOULD BE USING im2frame??>>>>>>>>>>>
            %%%
           % Movie(NumberExistingFrames+1).cdata = Image;



% %%% ATTEMPT TO ONLY SAVE AT THE END>>>
% %%% RIGHT NOW IT WON"T WORK IF WE ARE TRYING TO OVERWRITE AN OLD FILE.
%
% %%% See if this movie file already exists. If so,
% %%% retrieve the movie data accumulated so far from handles.
%
% if handles.Current.SetBeingAnalyzed == 1
%     Movie = avifile(NewFileAndPathName);
%     fieldname = ['Movie', ImageName];
%     handles.Pipeline.(fieldname) =  Movie;
% end
%
% %%% Add the frame to the movie.
% fieldname = ['Movie', ImageName];
% Movie = handles.Pipeline.(fieldname);
% Movie = addframe(Movie,Image);
%
% %%% Closes the file.
% if handles.Current.SetBeingAnalyzed == handles.Current.NumberOfImageSets
%     Movie = close(Movie);
% end


%%% FAILED ATTEMPT TO USE ADDFRAME.
% %%% See if this movie file already exists. If so, just
%         %%% retrieve the AviHandle from handles
%         SUCCESSFULHANDLERETIREVAL = 0;
%         if exist(NewFileAndPathName) ~= 0
%             try
%                 fieldname = ['AviHandle', ImageName];
%                 AviHandle = handles.Pipeline.(fieldname)
%                 AviHandle = addframe(AviHandle,Image);
% %                AviHandle = close(AviHandle);
%                 SUCCESSFULHANDLERETIREVAL = 1;
%             end
%         end
%
%         if SUCCESSFULHANDLERETIREVAL == 0
%             %%% If the movie does not exist already, create it using
%             %%% the avifile function and put the AviHandle into the handles.
%
%             AviHandle = avifile(NewFileAndPathName);
%             AviHandle = addframe(AviHandle,Image);
%             AviHandle = close(AviHandle);
%
%             fieldname = ['AviHandle', ImageName];
%             handles.Pipeline.(fieldname) =  AviHandle;
%         end
%

    else
        try eval(['imwrite(Image, NewFileAndPathName, FileFormat', FileSavingParameters,')']);
        catch
            error(['In the save images module, the image could not be saved to the hard drive for some reason. Check your settings, and see the Matlab imwrite function for details about parameters for each file format.  The error is: ', lasterr])
        end
    end
end