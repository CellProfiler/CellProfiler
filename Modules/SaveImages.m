function handles = SaveImages(handles)

% Help for the Save Images module:
% Category: File Processing
%
% SHORT DESCRIPTION:
% Save's any image produced during the image analysis, in any image format.
% Can be used as a file format converter to save images in a different
% format than the original.
% *************************************************************************
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
%   Anne Carpenter
%   Thouis Jones
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

%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);
ModuleName = char(handles.Settings.ModuleNames(CurrentModuleNum));

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

%textVAR04 = In what file format do you want to save images (figures must be saved as fig)?
%choiceVAR04 = bmp
%choiceVAR04 = cur
%choiceVAR04 = fts
%choiceVAR04 = fits
%choiceVAR04 = gif
%choiceVAR04 = hdf
%choiceVAR04 = ico
%choiceVAR04 = jpg
%choiceVAR04 = jpeg
%choiceVAR04 = pbm
%choiceVAR04 = pcx
%choiceVAR04 = pgm
%choiceVAR04 = png
%choiceVAR04 = pnm
%choiceVAR04 = ppm
%choiceVAR04 = rad
%choiceVAR04 = tif
%choiceVAR04 = tiff
%choiceVAR04 = xwd
%choiceVAR04 = mat
%choiceVAR04 = fig
%choiceVAR04 = avi
%inputtypeVAR04 = popupmenu
FileFormat = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%%% WE CANNOT PUT DIB OR STK HERE, BECAUSE WE CANNOT SAVE IN THAT FORMAT, RIGHT?

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

%textVAR09 = If you are saving in avi (movie) format, do you want to save the movie only after the last image set is processed (enter 'L'), or after every Nth image set (1,2,3...)? Saving movies is time-consuming. See the help for this module for more details.
%defaultVAR09 = L
SaveMovieWhen = char(handles.Settings.VariableValues{CurrentModuleNum,9});

%textVAR10 = Do you want to rescale the images to use a full 8 bit (256 graylevel) dynamic range (Y or N)?
%choiceVAR10 = No
%choiceVAR10 = Yes
RescaleImage = char(handles.Settings.VariableValues{CurrentModuleNum,10});
%inputtypeVAR10 = popupmenu

%textVAR11 = For grayscale images, specify the colormap to use (e.g. gray, jet, bone) if you are saving movie (avi) files.
%defaultVAR11 = gray
ColorMap = char(handles.Settings.VariableValues{CurrentModuleNum,11});

%textVAR12 = Enter any optional parameter's here ('Quality',1 or 'Quality',100 etc.) or leave / for no optional parameters.
%defaultVAR12 = /
OptionalParameters = char(handles.Settings.VariableValues{CurrentModuleNum,12});

%textVAR13 = Warning! It is possible to overwrite existing files using this module!

%%%VariableRevisionNumber = 11

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
    %%% Determines the figure number.
    fieldname = ['FigureNumberForModule',CurrentModule];
    ThisModuleFigureNumber = handles.Current.(fieldname);
    %%% The figure window is closed since there is nothing to display.
    try close(ThisModuleFigureNumber)
    end
end
drawnow

if strcmp(SaveWhen,'Every cycle') || strcmp(SaveWhen,'First cycle') && handles.Current.SetBeingAnalyzed == 1 || strcmp(SaveWhen,'Last cycle') && handles.Current.SetBeingAnalyzed == handles.Current.NumberOfImageSets
    try
        if iscell(handles.Pipeline.(['Filename', ImageFileName]))
            FileName = handles.Pipeline.(['Filename', ImageFileName]){handles.Current.SetBeingAnalyzed};
        else
            FileName = handles.Pipeline.(['Filename', ImageFileName]);
        end
        [temp FileName] = fileparts(FileName);
    catch
        if strcmp(ImageFileName,'N')
            FileName = TwoDigitString(handles.Current.SetBeingAnalyzed);
        else
            Spaces = isspace(FileName);
            if any(Spaces)
                error(['Image processing was canceled in the ', ModuleName, ' module because you have entered one or more spaces in the box of text for the filename of the image.'])
            end
            FileName = ImageFileName;
        end
    end

    if strcmp(Appendage,'N')
        FileName = [FileName TwoDigitString(handles.Current.SetBeingAnalyzed)];
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
            error(['Image processing was canceled in the ', ModuleName, ' module because the image file type entered is not recognized by Matlab. For a list of recognizable image file formats, type "CPimread" (no quotes) at the command line in Matlab, or see the help for this module.')
        end
    end

    FileAndPathName = fullfile(PathName, FileName);

    if strcmp(CheckOverwrite,'Yes') && ~strcmp(FileFormat,'avi')
        %%% Checks whether the new image name is going to overwrite the
        %%% original file. This check is not done here if this is an avi
        %%% (movie) file, because otherwise the check would be done on each
        %%% frame of the movie.
        if exist(FileAndPathName) == 2
            try
                Answer = CPquestdlg(['The settings in the ', ModuleName, ' module will cause the file "', FileAndPathName,'" to be overwritten. Do you want to continue or cancel?'], 'Warning', 'Continue','Skip Module','Cancel','Cancel');
            catch
                error(['Image processing was canceled in the ', ModuleName, ' module because the settings will cause the file "', FileAndPathName,'" to be overwritten and you have specified to not allow overwriting without confirming. When running on the cluster there is no way to confirm overwriting (no dialog boxes allowed), so image processing was canceled.'])
            end
            if strcmp(Answer,'Skip Module')
                return;
            end
            if strcmp(Answer,'Cancel')
                error('Image processing was canceled')
            end
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% SAVE IMAGE TO HARD DRIVE %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    drawnow

    FileSavingParameters = [];
    if ~strcmp(BitDepth,'8')
        FileSavingParameters = [',''bitdepth'', ', BitDepth,''];
        %%% In jpeg format at 12 and 16 bits, the mode must be set to
        %%% lossless to avoid failure of the imwrite function.
        if strcmp(FileFormat,'jpg') || strcmp(FileFormat,'jpeg')
            FileSavingParameters = [FileSavingParameters, ',''mode'', ''lossless'''];
        end
    end

    if ~strcmp(OptionalParameters,'/')
        FileSavingParameters = [',',OptionalParameters,FileSavingParameters]
    end

    if strcmp(FileFormat,'fig')
        if length(ImageName) == 1
            fieldname = ['FigureNumberForModule0',ImageName];
        elseif length(ImageName) == 2
            fieldname = ['FigureNumberForModule',ImageName];
        else
            error(['Image processing was canceled in the ', ModuleName, ' module because the figure number was not in XX format.']);
        end
        FigureHandle = handles.Current.(fieldname);
    end

    if strcmp(FileFormat,'mat')
        try
            eval(['save(''',FileAndPathName, ''',''Image'')']);
        catch
            error(['Image processing was canceled in the ', ModuleName, ' module because the image could not be saved to the hard drive for some reason. Check your settings.  The error is: ', lasterr])
        end
    elseif strcmp(FileFormat,'fig')
        try
            eval(['saveas(FigureHandle,FileAndPathName,''fig'')']);
        catch
            error(['Image processing was canceled in the ', ModuleName, ' module because the figure could not be saved to the hard drive for some reason. Check your settings.  The error is: ', lasterr])
        end
    elseif strcmpi(FileFormat,'avi')
        if handles.Current.SetBeingAnalyzed == 1 &&  strcmp(CheckOverwrite,'Y')
            %%% Checks whether the new image name is going to overwrite
            %%% the original file, but only on the first image set,
            %%% because otherwise the check would be done on each frame
            %%% of the movie.
            if exist(FileAndPathName) == 2
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
            if strncmpi(SaveMovieWhen,'L',1)
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
                try movie2avi(Movie,FileAndPathName)
                catch error(['Image processing was canceled in the ', ModuleName, ' module because there was an error saving the movie to the hard drive.'])
                end
            else
                %%% Specifying the size of the colormap is critical
                %%% to prevent a bunch of annoying weird errors. I assume
                %%% the avi format is always 8-bit (=256 levels).
                eval(['ChosenColormap = colormap(',ColorMap,'(256));']);
                try movie2avi(Movie,FileAndPathName,'colormap',ChosenColormap)
                catch error(['Image processing was canceled in the ', ModuleName, ' module because there was an error saving the movie to the hard drive.'])
                end
            end
        end
        
        %%%%%%% THIS IS FUNCTIONAL, BUT SLOW >>>>>>>>>>>
        %%% It opens the entire file from the hard drive and re-saves the whole
        %%% thing.
        %         %%% If this movie file already exists, open it.
        %         try
        %
        %             Movie = aviread(FileAndPathName);
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
        %         movie2avi(Movie,FileAndPathName,'colormap',colormap(gray(256)))

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


        %%% FAILED ATTEMPT TO USE ADDFRAME.
        % %%% See if this movie file already exists. If so, just
        %         %%% retrieve the AviHandle from handles
        %         SUCCESSFULHANDLERETIREVAL = 0;
        %         if exist(FileAndPathName) ~= 0
        %             try
        %                 fieldname = ['AviHandle', ImageName];
        %                 AviHandle = handles.Pipeline.(fieldname)
        %                 AviHandle = addframe(AviHandle,Image);
        %                 AviHandle = close(AviHandle);
        %                 SUCCESSFULHANDLERETIREVAL = 1;
        %             end
        %         end
        %
        %         if SUCCESSFULHANDLERETIREVAL == 0
        %             %%% If the movie does not exist already, create it using
        %             %%% the avifile function and put the AviHandle into the handles.
        %
        %             AviHandle = avifile(FileAndPathName);
        %             AviHandle = addframe(AviHandle,Image);
        %             AviHandle = close(AviHandle);
        %
        %             fieldname = ['AviHandle', ImageName];
        %             handles.Pipeline.(fieldname) =  AviHandle;
        %         end
        %

    else
        try eval(['imwrite(Image, FileAndPathName, FileFormat', FileSavingParameters,')']);
        catch
            error(['Image processing was canceled in the ', ModuleName, ' module because the image could not be saved to the hard drive for some reason. Check your settings, and see the Matlab imwrite function for details about parameters for each file format.  The error is: ', lasterr])
        end
    end
end

%%% SUBFUNCTION %%%
function twodigit = TwoDigitString(val)
%TwoDigitString is a function like num2str(int) but it returns a two digit
%representation of a string for our purposes.
if ((val > 99) || (val < 0)),
    error(['TwoDigitString: Can''t convert ' num2str(val) ' to a 2 digit number']);
end
twodigit = sprintf('%02d', val);