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

%textVAR01 = What did you call the images you want to save?
%defaultVAR01 = OrigBlue
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = Which image's original filename do you want to use as a base to create the new file name? Type N to use sequential numbers.
%defaultVAR02 = OrigBlue
ImageFileName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Enter text to append to the image name, or leave "N" to keep the name the same except for the file extension.
%defaultVAR03 = N
Appendage = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = In what file format do you want to save images? Do not include a period
%defaultVAR04 = tif
FileFormat = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = Enter the pathname to the directory where you want to save the images. Type a period (.) to save images in the default output directory, or type I to save images in the default image directory, or type S to save images in the same Subdirectory where the original files are located. #LongBox#
%defaultVAR05 = .
FileDirectory = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = Enter the bit depth at which to save the images (8, 12, or 16: some image formats do not support saving at a bit depth of 12 or 16; see Matlab's imwrite function for more details.)
%defaultVAR06 = 8
BitDepth = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = Do you want to always check whether you will be overwriting a file when saving images?
%defaultVAR07 = Y
CheckOverwrite = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = At what point in the pipeline do you want to save the image? Enter E for every time through the pipeline (every image set), F for first, and L for last.
%defaultVAR08 = E
SaveWhen = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%textVAR09 = If you are only saving the image once (e.g. last or first option), enter the filename to use (with no extension). To use the automatically determined filename (derived from the source images), enter A.
%defaultVAR09 = A
OverrideFileName = char(handles.Settings.VariableValues{CurrentModuleNum,9});

%textVAR10 = Warning! It is possible to overwrite existing files using this module!

%%%VariableRevisionNumber = 5

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%

if handles.Current.SetBeingAnalyzed == 1;
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
        FileDirectory = handles.Current.DefaultOutputDirectory;
        %%% Makes sure that the File Directory specified by the user exists.
        if isdir(FileDirectory) ~= 1
            error(['Image processing was canceled because the specified directory "', FileDirectory, '" in the Save Images module does not exist.']);
        end
    elseif strcmpi(FileDirectory,'I') == 1
        FileDirectory = handles.Current.DefaultImageDirectory;
        %%% Makes sure that the File Directory specified by the user exists.
        if isdir(FileDirectory) ~= 1
            error(['Image processing was canceled because the specified directory "', FileDirectory, '" in the Save Images module does not exist.']);
        end
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

    %%% Checks whether the file format the user entered is readable by Matlab.
    IsFormat = imformats(FileFormat);
    if isempty(IsFormat) == 1
        if strcmpi(FileFormat,'mat') ~= 1
            error('The image file type entered in the Save Images module is not recognized by Matlab. Or, you may have entered a period in the box. For a list of recognizable image file formats, type "imformats" (no quotes) at the command line in Matlab.')
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
        else
            %%% Determine the filename of the image to be analyzed.
            fieldname = ['Filename', ImageFileName];
            FileName = handles.Pipeline.(fieldname)(handles.Current.SetBeingAnalyzed);
            %%% If subdirectories are being analyzed, the filename will
            %%% include subdirectory pathnames.
            [SubdirectoryPathName,BareFileName,ext,versn] = fileparts(FileName{1});
            if strcmpi(FileDirectory,'S') == 1
                FileDirectory = fullfile(handles.Current.DefaultImageDirectory,SubdirectoryPathName);
            end
        end
        %%% Assembles the new image name.
        if strcmp(upper(Appendage), 'N') == 1
            Appendage = [];
        end
        NewImageName = [BareFileName,Appendage,'.',FileFormat];
    else
        %%% Otherwise, use the filename the user entered.
        NewImageName = [OverrideFileName,'.',FileFormat];
        Spaces = isspace(NewImageName);
        if any(Spaces) == 1
            error('Image processing was canceled because you have entered one or more spaces in the proposed filename in the Save Images module.')
        end
    end

    %%% Makes sure that the File Directory specified by the user exists.
    if isdir(FileDirectory) ~= 1
        error(['Image processing was canceled because the specified directory "', FileDirectory, '" in the Save Images module does not exist.']);
    end

    NewFileAndPathName = fullfile(FileDirectory, NewImageName);
    if strcmpi(CheckOverwrite,'Y') == 1
        %%% Checks whether the new image name is going to overwrite the
        %%% original file.
        if exist(NewFileAndPathName) == 2
            Answer = CPquestdlg(['The settings in the Save Images module will cause the file "', NewFileAndPathName,'" to be overwritten. Do you want to continue or cancel?'], 'Warning', 'Continue','Cancel','Cancel');
            if strcmp(Answer,'Cancel') == 1
                error('Image processing was canceled')
            end
        end
    end

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
    else
        try eval(['imwrite(Image, NewFileAndPathName, FileFormat', FileSavingParameters,')']);
        catch
            error(['In the save images module, the image could not be saved to the hard drive for some reason. Check your settings, and see the Matlab imwrite function for details about parameters for each file format.  The error is: ', lasterr])
        end
    end
end
% PROGRAMMING NOTES THAT ARE UNNECESSARY FOR THIS MODULE:
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