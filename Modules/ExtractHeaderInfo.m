function handles = ExtractHeaderInfo(handles)

% Help for the Extract Header Info module:
% Category: Measurement
%
% This module was written for an old version of CellProfiler and may
% not be functional anymore, but it serves as an example of how to
% extract header info from an unusual file format.  These are images
% acquired using ISee software from an automated microscope.
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
drawnow

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

%textVAR01 = What do you want to call the images saved in the first location?
%defaultVAR01 = CFP
FirstImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%textVAR02 = What do you want to call the images saved in the third location?
%defaultVAR02 = DAPI
ThirdImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%textVAR03 = What do you want to call the images saved in the fifth location?
%defaultVAR03 = YFP
FifthImageName = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%textVAR04 = Enter the directory path name where the images are saved.#LongBox#
%defaultVAR04 = Default directory - leave this text to retrieve images from the directory specified above
PathName = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% determines the set number being analyzed
SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;
ImagesPerSet = 5;
SpecifiedPathName = PathName;

%%% Stores the text the user entered into cell arrays.
NumberInSet{1} = 1;
NumberInSet{3} = 3;
NumberInSet{5} = 5;

ImageName{1} = FirstImageName;
ImageName{3} = ThirdImageName;
ImageName{5} = FifthImageName;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% FIRST IMAGE SET FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Extracting the list of files to be analyzed occurs only the first time
%%% through this module.
if SetBeingAnalyzed == 1
    %%% For all 3 image slots, the file names are extracted.
    for n = 1:2:5
        if strncmp(SpecifiedPathName, 'Default', 7) == 1
            PathName = handles.Current.DefaultImageDirectory;
            FileNames = handles.Current.FilenamesInImageDir;
            if SetBeingAnalyzed == 1
                %%% Determines the number of image sets to be analyzed.
                NumberOfImageSets = fix(length(FileNames)/ImagesPerSet);
                %%% The number of image sets is stored in the
                %%% handles structure.
                handles.Current.NumberOfImageSets = NumberOfImageSets;
            else NumberOfImageSets = handles.Current.NumberOfImageSets;
            end
            %%% Loops through the names in the FileNames listing,
            %%% creating a new list of files.
            for i = 1:NumberOfImageSets
                Number = (i - 1) .* ImagesPerSet + NumberInSet{n};
                FileList(i) = FileNames(Number);
            end
            %%% Saves the File Lists and Path Names to the handles structure.
            fieldname = ['dOTFileList', ImageName{n}];
            handles.(fieldname) = FileList;
            fieldname = ['dOTPathName', ImageName{n}];
            handles.(fieldname) = PathName;
            clear FileList
        else
            %%% If a directory was typed in, the filenames are retrieved
            %%% from the chosen directory.
            if exist(SpecifiedPathName) ~= 7
                error('Image processing was canceled because the directory typed into the Extract Header Info module does not exist. Be sure that no spaces or unusual characters exist in your typed entry and that the pathname of the directory begins with /.')
            else [handles, FileNames] = RetrieveImageFileNames(handles, SpecifiedPathName);
                if SetBeingAnalyzed == 1
                    %%% Determines the number of image sets to be analyzed.
                    NumberOfImageSets = fix(length(FileNames)/ImagesPerSet);
                    handles.Current.NumberOfImageSets = NumberOfImageSets;
                else NumberOfImageSets = handles.Current.NumberOfImageSets;
                end 
                %%% Loops through the names in the FileNames listing,
                %%% creating a new list of files.
                for i = 1:NumberOfImageSets
                    Number = (i - 1) .* ImagesPerSet + NumberInSet{n};
                    FileList(i) = FileNames(Number);
                end
                %%% Saves the File Lists and Path Names to the handles structure.
                fieldname = ['dOTFileList', ImageName{n}];
                handles.(fieldname) = FileList;
                fieldname = ['dOTPathName', ImageName{n}];
                handles.(fieldname) = PathName;
                clear FileList
            end
        end
        
    end  % Goes with: for n = 1:2:5
    %%% Update the handles structure.
    guidata(gcbo, handles);    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% LOADING IMAGES EACH TIME %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for n = 1:2:5
    %%% This try/catch will catch any problems in the EHI module.
    try 
        %%% Determine which image to analyze.
        fieldname = ['dOTFileList', ImageName{n}];
        FileList = handles.(fieldname);
        %%% Determine the file name of the image you want to analyze.
        CurrentFileName = FileList(SetBeingAnalyzed);
        %%% Determine the directory to switch to.
        fieldname = ['dOTPathName', ImageName{n}];
        PathName = handles.(fieldname);
        try
            %%% run the header info function on the loaded image
            [ExpTime, ExpNum, WorldXYZ, TimeDate] = ExtractHeaderInfo(fullfile(PathName,char(CurrentFileName)));
        catch error(['You encountered an error during the subfunction "ExtractHeaderInfo".  Not a good thing.'])
        end
        %%% Converts the WorldXYZ data into three separate variables
        WorldXYZChar = char(WorldXYZ);
        equals = '=';
        equalslocations = strfind(WorldXYZChar,equals);
        WorldX = WorldXYZChar(3:(equalslocations(2)-2));
        WorldY = WorldXYZChar((equalslocations(2)+1):(equalslocations(3)-2));
        WorldZ = WorldXYZChar((equalslocations(3)+1):end);
        %%% Saves the original image file name to the handles structure.  The field
        %%% is named appropriately based on the user's input, with the
        %%% 'dOT' prefix added so that this field will be deleted at
        %%% the end of the analysis batch.
        fieldname = ['dOTFilename', ImageName{n}];
        handles.(fieldname)(SetBeingAnalyzed) = CurrentFileName;
        %%% Saves the extracted header information to the handles
        %%% structure, naming it with dMT because there is a new one
        %%% for each image but it must be deleted from the handles
        %%% structure anyway.  The field name comes from the user's
        %%% input.
        fieldname = ['dMTExpTime',ImageName{n}];
        handles.(fieldname)(SetBeingAnalyzed) = {(ExpTime)};
        fieldname = ['dMTWorldX',ImageName{n}];
        handles.(fieldname)(SetBeingAnalyzed) = {(WorldX)};
        fieldname = ['dMTWorldY',ImageName{n}];
        handles.(fieldname)(SetBeingAnalyzed) = {(WorldY)};
        fieldname = ['dMTWorldZ',ImageName{n}];
        handles.(fieldname)(SetBeingAnalyzed) = {(WorldZ)};
        fieldname = ['dMTTimeDate',ImageName{n}];
        handles.(fieldname)(SetBeingAnalyzed) = {(TimeDate)};
    catch ErrorMessage = lasterr;
        ErrorNumber(1) = {'first'};
        ErrorNumber(2) = {'second'};
        ErrorNumber(3) = {'third'};
        error(['An error occurred when trying to load the ', ErrorNumber{n}, ' set of images using the Extract Header Information module. Please check the settings. A common problem is that there are non-image files in the directory you are trying to analyze. Matlab says the problem is: ', ErrorMessage])
    end % Goes with: catch
end

%%% Update the handles structure.
guidata(gcbo, handles);

%%%%%%%%%%%%%%%%%%%%
%%% FIGURE WINDOW %%%
%%%%%%%%%%%%%%%%%%%%

if SetBeingAnalyzed == 1
    %%% The figure window display is unnecessary for this module, so the figure
    %%% window is closed the first time through the module.
    %%% Determines the figure number.
    fieldname = ['FigureNumberForModule',CurrentModule];
    ThisModuleFigureNumber = handles.Current.(fieldname);
    %%% If the window is open, it is closed.
    if any(findobj == ThisModuleFigureNumber) == 1;
        close(ThisModuleFigureNumber)
    end
end

%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTIONS %%%
%%%%%%%%%%%%%%%%%%%%

function [exp_time_ms, exp_num, worldxyz, timedate] = ExtractHeaderInfo(filename)

fid = fopen(filename, 'r', 'l');

fseek(fid, 28, 'bof');
exp_time_ms = fread(fid, 1, 'int32');
exp_num = fread(fid, 1, 'int32');

fseek(fid, 44, 'bof');
timedate = char(fread(fid, 36, 'char')');

fseek(fid, 260, 'bof');
worldxyz = char(fread(fid, 36, 'char')');

fclose(fid);



function [handles, FileNames] = RetrieveImageFileNames(handles, PathName)
%%% Lists all the contents of that path into a structure which includes the
%%% name of each object as well as whether the object is a file or
%%% directory.
FilesAndDirsStructure = dir(PathName);
%%% Puts the names of each object into a list.
FileAndDirNames = sortrows({FilesAndDirsStructure.name}');
%%% Puts the logical value of whether each object is a directory into a list.
LogicalIsDirectory = [FilesAndDirsStructure.isdir];
%%% Eliminates directories from the list of file names.
FileNamesNoDir = FileAndDirNames(~LogicalIsDirectory);

if isempty(FileNamesNoDir) == 1
    errordlg('There are no files in the chosen directory')
    handles.Current.FilenamesInImageDir = [];
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
    DiscardLogical2Pre = regexpi(FileNamesNoDir, '.mat$','once');
    if strcmp(class(DiscardLogical2Pre), 'cell') == 1
        DiscardLogical2 = cellfun('prodofsize',DiscardLogical2Pre);
    else DiscardLogical2 = [];
    end
    DiscardLogical3Pre = regexpi(FileNamesNoDir, '.m$','once');
    if strcmp(class(DiscardLogical3Pre), 'cell') == 1
        DiscardLogical3 = cellfun('prodofsize',DiscardLogical3Pre);
    else DiscardLogical3 = [];
    end
    DiscardLogical4Pre = regexpi(FileNamesNoDir, '.m~$','once');
    if strcmp(class(DiscardLogical4Pre), 'cell') == 1
        DiscardLogical4 = cellfun('prodofsize',DiscardLogical4Pre);
    else DiscardLogical4 = [];
    end
    DiscardLogical5Pre = regexpi(FileNamesNoDir, '.frk$','once');
    if strcmp(class(DiscardLogical5Pre), 'cell') == 1
        DiscardLogical5 = cellfun('prodofsize',DiscardLogical5Pre);
    else DiscardLogical5 = [];
    end
    
    %%% Combines all of the DiscardLogical arrays into one.
    DiscardLogical = DiscardLogical1 | DiscardLogical2 | DiscardLogical3 | DiscardLogical4 | DiscardLogical5;
    %%% Eliminates filenames to be discarded.
    if isempty(DiscardLogical) == 1
        FileNames = FileNamesNoDir;
    else FileNames = FileNamesNoDir(~DiscardLogical);
    end
    %%% Checks whether any files are left.
    if isempty(FileNames) == 1
        errordlg('There are no image files in the chosen directory')
    end
end

% PROGRAMMING NOTES THAT ARE UNNECESSARY FOR THIS MODULE:
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
%
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
