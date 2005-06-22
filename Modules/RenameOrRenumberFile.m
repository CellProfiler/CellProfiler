function handles = RenameOrRenumberFile(handles)

% Help for the RenameOrRenumberFile module:
% Category: File Handling
%
% File renaming utility that deletes or adds text anywhere 
% within image file names.
% It can also serve as a File renumbering utility that converts numbers 
% within image file names to solve improper ordering of files on 
% Unix/Mac OSX systems.  For example:
% DrosDAPI_1.tif becomes DrosDAPI_001.tif
% DrosDAPI_10.tif	becomes DrosDAPI_010.tif

% Be very careful since you will be renaming (=
% overwriting) your files!! You will have the opportunity to
% confirm the name change for the first image set only.  The folder
% containing the files must not contain subfolders or the subfolders
% and their contents will also be renamed.

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

%infotypeVAR01 = imagegroup
%textVAR01 = What did you call the images you want to rename/renumber? Be very careful since you will be renaming (= overwriting) your files!! See the help for this module for other warnings.
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = How many characters at the beginning of the file name do you want to retain?
%defaultVAR02 = 6
NumberCharactersPrefix = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,2}));

%textVAR03 = How many characters at the end do you want to retain, including file extension?
%defaultVAR03 = 8
NumberCharactersSuffix = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,3}));

%textVAR04 = Enter any text you want to place between those two portions of filename. Leave "/" to leave as is.
%defaultVAR04 = /
TextToAdd = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = How many characters would you like to allow between those two portions of filename, for renumbering purposes? Leave / to leave as is.
%defaultVAR05 = /
NumberDigits = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Pathname = handles.Current.DefaultImageDirectory;
%%% Retrieves all the image file names and the number of
%%% images per set so they can be used by the module.
fieldname=['FileList',ImageName];
FileNames = handles.Pipeline.(fieldname);
if strcmp(class(TextToAdd), 'char') ~= 1
    try TextToAdd = num2str(TextToAdd);
    catch error('The text you tried to add could not be converted into text for some reason.')
    end
end

AmtDigits = str2double(NumberDigits);

for n = 1:length(FileNames)
    OldFilename = char(FileNames(n));
    Prefix = OldFilename(1:NumberCharactersPrefix);
    Suffix = OldFilename((end-NumberCharactersSuffix+1):end);
    
    %Renaming Stage
    if strcmp(TextToAdd,'/') == 0
        InterFilename = [Prefix,TextToAdd,Suffix];
    else
        InterFilename = OldFilename;
    end
    
    %Renumbering Stage
    if strcmp(NumberDigits,'/') == 0
        OldNumber = InterFilename(NumberCharactersPrefix+1:end-NumberCharactersSuffix);
        NumberOfZerosToAdd = AmtDigits - length(OldNumber);
        if NumberOfZerosToAdd < 0
            OldNumber = OldNumber(end-AmtDigits+1:end);
            NumberOfZerosToAdd=0;
        end
        ZerosToAdd = num2str(zeros(NumberOfZerosToAdd,1))';
        TextToAdd = [ZerosToAdd,OldNumber];
    else
        TextToAdd = InterFilename(NumberCharactersPrefix+1:end-NumberCharactersSuffix);
    end
    
    %Final renamed & renumbered name
    NewFilename = [Prefix,TextToAdd,Suffix];
    
    if n == 1
        DialogText = ['Confirm the file name change. For example, the first file''s name will change from ', OldFilename, ' to ', NewFilename, '.'];
        Answer = CPquestdlg(DialogText, 'Confirm file name change','OK','Cancel','Cancel');
        if strcmp(Answer, 'Cancel') == 1
            error('File renaming was canceled at your request.')
        end
    end
    if strcmp(OldFilename,NewFilename) ~= 1
        movefile(fullfile(Pathname,OldFilename),fullfile(Pathname,NewFilename)) 
    end    
    drawnow
end

%%% This line will "cancel" processing after the first time through this
%%% module.  All the files are renumbered the first time through. Without
%%% the following cancel line, the module will run X times, where X is the
%%% number of files in the current directory.  
set(handles.timertexthandle,'string','Cancel')

%%%%%%%%%%%%%%%%%%%%
%%% FIGURE WINDOW %%%
%%%%%%%%%%%%%%%%%%%%

%%% The figure window display is unnecessary for this module, so the figure
%%% window is closed if it was previously open.
%%% Determines the figure number.
fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
%%% If the window is open, it is closed.
if any(findobj == ThisModuleFigureNumber) == 1;
    delete(ThisModuleFigureNumber)
end
guidata(handles.figure1,handles)

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