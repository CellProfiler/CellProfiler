function handles = CreateBatchScripts(handles)

% Help for the Create Batch Scripts module:
% Category: Other
%
% This module writes a batch (set) of Matlab scripts that can be submitted
% in parallel to a cluster for faster processing.
%
% This module should be placed at the end of an image processing pipeline.
% It takes five values as input: the size of each batch that the full set
% of images should be split into, the path to the CellProfiler modules on
% the cluster machines, the path to the images on the remote machine, where
% in the local system to save the batch files, and finally, a prefix to put
% on the batch files. For jobs that you do not want to split into batches
% but simply want to run on a separate computer, set the batch size to a
% very large number (more than the number of image sets), which will create
% one large job.
%
% After the first image set is processed, batch files are created and
% saved on the local machine, by default in the
% current default output directory.  Each batch file is of the form
% Batch_X_to_Y.m (The prefix can be changed from Batch_ by the
% user), where X is the first image set to be processed in the particular
% batch file, and Y is the last.  There is also a Batch_data.mat file
% that each script uses to initialize the processing.
%
% After the batch files are created, they can be submitted
% individually to the remote machines.  Note that the batch files and
% Batch_data.mat file might have to be copied to the remote machines,
% first.  Details of how remote jobs will be started vary from
% location to location.  The output files will be written in the
% directory where the batch files are running, which may or may not be
% the directory where the batch scripts are located.  Please consult
% your local cluster experts.
%
% After batch processing is complete, the output files can be merged
% by the MergeBatchOutput module.  For the simplest behavior in
% merging, it is best to save output files to a unique and initially
% empty directory.
%
% If the batch processing fails for some reason, the handles structure
% in the output file will have a field BatchError, and the error will
% also be written to standard out.  Check the output from the batch
% processes to make sure all batches complete.  Batches that fail for
% transient reasons can be resubmitted.
%
% See also MERGEBATCHOUTPUT.

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

%textVAR01 = How many image sets should be in each batch?
%defaultVAR01 = 100
BatchSize = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,1}));

%textVAR02 = What is the path to the CellProfiler on the cluster machines?  Leave a period (.) to use the default module directory. (To change the default module directory, use the Set Preferences button).#LongBox#
%defaultVAR02 = .
BatchCellProfilerPath = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = What is the path to the image directory on the cluster machines? Leave a period (.) to use the default image directory.#LongBox#
%defaultVAR03 = .
BatchImagePath = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = What is the path to the directory where batch output should be written on the cluster machines? Leave a period (.) to use the default output directory.#LongBox#
%defaultVAR04 = .
BatchOutputPath = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = What is the path to the directory where you want to save the batch files (on the local machine)? Leave a period (.) to use the default output directory.#LongBox#
%defaultVAR05 = .
BatchSavePath = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = What is the path to the directory where the batch files will be saved on the cluster machines? Leave a period (.) to use the default output directory.#LongBox#
%defaultVAR06 = .
BatchRemotePath = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = What prefix should be added to the batch file names?
%defaultVAR07 = Batch_
BatchFilePrefix = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = WARNING: This module should be the last one in the analysis pipeline.

%%%VariableRevisionNumber = 05

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if strcmp(BatchSavePath, '.') == 1
    BatchSavePath = handles.Current.DefaultOutputDirectory;
end

if strcmp(BatchRemotePath, '.') == 1
    BatchRemotePath = handles.Current.DefaultOutputDirectory;
end

if strcmp(BatchImagePath, '.') == 1
    BatchImagePath = handles.Current.DefaultImageDirectory;
end

if strcmp(BatchOutputPath, '.') == 1
    BatchOutputPath = handles.Current.DefaultOutputDirectory;
end

if strcmp(BatchCellProfilerPath, '.') == 1,
    BatchCellProfilerPath = fullfile(handles.Preferences.DefaultModuleDirectory, '..');
end

%%% Save a snapshot of handles.
handles_in = handles;

%%% Check that this is the last module in the analysis path.
if (CurrentModuleNum ~= handles.Current.NumberOfModules),
    error(['CreateBatchFiles must be the last module in the pipeline.']);
end;

%%% If this isn't the first image set, we're probably running on the
%%% cluster, and should just continue.

if (handles.Current.SetBeingAnalyzed > 1),
    return;
end
        
%%% We need to rewrite the pathnames in the handles structure for the
%%% remote machines.
Fieldnames = fieldnames(handles.Pipeline);
PathFieldnames = Fieldnames(strncmp(Fieldnames,'Pathname',8)==1);
for i = 1:length(PathFieldnames),
    handles.Pipeline.(PathFieldnames{i}) = BatchImagePath;
end

%%% The remote machines need a copy of handles.
PathAndFileName = fullfile(BatchSavePath, [BatchFilePrefix 'data.mat']);
save(PathAndFileName, 'handles', '-v6');

%%% Create the individual batch files

if (BatchSize <= 0)
    BatchSize = 100;
end

for n = 2:BatchSize:handles.Current.NumberOfImageSets,
    StartImage = n;
    EndImage = min(StartImage + BatchSize - 1, handles.Current.NumberOfImageSets);
    BatchFileName = sprintf('%s/%s%d_to_%d.m', BatchSavePath, BatchFilePrefix, StartImage, EndImage);
    BatchFile = fopen(BatchFileName, 'wt');

    fprintf(BatchFile, 'path(''%s'',path);\n', fullfile(BatchCellProfilerPath, 'Modules'));
    fprintf(BatchFile, 'path(''%s'',path);\n', BatchCellProfilerPath);
    fprintf(BatchFile, 'BatchFilePrefix = ''%s'';\n', BatchFilePrefix);
    fprintf(BatchFile, 'StartImage = %d;\n', StartImage);
    fprintf(BatchFile, 'EndImage = %d;\n', EndImage);
    fprintf(BatchFile, 'tic;\n');
    fprintf(BatchFile, 'load([''%s/'' BatchFilePrefix ''data.mat'']);\n', BatchRemotePath);
    fprintf(BatchFile, 'for BatchSetBeingAnalyzed = StartImage:EndImage,\n');
    fprintf(BatchFile, '    disp(sprintf(''Analyzing set %%d'', BatchSetBeingAnalyzed));\n');
    fprintf(BatchFile, '    toc;\n');
    fprintf(BatchFile, '    break_outer_loop = 0;\n');
    fprintf(BatchFile, '    handles.Current.SetBeingAnalyzed = BatchSetBeingAnalyzed;\n');
    fprintf(BatchFile, '    setbeinganalyzed = handles.Current.SetBeingAnalyzed;\n');
    fprintf(BatchFile, '    for SlotNumber = 1:handles.Current.NumberOfModules,\n');
    fprintf(BatchFile, '        ModuleNumberAsString = sprintf(''%%02d'', SlotNumber);\n');
    fprintf(BatchFile, '        ModuleName = char(handles.Settings.ModuleNames(SlotNumber));\n');
    fprintf(BatchFile, '        if iscellstr(handles.Settings.ModuleNames(SlotNumber)) == 0\n');
    fprintf(BatchFile, '        else\n');
    fprintf(BatchFile, '            handles.Current.CurrentModuleNumber = ModuleNumberAsString;\n');
    fprintf(BatchFile, '            try\n');
    fprintf(BatchFile, '                eval([''handles = '',ModuleName,''(handles);''])\n');
    fprintf(BatchFile, '            catch\n');
    fprintf(BatchFile, '                handles.BatchError = [ModuleName '' '' lasterr];\n');
    fprintf(BatchFile, '                disp([''Batch Error: '' ModuleName '' '' lasterr]);\n');
    fprintf(BatchFile, '                break_outer_loop = 1;\n');
    fprintf(BatchFile, '                break;\n');
    fprintf(BatchFile, '            end\n');
    fprintf(BatchFile, '        end\n');
    fprintf(BatchFile, '    end\n');
    fprintf(BatchFile, '    if (break_outer_loop),\n');
    fprintf(BatchFile, '        break;\n');
    fprintf(BatchFile, '    end\n');
    fprintf(BatchFile, 'end\n');
    fprintf(BatchFile, 'cd(''%s'');\n', BatchOutputPath);
    fprintf(BatchFile, 'eval([''save '',sprintf(''%%s%%d_to_%%d_OUT'', BatchFilePrefix, StartImage, EndImage), '' handles;'']);\n');
    fclose(BatchFile);
end

helpdlg('Batch files have been written.  This analysis pipeline will now stop.  You should submit the invidual .m scripts for processing on your cluster.', 'BatchFilesDialog');

%%% This is the first image set, so this is the first time seeing this
%%% module.  It should cause a cancel so no further processing is done
%%% on this machine.
set(handles.timertexthandle,'string','Cancel')


%%% Undo the changes to handles.Pipeline, above.
handles = handles_in;


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

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%

%%% The figure window display is unnecessary for this module, so the figure
%%% window is closed if it was previously open.
%%% Determines the figure number.
fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
%%% If the window is open, it is closed.
if any(findobj == ThisModuleFigureNumber) == 1;
    delete(ThisModuleFigureNumber)
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