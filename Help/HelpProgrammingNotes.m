function HelpProgrammingNotes
helpdlg(help('HelpProgrammingNotes'))

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
% saved, and See also NameOfModule. The license/author
% information should be separated from the help lines with a blank line so
% that it does not show up in the help displays.  
%
% DRAWNOW:
% The 'drawnow' function allows figure windows to be updated and
% buttons to be pushed (like the pause, cancel, help, and view
% buttons).  The 'drawnow' function is sprinkled throughout the code
% so there are plenty of breaks where the figure windows/buttons can
% be interacted with.  This does theoretically slow the computation
% somewhat, so it might be reasonable to remove most of these lines
% when running jobs on a cluster where speed is important.
%
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
%
% DISPLAYING RESULTS:
% Each module checks whether its figure is open before calculating
% images that are for display only. This is done by examining all the
% figure handles for one whose handle is equal to the assigned figure
% number for this algorithm. If the figure is not open, everything
% between the "if" and "end" is ignored (to speed execution), so do
% not do any important calculations here. Otherwise an error message
% will be produced if the user has closed the window but you have
% attempted to access data that was supposed to be produced by this
% part of the code. If you plan to save images which are normally
% produced for display only, the corresponding lines should be moved
% outside this if statement.
%
% STEP 1: Find the appropriate figure window. If it is closed, usually none
% of the remaining steps are performed.
%   ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
%   if any(findobj == ThisModuleFigureNumber)
%
% STEP 2: Activate the appropriate figure window so subsequent steps are
% performed inside this window:
%   CPfigure(handles,'Image',ThisModuleFigureNumber);
% For figures that contain any images, choose 'Image', otherwise choose
% 'Text'. 'Image' figures will have the RGB checkboxes which allow
% displaying individual channels and they will also have the
% InteractiveZoom and CellProfiler Image Tools menu items.
%
% STEP 3: (only during starting image cycle) Make the figure the proper
% size:
%   if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
%     CPresizefigure('','NarrowText',ThisModuleFigureNumber)
%   end
% The figure is adjusted to fit the aspect ratio of the images, depending
% on how many rows and columns of images should be displayed. The choices
% are: OneByOne, TwoByOne, TwoByTwo, NarrowText. If a figure display is unnecessary for the module, skip STEP 2 and here use: close(ThisModuleFigureNumber) instead of CPresizefigure.
%
% STEP 4: Display your image:
%   ImageHandle = CPimagesc(Image,handles);
% This CPimagesc displays the image and also embeds an image tool bar which
% will appear when you click on the displayed image. The handles are passed
% in so the user's preferences for font size and colormap are used.
%
% 
% DRAWNOW BEFORE FIGURE COMMAND:
% The "drawnow" function executes any pending figure window-related
% commands.  In general, Matlab does not update figure windows until
% breaks between image analysis modules, or when a few select commands
% are used. "figure" and "drawnow" are two of the commands that allow
% Matlab to pause and carry out any pending figure window- related
% commands (like zooming, or pressing timer pause or cancel buttons or
% pressing a help button.)  If the drawnow command is not used
% immediately prior to the figure(ThisAlgFigureNumber) line, then
% immediately after the figure line executes, the other commands that
% have been waiting are executed in the other windows.  Then, when
% Matlab returns to this module and goes to the subplot line, the
% figure which is active is not necessarily the correct one. This
% results in strange things like the subplots appearing in the timer
% window or in the wrong figure window, or in help dialog boxes.
%
% FILENAMES AND FILELIST:
% The LoadImages module creates both handles.Pipeline.FilnamesIMAGENAME and
% handles.Pipeline.FileListIMAGENAME when loading an image or movie. For
% movies, the FileList field has the original name of the movie file and
% how many frames it contains. The Filenames field has the original movie
% file name and appends the frame number for every frame in the movie. This
% allows the names to be used in other modules such as SaveImages, which
% would otherwise over-write itself on every cycle using the original file
% name.
%
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
% This is where you should retrieve the PixelSize if necessary, not in
% handles.Preferences.
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
% should retrieve this information if needed within a module. Use the
% following context to call these variables:
% DefaultImageDirectory = handles.Current.DefaultImageDirectory
% DefaultModuleDirectory = handles.Current.DefaultModuleDirectory
% DefaultOutputDirectory = handles.Current.DefaultOutputDirectory
% FontSize = handles.Preferences.FontSize %%% Note this is in Preferences
% PixelSize = handles.Current.PixelSize
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
%
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
% also that the capitalization is as shown. We cannot indent the
% variables or they will not be read properly.

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
%
% VARIABLE ORDER:
% For CellProfiler to load modules and pipelines correctly, the order of
% variable information should be as follows:
%
% %textVAR01 = Whatever you want to say
% %defaultVAR01 = OrigBlue
% %infotypeVAR01 = imagegroup indep
% %inputtypeVAR01 = popupmenu
%
% In cases where the input type is "popupmenu custom", the choiceVAR01
% value should be after textVAR01 where defaultVAR01 is in our example.
% This order is necessary because the textVAR01 creates the VariableBox
% associated with a variable number. Also, the defaultVAR01 value will
% inadvertently overwrite saved settings when loading a saved pipeline if
% it is located after infotypeVAR01 or inputtypeVAR01.

%%% We are not using CPhelpdlg because this allows the help to be accessed
%%% from the command line of Matlab. The code of theis module (helpdlg) is
%%% never run from inside CP anyway.

% For CellProfiler Developer's version , properly
% formatted image analysis modules are Matlab m-files that end with .m.
%
% Error messages:
% In data tools & image tools:
% CPerrordlg(['The value you entered for the method to threshold ', GreaterOrLessThan, ' was not valid.  Acceptable entries are >, >=, =, <=, <.']);
%        return
% In modules and I think also in CPsubfunctions (no need for "return"):
% error('Your error message here.')
%
% CPsubfunctions:
% Depends whether the calling function has error handling. For functions
% called from modules, error() is fine; for functions called from data
% tools that nest the CPsubfunction in a try/catch with its own error
% handling, error() is fine. For functions called from data tools that do
% not have error handling, CPerrordlg(''), return is needed.
%
% In general, we should have our goal be to use mostly error('') within
% subfunctions and have the calling function handle errors appropriately.
%
%In case the settings file was created with an
% outdated version of a module, some of the behavior of settings may
% have changed, so CellProfiler warns you and guides you through
% converting your old settings file to something usable.