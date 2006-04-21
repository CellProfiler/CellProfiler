function HelpDeveloperInfo
helpdlg(help('HelpDeveloperInfo'))

% Programming Notes for CellProfiler Developer's version
%
% *** INTRODUCTION ***
%
% You can write your own modules, image tools, and data tools for
% CellProfiler - the easiest way is to modify an existing one. CellProfiler
% is modular: every module, image tool, and data tool is a single Matlab
% m-file (extension = .m). Upon startup, CellProfiler scans its Modules,
% DataTools, and ImageTools folders looking for files. Simply put your new
% file in the proper folder and it will appear in the proper place. They
% are automatically categorized, their help extracted, etc.
%
% If you have never tried computer programming or have not used Matlab,
% please do give it a try. Many beginners find this language easy to learn
% and the code for CellProfiler is heavily documented so that you can
% understand what each line does. It was designed so that biologists
% without programming experience could adapt it.
% 
% *** HELP SECTIONS AT THE BEGINNING OF EACH MODULE AND TOOL ***
%
% The first unbroken block of lines will be extracted as help by
% CellProfiler's 'Help for this analysis module' button, Help for image
% tools and data tools (Help menu in the main CellProfiler window) as well
% as Matlab's built in 'help' and 'doc' functions at the command line. It
% will also be used to automatically generate a pdf manual page for the
% module. An example image demonstrating the function of the module can
% also be saved in tif format, using the same name as the module, and it
% will automatically be included in the pdf manual page as well. Follow
% the convention of: Help for the XX module, Category (use an exact match
% of one of the categories so your module appears in the proper place in
% the "Add module" window), Short description, purpose of the module,
% description of the settings and acceptable range for each, how it works
% (technical description), and See also NameOfModule. The license/author
% information should be separated from the help lines with a blank line so
% that it does not show up in the help displays.
%
% *** SETTINGS (CALLED 'VARIABLES' IN THE CODE) ***
%
% Variables are automatically extracted from lines in a commented section
% near the beginning of each module. Even though they look like comments
% they are critical for the functioning of the code. The syntax here is
% critical - indenting lines or changing the spaces before and after the
% equals sign will affect the ability of the variables to be read properly.
%
% * The '%textVAR' lines contain the variable descriptions which are
% displayed in the CellProfiler main window next to each variable box. This
% text will wrap appropriately so it can be as long as desired, but it must
% be kept on a single line in the m-file (do not allow it to wrap). 
%
% * Whether the variable is entered into an edit box, chosen from a popup
% menu, or selected using browse buttons is determined by %inputtypeVAR
% lines and the %textVAR lines. The options are:
% - edit box (omit any %inputtypeVAR line for that variable number and use
% a %defaultVAR line to specify what text will appear in the box when the
% user first loads the module)
% - popup menu (use %inputtypeVAR = popupmenu and then use %choiceVAR
% lines, in the order you want them to appear, for each option that should
% appear in the popup menu)
% - popupmenu custom (this allows the user to choose from choices but also
% to have the option of typing in a custom entry. Use %inputtypeVAR =
% popupmenu custom and then use %choiceVAR lines, in the order you want
% them to appear, for each option that should appear in the popup menu)
% - pathname box + browse button (omit the %inputtypeVAR line and instead
% use %pathnametextVAR - the default shown in the edit box will be a
% period; this default is currently not alterable)
% - filename box + browse button (omit the %inputtypeVAR line and instead
% use %filenametextVAR - the default shown in the edit box will be the text
% "NO FILE LOADED"; this default is currently not alterable)
%
% * The %infotypeVAR lines specify the group that a particular entry will
% belong to. You will notice that many entries that the user types into the
% main window of CellProfiler are then available in popup menus in other
% modules. This works by classifying certain types of variable entries as
% follows:
% - imagegroup indep: the user's entry will be added to the imagegroup, and
% will therefore appear in the list of selectable images for variables
% whose type is 'imagegroup'. Usually used in combination with an edit
% box; i.e. no %inputtype line.
% - imagegroup: will display the user's image entries. Usually used in
% combination with a popupmenu.
% - objectgroup indep and objectgroup: Same idea as imagegroup, for passing 
% along object names.
% - outlinegroup indep and outlinegroup: Same idea as imagegroup, for
% passing along outline names.
% - datagroup indep and datagroup: Same idea as imagegroup, for passing
% along text/data names.
% - gridgroup indep and gridgroup: Same idea as imagegroup, for passing
% along grid names.
% 
% * The line of actual code within each group of variable lines is what
% actually extracts the value that the user has entered in the main window
% of CellProfiler (which is stored in the handles structure) and saves it
% as a variable in the workspace of this module with a meaningful name.
%
% * For CellProfiler to load modules and pipelines correctly, the order of
% variable information should be as follows:
% %textVAR01 = Whatever text description you want to appear
% %defaultVAR01 = Whatever text you want to appear 
% (OR, %choiceVAR01 = Whatever text)
% %infotypeVAR01 = imagegroup indep
% BlaBla = char(handles.Settings.VariableValues{CurrentModuleNum,1});
% %inputtypeVAR01 = popupmenu
%    In particular, when the input type is "popupmenu custom", the 
% choiceVAR01 line should be after textVAR01. This order is necessary
% because the textVAR01 creates a VariableBox associated with a variable
% number. Also, the defaultVAR01 value will inadvertently overwrite saved
% settings when loading a saved pipeline if it is located after
% infotypeVAR01 or inputtypeVAR01.
% 
% * CellProfiler uses VariableRevisionNumbers to help programmers notify
% users when something significant has changed about the variables.
% For example, if you have switched the position of two variables,
% loading a pipeline made with the old version of the module will not
% behave as expected when using the new version of the module, because
% the settings (variables) will be mixed up. The line should use this
% syntax:
% %%%VariableRevisionNumber = 1
%    If the module does not have this line, the VariableRevisionNumber is
% assumed to be 0.  This number need only be incremented when a change made
% to the modules will affect a user's previously saved settings. There is a
% revision number at the end of the license info at the top of the m-file
% for our source-control revisions - this revision number does not affect
% the user's previously saved settings files and you can ignore it.
%
% *** STORING AND RETRIEVING DATA: THE HANDLES STRUCTURE ***
% 
% In CellProfiler (and Matlab in general), each independent function
% (module) has its own workspace and is not able to 'see' variables
% produced by other modules. For data or images to be shared from one
% module to the next, they must be saved to what is called the 'handles
% structure'. This is a variable, whose class is 'structure', and whose
% name is handles. The contents of the handles structure can be printed out
% at the command line of Matlab using the Tech Diagnosis button and typing
% "handles" (no quotes). The only variables present in the *main* handles
% structure are handles to figures and GUI elements. Everything else should
% be saved in one of the following substructures:
%
% handles.Settings:
% Everything in handles.Settings is stored when the user uses File > Save
% pipeline, and these data are loaded into CellProfiler when the user uses
% File > Load pipeline. This substructure contains all necessary
% information to re-create a pipeline, including which modules were used
% (including variable revision numbers), their settings (variables), and
% the pixel size. Fields currently in handles.Settings: PixelSize,
% VariableValues, NumbersOfVariables, VariableInfoTypes,
% VariableRevisionNumbers, ModuleNames, SelectedOption. 
%    *** N.B. handles.Settings.PixelSize is where you should retrieve the
% PixelSize if needed, not in handles.Preferences!
%
% handles.Pipeline:
% This substructure is deleted at the beginning of the analysis run (see
% 'Which substructures are deleted prior to an analysis run?' below).
% handles.Pipeline is for storing data which must be retrieved by other
% modules. This data can be overwritten as each image cycle is processed,
% or it can be generated once and then retrieved during every subsequent
% image set's processing, or it can be saved for each image set by saving
% it according to which image cycle is being analyzed, depending on how it
% will be used by other modules. Example fields in handles.Pipeline:
% FileListOrigBlue, PathnameOrigBlue, FilenameOrigBlue, OrigBlue (which
% contains the actual image). Whether the handles.Pipeline structure is
% stored in the output file or not depends on whether you are in Fast Mode
% (see Help > HelpFastMode or File > SetPreferences).
%
% handles.Current:
% This substructure contains information needed for the main CellProfiler
% window display and for the various modules and help files to function. It
% does not contain any module-specific data (which is in handles.Pipeline).
% Example fields in handles.Current: NumberOfModules, StartupDirectory,
% DefaultOutputDirectory, DefaultImageDirectory, FilenamesInImageDir,
% CellProfilerPathname, CurrentHandles, ImageToolsFilenames, ImageToolHelp,
% DataToolsFilenames, DataToolHelp, HelpFilenames, Help, NumberOfImageSets,
% SetBeingAnalyzed, SaveOutputHowOften, TimeStarted, CurrentModuleNumber,
% FigureNumberForModuleXX.
%
% handles.Preferences: 
% Everything in handles.Preferences is stored in the file
% CellProfilerPreferences.mat when the user uses File > Set Preferences.
% These preferences are loaded upon launching CellProfiler, or individual
% preferences files can be loaded using File > Load Preferences. Fields in
% handles.Preferences: PixelSize, DefaultModuleDirectory,
% DefaultOutputDirectory, DefaultImageDirectory, IntensityColorMap,
% LabelColorMap, StripPipeline, SkipErrors, FontSize.
%    The PixelSize, DefaultImageDirectory, and DefaultOutputDirectory
% fields can be changed for the current session by the user using edit
% boxes in the main CellProfiler window, which changes their values in
% handles.Settings or handles.Current. Therefore:
%    *** N.B. handles.Settings.PixelSize is where you should retrieve the
% PixelSize if needed, not in handles.Preferences!
%    *** N.B. handles.Current.DefaultImageDirectory is where you should 
% retrieve the DefaultImageDirectory if needed, not in handles.Preferences!
%    *** N.B. handles.Current.DefaultOutputDirectory is where you should 
% retrieve the DefaultOutputDirectory if needed, not in
% handles.Preferences!
%
% handles.Measurements:
% Everything in handles.Measurements contains data specific to each image
% analyzed and is therefore accessed by the data tools. This substructure
% is deleted at the beginning of the analysis run (see 'Which substructures
% are deleted prior to an analysis run?' below).
%    Note that two types of measurements are typically made: Object
% and Image measurements.  Object measurements have one number for every
% object in the image (e.g. Object Area) and image measurements have one
% number for the entire image, which could come from one measurement from
% the entire image (e.g. Image TotalIntensity), or which could be an
% aggregate measurement based on individual object measurements (e.g. Image
% MeanArea).  Use the appropriate substructure to ensure that your data
% will be extracted properly. The relationships between objects can also be
% defined. For example, a nucleus might be associated with a particular
% cytoplasm and therefore each nucleus has a cytoplasm's number in the
% nucleus' measurement field which links the two. Or, for multiple speckles
% within a nucleus, each speckle will have a nucleus' number indicating
% which nucleus the speckle belongs to (see the Relate Objects module or
% Identify Secondary or Tertiary modules). Image measurements include a few
% standard fields: ModuleErrorFeatures, ModuleError, TimeElapsed,
% FileNamesText, FileNames, PathNamesText, PathNames.
%    The other measurement types have two entries: e.g.
% handles.Measurements.Nuclei.AreaShapeFeatures and
% handles.Measurements.Nuclei.AreaShape. The substructure ending in
% "Features" contains descriptive names of each measurement, e.g. "Area"
% "Perimeter", "Form Factor". These are essentially column headings. The
% companion substructure which lacks the "Feature" ending contains the
% actual numerical measurements themselves. For modules that measure the
% same objects in different ways (e.g. the Intensity module can measure
% intensities for Nuclei in two different images, blue and green), the
% identifying info becomes part of the substructure name, e.g.:
% handles.Measurements.Nuclei.Intensity_BlueFeatures
% handles.Measurements.Nuclei.Intensity_Blue
% handles.Measurements.Nuclei.Intensity_GreenFeatures
% handles.Measurements.Nuclei.Intensity_Green
%   Be sure to consider whether measurements you are storing will overwrite
% each other if more than one module is placed in the pipeline. You can
% differentiate measurements by including something specific in the name
% (e.g. Intensity modules include the image name (e.g. Blue or Green) in
% the substructure name). There are also several examples of modules where
% new measures are appended to the end of an existing substructure (i.e.
% forming a new column). See Calculate Ratios, which calls the
% CPaddmeasurements subfunction to do this.
%
% Why are file names stored in several places in the handles structure?
% The Load Images module creates both handles.Pipeline.FilenameIMAGENAME
% and handles.Pipeline.FileListIMAGENAME when loading an image or movie. In
% addition, file names are stored in handles.Measurements.Image.FileNames.
% They are present in Measurements so that they can be exported properly.
% For movies, the FileList field has the original name of the movie file
% and how many frames it contains. The Filenames field has the original
% movie file name and appends the frame number for every frame in the
% movie. This allows the names to be used in other modules such as
% SaveImages, which would otherwise over-write itself on every cycle using
% the original file name. The FileList location is created at the beginning
% of the run and contains all the images that will possibly be analyzed,
% whereas the Filename location is only populated as the images cycle
% through. We think there are good reasons for having filenames located in
% different places (especially when dealing with movie files) but we have
% not documented it thoroughly here!
%
% Which substructures are deleted prior to an analysis run?
% Anything stored in handles.Measurements or handles.Pipeline will be
% deleted at the beginning of the analysis run, whereas anything stored in
% handles.Settings, handles.Preferences, and handles.Current will be
% retained from one analysis to the next. It is important to think about
% which of these data should be deleted at the end of an analysis run
% because of the way Matlab saves variables: For example, a user might
% process 12 image sets of nuclei which results in a set of 12 measurements
% ("TotalStainedArea") stored in handles.Measurements.Image. In addition, a
% processed image of nuclei from the last image set is left in
% handles.Pipeline.SegmentedNuclei. Now, if the user uses a different
% module which happens to have the same measurement output name
% "TotalStainedArea" to analyze 4 image sets, the 4 measurements will
% overwrite the first 4 measurements of the previous analysis, but the
% remaining 8 measurements will still be present. So, the user will end up
% with 12 measurements from the 4 sets. Another potential problem is that
% if, in the second analysis run, the user runs only a module which depends
% on the output "SegmentedNuclei" but does not run a module that produces
% an image by that name, the module will run just fine: it will just
% repeatedly use the processed image of nuclei leftover from the last image
% set, which was left in handles.Pipeline.
%
% *** IMAGE ANALYSIS ***
%
% If you plan to use the same function in two different m-files (e.g. a
% module and a data tool, or two modules), it is helpful to write a
% CPsubfunction called by both m-files so that you have only one
% subfunction's code to maintain if any changes are necessary.
%
% Images loaded into CellProfiler are in the 0 to 1 range for consistency
% across modules. When retrieving images into your module, you can check
% the images for proper range, size, color/gray, etc using the
% CPretrieveimage subfunction.
%
% We have used many Matlab functions from the image processing toolbox.
% Currently, CellProfiler does not require any other toolboxes for
% processing.
%
% The 'drawnow' function allows figure windows to be updated and buttons to
% be pushed (like the pause, cancel, help, and view buttons).  The
% 'drawnow' function is sprinkled throughout the code so there are plenty
% of breaks where the figure windows/buttons can be interacted with.  This
% does theoretically slow the computation somewhat, so it might be
% reasonable to remove most of these lines when running jobs on a cluster
% where speed is important.
%
% *** ERROR HANDLING ***
%
% * In data tools & image tools:
%       CPerrordlg(['Image processing was canceled in the ',ModuleName,' 
%               module because your entry ',ValueX,' was invalid.'])
%       return
% 
% * In modules and I think also in CPsubfunctions (no need for "return"):
% error('Your error message here.')
%
% * In CPsubfunctions:
% Depends whether the calling function has error handling. For functions
% called from modules, error() is fine; for functions called from data
% tools that nest the CPsubfunction in a try/catch with its own error
% handling, error() is fine. For functions called from data tools that do
% not have error handling, CPerrordlg(''), return is needed. In general, we
% should have our goal be to use mostly error('') within CPsubfunctions and
% have the calling function handle errors appropriately.
%
% *** DISPLAYING RESULTS ***
%
% Each module checks whether its figure is open before calculating images
% that are for display only. This is done by examining all the figure
% handles for one whose handle is equal to the assigned figure number for
% this algorithm. If the figure is not open, everything between the "if"
% and "end" is ignored (to speed execution), so do not do any important
% calculations there. Otherwise an error message will be produced if the
% user has closed the window but you have attempted to access data that was
% supposed to be produced by this part of the code.  This is especially
% problematic when running on a cluster of computers with no displays. If
% you plan to save images which are normally produced for display only, the
% corresponding lines should be moved outside this if statement.
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
% are: OneByOne, TwoByOne, TwoByTwo, NarrowText. If a figure display is
% unnecessary for the module, skip STEP 2 and here use:
%   if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
%     close(ThisModuleFigureNumber)
%   end
%
% STEP 4: Display your image:
%   ImageHandle = CPimagesc(Image,handles);
% This CPimagesc displays the image and also embeds an image tool bar which
% will appear when you click on the displayed image. The handles are passed
% in so the user's preferences for font size and colormap are used.
%
% *** DEBUGGING HINTS ***
%
% * Use breakpoints in Matlab to stop your code at certain points and
% examine the intermediate results. 
%
% * To temporarily show an image during debugging, add lines like this to
% your code, or type them at the command line of Matlab:
%       CPfigure
%       CPimagesc(BlurredImage, [])
%       title('BlurredImage')
%
% * To temporarily save an intermediate image during debugging, try this:
%       imwrite(BlurredImage, 'FileName.tif', 'FileFormat');
% Note that you may have to alter the format of the image before
% saving. If the image is not saved correctly, for example, try
% adding the uint8 command:
%       imwrite(uint8(BlurredImage), 'FileName.tif', 'FileFormat');
% 
% * To routinely save images produced by this module, see the help in
% the SaveImages module.
%
% * If you want to save images that are produced by other modules but that
% are not given an official name in the settings boxes for that module,
% alter the code for the module to save those images to the handles
% structure and then use the Save Images module.
% The code should look like this:
% fieldname = ['SomeDescription(optional)',ImgOrObjNameFromSettingsBox];
% handles.Pipeline.(fieldname) = ImageProducedBytheModule;
% Example 1:
% fieldname = ['Segmented', ObjectName];
% handles.Pipeline.(fieldname) = SegmentedObjectImage;
% Example 2:
% fieldname = CroppedImageName;
% handles.Pipeline.(fieldname) = CroppedImage;
%
% For General help files:
% We have one line of actual code in these files so that the help is
% visible. We are not using CPhelpdlg because this allows the help to be
% accessed from the command line of Matlab. The code of theis module
% (helpdlg) is never run from inside CP anyway.
%
% *** RUNNING CELLPROFILER WITHOUT THE GRAPHICAL USER INTERFACE ***
%
% In order to run CellProfiler modules without the GUI you must have the 
% following variables:
% 
% handles.Settings.ModuleNames (for all modules in pipeline)
% handles.Settings.VariableValues (for all modules in pipeline)
% handles.Current.CurrentModuleNumber (must be consistent with pipeline)
% handles.Current.SetBeingAnalyzed (must be consistent with pipeline)
% handles.Current.FigureNumberForModuleXX (for all modules in pipeline)
% handles.Current.NumberOfImageSets (set by LoadImages, so if it is run
% first, you do not need to set it)
% handles.Current.DefaultOutputDirectory
% handles.Current.DefaultImageDirectory
% handles.Current.NumberOfModules
% handles.Preferences.IntensityColorMap (only used for display purposes)
% handles.Preferences.LabelColorMap (only used for display purposes)
% handles.Preferences.FontSize (only used for display purposes)
% 
% You will also need to have the CPsubfunctions folder, since our Modules
% call CP subfunctions for many tasks. The CurrentModuleNumber needs to be
% set correctly for each module in the pipeline since this is how the
% variable values are called. In order to see what all of these variables 
% look like, run a sample analysis and then go to File -> Tech Diagnosis.
% This will let you manipulate the handles variable in MatLab.