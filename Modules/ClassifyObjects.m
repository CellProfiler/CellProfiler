function handles = ClassifyObjects(handles)

% Help for the Classify Objects module:
% Category: Other
%
% This module classifies objects into a number of different
% classes according to the size of a measurement specified
% by the user.
%
%
% SAVING IMAGES: To save images using this module, use the SaveImages
% module with the images to be saved called 'ColorClassified' plus
% the object name you enter in the module, e.g.
% 'ColorClassifiedCells'.
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

%infotypeVAR01 = objectgroup
%textVAR01 = What did you call the segmented objects?
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = Enter the feature type, e.g. AreaShape, Texture, Intensity,...
%choiceVAR02 = AreaShape
%choiceVAR02 = Correlation
%choiceVAR02 = Texture
%choiceVAR02 = Intensity
%choiceVAR02 = Neighbors
%choiceVAR02 = Ratios
FeatureType = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu custom

%textVAR03 = Enter feature number
%defaultVAR03 = 1
FeatureNbr = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,3}));

%textVAR04 = Enter the lower limit of the lower bin
%defaultVAR04 = 0
LowerBinMin = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,4}));

%textVAR05 = Enter the upper limit for the upper bin
%defaultVAR05 = 100
UpperBinMax = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,5}));

%textVAR06 = Enter number of bins
%defaultVAR06 = 3
NbrOfBins = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,6}));


%%%VariableRevisionNumber = 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Retrieves the label matrix image that contains the segmented objects
fieldname = ['Segmented', ObjectName];

%%% Checks whether the image exists in the handles structure.
if ~isfield(handles.Pipeline, fieldname)
    errordlg(['Image processing has been canceled. Prior to running the ClassifyObject module, you must have previously run a module that generates an image with the objects identified.  You specified in the ClassifyObject module that the primary objects were named ',ObjectName,' which should have produced an image in the handles structure called ', fieldname, '. The ClassifyObject module cannot locate this image.']);
end
LabelMatrixImage = handles.Pipeline.(fieldname);

%%% Checks whether the feature type exists in the handles structure.
if ~isfield(handles.Measurements.(ObjectName),FeatureType)
    errordlg('The feature type entered in the ClassifyObjects module does not exist.');
end

if isempty(LowerBinMin) | isempty(UpperBinMax) | LowerBinMin > UpperBinMax
    errordlg('Image processing has been canceled because an error in the specification of the lower and upper limits was found in the ClassifyObjects module.');
end

if isempty(NbrOfBins) | NbrOfBins < 1
    errordlg('Image processing has been canceled because an error was found in the number of bins specification in the ClassifyObjects module.');
end


%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow

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

% Get Measurements
Measurements = handles.Measurements.(ObjectName).(FeatureType){handles.Current.SetBeingAnalyzed}(:,FeatureNbr);

% Quantize measurements
edges = linspace(LowerBinMin,UpperBinMax,NbrOfBins+1);
edges(1) = edges(1) - sqrt(eps);                               % Just a fix so that objects with a measurement that equals the lower bin edge of the lowest bin are counted
QuantizedMeasurements = zeros(size(Measurements));
bins = zeros(1,NbrOfBins);
for k = 1:NbrOfBins
    index = find(Measurements > edges(k) & Measurements <= edges(k+1));
    QuantizedMeasurements(index) = k;
    bins(k) = length(index);
end

% Produce image where the the objects are colored according to the original
% measurements and the quantized measurements
NonQuantizedImage = zeros(size(LabelMatrixImage));
NbrOfObjects = max(LabelMatrixImage(:));
props = regionprops(LabelMatrixImage,'PixelIdxList');              % Pixel indexes for objects fast
for k = 1:NbrOfObjects
    NonQuantizedImage(props(k).PixelIdxList) = Measurements(k);
end
QuantizedMeasurements = [0;QuantizedMeasurements];                 % Add a background class
QuantizedImage = QuantizedMeasurements(LabelMatrixImage+1);
cmap = [0 0 0;jet(length(bins))];
QuantizedRGBimage = ind2rgb(QuantizedImage+1,cmap);
    FeatureName = handles.Measurements.(ObjectName).([FeatureType,'Features']){FeatureNbr};


%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

% PROGRAMMING NOTE
% DISPLAYING RESULTS:
% Some calculations produce images that are used only for display or
% for saving to the hard drive, and are not used by downstream
% modules. To speed processing, these calculations are omitted if the
% figure window is closed and the user does not want to save the
% images.

fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
if any(findobj == ThisModuleFigureNumber) == 1;
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
    drawnow
    %%% Sets the width of the figure window to be appropriate (half width).
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        originalsize = get(ThisModuleFigureNumber, 'position');
        newsize = originalsize;
        newsize(3) = originalsize(3);
        set(ThisModuleFigureNumber, 'position', newsize);
    end
    %%% Activates the appropriate figure window.
    CPfigure(handles,ThisModuleFigureNumber);

    %%% A subplot of the figure window is set to display the original image.
    subplot(2,2,1)
    ImageHandle = imagesc(NonQuantizedImage,[min(Measurements) max(Measurements)]);
    set(ImageHandle,'ButtonDownFcn','ImageTool(gco)','Tag',sprintf('%s colored accoring to %s',ObjectName,FeatureName))
    colormap(hot),axis image
    set(gca,'Fontsize',get(0,'UserData'))
    title(sprintf('%s colored accoring to %s',ObjectName,FeatureName))

    %%% Produce and plot histogram of original data
    subplot(2,2,2)
    Nbins = min(round(NbrOfObjects/5),40);
    hist(Measurements,Nbins)
    set(get(gca,'Children'),'FaceVertexCData',hot(Nbins));
    set(gca,'Fontsize',get(0,'UserData'));
    xlabel(FeatureName),ylabel(['#',ObjectName]);
    title(sprintf('Histogram of %s',FeatureName));
    ylimits = ylim;
    axis tight
    xlimits = xlim;
    axis([xlimits ylimits])
    
    %%% A subplot of the figure window is set to display the quantized image.
    subplot(2,2,3)
    ImageHandle = image(QuantizedRGBimage);axis image
    set(ImageHandle,'ButtonDownFcn','ImageTool(gco)','Tag',['Classified ', ObjectName])
    set(gca,'Fontsize',get(0,'UserData'))
    title(['Classified ', ObjectName],'fontsize',get(0,'UserData'));

    %%% Produce and plot histogram
    subplot(2,2,4)
    x = edges(1:end-1) + (edges(2)-edges(1))/2;
    h = bar(x,bins,1);
    set(gca,'Fontsize',get(0,'UserData'));
    xlabel(FeatureName),ylabel(['#',ObjectName])
    title(sprintf('Histogram of %s',FeatureName))
    axis tight
    xlimits(1) = min(xlimits(1),LowerBinMin);                          % Extend limits if necessary and save them
    xlimits(2) = max(xlimits(2),UpperBinMax);                          % so they can be used for the second histogram
    axis([xlimits ylim])
    set(get(h,'Children'),'FaceVertexCData',jet(NbrOfBins));


end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

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

%%% Saves images to the handles structure so they can be saved to the hard
%%% drive, if the user requests.
fieldname = ['ColorClassified',ObjectName];
handles.Pipeline.(fieldname) = QuantizedRGBimage;

ClassifyFeatureNames = cell(1,NbrOfBins);
for k = 1:NbrOfBins
    ClassifyFeatureNames{k} = ['Bin ',num2str(k)];
end
FeatureName = FeatureName(~isspace(FeatureName));                    % Remove spaces in the feature name
handles.Measurements.Image.(['ClassifyObjects_',ObjectName,'_',FeatureName,'Features']) = ClassifyFeatureNames;
handles.Measurements.Image.(['ClassifyObjects_',ObjectName,'_',FeatureName])(handles.Current.SetBeingAnalyzed) = {bins};


