function handles = MeasureNeighbors(handles)

% Help for the Measure Neighbors module:
% Category: Measurement
%
% Given an image with objects identified (e.g. nuclei or cells), this
% module determines how many neighbors each object has. The user
% selects the distance within which objects should be considered
% neighbors.
%
% How it works:
% Retrieves a segmented image of the objects, in label matrix format.
% The objects are expanded by the number of pixels the user specifies,
% and then the module counts up how many other objects the object
% is overlapping. Alternately, the module can measure the number of
% neighbors each object has if every object were expanded up until the
% point where it hits another object.  To use this option, enter 0
% (the number zero) for the pixel distance.  Please note that
% currently the image of the objects, colored by how many neighbors
% each has, cannot be saved using the SaveImages module, because it is
% actually a black and white image displayed using a particular
% colormap
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
%
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


%textVAR01 = What did you call the objects whose neighbors you want to measure?
%defaultVAR01 = Cells
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%textVAR02 = Objects are considered neighbors if they are within this distance (pixels), or type 0 to find neighbors if each object were expanded until it touches others:
%defaultVAR02 = 0
NeighborDistance = str2num(handles.Settings.VariableValues{CurrentModuleNum,2});
%textVAR03 = If you are expanding objects until touching, what do you want to call these new objects?
%defaultVAR03 = ExpandedCells
ExpandedObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%textVAR04 = What do you want to call the image of the objects, colored by the number of neighbors?
%defaultVAR04 = ColoredNeighbors
ColoredNeighborsName = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a variable,
%%% "OrigImageToBeAnalyzed".
fieldname = ['Segmented',ObjectName];
%%% Checks whether the image exists in the handles structure.
if ~isfield(handles.Pipeline,fieldname)
    error(['Image processing has been canceled. Prior to running the Measure Neighbors module, you must have previously run a segmentation module.  You specified in the MeasureNeighbors module that the desired image was named ', IncomingLabelMatrixImageName(10:end), ', the Measure Neighbors module cannot locate this image.']);
end
IncomingLabelMatrixImage = handles.Pipeline.(fieldname);

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%

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

%%% Expands each object until almost 8-connected to its neighbors, if
%%% requested by the user.
if NeighborDistance == 0
    %%% The objects are thickened until they are one pixel shy of
    %%% being 8-connected.  This also turns the image binary rather
    %%% than a label matrix.
    ThickenedBinaryImage = bwmorph(IncomingLabelMatrixImage,'thicken',Inf);
    %%% The objects must be reconverted to a label matrix in a way
    %%% that preserves their prior labeling, so that any measurements
    %%% made on these objects correspond to measurements made by other
    %%% modules.
    ThickenedLabelMatrixImage = bwlabel(ThickenedBinaryImage);
    %%% For each object, one label and one label location is acquired and
    %%% stored.
    [LabelsUsed,LabelLocations] = unique(IncomingLabelMatrixImage);
    %%% The +1 increment accounts for the fact that there are zeros in the
    %%% image, while the LabelsUsed starts at 1.
    LabelsUsed(ThickenedLabelMatrixImage(LabelLocations(2:end))+1) = IncomingLabelMatrixImage(LabelLocations(2:end));
    FinalLabelMatrixImage = LabelsUsed(ThickenedLabelMatrixImage+1);
    IncomingLabelMatrixImage = FinalLabelMatrixImage;
    %%% The NeighborDistance is then set so that neighbors almost
    %%% 8-connected by the previous step are counted as neighbors.
    NeighborDistance = 4;
end


%%% Determines the neighbors for each object.
d = max(2,NeighborDistance+1);
[sr,sc] = size(IncomingLabelMatrixImage);
ImageOfNeighbors = -ones(sr,sc);
NumberOfNeighbors = zeros(max(IncomingLabelMatrixImage(:)),1);
IdentityOfNeighbors = cell(max(IncomingLabelMatrixImage(:)),1);
se = strel('disk',d,0);
for k = 1:max(IncomingLabelMatrixImage(:))
    % Cut patch
    [r,c] = find(IncomingLabelMatrixImage == k);
    rmax = min(sr,max(r) + (d+1));
    rmin = max(1,min(r) - (d+1));
    cmax = min(sc,max(c) + (d+1));
    cmin = max(1,min(c) - (d+1));
    p = IncomingLabelMatrixImage(rmin:rmax,cmin:cmax);
    % Extend cell boundary
    pextended = imdilate(p==k,se,'same');
    overlap = p.*pextended;
    IdentityOfNeighbors{k} = setdiff(unique(overlap(:)),[0,k]);
    NumberOfNeighbors(k) = length(IdentityOfNeighbors{k});
    ImageOfNeighbors(sub2ind([sr sc],r,c)) = NumberOfNeighbors(k);
end

%%% Calculates the ColoredLabelMatrixImage for displaying in the figure
%%% window and saving to the handles structure.
%%% Note that the label2rgb function doesn't work when there are no objects
%%% in the label matrix image, so there is an "if".
%%% Note: this is the expanded version of the objects, if the user
%%% requested expansion.

if sum(sum(IncomingLabelMatrixImage)) >= 1
    ColoredLabelMatrixImage = label2rgb(IncomingLabelMatrixImage,'jet', 'k', 'shuffle');
else  ColoredLabelMatrixImage = IncomingLabelMatrixImage;
end

%%% Does the same for the ImageOfNeighbors.  For some reason, this
%%% does not exactly match the results of the display window. Not sure
%%% why.
if sum(sum(ImageOfNeighbors)) >= 1
    ColoredImageOfNeighbors = ind2rgb(ImageOfNeighbors,[0 0 0;jet(max(ImageOfNeighbors(:)))]);
else  ColoredImageOfNeighbors = ImageOfNeighbors;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% MAKE MEASUREMENTS & SAVE TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

%%% Saves neighbor measurements to handles structure.
handles.Measurements.(ObjectName).NumberNeighbors(handles.Current.SetBeingAnalyzed) = {NumberOfNeighbors};
handles.Measurements.(ObjectName).NumberNeighborsFeatures = {'Number of neighbors'};

% This field is different from the usual measurements. To avoid problems with export modules etc we don't
% add a IdentityOfNeighborsFeatures field. It will then be "invisible" to
% export modules, which look for fields with 'Features' in the name.
handles.Measurements.(ObjectName).IdentityOfNeighbors(handles.Current.SetBeingAnalyzed) = {IdentityOfNeighbors};


%%% Example: To extract the number of neighbor for objects called Cells, use code like this:
%%% handles.Measurements.Cells.IdentityOfNeighborsCells{1}{3}
%%% where 1 is the image number and 3 is the object number. This
%%% yields a list of the objects who are neighbors with Cell object 3.

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
if any(findobj == ThisModuleFigureNumber) == 1
    FontSize = handles.Current.FontSize;
    %%% Sets the width of the figure window to be appropriate (half width).
    if handles.Current.SetBeingAnalyzed == 1
        originalsize = get(ThisModuleFigureNumber, 'position');
        newsize = originalsize;
        newsize(3) = 0.5*originalsize(3);
        set(ThisModuleFigureNumber, 'position', newsize);
    end
    drawnow
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
    figure(ThisModuleFigureNumber)
    subplot(2,1,1)
    imagesc(ColoredLabelMatrixImage)
    title('Cells colored according to their original colors','FontSize',FontSize)
    set(gca,'FontSize',FontSize)
    subplot(2,1,2)
    imagesc(ImageOfNeighbors)
    colormap([0 0 0;jet(max(ImageOfNeighbors(:)))]);
    colorbar('SouthOutside','FontSize',FontSize)
    title('Cells colored according to the number of neighbors','FontSize',FontSize)
    set(gca,'FontSize',FontSize)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE IMAGES TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% To make this module produce results similar to an IdentifyPrim
%%% module, we will save the image IncomingLabelMatrixImage (which may
%%% have been expanded until objects touch) to the handles
%%% structure, even though the editing for size and for edge-touching
%%% has not been performed in this module's case.

%%% Saves the segmented image, not edited for objects along the edges or
%%% for size, to the handles structure.
fieldname = ['PrelimSegmented',ExpandedObjectName];
handles.Pipeline.(fieldname) = IncomingLabelMatrixImage;

%%% Saves the segmented image, only edited for small objects, to the
%%% handles structure.
fieldname = ['PrelimSmallSegmented',ExpandedObjectName];
handles.Pipeline.(fieldname) = IncomingLabelMatrixImage;

%%% Saves the final segmented label matrix image to the handles structure.
fieldname = ['Segmented',ExpandedObjectName];
handles.Pipeline.(fieldname) = IncomingLabelMatrixImage;

%%% Saves the colored version of images to the handles structure so
%%% they can be saved to the hard drive, if the user requests.
fieldname = ['Colored',ExpandedObjectName];
handles.Pipeline.(fieldname) = ColoredLabelMatrixImage;
fieldname = ['Colored',ColoredNeighborsName];
handles.Pipeline.(fieldname) = ColoredImageOfNeighbors;