function handles = IdentifyPrimManually(handles)

% Help for the Identify Primary Manually module:
% Category: Object Identification
%
% This module allows the user to identify an single object by manually
% outlining it by using the mouse to click at multiple points around
% the object.
%
% What does Primary mean?
% Identify Primary modules identify objects without relying on any
% information other than a single grayscale input image (e.g. nuclei
% are typically primary objects). Identify Secondary modules require a
% grayscale image plus an image where primary objects have already
% been identified, because the secondary objects' locations are
% determined in part based on the primary objects (e.g. cells can be
% secondary objects). Identify Tertiary modules require images where
% two sets of objects have already been identified (e.g. nuclei and
% cell regions are used to define the cytoplasm objects, which are
% tertiary objects).
%
% SAVING IMAGES: In addition to the object outlines and the
% pseudo-colored object images that can be saved using the
% instructions in the main CellProfiler window for this module,
% this module produces several additional images which can be
% easily saved using the Save Images module. These will be grayscale
% images where each object is a different intensity. (1) The
% preliminary segmented image, which includes objects on the edge of
% the image and objects that are outside the size range can be saved
% using the name: PrelimSegmented + whatever you called the objects
% (e.g. PrelimSegmentedNuclei). (2) The preliminary segmented image
% which excludes objects smaller than your selected size range can be
% saved using the name: PrelimSmallSegmented + whatever you called the
% objects (e.g. PrelimSmallSegmented Nuclei) (3) The final segmented
% image which excludes objects on the edge of the image and excludes
% objects outside the size range can be saved using the name:
% Segmented + whatever you called the objects (e.g. SegmentedNuclei)
%
% Additional image(s) are normally calculated for display only,
% including the object outlines alone. These images can be saved by
% altering the code for this module to save those images to the
% handles structure (see the SaveImages module help) and then using
% the Save Images module.
%
% See also IDENTIFYPRIMTHRESHOLD,
% IDENTIFYPRIMADAPTTHRESHOLDA,
% IDENTIFYPRIMADAPTTHRESHOLDB,
% IDENTIFYPRIMADAPTTHRESHOLDC,
% IDENTIFYPRIMADAPTTHRESHOLDD,
% IDENTIFYPRIMSHAPEDIST,
% IDENTIFYPRIMSHAPEINTENS,
% IDENTIFYPRIMINTENSINTENS.

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

%textVAR01 = What did you call the images you want to use to manually identify an object?
%defaultVAR01 = OrigBlue
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = What do you want to call the objects identified by this module?
%defaultVAR02 = Nuclei
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Will you want to save the outlines of the objects (Yes or No)? If yes, use a Save Images module and type "OutlinedOBJECTNAME" in the first box, where OBJECTNAME is whatever you have called the objects identified by this module.
%defaultVAR03 = No
SaveOutlined = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 =  Will you want to save the image of the pseudo-colored objects (Yes or No)? If yes, use a Save Images module and type "ColoredOBJECTNAME" in the first box, where OBJECTNAME is whatever you have called the objects identified by this module.
%defaultVAR04 = No
SaveColored = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%%%VariableRevisionNumber = 01

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a variable,
%%% "OrigImage".
%%% Checks whether the image exists in the handles structure.
if isfield(handles.Pipeline, ImageName) == 0
    error(['Image processing has been canceled. Prior to running the Identify Primary Manually module, you must have previously run a module to load an image. You specified in the Identify Primary Manually module that this image was called ', ImageName, ' which should have produced a field in the handles structure called ', ImageName, '. The Identify Primary Manually module cannot find this image.']);
end
OrigImage = handles.Pipeline.(ImageName);

% Use a low resolution image for outlining the primary region
MaxSize = max(size(OrigImage));
if MaxSize > 512
    LowResOrigImage = imresize(OrigImage,512/MaxSize);
else
    LowResOrigImage = OrigImage;
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

%%% Displays the image in a new figure window.
FigureHandle = figure;
ImageHandle = imagesc(LowResOrigImage);axis off
[nrows,ncols,ncolors] = size(LowResOrigImage);
if ncolors == 1
    colormap(gray)
end
AxisHandle = gca;
title({['Image Set # ', num2str(handles.Current.SetBeingAnalyzed)], 'Click on consecutive points to outline the region of interest, then press enter.', 'The first and last points will be connected automatically.'})

%%% The following code was written by RonaldÊOuwerkerk of John Hopkins
%%% University and was retrieved from Mathworks Central as the file
%%% ImROI and ImROIdemo and modified for use in CellProfiler.

%============================================================================
% Get the ROI interactively
%============================================================================
%%% See local function 'getpoints' below.
[x , y, linehandle] = getpoints(AxisHandle);
close(FigureHandle)
[X,Y] = meshgrid(1:ncols,1:nrows);
LowResInterior = inpolygon(X,Y, x,y);
FinalLabelMatrixImage = double(imresize(LowResInterior,size(OrigImage)) > 0.5);

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
if any(findobj == ThisModuleFigureNumber) == 1 | strncmpi(SaveColored,'Y',1) == 1 | strncmpi(SaveOutlined,'Y',1) == 1
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
    CPfigure(ThisModuleFigureNumber);
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,2,1);imagesc(LowResOrigImage); title(['Original Image, Image Set # ', num2str(handles.Current.SetBeingAnalyzed)]); colormap(gray);
    %%% A subplot of the figure window is set to display the colored label
    %%% matrix image.
    subplot(2,2,2); imagesc(LowResInterior); title(['Manually Identified ',ObjectName]);
    %%% A subplot of the figure window is set to display the inverted original
    %%% image with outlines drawn on top.
    subplot(2,2,3); imagesc(LowResOrigImage);colormap(gray); title([ObjectName, ' Outline on Input Image']);
    hold on, plot(x,y,'r'),hold off
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

%%% Saves the segmented image, not edited for objects along the edges or
%%% for size, to the handles structure.
%%% Makes this module comparable to other Identify Primary modules,
%%% even though in this case the object was not edited for objects
%fieldname = ['PrelimSegmented',ObjectName];
%handles.Pipeline.(fieldname) = FinalLabelMatrixImage;

%%% Saves the segmented image, only edited for small objects, to the
%%% handles structure.
%%% Makes this module comparable to other Identify Primary modules,
%%% even though in this case the object was not edited for objects
%fieldname = ['PrelimSmallSegmented',ObjectName];
%handles.Pipeline.(fieldname) = FinalLabelMatrixImage;

%%% Saves the final segmented label matrix image to the handles structure.
fieldname = ['Segmented',ObjectName];
handles.Pipeline.(fieldname) = FinalLabelMatrixImage;

%%% Saves the ObjectCount, i.e. the number of segmented objects.
if ~isfield(handles.Measurements.Image,'ObjectCountFeatures')
    handles.Measurements.Image.ObjectCountFeatures = {};
    handles.Measurements.Image.ObjectCount = {};
end
column = find(~cellfun('isempty',strfind(handles.Measurements.Image.ObjectCountFeatures,ObjectName)));
if isempty(column)
    handles.Measurements.Image.ObjectCountFeatures(end+1) = {['ObjectCount ' ObjectName]};
    column = length(handles.Measurements.Image.ObjectCountFeatures);
end
handles.Measurements.Image.ObjectCount{handles.Current.SetBeingAnalyzed}(1,column) = max(FinalLabelMatrixImage(:));

%%% Saves the location of each segmented object
handles.Measurements.(ObjectName).LocationFeatures = {'CenterX','CenterY'};
tmp = regionprops(FinalLabelMatrixImage,'Centroid');
Centroid = cat(1,tmp.Centroid);
handles.Measurements.(ObjectName).Location(handles.Current.SetBeingAnalyzed) = {Centroid};


%%% Saves images to the handles structure so they can be saved to the hard
%%% drive, if the user requested.
try
    if strncmpi(SaveColored,'Y',1) == 1
        fieldname = ['Colored',ObjectName];
        handles.Pipeline.(fieldname) = FinalLabelMatrixImage;
    end
    if strncmpi(SaveOutlined,'Y',1) == 1
        fieldname = ['Outlined',ObjectName];
        handles.Pipeline.(fieldname) = ObjectOutlinesOnOrigImage;
    end
    %%% I am pretty sure this try/catch is no longer necessary, but will
    %%% leave in just in case.
catch errordlg('The object outlines or colored objects were not calculated by an identify module (possibly because the window is closed) so these images could not be saved to the handles structure. The Save Images module will therefore not function on these images.')
end
%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTION %%%
%%%%%%%%%%%%%%%%%%

function [xs,ys, linehandle] = getpoints(AxisHandle)
%%% The following code was written by RonaldÊOuwerkerk of John Hopkins
%%% University and was retrieved from Mathworks Central as the file
%%% ImROI and ImROIdemo and modified for use in CellProfiler.

%============================================================================
% Find parent figure for the argument axishandle
%============================================================================
FigureHandle = (get(AxisHandle, 'Parent'));

%===========================================================================
% Prepare for interactive collection of ROI boundary points
%===========================================================================
hold on
pointhandles = [];
xpts = [];
ypts = [];
splinehandle= [];
n = 0;
but = 1;
BUTN = 0;
KEYB = 1;
done =0;
%===========================================================================
% Loop until right hand mouse button or keyboard is pressed
%===========================================================================
while ~done;
    %===========================================================================
    % Analyze each buttonpressed event
    %===========================================================================
    keyb_or_butn = waitforbuttonpress;
    if keyb_or_butn == BUTN;
        currpt = get(AxisHandle, 'CurrentPoint');
        seltype = get(FigureHandle,'SelectionType');
        switch seltype
            case 'normal',
                but = 1;
            case 'alt',
                but = 2;
            otherwise,
                but = 2;
        end;
    elseif keyb_or_butn == KEYB
        but = 2;
    end;

    %===========================================================================
    % Get coordinates of the last buttonpressed event
    %===========================================================================
    xi = currpt(2,1);
    yi = currpt(2,2);
    %===========================================================================
    % Start a spline throught the points or
    % update the line through the points with a new spline
    %===========================================================================

    if but ==1
        if ~isempty(splinehandle)
            delete(splinehandle);
        end;
        pointhandles(n+1) = plot(xi,yi,'ro');
        n = n+1;
        xpts(n,1) = xi;
        ypts(n,1) = yi;

        %===========================================================================
        % Draw a spline line through the points
        %===========================================================================
        if n > 1
            t = 1:n;
            ts = 1: 0.1 : n;
            xs = spline(t, xpts, ts);
            ys = spline(t, ypts, ts);
            splinehandle = plot(xs,ys,'r-');
        end;

    elseif but > 1
        %===========================================================================
        % Exit for right hand mouse button or keyboard input
        %===========================================================================
        done = 1;
    end;
end;

%===========================================================================
% Add first point to the end of the vector for spline
%===========================================================================
xpts(n+1,1) = xpts(1,1);
ypts(n+1,1) = ypts(1,1);

%===========================================================================
% (re)draw the final spline
%===========================================================================
if ~ isempty(splinehandle)
    delete(splinehandle);
end;

t = 1:n+1;
ts = 1: 0.25 : n+1;
xs = spline(t, xpts, ts);
ys = spline(t, ypts, ts);

linehandle = plot(xs,ys,'r-');
drawnow;
%===========================================================================
% Delete the point markers
%===========================================================================
if ~isempty(pointhandles)
    delete(pointhandles)
end;

%===========================================================================
% END OF LOCAL FUNCTION GETPOINTS
%=====================================================================