function handles = AlgShrinkPrimaryObjects(handles)

% Help for the Shrink Primary Objects module: 
% Category: Object Identification
%
% The module shrinks primary objects by removing border pixels. The
% user can specify a certain number of times the border pixels are
% removed, or type 'Inf' to shrink objects down to a point. Objects
% are never lost using this module (shrinking stops when an object
% becomes a single pixel). Sometimes when identifying secondary
% objects (e.g. cell edges), it is useful to shrink the primary
% objects (e.g. nuclei) a bit in case the nuclei overlap the cell
% edges slightly, since the secondary object identifiers demand that
% the secondary objects completely enclose primary objects. This is
% handy when the two images are not aligned perfectly, for example.
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
% SAVING IMAGES: This module produces several images which can be
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
% Several additional images are normally calculated for display only,
% including the colored label matrix image (the objects displayed as
% arbitrary colors), object outlines, and object outlines overlaid on
% the original image. These images can be saved by altering the code
% for this module to save those images to the handles structure (see
% the SaveImages module help) and then using the Save Images module.
% Important note: The calculations of these display images are only
% performed if the figure window is open, so the figure window must be
% left open or the Save Images module will fail.  If you are running
% the job on a cluster, figure windows are not open, so the Save
% Images module will also fail, unless you go into the code for this
% module and remove the 'if/end' statement surrounding the DISPLAY
% RESULTS section.
%
% See also any identify primary module.

% The contents of this file are subject to the Mozilla Public License Version 
% 1.1 (the "License"); you may not use this file except in compliance with 
% the License. You may obtain a copy of the License at 
% http://www.mozilla.org/MPL/
% 
% Software distributed under the License is distributed on an "AS IS" basis,
% WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
% for the specific language governing rights and limitations under the
% License.
% 
% 
% The Original Code is the Shrink Primary Objects Module.
% 
% The Initial Developer of the Original Code is
% Whitehead Institute for Biomedical Research
% Portions created by the Initial Developer are Copyright (C) 2003,2004
% the Initial Developer. All Rights Reserved.
% 
% Contributor(s):
%   Anne Carpenter <carpenter@wi.mit.edu>
%   Thouis Jones   <thouis@csail.mit.edu>
%   In Han Kang    <inthek@mit.edu>
%
% $Revision$

% PROGRAMMING NOTE
% HELP:
% The first unbroken block of lines will be extracted as help by
% CellProfiler's 'Help for this analysis module' button as well as
% Matlab's built in 'help' and 'doc' functions at the command line. It
% will also be used to automatically generate a manual page for the
% module. An example image demonstrating the function of the module
% can also be saved in tif format, using the same name as the
% algorithm (minus Alg), and it will automatically be included in the
% manual page as well.  Follow the convention of: purpose of the
% module, description of the variables and acceptable range for each,
% how it works (technical description), info on which images can be 
% saved, and See also CAPITALLETTEROTHERALGORITHMS. The license/author
% information should be separated from the help lines with a blank
% line so that it does not show up in the help displays.  Do not
% change the programming notes in any modules! These are standard
% across all modules for maintenance purposes, so anything
% module-specific should be kept separate.

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
% The '%textVAR' lines contain the text which is displayed in the GUI
% next to each variable box. The '%defaultVAR' lines contain the
% default values which are displayed in the variable boxes when the
% user loads the algorithm. The line of code after the textVAR and
% defaultVAR extracts the value that the user has entered from the
% handles structure and saves it as a variable in the workspace of
% this algorithm with a descriptive name. The syntax is important for
% the %textVAR and %defaultVAR lines: be sure there is a space before
% and after the equals sign and also that the capitalization is as
% shown.  Don't allow the text to wrap around to another line; the
% second line will not be displayed.  If you need more space to
% describe a variable, you can refer the user to the help file, or you
% can put text in the %textVAR line above or below the one of
% interest, and do not include a %defaultVAR line so that the variable
% edit box for that variable will not be displayed; the text will
% still be displayed. CellProfiler is currently being restructured to
% handle more than 11 variable boxes. Keep in mind that you can have
% several inputs into the same box: for example, a box could be
% designed to receive two numbers separated by a comma, as long as you
% write a little extraction algorithm that separates the input into
% two distinct variables.  Any extraction algorithms like this should
% be within the VARIABLES section of the code, at the end.

%%% Reads the current algorithm number, since this is needed to find 
%%% the variable values that the user entered.
CurrentAlgorithm = handles.currentalgorithm;
CurrentAlgorithmNum = str2double(handles.currentalgorithm);

%textVAR01 = What did you call the objects that you want to shrink?
%defaultVAR01 = Nuclei
ObjectName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,1});

%textVAR02 = What do you want to call the shrunken objects?
%defaultVAR02 = ShrunkenNuclei
ShrunkenObjectName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,2});

%textVAR03 = Enter the number of pixels by which to shrink the objects
%textVAR04 = (Positive number, or "Inf" to shrink to a point)
%defaultVAR03 = 1
ShrinkingNumber = char(handles.Settings.Vvariable{CurrentAlgorithmNum,3});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Retrieves the segmented image, not edited for objects along the edges or
%%% for size.
fieldname = ['PrelimSegmented', ObjectName];
%%% Checks whether the image to be analyzed exists in the handles structure.
if isfield(handles.Pipeline, fieldname)==0,
    error(['Image processing was canceled because the Shrink Primary Objects module could not find the input image.  It was supposed to be produced by an Identify Primary module in which the objects were named ', ObjectName, '.  Perhaps there is a typo in the name.'])
end
PrelimSegmentedImage = handles.Pipeline.(fieldname);


%%% Retrieves the segmented image, only edited for small objects.
fieldname = ['PrelimSmallSegmented', ObjectName];
%%% Checks whether the image to be analyzed exists in the handles structure.
if isfield(handles.Pipeline, fieldname)==0,
    error(['Image processing was canceled because the Shrink Primary Objects module could not find the input image.  It was supposed to be produced by an Identify Primary module in which the objects were named ', ObjectName, '.  Perhaps there is a typo in the name.'])
end
PrelimSmallSegmentedImage = handles.Pipeline.(fieldname);


%%% Retrieves the final segmented label matrix image.
fieldname = ['Segmented', ObjectName];
%%% Checks whether the image to be analyzed exists in the handles structure.
if isfield(handles.Pipeline, fieldname)==0,
    error(['Image processing was canceled because the Shrink Primary Objects module could not find the input image.  It was supposed to be produced by an Identify Primary module in which the objects were named ', ObjectName, '.  Perhaps there is a typo in the name.'])
end
SegmentedImage = handles.Pipeline.(fieldname);


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

%%% Shrinks the three incoming images.  The "thin" option nicely removes
%%% one pixel border from objects with each iteration.  When carried out
%%% for an infinite number of iterations, however, it produces one-pixel
%%% width objects (points, lines, or branched lines) rather than a single
%%% pixel.  The "shrink" option uses a peculiar algorithm to remove border
%%% pixels that does not result in nice uniform shrinking of objects, but
%%% it does have the capability, when used with an infinite number of
%%% iterations, to reduce objects to a single point (one pixel).
%%% Therefore, if the user wants a single pixel for each object, the
%%% "shrink" option is used; otherwise, the "thin" option is used.
if strcmp(upper(ShrinkingNumber),'INF') == 1
    ShrunkenPrelimSegmentedImage = bwmorph(PrelimSegmentedImage, 'shrink', Inf);
    ShrunkenPrelimSmallSegmentedImage = bwmorph(PrelimSmallSegmentedImage, 'shrink', Inf);
    ShrunkenSegmentedImage = bwmorph(SegmentedImage, 'shrink', Inf);
else 
    try ShrinkingNumber = str2double(ShrinkingNumber);
        ShrunkenPrelimSegmentedImage = bwmorph(PrelimSegmentedImage, 'thin', ShrinkingNumber);
        ShrunkenPrelimSmallSegmentedImage = bwmorph(PrelimSmallSegmentedImage, 'thin', ShrinkingNumber);
        ShrunkenSegmentedImage = bwmorph(SegmentedImage, 'thin', ShrinkingNumber);
    catch error('Image processing was canceled because the value entered in the Shrink Primary Objects module must either be a number or the text "Inf" (no quotes).')
    end
end

%%% For the ShrunkenSegmentedImage, the objects are relabeled so that their
%%% numbers correspond to the numbers used for nuclei.  This is important
%%% so that if the user has made measurements on the non-shrunk objects,
%%% the order of these objects will be exactly the same as the shrunk
%%% objects, which may go on to be used to identify secondary objects.
FinalShrunkenPrelimSegmentedImage = ShrunkenPrelimSegmentedImage.*PrelimSegmentedImage;
FinalShrunkenPrelimSmallSegmentedImage = ShrunkenPrelimSmallSegmentedImage.*PrelimSmallSegmentedImage;
FinalShrunkenSegmentedImage = ShrunkenSegmentedImage.*SegmentedImage;

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

% PROGRAMMING NOTE
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

fieldname = ['figurealgorithm',CurrentAlgorithm];
ThisAlgFigureNumber = handles.(fieldname);
if any(findobj == ThisAlgFigureNumber) == 1;
    %%% Calculates the OriginalColoredLabelMatrixImage for displaying in the figure
    %%% window in subplot(2,1,1).
    %%% Note that the label2rgb function doesn't work when there are no objects
    %%% in the label matrix image, so there is an "if".
    if sum(sum(SegmentedImage)) >= 1
        OriginalColoredLabelMatrixImage = label2rgb(SegmentedImage,'jet', 'k', 'shuffle');
    else  OriginalColoredLabelMatrixImage = SegmentedImage;
    end
    %%% Calculates the ShrunkenColoredLabelMatrixImage for displaying in the figure
    %%% window in subplot(2,1,2).
    %%% Note that the label2rgb function doesn't work when there are no objects
    %%% in the label matrix image, so there is an "if".
    if sum(sum(SegmentedImage)) >= 1
        ShrunkenColoredLabelMatrixImage = label2rgb(FinalShrunkenSegmentedImage,'jet', 'k', 'shuffle');
    else  ShrunkenColoredLabelMatrixImage = FinalShrunkenSegmentedImage;
    end
% PROGRAMMING NOTE
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
    drawnow
    %%% Sets the width of the figure window to be appropriate (half width).
    if handles.setbeinganalyzed == 1
        originalsize = get(ThisAlgFigureNumber, 'position');
        newsize = originalsize;
        newsize(3) = 0.5*originalsize(3);
        set(ThisAlgFigureNumber, 'position', newsize);
    end
    %%% Activates the appropriate figure window.
    figure(ThisAlgFigureNumber);
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,1,1); imagesc(OriginalColoredLabelMatrixImage);colormap(gray);
    title([ObjectName, ' Image Set # ',num2str(handles.setbeinganalyzed)]);
    subplot(2,1,2); imagesc(ShrunkenColoredLabelMatrixImage); title(ShrunkenObjectName);colormap(gray);
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
% 'structure', and whose name is handles. Data which should be saved
% to the handles structure within each module includes: any images,
% data or measurements which are to be eventually saved to the hard
% drive (either in an output file, or using the SaveImages module) or
% which are to be used by a later module in the analysis pipeline. Any
% module which produces or passes on an image needs to also pass along
% the original filename of the image, named after the new image name,
% so that if the SaveImages module attempts to save the resulting
% image, it can be named by appending text to the original file name.
% handles.Pipeline is for storing data which must be retrieved by other modules.
% This data can be overwritten as each image set is processed, or it
% can be generated once and then retrieved during every subsequent image
% set's processing, or it can be saved for each image set by
% saving it according to which image set is being analyzed.
%       Anything stored in handles.Measurements or handles.Pipeline
% will be deleted at the end of the analysis run, whereas anything
% stored in handles.Settings will be retained from one analysis to the
% next. It is important to think about which of these data should be
% deleted at the end of an analysis run because of the way Matlab
% saves variables: For example, a user might process 12 image sets of
% nuclei which results in a set of 12 measurements ("TotalNucArea")
% stored in the handles structure. In addition, a processed image of
% nuclei from the last image set is left in the handles structure
% ("SegmNucImg"). Now, if the user uses a different algorithm which
% happens to have the same measurement output name "TotalNucArea" to
% analyze 4 image sets, the 4 measurements will overwrite the first 4
% measurements of the previous analysis, but the remaining 8
% measurements will still be present. So, the user will end up with 12
% measurements from the 4 sets. Another potential problem is that if,
% in the second analysis run, the user runs only an algorithm which
% depends on the output "SegmNucImg" but does not run an algorithm
% that produces an image by that name, the algorithm will run just
% fine: it will just repeatedly use the processed image of nuclei
% leftover from the last image set, which was left in the handles
% structure ("SegmNucImg").
%       Note that two types of measurements are typically made: Object
% and Image measurements.  Object measurements have one number for
% every object in the image (e.g. ObjectArea) and image measurements
% have one number for the entire image, which could come from one
% measurement from the entire image (e.g. ImageTotalIntensity), or
% which could be an aggregate measurement based on individual object
% measurements (e.g. ImageMeanArea).  Use the appropriate prefix to
% ensure that your data will be extracted properly.
%       Saving measurements: The data extraction functions of
% CellProfiler are designed to deal with only one "column" of data per
% named measurement field. So, for example, instead of creating a
% field of XY locations stored in pairs, they should be split into a field
% of X locations and a field of Y locations. Measurements must be
% stored in double format, because the extraction part of the program
% is designed to deal with that type of array only, not cell or
% structure arrays. It is wise to include the user's input for
% 'ObjectName' as part of the fieldname in the handles structure so
% that multiple modules can be run and their data will not overwrite
% each other.
%       Extracting measurements: handles.Measurements.CenterXNuclei{1}(2) gives
% the X position for the second object in the first image.
% handles.Measurements.AreaNuclei{2}(1) gives the area of the first object in
% the second image.

%%% Saves the segmented image, not edited for objects along the edges or
%%% for size, to the handles structure.
fieldname = ['PrelimSegmented',ShrunkenObjectName];
handles.Pipeline.(fieldname) = FinalShrunkenPrelimSegmentedImage;

%%% Saves the segmented image, only edited for small objects, to the
%%% handles structure.
fieldname = ['PrelimSmallSegmented',ShrunkenObjectName];
handles.Pipeline.(fieldname) = FinalShrunkenPrelimSmallSegmentedImage;

%%% Saves the final segmented label matrix image to the handles structure.
fieldname = ['Segmented',ShrunkenObjectName];
handles.Pipeline.(fieldname) = FinalShrunkenSegmentedImage;

%%% Determines the filename of the image to be analyzed.
fieldname = ['Filename', ObjectName];
FileName = handles.Pipeline.(fieldname)(handles.setbeinganalyzed);
%%% Saves the filename of the objects created.
fieldname = ['Filename', ShrunkenObjectName];
handles.Pipeline.(fieldname)(handles.setbeinganalyzed) = FileName;