function handles = AlgIdentifyPrimThreshold(handles)

% Help for the Identify Primary Threshold module:
% Category: Object Identification
%
% This image analysis module identifies objects by simply thresholding
% the image, with a single, global threshold used across the image.
% This module does not separate clumps well, although it is fast; it
% is ideal for well-separated nuclei. The threshold can be (1)
% user-specified (2) based on each individual image, or (3) based on
% all of the images in the set.  Objects on the border of the image
% are ignored, and the user can select a size range, outside which
% objects will be ignored.
%
% Settings:
%
% Size range: You may exclude objects that are smaller or bigger than
% the size range you specify. A comma should be placed between the
% lower size limit and the upper size limit. The units here are pixels
% so that it is easy to zoom in on found objects and determine the
% size of what you think should be excluded.
%
% Threshold: The threshold affects the stringency of the lines between
% the objects and the background. You may enter an absolute number
% between 0 and 1 for the threshold (use 'Show pixel data' to see the
% pixel intensities for your images in the appropriate range of 0 to
% 1), or you may have it calculated for each image individually. There
% are advantages either way.  An absolute number treats every image
% identically, but an automatically calculated threshold is more
% realistic/accurate, though occasionally subject to artifacts.  The
% threshold which is used for each image is recorded as a measurement
% in the output file, so if you find unusual measurements from one of
% your images, you might check whether the automatically calculated
% threshold was unusually high or low compared to the remaining
% images.  When an automatic threshold is selected, it may
% consistently be too stringent or too lenient, so an adjustment
% factor can be entered as well. The number 1 means no adjustment, 0
% to 1 makes the threshold more lenient and greater than 1 (e.g. 1.3)
% makes the threshold more stringent. The automatic threshold has two
% options: All or Each, depending on whether you want a different
% threshold for each image or one threshold for all images, based on
% all images.
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
% See also ALGIDENTIFYPRIMADAPTTHRESHOLDA,
% ALGIDENTIFYPRIMADAPTTHRESHOLDB,
% ALGIDENTIFYPRIMADAPTTHRESHOLDC, 
% ALGIDENTIFYPRIMADAPTTHRESHOLDD, 
% ALGIDENTIFYPRIMDISTDIST,
% ALGIDENTIFYPRIMDISTINTENS, 
% ALGIDENTIFYPRIMINTENSINTENS.

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
% The Original Code is the Identify Primary Threshold module.
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

%textVAR01 = What did you call the images you want to process? 
%defaultVAR01 = OrigBlue
ImageName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,1});

%textVAR02 = What do you want to call the objects identified by this algorithm?
%defaultVAR02 = Nuclei
ObjectName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,2});

%textVAR03 = Size range (in pixels) of objects to include (1,99999 = do not discard any)
%defaultVAR03 = 1,99999
SizeRange = char(handles.Settings.Vvariable{CurrentAlgorithmNum,3});

%textVAR04 = Set the Threshold (Between 0 and 1). Or, type "All" to calculate the threshold based 
%defaultVAR04 = All
Threshold = char(handles.Settings.Vvariable{CurrentAlgorithmNum,4});

%textVAR05 = on all of the images or type "Each" to calculate the threshold for each individual
%textVAR06 = image and enter an adjustment factor here (Positive number):
%defaultVAR06 = 1
ThresholdAdjustmentFactor = str2double(char(handles.Settings.Vvariable{CurrentAlgorithmNum,6}));


%%% Determines what the user entered for the size range.
SizeRangeNumerical = str2num(SizeRange);  %#ok We want to ignore MLint error checking for this line.
MinSize = SizeRangeNumerical(1);
MaxSize = SizeRangeNumerical(2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a variable,
%%% "OrigImageToBeAnalyzed".
%%% Checks whether the image exists in the handles structure.
    if isfield(handles.Pipeline, ImageName) == 0
    error(['Image processing has been canceled. Prior to running the Identify Primary Threshold module, you must have previously run an algorithm to load an image. You specified in the Identify Primary Threshold module that this image was called ', ImageName, ' which should have produced a field in the handles structure called ', ImageName, '. The Identify Primary Threshold module cannot find this image.']);
    end
OrigImageToBeAnalyzed = handles.Pipeline.(ImageName);

%%% Checks that the original image is two-dimensional (i.e. not a color
%%% image), which would disrupt several of the image functions.
if ndims(OrigImageToBeAnalyzed) ~= 2
    error('Image processing was canceled because the Identify Primary Threshold module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
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

if strcmp(upper(Threshold), 'ALL') == 1
    if handles.setbeinganalyzed == 1
        try
            %%% Makes note of the current directory so the module can return to it
            %%% at the end of this module.
            CurrentDirectory = cd;
            %%% Notifies the user that the first image set will take much longer than
            %%% subsequent sets. 
            %%% Obtains the screen size.
            ScreenSize = get(0,'ScreenSize');
            ScreenHeight = ScreenSize(4);
            PotentialBottom = [0, (ScreenHeight-720)];
            BottomOfMsgBox = max(PotentialBottom);
            PositionMsgBox = [500 BottomOfMsgBox 350 100];
            h = msgbox('Preliminary calculations are under way for the Identify Primary Threshold module.  Subsequent image sets will be processed much more quickly than the first image set.');
            set(h, 'Position', PositionMsgBox)
            drawnow
            %%% Retrieves the path where the images are stored from the handles
            %%% structure.
            fieldname = ['Pathname', ImageName];
            try Pathname = handles.Pipeline.(fieldname);
            catch error('Image processing was canceled because the Identify Primary Threshold module must be run using images straight from a load images module (i.e. the images cannot have been altered by other image processing modules). This is because you have asked the Identify Primary Threshold module to calculate a threshold based on all of the images before identifying objects within each individual image as CellProfiler cycles through them. One solution is to process the entire batch of images using the image analysis modules preceding this module and save the resulting images to the hard drive, then start a new stage of processing from this Identify Primary Threshold module onward.')
            end
            %%% Changes to that directory.
            cd(Pathname)
            %%% Retrieves the list of filenames where the images are stored from the
            %%% handles structure.
            fieldname = ['FileList', ImageName];
            FileList = handles.Pipeline.(fieldname);
            %%% Calculates the threshold based on all of the images.
            Counts = zeros(256,1);
            NumberOfBins = 256;
            for i=1:length(FileList)
                Image = imread(char(FileList(i)));
                Counts = Counts + imhist(im2uint8(Image(:)), NumberOfBins);
                drawnow
            end
            % Variables names are chosen to be similar to the formulas in
            % the Otsu paper.
            P = Counts / sum(Counts);
            Omega = cumsum(P);
            Mu = cumsum(P .* (1:NumberOfBins)');
            Mu_t = Mu(end);
            % Saves the warning state and disable warnings to prevent divide-by-zero
            % warnings.
            State = warning;
warning off Matlab:DivideByZero
SigmaBSquared = (Mu_t * Omega - Mu).^2 ./ (Omega .* (1 - Omega));
            % Restores the warning state.
            warning(State);
            % Finds the location of the maximum value of sigma_b_squared.
            % The maximum may extend over several bins, so average together the
            % locations.  If maxval is NaN, meaning that sigma_b_squared is all NaN,
            % then return 0.
            Maxval = max(SigmaBSquared);
            if isfinite(Maxval)
                Idx = mean(find(SigmaBSquared == Maxval));
                % Normalizes the threshold to the range [0, 1].
                Threshold = (Idx - 1) / (NumberOfBins - 1);
            else
                Threshold = 0.0;
            end
        catch [ErrorMessage, ErrorMessage2] = lasterr;
            error(['An error occurred in the Identify Primary Threshold module. Matlab says the problem is: ', ErrorMessage, ErrorMessage2])
        end
        fieldname = ['Threshold', ImageName];
        handles.Pipeline.(fieldname) = Threshold;
        cd(CurrentDirectory)
    else fieldname = ['Threshold', ImageName];
        Threshold = handles.Pipeline.(fieldname);
    end
elseif strcmp(upper(Threshold), 'EACH') == 1
    Threshold = ThresholdAdjustmentFactor*graythresh(OrigImageToBeAnalyzed);
else Threshold = str2double(Threshold);
end
ThresholdedImage = im2bw(OrigImageToBeAnalyzed, Threshold);
drawnow
%%% Fills holes in the ThresholdedImage image.
ThresholdedImage = imfill(ThresholdedImage, 'holes');
%%% Identifies objects in the binary image.
PrelimLabelMatrixImage1 = bwlabel(ThresholdedImage);
%%% Finds objects larger and smaller than the user-specified size.
%%% Finds the locations and labels for the pixels that are part of an object.
AreaLocations = find(PrelimLabelMatrixImage1);
AreaLabels = PrelimLabelMatrixImage1(AreaLocations);
drawnow
%%% Creates a sparse matrix with column as label and row as location, with
%%% a 1 at (A,B) if location A has label B.  Summing the columns gives the
%%% count of area pixels with a given label.  E.g. Areas(L) is the number
%%% of pixels with label L.
Areas = full(sum(sparse(AreaLocations, AreaLabels, 1)));
Map = [0,Areas];
AreasImage = Map(PrelimLabelMatrixImage1 + 1);
%%% Overwrites the small objects with zeros.
PrelimLabelMatrixImage2 = PrelimLabelMatrixImage1;
PrelimLabelMatrixImage2(AreasImage < MinSize) = 0;
%%% Relabels so that labels are consecutive. This is important for
%%% downstream modules (IdentifySec).
PrelimLabelMatrixImage2 = bwlabel(im2bw(PrelimLabelMatrixImage2,.1));
drawnow
%%% Overwrites the large objects with zeros.
PrelimLabelMatrixImage3 = PrelimLabelMatrixImage2;
if MaxSize ~= 99999
    PrelimLabelMatrixImage3(AreasImage > MaxSize) = 0;
end
%%% Removes objects that are touching the edge of the image, since they
%%% won't be measured properly.
PrelimLabelMatrixImage4 = imclearborder(PrelimLabelMatrixImage3,8);
%%% Converts PrelimLabelMatrixImage4 to binary.
FinalBinaryPre = im2bw(PrelimLabelMatrixImage4,1);
%%% Converts the image to label matrix format. It is necessary to do this
%%% in order to "compact" the label matrix: this way, each number
%%% corresponds to an object, with no numbers skipped.
FinalLabelMatrixImage = bwlabel(FinalBinaryPre);

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
    %%% Calculates the ColoredLabelMatrixImage for displaying in the figure
    %%% window in subplot(2,2,2).
    %%% Note that the label2rgb function doesn't work when there are no objects
    %%% in the label matrix image, so there is an "if".
    if sum(sum(FinalLabelMatrixImage)) >= 1
        ColoredLabelMatrixImage = label2rgb(FinalLabelMatrixImage, 'jet', 'k', 'shuffle');
    else  ColoredLabelMatrixImage = FinalLabelMatrixImage;
    end
    %%% Calculates the object outlines, which are overlaid on the original
    %%% image and displayed in figure subplot (2,2,4).
    %%% Creates the structuring element that will be used for dilation.
    StructuringElement = strel('square',3);
    %%% Converts the FinalLabelMatrixImage to binary.
    FinalBinaryImage = im2bw(FinalLabelMatrixImage,1);
    %%% Dilates the FinalBinaryImage by one pixel (8 neighborhood).
    DilatedBinaryImage = imdilate(FinalBinaryImage, StructuringElement);
    %%% Subtracts the FinalBinaryImage from the DilatedBinaryImage,
    %%% which leaves the PrimaryObjectOutlines.
    PrimaryObjectOutlines = DilatedBinaryImage - FinalBinaryImage;
    %%% Overlays the object outlines on the original image.
    ObjectOutlinesOnOriginalImage = OrigImageToBeAnalyzed;
    %%% Determines the grayscale intensity to use for the cell outlines.
    LineIntensity = max(OrigImageToBeAnalyzed(:));
    ObjectOutlinesOnOriginalImage(PrimaryObjectOutlines == 1) = LineIntensity;
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
    figure(ThisAlgFigureNumber);
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,2,1); imagesc(OrigImageToBeAnalyzed);colormap(gray);
    title(['Input Image, Image Set # ',num2str(handles.setbeinganalyzed), ', Threshold used = ', num2str(Threshold)]);
    %%% A subplot of the figure window is set to display the colored label
    %%% matrix image.
    subplot(2,2,2); imagesc(ColoredLabelMatrixImage); title(['Segmented ',ObjectName]);
    %%% A subplot of the figure window is set to display the inverted original
    %%% image with outlines drawn on top.
    subplot(2,2,3); imagesc(ObjectOutlinesOnOriginalImage);colormap(gray); title([ObjectName, ' Outlines on Input Image']);
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
fieldname = ['PrelimSegmented',ObjectName];
handles.Pipeline.(fieldname) = PrelimLabelMatrixImage1;

%%% Saves the segmented image, only edited for small objects, to the
%%% handles structure.
fieldname = ['PrelimSmallSegmented',ObjectName];
handles.Pipeline.(fieldname) = PrelimLabelMatrixImage2;

%%% Saves the final segmented label matrix image to the handles structure.
fieldname = ['Segmented',ObjectName];
handles.Pipeline.(fieldname) = FinalLabelMatrixImage;

%%% Saves the Threshold value to the handles structure.
fieldname = ['ImageThreshold', ObjectName];
handles.Measurements.(fieldname)(handles.setbeinganalyzed) = {Threshold};

%%% Determines the filename of the image to be analyzed.
fieldname = ['Filename', ImageName];
FileName = handles.Pipeline.(fieldname)(handles.setbeinganalyzed);
%%% Saves the filename of the image to be analyzed.
fieldname = ['Filename', ObjectName];
handles.Pipeline.(fieldname)(handles.setbeinganalyzed) = FileName;