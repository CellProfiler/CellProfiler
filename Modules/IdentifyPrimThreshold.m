function handles = AlgIdentifyPrimThreshold(handles)

% Help for the Identify Primary Threshold module:
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

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow 

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
fieldname = ['dOT', ImageName];
%%% Checks whether the image exists in the handles structure.
    if isfield(handles, fieldname) == 0
    error(['Image processing has been canceled. Prior to running the Identify Primary Threshold module, you must have previously run an algorithm to load an image. You specified in the Identify Primary Threshold module that this image was called ', ImageName, ' which should have produced a field in the handles structure called ', fieldname, '. The Identify Primary Threshold module cannot find this image.']);
    end
OrigImageToBeAnalyzed = handles.(fieldname);

%%% Checks that the original image is two-dimensional (i.e. not a color
%%% image), which would disrupt several of the image functions.
if ndims(OrigImageToBeAnalyzed) ~= 2
    error('Image processing was canceled because the Identify Primary Threshold module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
end

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow

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
            fieldname = ['dOTPathName', ImageName];
            try PathName = handles.(fieldname);
            catch error('Image processing was canceled because the Identify Primary Threshold module must be run using images straight from a load images module (i.e. the images cannot have been altered by other image processing modules). This is because you have asked the Identify Primary Threshold module to calculate a threshold based on all of the images before identifying objects within each individual image as CellProfiler cycles through them. One solution is to process the entire batch of images using the image analysis modules preceding this module and save the resulting images to the hard drive, then start a new stage of processing from this Identify Primary Threshold module onward.')
            end
            %%% Changes to that directory.
            cd(PathName)
            %%% Retrieves the list of filenames where the images are stored from the
            %%% handles structure.
            fieldname = ['dOTFileList', ImageName];
            FileList = handles.(fieldname);
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
        fieldname = ['dOTThreshold', ImageName];
        handles.(fieldname) = Threshold;
        cd(CurrentDirectory)
    else fieldname = ['dOTThreshold', ImageName];
        Threshold = handles.(fieldname);
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

%%% Determines the figure number to display in.
fieldname = ['figurealgorithm',CurrentAlgorithm];
ThisAlgFigureNumber = handles.(fieldname);
%%% Checks whether that figure is open. This checks all the figure handles
%%% for one whose handle is equal to the figure number for this algorithm.
%%% Note: Everything between the "if" and "end" is not carried out if the
%%% user has closed the figure window, so do not do any important
%%% calculations here. Otherwise an error message will be produced if the
%%% user has closed the window but you have attempted to access data that
%%% was supposed to be produced by this part of the code.
if any(findobj == ThisAlgFigureNumber) == 1;
    %%% THE FOLLOWING CALCULATIONS ARE FOR DISPLAY PURPOSES ONLY: The
    %%% resulting images are shown in the figure window (if open), or saved
    %%% to the hard drive (if desired).  To speed execution, all of this
    %%% code has been moved to within the if statement in the figure window
    %%% display section and then after starting image analysis, the figure
    %%% window can be closed.  Just remember that when the figure window is
    %%% closed, nothing within the if loop is carried out, so you would not
    %%% be able to save images depending on these lines to the hard drive,
    %%% for example.  If you plan to save images, these lines should be
    %%% moved outside this if statement.

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
    %%% The "drawnow" function executes any pending figure window-related
    %%% commands.  In general, Matlab does not update figure windows
    %%% until breaks between image analysis modules, or when a few select
    %%% commands are used. "figure" and "drawnow" are two of the commands
    %%% that allow Matlab to pause and carry out any pending figure window-
    %%% related commands (like zooming, or pressing timer pause or cancel
    %%% buttons or pressing a help button.)  If the drawnow command is not
    %%% used immediately prior to the figure(ThisAlgFigureNumber) line,
    %%% then immediately after the figure line executes, the other commands
    %%% that have been waiting are executed in the other windows.  Then,
    %%% when Matlab returns to this module and goes to the subplot line,
    %%% the figure which is active is not necessarily the correct one.
    %%% This results in strange things like the subplots appearing in the
    %%% timer window or in the wrong figure window, or in help dialog boxes.
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

%%% Saves the segmented image, not edited for objects along the edges or
%%% for size, to the handles structure.
fieldname = ['dOTPrelimSegmented',ObjectName];
handles.(fieldname) = PrelimLabelMatrixImage1;

%%% Saves the segmented image, only edited for small objects, to the
%%% handles structure.
fieldname = ['dOTPrelimSmallSegmented',ObjectName];
handles.(fieldname) = PrelimLabelMatrixImage2;

%%% Saves the final segmented label matrix image to the handles structure.
fieldname = ['dOTSegmented',ObjectName];
handles.(fieldname) = FinalLabelMatrixImage;

%%% Saves the Threshold value to the handles structure.
fieldname = ['dMTThreshold', ObjectName];
handles.(fieldname)(handles.setbeinganalyzed) = {Threshold};

%%% Determines the filename of the image to be analyzed.
fieldname = ['dOTFilename', ImageName];
FileName = handles.(fieldname)(handles.setbeinganalyzed);
%%% Saves the filename of the image to be analyzed.
fieldname = ['dOTFilename', ObjectName];
handles.(fieldname)(handles.setbeinganalyzed) = FileName;