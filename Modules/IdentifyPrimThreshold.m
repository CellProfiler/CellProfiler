function handles = AlgIdentifyPrimThreshold5(handles)
%%% This image analysis module identifies objects by simply thresholding
%%% the image.  The threshold can be (1) user-specified (2) based on each
%%% individual image, or (3) based on all of the images in the set.

%%% Reads the current algorithm number, since this is needed to find 
%%% the variable values that the user entered.
CurrentAlgorithm = handles.currentalgorithm;

%%% The "drawnow" function allows figure windows to be updated and buttons
%%% to be pushed (like the pause, cancel, help, and view buttons).  The
%%% "drawnow" function is sprinkled throughout the algorithm so there are
%%% plenty of breaks where the figure windows/buttons can be interacted
%%% with.
drawnow 

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%

%textVAR01 = What did you call the images you want to process? 
%defaultVAR01 = OrigBlue
fieldname = ['Vvariable',CurrentAlgorithm,'_01'];
ImageName = handles.(fieldname);
%textVAR02 = What do you want to call the objects identified by this algorithm?
%defaultVAR02 = Nuclei
fieldname = ['Vvariable',CurrentAlgorithm,'_02'];
ObjectName = handles.(fieldname);
%textVAR03 = Size range (in pixels) of objects to include (1,99999 = do not discard any)
%defaultVAR03 = 1,99999
fieldname = ['Vvariable',CurrentAlgorithm,'_03'];
SizeRange = handles.(fieldname);
%textVAR04 = Set the Threshold (Between 0 and 1). Or, type "All" to calculate the threshold based 
%defaultVAR04 = All
fieldname = ['Vvariable',CurrentAlgorithm,'_04'];
Threshold = handles.(fieldname);
%textVAR05 = on all of the images or type "Each" to calculate the threshold for each individual
%textVAR06 = image and enter an adjustment factor here (Positive number):
%defaultVAR06 = 1
fieldname = ['Vvariable',CurrentAlgorithm,'_06'];
ThresholdAdjustmentFactor = str2num(handles.(fieldname));

%textVAR08 = To save object outlines as an image, enter text to append to the name 
%defaultVAR08 = N
fieldname = ['Vvariable',CurrentAlgorithm,'_08'];
SaveObjectOutlines = handles.(fieldname);
%textVAR09 = To save colored object blocks as an image, enter text to append to the name 
%defaultVAR09 = N
fieldname = ['Vvariable',CurrentAlgorithm,'_09'];
SaveColoredObjects = handles.(fieldname);
%textVAR10 = Otherwise, leave as "N". To save or display other images, press Help button
%textVAR11 = If saving images, what file format do you want to use? Do not include a period.
%defaultVAR11 = tif
fieldname = ['Vvariable',CurrentAlgorithm,'_11'];
FileFormat = handles.(fieldname);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Determines what the user entered for the size range.
SizeRangeNumerical = str2num(SizeRange);
MinSize = SizeRangeNumerical(1);
MaxSize = SizeRangeNumerical(2);

%%% Checks whether the file format the user entered is readable by Matlab.
IsFormat = imformats(FileFormat);
if isempty(IsFormat) == 1
    error('The image file type entered in the Identify Primary Threshold module is not recognized by Matlab. Or, you may have entered a period in the box. For a list of recognizable image file formats, type "imformats" (no quotes) at the command line in Matlab.','Error')
end
%%% Read (open) the image you want to analyze and assign it to a variable,
%%% "OrigImageToBeAnalyzed".
fieldname = ['dOT', ImageName];
%%% Checks whether the image exists in the handles structure.
    if isfield(handles, fieldname) == 0
    error(['Image processing has been canceled. Prior to running the Identify Primary Threshold module, you must have previously run an algorithm to load an image. You specified in the Identify Primary Threshold module that this image was called ', ImageName, ' which should have produced a field in the handles structure called ', fieldname, '. The Identify Primary Threshold module cannot find this image.']);
    end
OrigImageToBeAnalyzed = handles.(fieldname);
%%% Update the handles structure.
%%% Removed for parallel: guidata(gcbo, handles);

%%% Check whether the appendages to be added to the file names of images
%%% will result in overwriting the original file, or in a file name that
%%% contains spaces.
        
%%% Determine the filename of the image to be analyzed.
fieldname = ['dOTFilename', ImageName];
FileName = handles.(fieldname)(handles.setbeinganalyzed);
%%% Find and remove the file format extension within the original file
%%% name, but only if it is at the end. Strip the original file format extension 
%%% off of the file name, if it is present, otherwise, leave the original
%%% name intact.
CharFileName = char(FileName);
PotentialDot = CharFileName(end-3:end-3);
if strcmp(PotentialDot,'.') == 1
    BareFileName = CharFileName(1:end-4);
else BareFileName = CharFileName;
end

%%% Assemble the new image name.
NewImageNameSaveObjectOutlines = [BareFileName,SaveObjectOutlines,'.',FileFormat];
%%% Check whether the new image name is going to result in a name with
%%% spaces.
A = isspace(SaveObjectOutlines);
if any(A) == 1
    error('Image processing was canceled because you have entered one or more spaces in the box of text to append to the object outlines image name in the Identify Primary Threshold module.  If you do not want to save the object outlines image to the hard drive, type "N" into the appropriate box.')
end
%%% Check whether the new image name is going to result in overwriting the
%%% original file.
B = strcmp(upper(CharFileName), upper(NewImageNameSaveObjectOutlines));
if B == 1
    error('Image processing was canceled because you have not entered text to append to the object outlines image name in the Identify Primary Threshold module.  If you do not want to save the object outlines image to the hard drive, type "N" into the appropriate box.')
end

%%% Repeat the above for the other image to be saved: 
NewImageNameSaveColoredObjects = [BareFileName,SaveColoredObjects,'.',FileFormat];
A = isspace(SaveColoredObjects);
if any(A) == 1
    error('Image processing was canceled because you have entered one or more spaces in the box of text to append to the colored objects image name in the Identify Primary Threshold module.  If you do not want to save the colored objects image to the hard drive, type "N" into the appropriate box.')
end
B = strcmp(upper(CharFileName), upper(NewImageNameSaveColoredObjects));
if B == 1
    error('Image processing was canceled because you have not entered text to append to the colored objects image name in the Identify Primary Threshold module.  If you do not want to save the colored objects image to the hard drive, type "N" into the appropriate box.')
end

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%

%%% Checks that the original image is two-dimensional (i.e. not a color
%%% image), which would disrupt several of the image functions.
if ndims(OrigImageToBeAnalyzed) ~= 2
    error('Image processing was canceled because the Identify Primary Threshold module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
end

if strcmp(upper(Threshold), 'ALL') == 1
    fieldname = ['dOTThreshold', ImageName];
    if handles.setbeinganalyzed == 1
        try
            %%% Make note of the current directory so the module can return to it
            %%% at the end of this module.
            CurrentDirectory = cd;
            %%% Notify the user that the first image set will take much longer than
            %%% subsequent sets. 
            %%% Obtains the screen size.
            ScreenSize = get(0,'ScreenSize');
            ScreenHeight = ScreenSize(4);
            PotentialBottom = [0, (ScreenHeight-720)];
            BottomOfMsgBox = max(PotentialBottom);
            PositionMsgBox = [500 BottomOfMsgBox 350 100];
            h = msgbox('Preliminary calculations are under way for the Identify Primary Threshold module.  Subsequent image sets will be processed much more quickly than the first image set.');
            set(h, 'Position', [PositionMsgBox])
            drawnow
            %%% Retrieve the path where the images are stored from the handles
            %%% structure.
            fieldname = ['dOTPathName', ImageName];
            try PathName = handles.(fieldname);
            catch error('Image processing was canceled because the Identify Primary Threshold module must be run using images straight from a load images module (i.e. the images cannot have been altered by other image processing modules). This is because you have asked the Identify Primary Threshold module to calculate a threshold based on all of the images before identifying objects within each individual image as CellProfiler cycles through them. One solution is to process the entire batch of images using the image analysis modules preceding this module and save the resulting images to the hard drive, then start a new stage of processing from this Identify Primary Threshold module onward.')
            end
            %%% Change to that directory.
            cd(PathName)
            %%% Retrieve the list of filenames where the images are stored from the
            %%% handles structure.
            fieldname = ['dOTFileList', ImageName];
            FileList = handles.(fieldname);
            %%% Calculates the threshold based on all of the images.
            Counts = zeros(256,1);
            NumberOfBins = 256;
            for i=1:length(FileList)
                Image = imread(char(FileList(i)));
                Counts = Counts + imhist(im2uint8(Image(:)), NumberOfBins);
            end
            % Variables names are chosen to be similar to the formulas in
            % the Otsu paper.
            P = Counts / sum(Counts);
            Omega = cumsum(P);
            Mu = cumsum(P .* (1:NumberOfBins)');
            Mu_t = Mu(end);
            % Save the warning state and disable warnings to prevent divide-by-zero
            % warnings.
            State = warning;
            warning off;
            SigmaBSquared = (Mu_t * Omega - Mu).^2 ./ (Omega .* (1 - Omega));
            % Restore the warning state.
            warning(State);
            % Find the location of the maximum value of sigma_b_squared.
            % The maximum may extend over several bins, so average together the
            % locations.  If maxval is NaN, meaning that sigma_b_squared is all NaN,
            % then return 0.
            Maxval = max(SigmaBSquared);
            if isfinite(Maxval)
                Idx = mean(find(SigmaBSquared == Maxval));
                % Normalize the threshold to the range [0, 1].
                Threshold = (Idx - 1) / (NumberOfBins - 1);
            else
                Threshold = 0.0;
            end
        catch [ErrorMessage, ErrorMessage2] = lasterr;
            error(['An error occurred in the Identify Primary Threshold module. Matlab says the problem is: ', ErrorMessage, ErrorMessage2])
        end
        fieldname = ['dOTThreshold', ImageName];
        handles.(fieldname) = Threshold;
        %%% Update the handles structure.
        %%% Removed for parallel: guidata(gcbo, handles);
        cd(CurrentDirectory)
    else fieldname = ['dOTThreshold', ImageName];
        Threshold = handles.(fieldname);
    end
elseif strcmp(upper(Threshold), 'EACH') == 1
    Threshold = ThresholdAdjustmentFactor*graythresh(OrigImageToBeAnalyzed);
else Threshold = str2num(Threshold);
end
ThresholdedImage = im2bw(OrigImageToBeAnalyzed, Threshold);
    % figure, imshow(ThresholdedImage, []), title('ThresholdedImage')
    % imwrite(ThresholdedImage, [BareFileName,'TI','.',FileFormat], FileFormat);
%%% Holes in the ThresholdedImage image are filled in.
ThresholdedImage = imfill(ThresholdedImage, 'holes');
%%% Identifies objects in the binary image.
PrelimLabelMatrixImage1 = bwlabel(ThresholdedImage);
    % figure, imshow(PrelimLabelMatrixImage1, []), title('PrelimLabelMatrixImage1')
    % imwrite(PrelimLabelMatrixImage1, [BareFileName,'PLMI1','.',FileFormat], FileFormat);
%%% Finds objects larger and smaller than the user-specified size.
%%% Finds the locations and labels for the pixels that are part of an object.
AreaLocations = find(PrelimLabelMatrixImage1);
AreaLabels = PrelimLabelMatrixImage1(AreaLocations);
%%% Creates a sparse matrix with column as label and row as location,
%%% with a 1 at (A,B) if location A has label B.  Summing the columns
%%% gives the count of area pixels with a given label.  E.g. Areas(L) is the
%%% number of pixels with label L.
Areas = full(sum(sparse(AreaLocations, AreaLabels, 1)));
Map = [0,Areas];
AreasImage = Map(PrelimLabelMatrixImage1 + 1);
    % figure, imshow(AreasImage, []), title('AreasImage')
    % imwrite(AreasImage, [BareFileName,'AI','.',FileFormat], FileFormat);
%%% The small objects are overwritten with zeros.
PrelimLabelMatrixImage2 = PrelimLabelMatrixImage1;
PrelimLabelMatrixImage2(AreasImage < MinSize) = 0;
%%% Relabels so that labels are consecutive. This is important for
%%% downstream modules (IdentifySec).
PrelimLabelMatrixImage2 = bwlabel(im2bw(PrelimLabelMatrixImage2,.1));
%%% The large objects are overwritten with zeros.
PrelimLabelMatrixImage3 = PrelimLabelMatrixImage2;
if MaxSize ~= 99999
    PrelimLabelMatrixImage3(AreasImage > MaxSize) = 0;
        % figure, imshow(PrelimLabelMatrixImage3, []), title('PrelimLabelMatrixImage3')
        % imwrite(PrelimLabelMatrixImage3, [BareFileName,'PLMI3','.',FileFormat], FileFormat);
end
%%% Removes objects that are touching the edge of the image, since they
%%% won't be measured properly.
PrelimLabelMatrixImage4 = imclearborder(PrelimLabelMatrixImage3,8);
    % figure, imshow(PrelimLabelMatrixImage4, []), title('PrelimLabelMatrixImage4')
    % imwrite(PrelimLabelMatrixImage4, [BareFileName,'PLMI4','.',FileFormat], FileFormat);
%%% The PrelimLabelMatrixImage4 is converted to binary.
FinalBinaryPre = im2bw(PrelimLabelMatrixImage4,1);
% figure, imshow(FinalBinaryPre, []), title('FinalBinaryPre')
% imwrite(FinalBinaryPre, [BareFileName,'FBP','.',FileFormat], FileFormat);
%%% The image is converted to label matrix format. It is necessary to do this in order
%%% to "compact" the label matrix: this way, each number corresponds to an
%%% object, with no numbers skipped.
FinalLabelMatrixImage = bwlabel(FinalBinaryPre);
% figure, imshow(FinalLabelMatrixImage, []), title('FinalLabelMatrixImage')
% imwrite(FinalLabelMatrixImage, [BareFileName,'FLMInuc','.',FileFormat], FileFormat);
drawnow 

%%% THE FOLLOWING CALCULATIONS ARE FOR DISPLAY PURPOSES ONLY: The resulting
%%% images are shown in the figure window (if open), or saved to the hard
%%% drive (if desired).  To speed execution, these lines can be removed (or
%%% have a % sign placed in front of them) as long as all the lines which
%%% depend on the resulting images are also removed (e.g. in the figure
%%% window display section).  Alternately, all of this code can be moved to
%%% within the if loop in the figure window display section and then after
%%% starting image analysis the figure window can be closed.  Just remember
%%% that when the figure window is closed, nothing within the if loop is
%%% carried out, so you would not be able to use the imwrite lines below to
%%% save images to the hard drive, for example.

%%% Calculates the ColoredLabelMatrixImage for displaying in the figure
%%% window in subplot(2,2,2).  
%%% Note that the label2rgb function doesn't work when there are no objects
%%% in the label matrix image, so there is an "if".
if sum(sum(FinalLabelMatrixImage)) >= 1
    ColoredLabelMatrixImage = label2rgb(FinalLabelMatrixImage, 'jet', 'k', 'shuffle');
    % figure, imshow(ColoredLabelMatrixImage, []), title('ColoredLabelMatrixImage')
    % imwrite(ColoredLabelMatrixImage, [BareFileName,'CLMI','.',FileFormat], FileFormat);
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
        % figure, imshow(DilatedBinaryImage, []), title('DilatedBinaryImage')
        % imwrite(DilatedBinaryImage, [BareFileName,'DBI','.',FileFormat], FileFormat);
%%% Subtracts the FinalBinaryImage from the DilatedBinaryImage,
%%% which leaves the PrimaryObjectOutlines.
PrimaryObjectOutlines = DilatedBinaryImage - FinalBinaryImage;
        % figure, imshow(PrimaryObjectOutlines, []), title('PrimaryObjectOutlines')
        % imwrite(PrimaryObjectOutlines, [BareFileName,'POO','.',FileFormat], FileFormat);
%%% Overlays the object outlines on the original image.
ObjectOutlinesOnOriginalImage = OrigImageToBeAnalyzed;
    %%% Determines the grayscale intensity to use for the cell outlines.
    LineIntensity = max(OrigImageToBeAnalyzed(:));
ObjectOutlinesOnOriginalImage(PrimaryObjectOutlines == 1) = LineIntensity;
        % figure, imshow(ObjectOutlinesOnOriginalImage, []), title('ObjectOutlinesOnOriginalImage')
        % imwrite(ObjectOutlinesOnOriginalImage, [BareFileName,'OOOOI','.',FileFormat], FileFormat);

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow 

%%% Note: Everything between the "if" and "end" is not carried out if the 
%%% user has closed
%%% the figure window, so do not do any important calculations here.
%%% Otherwise an error message will be produced if the user has closed the
%%% window but you have attempted to access data that was supposed to be
%%% produced by this part of the code.

%%% Determines the figure number to display in.
fieldname = ['figurealgorithm',CurrentAlgorithm];
ThisAlgFigureNumber = handles.(fieldname);
%%% Check whether that figure is open. This checks all the figure handles
%%% for one whose handle is equal to the figure number for this algorithm.
if any(findobj == ThisAlgFigureNumber) == 1;
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
%%% Executes pending figure-related commands so that the results are
%%% displayed.
drawnow

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

%%% Update the handles structure.
%%% Removed for parallel: guidata(gcbo, handles);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE PROCESSED IMAGE TO HARD DRIVE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Determine whether the user wanted to save the image of object outlines
%%% by comparing their entry "SaveObjectOutlines" with "N" (after
%%% converting SaveObjectOutlines to uppercase).  The appropriate names
%%% were determined towards the beginning of the module during error
%%% checking.
if strcmp(upper(SaveObjectOutlines),'N') ~= 1
%%% Save the image to the hard drive.    
imwrite(PrimaryObjectOutlines, NewImageNameSaveObjectOutlines, FileFormat);
end
%%% Same for the SaveColoredObjects image.
if strcmp(upper(SaveColoredObjects),'N') ~= 1
%%% Save the image to the hard drive.    
imwrite(ColoredLabelMatrixImage, NewImageNameSaveColoredObjects, FileFormat);
end

drawnow 

%%%%%%%%%%%
%%% HELP %%%
%%%%%%%%%%%

%%%%% Help for Identify Primary Threshold module: 
%%%%% .
%%%%% 
%%%%% .
%%%%% DISPLAYING AND SAVING PROCESSED IMAGES 
%%%%% PRODUCED BY THIS IMAGE ANALYSIS MODULE:
%%%%% Note: Images saved using the boxes in the main CellProfiler window
%%%%% will be saved in the default directory specified in STEP 1.
%%%%% If you want to save other processed images, open the m-file for this 
%%%%% image analysis module, go to the line in the
%%%%% m-file where the image is generated, and there should be 2 lines
%%%%% which have been inactivated.  These are green comment lines that are
%%%%% indented. To display an image, remove the percent sign before
%%%%% the line that says "figure, imshow...". This will cause the image to
%%%%% appear in a fresh display window for every image set. To save an
%%%%% image to the hard drive, remove the percent sign before the line
%%%%% that says "imwrite..." and adjust the file type and appendage to the
%%%%% file name as desired.  When you have finished removing the percent
%%%%% signs, go to File > Save As and save the m file with a new name.
%%%%% Then load the new image analysis module into the CellProfiler as
%%%%% usual.
%%%%% Please note that not all of these imwrite lines have been checked for
%%%%% functionality: it may be that you will have to alter the format of
%%%%% the image before saving.  Try, for example, adding the uint8 command:
%%%%% uint8(Image) surrounding the image prior to using the imwrite command
%%%%% if the image is not saved correctly.


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
% The Original Code is the ______________________.
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