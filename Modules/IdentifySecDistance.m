function handles = AlgIdentifySecDistance6(handles)

%%% Reads the current algorithm number, since this is needed to find 
%%% the variable values that the user entered.
CurrentAlgorithm = handles.currentalgorithm;

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow

%textVAR1 = What did you call the primary objects you want to create secondary objects around? 
%defaultVAR1 = Nuclei
fieldname = ['Vvariable',CurrentAlgorithm,'_01'];
PrimaryObjectName = handles.(fieldname);
%textVAR2 = What do you want to call the secondary objects identified by this algorithm?
%defaultVAR2 = Cells
fieldname = ['Vvariable',CurrentAlgorithm,'_02'];
SecondaryObjectName = handles.(fieldname);
%textVAR3 = On which image would you like to display the outlines of the secondary objects?
%defaultVAR3 = OrigGreen
fieldname = ['Vvariable',CurrentAlgorithm,'_03'];
OrigImageName = handles.(fieldname);
%textVAR5 = Set the number of pixels by which to expand the primary objects [Positive number]
%defaultVAR5 = 10
fieldname = ['Vvariable',CurrentAlgorithm,'_05'];
DistanceToDilate = str2num(handles.(fieldname));

%textVAR8 = To save object outlines as an image, enter text to append to the name 
%defaultVAR8 = N
fieldname = ['Vvariable',CurrentAlgorithm,'_08'];
SaveObjectOutlines = handles.(fieldname);
%textVAR9 = To save colored object blocks as an image, enter text to append to the name 
%defaultVAR9 = N
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
drawnow

%%% Checks whether the file format the user entered is readable by Matlab.
IsFormat = imformats(FileFormat);
if isempty(IsFormat) == 1
    error('The image file type entered in the Identify Secondary Distance module is not recognized by Matlab. Or, you may have entered a period in the box. For a list of recognizable image file formats, type "imformats" (no quotes) at the command line in Matlab.','Error')
end

%%% Retrieve the label matrix image that contains the edited primary
%%% segmented objects which will be used for dilation. Checks first to see
%%% whether the appropriate image exists.
fieldname = ['dOTSegmented',PrimaryObjectName];
%%% Checks whether the image exists in the handles structure.
if isfield(handles, fieldname) == 0
    error(['Image processing has been canceled. Prior to running the Identify Secondary Distance module, you must have previously run an algorithm that generates an image with the primary objects identified.  You specified in the Identify Secondary Distance module that the primary objects were named ', PrimaryObjectName, ' as a result of the previous algorithm, which should have produced an image called ', fieldname, ' in the handles structure.  The Identify Secondary Distance module cannot locate this image.']);
end
PrimaryLabelMatrixImage = handles.(fieldname);
    % figure, imshow(PrimaryLabelMatrixImage), title('PrimaryLabelMatrixImage')

%%% Retrieves the preliminary label matrix image that contains the primary
%%% segmented objects which have only been edited to discard objects 
%%% that are smaller than a certain size.  This image
%%% will be used as markers to segment the secondary objects with this
%%% algorithm.  Checks first to see whether the appropriate image exists.
fieldname = ['dOTPrelimSmallSegmented',PrimaryObjectName];
%%% Checks whether the image exists in the handles structure.
    if isfield(handles, fieldname) == 0
    error(['Image processing has been canceled. Prior to running the Identify Secondary Distance module, you must have previously run an algorithm that generates an image with the preliminary primary objects identified.  You specified in the Identify Secondary Distance module that the primary objects were named ', PrimaryObjectName, ' as a result of the previous algorithm, which should have produced an image called ', fieldname, ' in the handles structure.  The Identify Secondary Distance module cannot locate this image.']);
    end
PrelimPrimaryLabelMatrixImage = handles.(fieldname);
     % figure, imshow(PrelimPrimaryLabelMatrixImage), title('PrelimPrimaryLabelMatrixImage')
        
%%% Read (open) the image you want to analyze and assign it to a variable,
%%% "OrigImage".
fieldname = ['dOT', OrigImageName];
%%% Check whether the image to be analyzed exists in the handles structure.
if isfield(handles, fieldname) == 0
    %%% If the image is not there, an error message is produced.  The error
    %%% is not displayed: The error function halts the current function and
    %%% returns control to the calling function (the analyze all images
    %%% button callback.)  That callback recognizes that an error was
    %%% produced because of its try/catch loop and breaks out of the image
    %%% analysis loop without attempting further modules.
    error(['Image processing was canceled because the Identify Secondary Distance module could not find the input image.  It was supposed to be named ', OrigImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
OrigImage = handles.(fieldname);
% figure, imshow(OrigImage), title('OrigImage')

%%% Check whether the appendages to be added to the file names of images
%%% will result in overwriting the original file, or in a file name that
%%% contains spaces.
%%% Determine the filename of the image to be analyzed.
fieldname = ['dOTFilename', OrigImageName];
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
    error('Image processing was canceled because you have entered one or more spaces in the box of text to append to the object outlines image name in the Identify Secondary Distance module.  If you do not want to save the object outlines image to the hard drive, type "N" into the appropriate box.')
end
%%% Check whether the new image name is going to result in overwriting the
%%% original file.
B = strcmp(upper(CharFileName), upper(NewImageNameSaveObjectOutlines));
if B == 1
    error('Image processing was canceled because you have not entered text to append to the object outlines image name in the Identify Secondary Distance module.  If you do not want to save the object outlines image to the hard drive, type "N" into the appropriate box.')
end

%%% Repeat the above for the other image to be saved: 
NewImageNameSaveColoredObjects = [BareFileName,SaveColoredObjects,'.',FileFormat];
A = isspace(SaveColoredObjects);
if any(A) == 1
    error('Image processing was canceled because you have entered one or more spaces in the box of text to append to the colored objects image name in the Identify Secondary Distance module.  If you do not want to save the colored objects image to the hard drive, type "N" into the appropriate box.')
end
B = strcmp(upper(CharFileName), upper(NewImageNameSaveColoredObjects));
if B == 1
    error('Image processing was canceled because you have not entered text to append to the colored objects image name in the Identify Secondary Distance module.  If you do not want to save the colored objects image to the hard drive, type "N" into the appropriate box.')
end

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%

%%% Creates the structuring element using the user-specified size.
StructuringElement = strel('disk', DistanceToDilate);
%%% Dilates the preliminary label matrix image (edited for small only).
DilatedPrelimSecObjectLabelMatrixImage = imdilate(PrelimPrimaryLabelMatrixImage, StructuringElement);
    % figure, imshow(DilatedPrelimSecObjectLabelMatrixImage, []), title('DilatedPrelimSecObjectLabelMatrixImage')
    % imwrite(DilatedPrelimSecObjectLabelMatrixImage, [BareFileName,'DSOLMI','.',FileFormat], FileFormat);
%%% Converts to binary.
DilatedPrelimSecObjectBinaryImage = im2bw(DilatedPrelimSecObjectLabelMatrixImage,0.1);
    % figure, imshow(DilatedPrelimSecObjectBinaryImage, []), title('DilatedPrelimSecObjectBinaryImage')
    % imwrite(DilatedPrelimSecObjectBinaryImage, [BareFileName,'DSOBI','.',FileFormat], FileFormat);
%%% Computes nearest neighbor image of nuclei centers so that the dividing
%%% line between secondary objects is halfway between them rather than
%%% favoring the primary object with the greater label number.
[ignore, Labels] = bwdist(full(PrelimPrimaryLabelMatrixImage>0));
%%% Remaps labels in Labels to labels in PrelimPrimaryLabelMatrixImage.
ExpandedRelabeledDilatedPrelimSecObjectImage = PrelimPrimaryLabelMatrixImage(Labels);
    % figure, imshow(ExpandedRelabeledDilatedPrelimSecObjectImage, []), title('ExpandedRelabeledDilatedPrelimSecObjectImage')
    % imwrite(ExpandedRelabeledDilatedPrelimSecObjectImage, [BareFileName,'RI','.',FileFormat], FileFormat);
%%% Removes the background pixels (those not labeled as foreground in the
%%% DilatedPrelimSecObjectBinaryImage). This is necessary because the
%%% nearest neighbor function assigns *every* pixel to a nucleus, not just
%%% the pixels that are part of a secondary object.
RelabeledDilatedPrelimSecObjectImage = zeros(size(ExpandedRelabeledDilatedPrelimSecObjectImage));
RelabeledDilatedPrelimSecObjectImage(DilatedPrelimSecObjectBinaryImage) = ExpandedRelabeledDilatedPrelimSecObjectImage(DilatedPrelimSecObjectBinaryImage);
    % figure, imshow(RelabeledDilatedPrelimSecObjectImage, []), title('RelabeledDilatedPrelimSecObjectImage')
    % imwrite(RelabeledDilatedPrelimSecObjectImage, [BareFileName,'FSOLMI','.',FileFormat], FileFormat);

%%% Now, remove objects that are not in the edited PrimaryLabelMatrixImage.
LookUpTable = sortrows(unique([PrelimPrimaryLabelMatrixImage(:) PrimaryLabelMatrixImage(:)],'rows'),[1]);
LookUpColumn = LookUpTable(:,2);

FinalSecObjectsLabelMatrixImage = LookUpColumn(RelabeledDilatedPrelimSecObjectImage+1);
       % figure, imshow(FinalSecObjectsLabelMatrixImage, []), title('FinalSecObjectsLabelMatrixImage'), 

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

%%% Calculate the ColoredLabelMatrixImage for displaying in the figure
%%% window in subplot(2,2,2).
%%% Note that the label2rgb function doesn't work when there are no objects
%%% in the label matrix image, so there is an "if".
if sum(sum(FinalSecObjectsLabelMatrixImage)) >= 1
    ColoredLabelMatrixImage = label2rgb(FinalSecObjectsLabelMatrixImage,'jet', 'k', 'shuffle');
    % figure, imshow(ColoredLabelMatrixImage, []), title('ColoredLabelMatrixImage')
    % imwrite(ColoredLabelMatrixImage, [BareFileName,'CLMI2','.',FileFormat], FileFormat);
else ColoredLabelMatrixImage = FinalSecObjectsLabelMatrixImage;
end

%%% Calculate OutlinesOnOriginalImage for displaying in the figure
%%% window in subplot(2,2,3).        
StructuringElement3 = [0 0 0; 0 1 -1; 0 0 0];
OutlinesDirection1 = filter2(StructuringElement3, FinalSecObjectsLabelMatrixImage);
OutlinesDirection2 = filter2(StructuringElement3', FinalSecObjectsLabelMatrixImage);
SecondaryObjectOutlines = OutlinesDirection1 | OutlinesDirection2;
    % figure, imshow(SecondaryObjectOutlines, []), title('SecondaryObjectOutlines')
    % imwrite(SecondaryObjectOutlines, [BareFileName,'SOO','.',FileFormat], FileFormat);
%%% Overlay the watershed lines on the original image.
OutlinesOnOriginalImage = OrigImage;
%%% Determines the grayscale intensity to use for the cell outlines.
LineIntensity = max(OrigImage(:));
OutlinesOnOriginalImage(SecondaryObjectOutlines == 1) = LineIntensity;
    % figure, imshow(OutlinesOnOriginalImage, []), title('OutlinesOnOriginalImage')
    % imwrite(OutlinesOnOriginalImage, [BareFileName,'OOOI','.',FileFormat], FileFormat);

%%% Calculate BothOutlinesOnOriginalImage for displaying in the figure
%%% window in subplot(2,2,4).
%%% Converts the PrimaryLabelMatrixImage to binary.
PrimaryBinaryImage = im2bw(PrimaryLabelMatrixImage,.1);
%%% Dilates the Primary Binary Image by one pixel (8 neighborhood).
StructuringElement2 = strel('square',3);
DilatedPrimaryBinaryImage = imdilate(PrimaryBinaryImage, StructuringElement2);
        % figure, imshow(DilatedPrimaryBinaryImage, []), title('DilatedPrimaryBinaryImage')
        % imwrite(DilatedPrimaryBinaryImage, [BareFileName,'DPBI','.',FileFormat], FileFormat);
%%% Subtracts the PrimaryBinaryImage from the DilatedPrimaryBinaryImage,
%%% which leaves the PrimaryObjectOutlines.
PrimaryObjectOutlines = DilatedPrimaryBinaryImage - PrimaryBinaryImage;
         % figure, imshow(PrimaryObjectOutlines, []), title('NoneditedPrimaryObjectOutlines')
         % imwrite(PrimaryObjectOutlines, [BareFileName,'POO','.',FileFormat], FileFormat);
%%% Writes the outlines onto the original image.
BothOutlinesOnOriginalImage = OutlinesOnOriginalImage;
BothOutlinesOnOriginalImage(PrimaryObjectOutlines == 1) = LineIntensity;
        % figure, imshow(BothOutlinesOnOriginalImage, []), title('BothOutlinesOnOriginalImage')
        % imwrite(BothOutlinesOnOriginalImage, [BareFileName,'BOOOI','.',FileFormat], FileFormat);

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
    %%% Activates the appropriate figure window.
    figure(ThisAlgFigureNumber);
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,2,1); imagesc(OrigImage);colormap(gray);
    title(['Input Image, Image Set # ',num2str(handles.setbeinganalyzed)]);
    %%% A subplot of the figure window is set to display the colored label
    %%% matrix image.
    subplot(2,2,2); imagesc(ColoredLabelMatrixImage); title(['Segmented ',SecondaryObjectName]);
    %%% A subplot of the figure window is set to display the original image
    %%% with outlines drawn on top.
    subplot(2,2,3); imagesc(OutlinesOnOriginalImage); colormap(gray); title([SecondaryObjectName, ' Outlines on Input Image']);
    %%% A subplot of the figure window is set to display the original image
    %%% with outlines drawn on top.
    subplot(2,2,4); imagesc(BothOutlinesOnOriginalImage); colormap(gray); title(['Outlines of ', PrimaryObjectName, ' and ', SecondaryObjectName, ' on Input Image']);
end
%%% Executes pending figure-related commands so that the results are
%%% displayed.
drawnow

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% The final, segmented label matrix image of secondary objects is saved to the
%%% handles structure so it can be used by subsequent algorithms.
fieldname = ['dOTSegmented',SecondaryObjectName];
handles.(fieldname) = FinalSecObjectsLabelMatrixImage;
%%% Removed for parallel: guidata(gcbo, handles);

%%% Save the filename of the image to be analyzed.
fieldname = ['dOTFilename', SecondaryObjectName];
handles.(fieldname)(handles.setbeinganalyzed) = FileName;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE PROCESSED IMAGE TO HARD DRIVE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Determine whether the user wanted to save the image of object outlines
%%% by comparing their entry "SaveObjectOutlines" with "N" (after
%%% converting SaveObjectOutlines to uppercase).
if strcmp(upper(SaveObjectOutlines),'N') ~= 1
%%% Save the image to the hard drive.    
imwrite(SecondaryObjectOutlines, NewImageNameSaveObjectOutlines, FileFormat);
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

%%%%% Help for the Identify Secondary Distance module: 
%%%%% .
%%%%% SETTINGS:
%%%%% 
%%%%% .

%%%%% DISPLAYING AND SAVING PROCESSED IMAGES 
%%%%% PRODUCED BY THIS IMAGE ANALYSIS MODULE:
%%%%% Note: Images saved using the boxes in the main CellProfiler window
%%%%% will be saved in the default directory specified in STEP 1.
%%%%% .
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