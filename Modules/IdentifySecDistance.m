function handles = AlgIdentifySecDistance(handles)

% Help for the Identify Secondary Distance module:
% Category: Object Identification
%
% Based on another module's identification of primary objects, this
% module identifies secondary objects when no specific staining is
% available.  The edges of the primary objects are simply expanded a
% particular distance to create the secondary objects. For example, if
% nuclei are labeled but there is no stain to help locate cell edges,
% the nuclei can simply be expanded in order to estimate the cell's
% location.  This is a standard module used in commercial software and
% is known as the 'donut' or 'annulus' approach for identifying the
% cytoplasmic compartment.
%
% What does Secondary mean?
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
% SAVING IMAGES: The images of the objects produced by this module can
% be easily saved using the Save Images module using the name:
% Segmented + whatever you called the objects (e.g. SegmentedCells).
% This will be a grayscale image where each object is a different
% intensity.
% 
% Several additional images are normally calculated for display only,
% including the colored label matrix image (the objects displayed as
% arbitrary colors), object outlines, and object outlines overlaid on
% the original image, and object outlines plus primary object outlines
% on the original image. These images can be saved by altering the
% code for this module to save those images to the handles structure
% (see the SaveImages module help) and then using the Save Images
% module.  Important note: The calculations of these display images
% are only performed if the figure window is open, so the figure
% window must be left open or the Save Images module will fail.  If
% you are running the job on a cluster, figure windows are not open,
% so the Save Images module will also fail, unless you go into the
% code for this module and remove the 'if/end' statement surrounding
% the DISPLAY RESULTS section.
%
% See also ALGIDENTIFYSECPROPAGATE, ALGIDENTIFYSECWATERSHED.

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
% The Original Code is the Identify Secondary Distance module.
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

%textVAR01 = What did you call the primary objects you want to create secondary objects around? 
%defaultVAR01 = Nuclei
PrimaryObjectName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,1});

%textVAR02 = What do you want to call the secondary objects identified by this algorithm?
%defaultVAR02 = Cells
SecondaryObjectName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,2});

%textVAR03 = On which image would you like to display the outlines of the secondary objects?
%defaultVAR03 = OrigGreen
OrigImageName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,3});

%textVAR04 = Set the number of pixels by which to expand the primary objects [Positive number]
%defaultVAR04 = 10
DistanceToDilate = str2double(char(handles.Settings.Vvariable{CurrentAlgorithmNum,4}));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a variable,
%%% "OrigImage".
fieldname = ['dOT', OrigImageName];
%%% Checks whether the image to be analyzed exists in the handles structure.
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

%%% Retrieves the label matrix image that contains the edited primary
%%% segmented objects which will be used for dilation. Checks first to see
%%% whether the appropriate image exists.
fieldname = ['dOTSegmented',PrimaryObjectName];
%%% Checks whether the image exists in the handles structure.
if isfield(handles, fieldname) == 0
    error(['Image processing has been canceled. Prior to running the Identify Secondary Distance module, you must have previously run an algorithm that generates an image with the primary objects identified.  You specified in the Identify Secondary Distance module that the primary objects were named ', PrimaryObjectName, ' as a result of the previous algorithm, which should have produced an image called ', fieldname, ' in the handles structure.  The Identify Secondary Distance module cannot locate this image.']);
end
PrimaryLabelMatrixImage = handles.(fieldname);

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

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Creates the structuring element using the user-specified size.
StructuringElement = strel('disk', DistanceToDilate);
%%% Dilates the preliminary label matrix image (edited for small only).
DilatedPrelimSecObjectLabelMatrixImage = imdilate(PrelimPrimaryLabelMatrixImage, StructuringElement);
%%% Converts to binary.
DilatedPrelimSecObjectBinaryImage = im2bw(DilatedPrelimSecObjectLabelMatrixImage,0.1);
%%% Computes nearest neighbor image of nuclei centers so that the dividing
%%% line between secondary objects is halfway between them rather than
%%% favoring the primary object with the greater label number.
[ignore, Labels] = bwdist(full(PrelimPrimaryLabelMatrixImage>0)); %#ok We want to ignore MLint error checking for this line.
drawnow
%%% Remaps labels in Labels to labels in PrelimPrimaryLabelMatrixImage.
ExpandedRelabeledDilatedPrelimSecObjectImage = PrelimPrimaryLabelMatrixImage(Labels);
%%% Removes the background pixels (those not labeled as foreground in the
%%% DilatedPrelimSecObjectBinaryImage). This is necessary because the
%%% nearest neighbor function assigns *every* pixel to a nucleus, not just
%%% the pixels that are part of a secondary object.
RelabeledDilatedPrelimSecObjectImage = zeros(size(ExpandedRelabeledDilatedPrelimSecObjectImage));
RelabeledDilatedPrelimSecObjectImage(DilatedPrelimSecObjectBinaryImage) = ExpandedRelabeledDilatedPrelimSecObjectImage(DilatedPrelimSecObjectBinaryImage);
drawnow
%%% Removes objects that are not in the edited PrimaryLabelMatrixImage.
LookUpTable = sortrows(unique([PrelimPrimaryLabelMatrixImage(:) PrimaryLabelMatrixImage(:)],'rows'),1);
LookUpColumn = LookUpTable(:,2);
FinalSecObjectsLabelMatrixImage = LookUpColumn(RelabeledDilatedPrelimSecObjectImage+1);

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
    if sum(sum(FinalSecObjectsLabelMatrixImage)) >= 1
        ColoredLabelMatrixImage = label2rgb(FinalSecObjectsLabelMatrixImage,'jet', 'k', 'shuffle');
    else ColoredLabelMatrixImage = FinalSecObjectsLabelMatrixImage;
    end

    %%% Calculates OutlinesOnOriginalImage for displaying in the figure
    %%% window in subplot(2,2,3).
    StructuringElement3 = [0 0 0; 0 1 -1; 0 0 0];
    OutlinesDirection1 = filter2(StructuringElement3, FinalSecObjectsLabelMatrixImage);
    OutlinesDirection2 = filter2(StructuringElement3', FinalSecObjectsLabelMatrixImage);
    SecondaryObjectOutlines = OutlinesDirection1 | OutlinesDirection2;
    %%% Overlay the watershed lines on the original image.
    OutlinesOnOriginalImage = OrigImage;
    %%% Determines the grayscale intensity to use for the cell outlines.
    LineIntensity = max(OrigImage(:));
    OutlinesOnOriginalImage(SecondaryObjectOutlines == 1) = LineIntensity;

    %%% Calculates BothOutlinesOnOriginalImage for displaying in the figure
    %%% window in subplot(2,2,4).
    %%% Converts the PrimaryLabelMatrixImage to binary.
    PrimaryBinaryImage = im2bw(PrimaryLabelMatrixImage,.1);
    %%% Dilates the Primary Binary Image by one pixel (8 neighborhood).
    StructuringElement2 = strel('square',3);
    DilatedPrimaryBinaryImage = imdilate(PrimaryBinaryImage, StructuringElement2);
    %%% Subtracts the PrimaryBinaryImage from the DilatedPrimaryBinaryImage,
    %%% which leaves the PrimaryObjectOutlines.
    PrimaryObjectOutlines = DilatedPrimaryBinaryImage - PrimaryBinaryImage;
    %%% Writes the outlines onto the original image.
    BothOutlinesOnOriginalImage = OutlinesOnOriginalImage;
    BothOutlinesOnOriginalImage(PrimaryObjectOutlines == 1) = LineIntensity;
    
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the final, segmented label matrix image of secondary objects to
%%% the handles structure so it can be used by subsequent algorithms.
fieldname = ['dOTSegmented',SecondaryObjectName];
handles.(fieldname) = FinalSecObjectsLabelMatrixImage;

%%% Determines the filename of the image that was analyzed.
%%% This is not entirely necessary, because this image was not actually
%%% used for analysis, it was only used for display, but it allows this
%%% module to be consistent with the other secondary object-identifying
%%% modules.
fieldname = ['dOTFilename', OrigImageName];
FileName = handles.(fieldname)(handles.setbeinganalyzed);
%%% Saves the filename of the image that was analyzed.
fieldname = ['dOTFilename', SecondaryObjectName];
handles.(fieldname)(handles.setbeinganalyzed) = FileName;