function handles = AlgShrinkPrimaryObjects(handles)

% Help for the Shrink Primary Objects module: 
%
% The module shrinks primary objects by removing border pixels. The
% user can specify a certain number of times the border pixels are
% removed, or type “Inf” to shrink objects down to a point. Objects
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

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow

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
fieldname = ['dOTPrelimSegmented',ObjectName];
%%% Checks whether the image to be analyzed exists in the handles structure.
if isfield(handles, fieldname) == 0
    error(['Image processing was canceled because the Shrink Primary Objects module could not find the input image.  It was supposed to be produced by an Identify Primary module in which the objects were named ', ObjectName, '.  Perhaps there is a typo in the name.'])
end
PrelimSegmentedImage = handles.(fieldname);

%%% Retrieves the segmented image, only edited for small objects.
fieldname = ['dOTPrelimSmallSegmented',ObjectName];
%%% Checks whether the image to be analyzed exists in the handles structure.
if isfield(handles, fieldname) == 0
    error(['Image processing was canceled because the Shrink Primary Objects module could not find the input image.  It was supposed to be produced by an Identify Primary module in which the objects were named ', ObjectName, '.  Perhaps there is a typo in the name.'])
end
PrelimSmallSegmentedImage = handles.(fieldname);

%%% Retrieves the final segmented label matrix image.
fieldname = ['dOTSegmented',ObjectName];
%%% Checks whether the image to be analyzed exists in the handles structure.
if isfield(handles, fieldname) == 0
    error(['Image processing was canceled because the Shrink Primary Objects module could not find the input image.  It was supposed to be produced by an Identify Primary module in which the objects were named ', ObjectName, '.  Perhaps there is a typo in the name.'])
end
SegmentedImage = handles.(fieldname);

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow

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

%%% Saves the segmented image, not edited for objects along the edges or
%%% for size, to the handles structure.
fieldname = ['dOTPrelimSegmented',ShrunkenObjectName];
handles.(fieldname) = FinalShrunkenPrelimSegmentedImage;

%%% Saves the segmented image, only edited for small objects, to the
%%% handles structure.
fieldname = ['dOTPrelimSmallSegmented',ShrunkenObjectName];
handles.(fieldname) = FinalShrunkenPrelimSmallSegmentedImage;

%%% Saves the final segmented label matrix image to the handles structure.
fieldname = ['dOTSegmented',ShrunkenObjectName];
handles.(fieldname) = FinalShrunkenSegmentedImage;

%%% Determines the filename of the image to be analyzed.
fieldname = ['dOTFilename', ObjectName];
FileName = handles.(fieldname)(handles.setbeinganalyzed);
%%% Saves the filename of the objects created.
fieldname = ['dOTFilename', ShrunkenObjectName];
handles.(fieldname)(handles.setbeinganalyzed) = FileName;