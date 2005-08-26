function handles = Exclude(handles)

% Help for the Exclude Objects module:
% Category: Object Identification and Modification
%
% This image analysis module allows you to delete the objects and
% portions of objects that are outside of a region you specify (e.g.
% nuclei outside of a tissue region).  The objects and the region
% should both result from any Identify module (Primary, Secondary, and
% Tertiary modules will all work). Once the remaining objects are
% identified, the user has the option to retain their original number
% or renumber them consecutively. Retaining their original number
% might be important if you intend to correlate measurements made on
% the remaining objects with measurements made on the original
% objects. Renumbering, on the other hand, makes the output file more
% compact and the processing quicker. In addition, some subsequent
% modules may not expect to have gaps in the list of objects (since
% the objects no longer exist) so they may not run properly if the
% objects are not renumbered.
%
% SAVING IMAGES: In addition to the object outlines and the
% pseudo-colored object images that can be saved using the
% instructions in the main CellProfiler window for this module,
% this module produces several additional images which can be
% easily saved using the Save Images module. These will be grayscale
% images where each object is a different intensity. (1) The
% preliminary segmented image, which includes objects on the edge of
% the image and objects that are outside the size range can be saved
% using the name: UneditedSegmented + whatever you called the objects
% (e.g. UneditedSegmentedNuclei). (2) The preliminary segmented image
% which excludes objects smaller than your selected size range can be
% saved using the name: SmallRemovedSegmented + whatever you called the
% objects (e.g. SmallRemovedSegmented Nuclei) (3) The final segmented
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




drawnow

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%



%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

%textVAR01 = Ignore the objects you called
%infotypeVAR01 = objectgroup
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = If they are outside the region(s) called
%infotypeVAR02 = objectgroup
MaskRegionName = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 = What do you want to call the remaining objects?
%infotypeVAR03 = objectgroup indep
%defaultVAR03 = EditedStaining
RemainingObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = For the remaining objects, do you want to retain their original number or renumber them consecutively (Retain or Renumber)? Retaining their original number might be important if you intend to correlate measurements made on the remaining objects with measurements made on the original objects.  Renumbering, on the other hand, makes the output file more compact and the processing quicker.
%choiceVAR04 = Renumber
%choiceVAR04 = Retain
Renumber = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%%%VariableRevisionNumber = 01

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the images to be used for analysis.
%%% Checks whether the images exist in the handles structure.
fieldname = ['Segmented',ObjectName];
if isfield(handles.Pipeline, fieldname) == 0
    error(['Image processing has been canceled. Prior to running the Exclude Primary Object module, you must have previously run a module to identify primary objects. You specified in the Exclude Primary Object module that these objects were called ', ObjectName, ' which should have produced a field in the handles structure called ', fieldname, '. The Exclude Objects module cannot find this image.']);
end
SegmentedObjectImage = handles.Pipeline.(fieldname);

%%% The following is only relevant for objects identified using
%%% Identify Primary modules, not Identify Secondary modules.
fieldname = ['UneditedSegmented',ObjectName];
if isfield(handles.Pipeline, fieldname) == 1
    UneditedSegmentedObjectImage = handles.Pipeline.(fieldname);
end

fieldname = ['SmallRemovedSegmented',ObjectName];
if isfield(handles.Pipeline, fieldname) == 1
    SmallRemovedSegmentedObjectImage = handles.Pipeline.(fieldname);
end

%%% The final, edited version of the Masked objects is the only one
%%% which must be loaded here.
fieldname = ['Segmented',MaskRegionName];
if isfield(handles.Pipeline, fieldname) == 0
    error(['Image processing has been canceled. Prior to running the Exclude Primary Object module, you must have previously run a module to identify primary objects. You specified in the Exclude Primary Object module that these objects were called ', MaskRegionName, ' which should have produced a field in the handles structure called ', fieldname, '. The Exclude Objects module cannot find this image.']);
end
MaskRegionObjectImage = handles.Pipeline.(fieldname);

if size(SegmentedObjectImage) ~= size(MaskRegionObjectImage)
    error(['Image processing has been canceled because in the Exclude Primary Object module, the two images in which primary objects were identified (', MaskRegionName, ' and ', ObjectName, ') are not the same size.']);
end

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow



%%% Pixels in the objects are deleted if they are not
%%% within the regions identified in the MaskRegionObjectImage.
NewSegmentedObjectImage = SegmentedObjectImage;
NewSegmentedObjectImage(MaskRegionObjectImage == 0) = 0;
if strcmp(Renumber,'Renumber') == 1
    %%% In case some objects are entirely deleted, the label matrix is
    %%% compacted so that labels are not skipped. This is done by
    %%% relabeling. The default connectivity is used, to be consistent
    %%% with the connectivity used in the IdentifyPrim modules.
    NewSegmentedObjectImage = bwlabel(NewSegmentedObjectImage);
end

%%% The following is only relevant for objects identified using
%%% Identify Primary modules, not Identify Secondary modules.
if exist('UneditedSegmentedObjectImage','var') == 1
    NewUneditedSegmentedObjectImage = UneditedSegmentedObjectImage;
    NewUneditedSegmentedObjectImage(MaskRegionObjectImage == 0) = 0;
    if strcmp(Renumber,'Renumber') == 1
        NewUneditedSegmentedObjectImage = bwlabel(NewUneditedSegmentedObjectImage);
    end
end
if exist('SmallRemovedSegmentedObjectImage','var') == 1
    NewSmallRemovedSegmentedObjectImage = SmallRemovedSegmentedObjectImage;
    NewSmallRemovedSegmentedObjectImage(MaskRegionObjectImage == 0) = 0;
    if strcmp(Renumber,'Renumber') == 1
        NewSmallRemovedSegmentedObjectImage = bwlabel(NewSmallRemovedSegmentedObjectImage);
    end
end

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow



fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
if any(findobj == ThisModuleFigureNumber) == 1;
    %%% Calculates the ColoredLabelMatrixImage for displaying in the figure
    %%% window.
    %%% Note that the label2rgb function doesn't work when there are no objects
    %%% in the label matrix image, so there is an "if".
    if sum(sum(NewSegmentedObjectImage)) >= 1
        cmap = jet(max(64,max(NewSegmentedObjectImage(:))));
        ColoredNewSegmentedObjectImage = label2rgb(NewSegmentedObjectImage, cmap, 'k', 'shuffle');
    else  ColoredNewSegmentedObjectImage = NewSegmentedObjectImage;
    end
    if sum(sum(MaskRegionObjectImage)) >= 1
        cmap = jet(max(64,max(MaskRegionObjectImage(:))));
        ColoredMaskRegionObjectImage = label2rgb(MaskRegionObjectImage, cmap, 'k', 'shuffle');
    else  ColoredMaskRegionObjectImage = MaskRegionObjectImage;
    end
    if sum(sum(SegmentedObjectImage)) >= 1
        cmap = jet(max(64,max(SegmentedObjectImage(:))));
        ColoredSegmentedObjectImage = label2rgb(SegmentedObjectImage, cmap, 'k', 'shuffle');
    else  ColoredSegmentedObjectImage = SegmentedObjectImage;
    end


    drawnow
    CPfigure(handles,ThisModuleFigureNumber);
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,2,1); imagesc(ColoredSegmentedObjectImage);
    title(['Previously identified ', ObjectName,', Image Set # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the inverted original
    %%% image with outlines drawn on top.
    subplot(2,2,2); imagesc(ColoredNewSegmentedObjectImage);
    title(RemainingObjectName);
    %%% A subplot of the figure window is set to display the colored label
    %%% matrix image.
    subplot(2,2,3); imagesc(ColoredMaskRegionObjectImage);
    title(['Previously identified ', MaskRegionName,', Image Set # ',num2str(handles.Current.SetBeingAnalyzed)]);
    CPFixAspectRatio(ColoredSegmentedObjectImage);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow



%%% Saves the final segmented label matrix image to the handles structure.
fieldname = ['Segmented',RemainingObjectName];
handles.Pipeline.(fieldname) = NewSegmentedObjectImage;

%%% The following is only relevant for objects identified using
%%% Identify Primary modules, not Identify Secondary modules.
if exist('NewUneditedSegmentedObjectImage','var') == 1
%%% Saves the segmented image, not edited for objects along the edges or
%%% for size, to the handles structure.
fieldname = ['UneditedSegmented',RemainingObjectName];
handles.Pipeline.(fieldname) = NewUneditedSegmentedObjectImage;
end
if exist('NewSmallRemovedSegmentedObjectImage','var') == 1
%%% Saves the segmented image, only edited for small objects, to the
%%% handles structure.
fieldname = ['SmallRemovedSegmented',RemainingObjectName];
handles.Pipeline.(fieldname) = NewSmallRemovedSegmentedObjectImage;
end
