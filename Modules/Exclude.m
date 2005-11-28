function handles = Exclude(handles)

% Help for the Exclude Objects module:
% Category: Object Processing
%
% SHORT DESCRIPTION:
% Removes objects outside of specified region.
% *************************************************************************
%
% This image analysis module allows you to delete the objects and
% portions of objects that are outside of a region you specify (e.g.
% nuclei outside of a tissue region).  The objects and the region
% should both result from any Identify module (Primary, Secondary, or
% Tertiary).
%
% Retain or renumber:
% Retaining objects' original numbers might be important if you intend to
% correlate measurements made on the remaining objects with measurements
% made on the original objects. Note that retaining original numbers will
% produce gaps in the numbered list of objects (since some objects no
% longer exist). This may cause errors with certain exporting tools or with
% downstream modules that expect object numbers to not have gaps.
% Renumbering, on the other hand, makes the output file more compact, the
% processing quicker, and is also guaranteed to work with exporting and
% data analysis tools.
%
% Special note on saving images: Using the settings in this module, object
% outlines can be passed along to the module OverlayOutlines and then saved
% with SaveImages. Objects themselves can be passed along to the object
% processing module ConvertToImage and then saved with SaveImages. This
% module produces several additional types of objects with names that are
% automatically passed along with the following naming structure: (1) The
% unedited segmented image, which includes objects on the edge of the image
% and objects that are outside the size range, can be saved using the name:
% UneditedSegmented + whatever you called the objects (e.g.
% UneditedSegmentedNuclei). (2) The segmented image which excludes objects
% smaller than your selected size range can be saved using the name:
% SmallRemovedSegmented + whatever you called the objects (e.g.
% SmallRemovedSegmented Nuclei).
%
% See also <nothing relevant>.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne Carpenter
%   Thouis Jones
%   In Han Kang
%   Ola Friman
%   Steve Lowe
%   Joo Han Chang
%   Colin Clarke
%   Mike Lamprecht
%   Susan Ma
%   Wyman Li
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the objects you want to filter?
%infotypeVAR01 = objectgroup
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What did you call the region outside of which objects should be excluded?
%infotypeVAR02 = objectgroup
MaskRegionName = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 = What do you want to call the remaining objects?
%defaultVAR03 = FilteredObjects
%infotypeVAR03 = objectgroup indep
RemainingObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = For the remaining objects, do you want to retain their original number or renumber them consecutively?
%choiceVAR04 = Retain
%choiceVAR04 = Renumber
Renumber = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%textVAR05 = What do you want to call the outlines of the remaining objects (optional)?
%defaultVAR05 = Do not save
%infotypeVAR05 = outlinegroup indep
SaveOutlines = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the images to be used for analysis.
%%% Checks whether the images exist in the handles structure.
fieldname = ['Segmented',ObjectName];
if isfield(handles.Pipeline, fieldname) == 0
    error(['Image processing was canceled in the ', ModuleName, ' module. Prior to running the Exclude Primary Object module, you must have previously run a module to identify primary objects. You specified in the Exclude Primary Object module that these objects were called ', ObjectName, ' which should have produced a field in the handles structure called ', fieldname, '. The ', ModuleName, ' module cannot find this image.']);
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
    error(['Image processing was canceled in the ', ModuleName, ' module. Prior to running the Exclude Primary Object module, you must have previously run a module to identify primary objects. You specified in the Exclude Primary Object module that these objects were called ', MaskRegionName, ' which should have produced a field in the handles structure called ', fieldname, '. The Exclude Objects module cannot find this image.']);
end
MaskRegionObjectImage = handles.Pipeline.(fieldname);

if size(SegmentedObjectImage) ~= size(MaskRegionObjectImage)
    error(['Image processing was canceled in the ', ModuleName, ' module because the two images in which primary objects were identified (', MaskRegionName, ' and ', ObjectName, ') are not the same size.']);
end

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
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

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber) | ~strcmpi(SaveOutlines,'Do not save') %#ok Ignore MLint
    %%% Calculates the ColoredLabelMatrixImage for displaying in the figure
    %%% window.
    %%% Note that the label2rgb function doesn't work when there are no objects
    %%% in the label matrix image, so there is an "if".
    if sum(sum(NewSegmentedObjectImage)) >= 1
        ColoredNewSegmentedObjectImage = CPlabel2rgb(handles,NewSegmentedObjectImage);
    else  ColoredNewSegmentedObjectImage = NewSegmentedObjectImage;
    end
    if sum(sum(MaskRegionObjectImage)) >= 1
        ColoredMaskRegionObjectImage = CPlabel2rgb(handles,MaskRegionObjectImage);
    else  ColoredMaskRegionObjectImage = MaskRegionObjectImage;
    end
    if sum(sum(SegmentedObjectImage)) >= 1
        ColoredSegmentedObjectImage = CPlabel2rgb(handles,SegmentedObjectImage);
    else  ColoredSegmentedObjectImage = SegmentedObjectImage;
    end
    
    %%% Calculates the object outlines, which are overlaid on the original
    %%% image and displayed in figure subplot (2,2,4).
    %%% Creates the structuring element that will be used for dilation.
    StructuringElement = strel('square',3);
    %%% Converts the FinalLabelMatrixImage to binary.
    FinalBinaryImage = im2bw(NewSegmentedObjectImage,.5);
    %%% Dilates the FinalBinaryImage by one pixel (8 neighborhood).
    DilatedBinaryImage = imdilate(FinalBinaryImage, StructuringElement);
    %%% Subtracts the FinalBinaryImage from the DilatedBinaryImage,
    %%% which leaves the PrimaryObjectOutlines.
    PrimaryObjectOutlines = DilatedBinaryImage - FinalBinaryImage;
    %%% Overlays the object outlines on the mask region image.
    ObjectOutlinesOnOrigImage = ColoredMaskRegionObjectImage;
    %%% Determines the grayscale intensity to use for the cell outlines.
    LineIntensity = max(ColoredMaskRegionObjectImage(:));
    ObjectOutlinesOnOrigImage(PrimaryObjectOutlines == 1) = LineIntensity;

    drawnow
    CPfigure(handles,ThisModuleFigureNumber);
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,2,1);
    CPimagesc(ColoredSegmentedObjectImage);
    title(['Previously identified ', ObjectName,', cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the inverted original
    %%% image with outlines drawn on top.
    subplot(2,2,2);
    CPimagesc(ColoredNewSegmentedObjectImage);
    title(RemainingObjectName);
    %%% A subplot of the figure window is set to display the colored label
    %%% matrix image.
    subplot(2,2,3);
    CPimagesc(ColoredMaskRegionObjectImage);
    title(['Previously identified ', MaskRegionName,', cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    subplot(2,2,4); 
    CPimagesc(ObjectOutlinesOnOrigImage); 
    title([ObjectName, ' Outlines on Input Image']);
    CPFixAspectRatio(ColoredSegmentedObjectImage);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the final segmented label matrix image to the handles structure.
fieldname = ['Segmented',RemainingObjectName];
handles.Pipeline.(fieldname) = NewSegmentedObjectImage;

%%% Saves images to the handles structure so they can be saved to the hard
%%% drive, if the user requested.
if ~strcmp(SaveOutlines,'Do not save')
    try    handles.Pipeline.(SaveOutlines) = FinalOutline;
    catch error(['The object outlines were not calculated by the ', ModuleName, ' module, so these images were not saved to the handles structure. The Save Images module will therefore not function on these images. This is just for your information - image processing is still in progress, but the Save Images module will fail if you attempted to save these images.'])
    end
end

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