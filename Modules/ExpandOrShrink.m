function handles = ExpandOrShrink(handles)

% Help for the Expand Or Shrink Primary Objects module:
% Category: Object Processing
%
% SHORT DESCRIPTION:
% Expands or shrinks identified objects by a defined distance.
% *************************************************************************
%
% The module expands or shrinks primary objects by adding or removing
% border pixels. The user can specify a certain number of times the
% border pixels are added or removed, or type 'Inf' to expand objects
% until they are almost touching or to shrink objects down to a point.
% Objects are never lost using this module (shrinking stops when an
% object becomes a single pixel). Sometimes when identifying secondary
% objects (e.g. cell edges), it is useful to shrink the primary
% objects (e.g. nuclei) a bit in case the nuclei overlap the cell
% edges slightly, since the secondary object identifiers demand that
% the secondary objects completely enclose primary objects. This is
% handy when the two images are not aligned perfectly, for example.
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
% See also any identify primary module.

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
% $Revision: 1718 $

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the objects that you want to expand or shrink?
%infotypeVAR01 = objectgroup
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the expanded or shrunken objects?
%defaultVAR02 = ShrunkenNuclei
%infotypeVAR02 = objectgroup indep
ShrunkenObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Were the objects identified using an IdentifyPrimary or IdentifySecondary module?
%choiceVAR03 = Primary
%choiceVAR03 = Secondary
%inputtypeVAR03 = popupmenu
ObjectChoice = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = Choose expand or shrink:
%choiceVAR04 = Shrink
%choiceVAR04 = Expand
ShrinkOrExpand = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%textVAR05 = Enter the number of pixels by which to expand or shrink the objects (or "Inf" to either shrink to a point or expand until almost touching). Or type 0 (the number zero) to simply add partial dividing lines between objects that are touching (experimental feature).
%choiceVAR05 = 1
%choiceVAR05 = 2
%choiceVAR05 = 3
%choiceVAR05 = Inf
ShrinkingNumber = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu custom

%textVAR06 = What do you want to call the image of the outlines of the objects?
%choiceVAR06 = Do not save
%choiceVAR06 = OutlinedObjects
%infotypeVAR06 = outlinegroup indep
SaveOutlined = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%inputtypeVAR06 = popupmenu custom

%%%VariableRevisionNumber = 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

if strcmp(ObjectChoice,'Primary')
    %%% Retrieves the segmented image, not edited for objects along the edges or
    %%% for size.
    fieldname = ['UneditedSegmented', ObjectName];
    %%% Checks whether the image to be analyzed exists in the handles structure.
    if isfield(handles.Pipeline, fieldname)==0,
        error(['Image processing was canceled in the ', ModuleName, ' module because the Expand Or Shrink Primary Objects module could not find the input image.  It was supposed to be produced by an Identify Primary module in which the objects were named ', ObjectName, '.  Perhaps there is a typo in the name.'])
    end
    UneditedSegmentedImage = handles.Pipeline.(fieldname);

    %%% Retrieves the segmented image, only edited for small objects.
    fieldname = ['SmallRemovedSegmented', ObjectName];
    %%% Checks whether the image to be analyzed exists in the handles structure.
    if isfield(handles.Pipeline, fieldname)==0,
        error(['Image processing was canceled in the ', ModuleName, ' module because the Expand Or Shrink Primary Objects module could not find the input image.  It was supposed to be produced by an Identify Primary module in which the objects were named ', ObjectName, '.  Perhaps there is a typo in the name.'])
    end
    SmallRemovedSegmentedImage = handles.Pipeline.(fieldname);
end

%%% Retrieves the final segmented label matrix image.
fieldname = ['Segmented', ObjectName];
%%% Checks whether the image to be analyzed exists in the handles structure.
if isfield(handles.Pipeline, fieldname)==0,
    error(['Image processing was canceled in the ', ModuleName, ' module because the Expand Or Shrink Primary Objects module could not find the input image.  It was supposed to be produced by an Identify Primary module in which the objects were named ', ObjectName, '.  Perhaps there is a typo in the name.'])
end
SegmentedImage = handles.Pipeline.(fieldname);
OrigSegmentedImage = SegmentedImage;

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

if strcmp(ShrinkOrExpand,'Shrink') == 1
    %%% Secondary objects physically touch each other and then the
    %%% shrinking function does not work. So this is a quick fix to
    %%% put in some edges. It doesn't catch all of the edges but at
    %%% least it puts most in so that some shrinking can occur.
    if strcmp(ObjectChoice,'Secondary')
        %%% Calculates object outlines
        MaxFilteredImage = ordfilt2(SegmentedImage,9,ones(3,3),'symmetric');
        %%% Determines the outlines.
        IntensityOutlines = SegmentedImage - MaxFilteredImage;
        %%% Converts to logical.
        warning off MATLAB:conversionToLogical
        FinalOutline = logical(IntensityOutlines);
        warning on MATLAB:conversionToLogical
        SegmentedImage(FinalOutline) = 0;
    end
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
    if strcmp(ShrinkingNumber,'Inf') == 1
        if strcmp(ObjectChoice,'Primary')
            ShrunkenUneditedSegmentedImage = bwmorph(UneditedSegmentedImage, 'shrink', Inf);
            ShrunkenSmallRemovedSegmentedImage = bwmorph(SmallRemovedSegmentedImage, 'shrink', Inf);
        end
        ShrunkenSegmentedImage = bwmorph(SegmentedImage, 'shrink', Inf);
    else
        try
            ShrinkingNumber = str2double(ShrinkingNumber);
            if strcmp(ObjectChoice,'Primary')
                ShrunkenUneditedSegmentedImage = bwmorph(UneditedSegmentedImage, 'thin', ShrinkingNumber);
                ShrunkenSmallRemovedSegmentedImage = bwmorph(SmallRemovedSegmentedImage, 'thin', ShrinkingNumber);
            end
            ShrunkenSegmentedImage = bwmorph(SegmentedImage, 'thin', ShrinkingNumber);
        catch error(['Image processing was canceled in the ', ModuleName, ' module because the value entered in the Expand Or Shrink Primary Objects module must either be a number or the text "Inf" (no quotes).'])
        end
    end
elseif strcmp(ShrinkOrExpand,'Expand')
    %%% Converts the ShrinkingNumber entry to a number if possible
    %%% (or leaves it as Inf otherwise).
    try
        ShrinkingNumber = str2double(ShrinkingNumber);
        if strcmp(ObjectChoice,'Primary')
            ShrunkenUneditedSegmentedImage = bwmorph(UneditedSegmentedImage, 'thicken', ShrinkingNumber);
            ShrunkenSmallRemovedSegmentedImage = bwmorph(SmallRemovedSegmentedImage, 'thicken', ShrinkingNumber);
        end
        ShrunkenSegmentedImage = bwmorph(SegmentedImage, 'thicken', ShrinkingNumber);
    catch
        error(['Image processing was canceled in the ', ModuleName, ' module because the value entered in the Expand Or Shrink Primary Objects module must either be a number or the text "Inf" (no quotes).']);
    end
end

%%% For the ShrunkenSegmentedImage, the objects are relabeled so that their
%%% numbers correspond to the numbers used for nuclei.  This is important
%%% so that if the user has made measurements on the non-shrunk objects,
%%% the order of these objects will be exactly the same as the shrunk
%%% objects, which may go on to be used to identify secondary objects.
if strcmp(ShrinkOrExpand,'Shrink')
    if strcmp(ObjectChoice,'Primary')
        FinalShrunkenUneditedSegmentedImage = ShrunkenUneditedSegmentedImage.*UneditedSegmentedImage;
        FinalShrunkenSmallRemovedSegmentedImage = ShrunkenSmallRemovedSegmentedImage.*SmallRemovedSegmentedImage;
    end
    FinalShrunkenSegmentedImage = ShrunkenSegmentedImage.*SegmentedImage;
elseif strcmp(ShrinkOrExpand,'Expand')
    if strcmp(ObjectChoice,'Primary')
        [L,num] = bwlabel(ShrunkenUneditedSegmentedImage);     % Generate new temporal labeling of the expanded objects
        FinalShrunkenUneditedSegmentedImage = zeros(size(ShrunkenUneditedSegmentedImage));
        for k = 1:num                                          % Loop over the objects to give them a new label
            index = find(L==k);                                % Get index for expanded object temporarily numbered k
            OriginalLabel = UneditedSegmentedImage(index);       % In the original labeled image, index indexes either zeros or the original label
            fooindex = find(OriginalLabel);                    % Find index to a nonzero element, i.e. to the original label number
            FinalShrunkenUneditedSegmentedImage(index) = OriginalLabel(fooindex(1)); % Put new label on expanded object
        end

        [L,num] = bwlabel(ShrunkenSmallRemovedSegmentedImage);
        FinalShrunkenSmallRemovedSegmentedImage = zeros(size(ShrunkenSmallRemovedSegmentedImage));
        for k = 1:num
            index = find(L==k);                                 % Get index for expanded object temporarily numbered k
            OriginalLabel = SmallRemovedSegmentedImage(index);   % In the original labeled image, index indexes either zeros or the original label
            fooindex = find(OriginalLabel);                     % Find index to a nonzero element, i.e. to the original label number
            FinalShrunkenSmallRemovedSegmentedImage(index) = OriginalLabel(fooindex(1)); % Put new label on expanded object
        end
    end
    [L,num] = bwlabel(ShrunkenSegmentedImage);
    FinalShrunkenSegmentedImage = zeros(size(ShrunkenSegmentedImage));
    for k = 1:num
        index = find(L==k);                             % Get index for expanded object temporarily numbered k
        OriginalLabel = SegmentedImage(index);          % In the original labeled image, index indexes either zeros or the original label
        fooindex = find(OriginalLabel);                 % Find index to a nonzero element, i.e. to the original label number
        FinalShrunkenSegmentedImage(index) = OriginalLabel(fooindex(1)); % Put new label on expanded object
    end
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber) == 1;
    %%% Calculates the OriginalColoredLabelMatrixImage for displaying in the figure
    %%% window in subplot(2,1,1).
    %%% Note that the label2rgb function doesn't work when there are no objects
    %%% in the label matrix image, so there is an "if".
    if sum(sum(SegmentedImage)) >= 1
        OriginalColoredLabelMatrixImage = CPlabel2rgb(handles,OrigSegmentedImage);
    else  OriginalColoredLabelMatrixImage = OrigSegmentedImage;
    end
    %%% Calculates the ShrunkenColoredLabelMatrixImage for displaying in the figure
    %%% window in subplot(2,1,2).
    %%% Note that the label2rgb function doesn't work when there are no objects
    %%% in the label matrix image, so there is an "if".
    if sum(sum(SegmentedImage)) >= 1
        ShrunkenColoredLabelMatrixImage = CPlabel2rgb(handles,FinalShrunkenSegmentedImage);
    else  ShrunkenColoredLabelMatrixImage = FinalShrunkenSegmentedImage;
    end

    drawnow
    %%% Sets the width of the figure window to be appropriate (half width).
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        originalsize = get(ThisModuleFigureNumber, 'position');
        newsize = originalsize;
        newsize(3) = 0.5*originalsize(3);
        set(ThisModuleFigureNumber, 'position', newsize);
    end
    %%% Activates the appropriate figure window.
    CPfigure(handles,ThisModuleFigureNumber);
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,1,1);
    CPimagesc(OriginalColoredLabelMatrixImage);
    title([ObjectName, ' cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    subplot(2,1,2);
    CPimagesc(ShrunkenColoredLabelMatrixImage);
    title(ShrunkenObjectName);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

if strcmp(ObjectChoice,'Primary')
    %%% Saves the segmented image, not edited for objects along the edges or
    %%% for size, to the handles structure.
    fieldname = ['UneditedSegmented',ShrunkenObjectName];
    handles.Pipeline.(fieldname) = FinalShrunkenUneditedSegmentedImage;

    %%% Saves the segmented image, only edited for small objects, to the
    %%% handles structure.
    fieldname = ['SmallRemovedSegmented',ShrunkenObjectName];
    handles.Pipeline.(fieldname) = FinalShrunkenSmallRemovedSegmentedImage;
end

%%% Saves the final segmented label matrix image to the handles structure.
fieldname = ['Segmented',ShrunkenObjectName];
handles.Pipeline.(fieldname) = FinalShrunkenSegmentedImage;

if ~strcmpi(SaveOutlined,'Do not save')
    %%% Calculates object outlines
    MaxFilteredImage = ordfilt2(FinalShrunkenSegmentedImage,9,ones(3,3),'symmetric');
    %%% Determines the outlines.
    IntensityOutlines = FinalShrunkenSegmentedImage - MaxFilteredImage;
    %%% Converts to logical.
    warning off MATLAB:conversionToLogical
    LogicalOutlines = logical(IntensityOutlines);
    warning on MATLAB:conversionToLogical
    handles.Pipeline.(SaveOutlined) = LogicalOutlines;
end

%%% Saves the ObjectCount, i.e., the number of segmented objects.
%%% See comments for the Threshold saving above
if ~isfield(handles.Measurements.Image,'ObjectCountFeatures')
    handles.Measurements.Image.ObjectCountFeatures = {};
    handles.Measurements.Image.ObjectCount = {};
end
column = find(~cellfun('isempty',strfind(handles.Measurements.Image.ObjectCountFeatures,ShrunkenObjectName)));
if isempty(column)
    handles.Measurements.Image.ObjectCountFeatures(end+1) = {['ObjectCount ' ShrunkenObjectName]};
    column = length(handles.Measurements.Image.ObjectCountFeatures);
end
handles.Measurements.Image.ObjectCount{handles.Current.SetBeingAnalyzed}(1,column) = max(FinalShrunkenSegmentedImage(:));

%%% Saves the location of each segmented object
handles.Measurements.(ShrunkenObjectName).LocationFeatures = {'CenterX','CenterY'};
tmp = regionprops(FinalShrunkenSegmentedImage,'Centroid');
Centroid = cat(1,tmp.Centroid);
handles.Measurements.(ShrunkenObjectName).Location(handles.Current.SetBeingAnalyzed) = {Centroid};