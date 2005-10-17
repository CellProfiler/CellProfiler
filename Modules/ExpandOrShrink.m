function handles = ExpandOrShrinkPrim(handles)

% Help for the Expand Or Shrink Primary Objects module:
% Category: Object Processing
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
%    Additional image(s) are calculated by this module and can be 
% saved by altering the code for the module (see the SaveImages module
% help for instructions).
%
% See also any identify primary module.

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
% $Revision: 1718 $

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow

%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);
ModuleName = char(handles.Settings.ModuleNames(CurrentModuleNum));

%textVAR01 = What did you call the objects that you want to expand or shrink?
%infotypeVAR01 = objectgroup
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the expanded or shrunken objects?
%defaultVAR02 = ShrunkenNuclei
%infotypeVAR02 = objectgroup indep
ShrunkenObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Are the Objects Primary or Secondary?
%choiceVAR03 = Primary
%choiceVAR03 = Secondary
%inputtypeVAR02 = popupmenu
ObjectChoice = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = Enter E for Expand or S for Shrink.
%choiceVAR04 = Shrink
%choiceVAR04 = Expand
ShrinkOrExpand = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%textVAR05 = Enter the number of pixels by which to expand or shrink the objects (or "Inf" to either shrink to a point or expand until almost touching).
%choiceVAR05 = 1
%choiceVAR05 = 2
%choiceVAR05 = 3
%choiceVAR05 = Inf
ShrinkingNumber = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu custom

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow

if strcmp(ShrinkOrExpand),'Shrink') == 1
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
        try ShrinkingNumber = str2double(ShrinkingNumber)
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
    try ShrinkingNumber = str2double(ShrinkingNumber); end
    if strcmp(ObjectChoice,'Primary')
        ShrunkenUneditedSegmentedImage = bwmorph(UneditedSegmentedImage, 'thicken', ShrinkingNumber);
        ShrunkenSmallRemovedSegmentedImage = bwmorph(SmallRemovedSegmentedImage, 'thicken', ShrinkingNumber);
    end
    ShrunkenSegmentedImage = bwmorph(SegmentedImage, 'thicken', ShrinkingNumber);
    catch error(['Image processing was canceled in the ', ModuleName, ' module because the value entered in the Expand Or Shrink Primary Objects module must either be a number or the text "Inf" (no quotes).'])
end

%%% TODO >>> The following quickly relabels the three binary images so
%%% their labels match the incoming labels, but it only works for
%%% shrunken objects, not expanded ones >>>

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

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
if any(findobj == ThisModuleFigureNumber) == 1;
    %%% Calculates the OriginalColoredLabelMatrixImage for displaying in the figure
    %%% window in subplot(2,1,1).
    %%% Note that the label2rgb function doesn't work when there are no objects
    %%% in the label matrix image, so there is an "if".
    if sum(sum(SegmentedImage)) >= 1
        OriginalColoredLabelMatrixImage = CPlabel2rgb(handles,SegmentedImage);
    else  OriginalColoredLabelMatrixImage = SegmentedImage;
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
    subplot(2,1,1); imagesc(OriginalColoredLabelMatrixImage);
    title([ObjectName, ' Image Set # ',num2str(handles.Current.SetBeingAnalyzed)]);
    subplot(2,1,2); imagesc(ShrunkenColoredLabelMatrixImage); title(ShrunkenObjectName);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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