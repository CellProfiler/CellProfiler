function handles = ExpandOrShrink(handles)

% Help for the Expand Or Shrink module:
% Category: Object Processing
%
% SHORT DESCRIPTION:
% Expands or shrinks identified objects by a defined distance.
% *************************************************************************
%
% The module expands or shrinks objects by adding or removing border
% pixels. The user can specify a certain number of border pixels to be
% added or removed, or use 'Inf' to expand objects until they are almost
% touching or to shrink objects down to a point. Objects are never lost
% using this module (shrinking stops when an object becomes a single
% pixel). An experimental feature is able to allow shrinking with secondary
% objects - it adds partial dividing lines between objects which are
% touching before the shrinking step so it is not perfect. It would be nice
% to improve this code to draw complete dividing lines, but we have only
% implemented a partial fix.
%
% Special note on saving images: Using the settings in this module, object
% outlines can be passed along to the module OverlayOutlines and then saved
% with the SaveImages module. Objects themselves can be passed along to the
% object processing module ConvertToImage and then saved with the
% SaveImages module.
%
% This module produces several additional types of objects with names
% that are automatically passed along with the following naming
% structure: (1) The unedited segmented image, which includes objects
% on the edge of the image and objects that are outside the size
% range, can be saved using the name: UneditedSegmented + whatever you
% called the objects (e.g. UneditedSegmentedNuclei). (2) The segmented
% image which excludes objects smaller than your selected size range
% can be saved using the name: SmallRemovedSegmented + whatever you
% called the objects (e.g. SmallRemovedSegmented Nuclei).
%
% See also IdentifyPrimAutomatic, IdentifyPrimManual, IdentifySecondary.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne E. Carpenter
%   Thouis Ray Jones
%   In Han Kang
%   Ola Friman
%   Steve Lowe
%   Joo Han Chang
%   Colin Clarke
%   Mike Lamprecht
%   Peter Swire
%   Rodrigo Ipince
%   Vicky Lay
%   Jun Liu
%   Chris Gang
%
% Website: http://www.cellprofiler.org
%
% $Revision$

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

%textVAR03 = Were the objects identified using an Identify Primary or Identify Secondary module (note: shrinking results are not perfect with Secondary objects)?
%choiceVAR03 = Primary
%choiceVAR03 = Secondary
%choiceVAR03 = Other
%inputtypeVAR03 = popupmenu
ObjectChoice = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = Do you want to expand or shrink the objects?
%choiceVAR04 = Shrink
%choiceVAR04 = Expand
ShrinkOrExpand = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%textVAR05 = Enter the number of pixels by which to expand or shrink the objects, or "Inf" to either shrink to a point or expand until almost touching, or 0 (the number zero) to simply add partial dividing lines between objects that are touching (experimental feature).
%choiceVAR05 = 1
%choiceVAR05 = 2
%choiceVAR05 = 3
%choiceVAR05 = Inf
%choiceVAR05 = 0
ShrinkingNumber = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu custom

%textVAR06 = What do you want to call the outlines of the identified objects (optional)?
%defaultVAR06 = Do not save
%infotypeVAR06 = outlinegroup indep
SaveOutlines = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%%%VariableRevisionNumber = 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

if strcmp(ObjectChoice,'Primary')
    UneditedSegmentedImage = CPretrieveimage(handles,['UneditedSegmented', ObjectName],ModuleName);
    SmallRemovedSegmentedImage = CPretrieveimage(handles,['SmallRemovedSegmented', ObjectName],ModuleName);
end

SegmentedImage = CPretrieveimage(handles,['Segmented', ObjectName],ModuleName);
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
        catch
            error(['Image processing was canceled in the ', ModuleName, ' module because the value entered for number of pixels to shrink or expand must either be a number or the text "Inf" (no quotes).'])
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
        error(['Image processing was canceled in the ', ModuleName, ' module because the value entered for number of pixels to shrink or expand must either be a number or the text "Inf" (no quotes).'])
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
if any(findobj == ThisModuleFigureNumber)
    %%% Calculates the OriginalColoredLabelMatrixImage for displaying in the figure
    %%% window in subplot(2,1,1).
    OriginalColoredLabelMatrixImage = CPlabel2rgb(handles,OrigSegmentedImage);
    %%% Calculates the ShrunkenColoredLabelMatrixImage for displaying in the figure
    %%% window in subplot(2,1,2).
    ShrunkenColoredLabelMatrixImage = CPlabel2rgb(handles,FinalShrunkenSegmentedImage);

    %%% Activates the appropriate figure window.
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure(OriginalColoredLabelMatrixImage,'TwoByOne',ThisModuleFigureNumber)
    end%%% A subplot of the figure window is set to display the original image.
    subplot(2,1,1);
    CPimagesc(OriginalColoredLabelMatrixImage,handles);
    title([ObjectName, ' cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    subplot(2,1,2);
    CPimagesc(ShrunkenColoredLabelMatrixImage,handles);
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

if ~strcmpi(SaveOutlines,'Do not save')
    %%% Calculates object outlines
    MaxFilteredImage = ordfilt2(FinalShrunkenSegmentedImage,9,ones(3,3),'symmetric');
    %%% Determines the outlines.
    IntensityOutlines = FinalShrunkenSegmentedImage - MaxFilteredImage;
    %%% Converts to logical.
    warning off MATLAB:conversionToLogical
    LogicalOutlines = logical(IntensityOutlines);
    warning on MATLAB:conversionToLogical
    handles.Pipeline.(SaveOutlines) = LogicalOutlines;
end

handles = CPsaveObjectCount(handles, ShrunkenObjectName, FinalShrunkenSegmentedImage);
handles = CPsaveObjectLocations(handles, ShrunkenObjectName, FinalShrunkenSegmentedImage);
