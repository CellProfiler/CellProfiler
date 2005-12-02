function handles = MeasureObjectNeighbors(handles)

% Help for the Measure Object Neighbors module:
% Category: Measurement
%
% SHORT DESCRIPTION:
% Calculates how many neighbors each object has.
% *************************************************************************
%
% Given an image with objects identified (e.g. nuclei or cells), this
% module determines how many neighbors each object has. The user selects
% the distance within which objects should be considered neighbors. The
% module can measure the number of neighbors each object has if every
% object were expanded up until the point where it hits another object. To
% use this option, enter 0 (the number zero) for the pixel distance.
%
% How it works:
% Retrieves objects in label matrix format. The objects are expanded by the
% number of pixels the user specifies, and then the module counts up how
% many other objects the object is overlapping.

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

%textVAR01 = What did you call the objects whose neighbors you want to measure?
%infotypeVAR01 = objectgroup
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = Objects are considered neighbors if they are within this distance, in pixels. Or, enter 0 (zero) to expand objects until touching and then count their neighbors:
%defaultVAR02 = 0
NeighborDistance = str2double(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = What do you want to call the image of the objects with grayscale values corresponding to the number of neighbors, which is compatible for saving in .mat format using the Save Images module for further analysis in Matlab?
%defaultVAR03 = Do not save
%infotypeVAR03 = imagegroup indep
GrayscaleNeighborsName = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = What do you want to call the objects colored by number of neighbors, which are compatible for converting to a color image using the Convert To Image and Save Images modules?
%defaultVAR04 = Do not save
%infotypeVAR04 = objectgroup indep
ColoredNeighborsName = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%%%VariableRevisionNumber = 3

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a variable,
%%% "OrigImage".
IncomingLabelMatrixImage = CPretrieveimage(handles,['Segmented', ObjectName],ModuleName,2,0);

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines the neighbors for each object.
d = max(2,NeighborDistance+1);
[sr,sc] = size(IncomingLabelMatrixImage);
ImageOfNeighbors = -ones(sr,sc);
NumberOfNeighbors = zeros(max(IncomingLabelMatrixImage(:)),1);
IdentityOfNeighbors = cell(max(IncomingLabelMatrixImage(:)),1);
se = strel('disk',d,0);
props = regionprops(IncomingLabelMatrixImage,'PixelIdxList');
for k = 1:max(IncomingLabelMatrixImage(:))
    % Cut patch
    [r,c] = ind2sub([sr sc],props(k).PixelIdxList);
    rmax = min(sr,max(r) + (d+1));
    rmin = max(1,min(r) - (d+1));
    cmax = min(sc,max(c) + (d+1));
    cmin = max(1,min(c) - (d+1));
    p = IncomingLabelMatrixImage(rmin:rmax,cmin:cmax);
    % Extend cell boundary
    pextended = imdilate(p==k,se,'same');
    overlap = p.*pextended;
    IdentityOfNeighbors{k} = setdiff(unique(overlap(:)),[0,k]);
    NumberOfNeighbors(k) = length(IdentityOfNeighbors{k});
    ImageOfNeighbors(sub2ind([sr sc],r,c)) = NumberOfNeighbors(k);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% MAKE MEASUREMENTS & SAVE TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves neighbor measurements to handles structure.
handles.Measurements.(ObjectName).NumberNeighbors(handles.Current.SetBeingAnalyzed) = {NumberOfNeighbors};
handles.Measurements.(ObjectName).NumberNeighborsFeatures = {'Number of neighbors'};

% This field is different from the usual measurements. To avoid problems with export modules etc we don't
% add a IdentityOfNeighborsFeatures field. It will then be "invisible" to
% export modules, which look for fields with 'Features' in the name.
handles.Measurements.Neighbors.IdentityOfNeighbors(handles.Current.SetBeingAnalyzed) = {IdentityOfNeighbors};

%%% Example: To extract the number of neighbor for objects called Cells, use code like this:
%%% handles.Measurements.Neighbors.IdentityOfNeighborsCells{1}{3}
%%% where 1 is the image number and 3 is the object number. This
%%% yields a list of the objects who are neighbors with Cell object 3.

%%% For some reason, this does not exactly match the results of the display
%%% window. Not sure why.
if sum(sum(ImageOfNeighbors)) >= 1
    handlescmap = handles.Preferences.LabelColorMap;
    cmap = feval(handlescmap,max(64,max(ImageOfNeighbors(:))));
    ColoredImageOfNeighbors = ind2rgb(ImageOfNeighbors,[0 0 0;cmap]);
else  ColoredImageOfNeighbors = ImageOfNeighbors;
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    %%% Calculates the ColoredIncomingObjectsImage for displaying in the figure
    %%% window and saving to the handles structure.
    %%% Note that the label2rgb function doesn't work when there are no objects
    %%% in the label matrix image, so there is an "if".
    if sum(sum(IncomingLabelMatrixImage)) >= 1
        handlescmap = handles.Preferences.LabelColorMap;
        cmap = feval(handlescmap,max(64,max(IncomingLabelMatrixImage(:))));
        ColoredIncomingObjectsImage = label2rgb(IncomingLabelMatrixImage,cmap,'k','shuffle');
    else  ColoredIncomingObjectsImage = IncomingLabelMatrixImage;
    end

    FontSize = handles.Preferences.FontSize;
    %%% Activates the appropriate figure window.
    CPfigure(handles,ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure(IncomingLabelMatrixImage,'TwoByOne')
    end
    subplot(2,1,1); CPimagesc(ColoredIncomingObjectsImage); 
    title(ObjectName,'FontSize',FontSize)
    subplot(2,1,2); CPimagesc(ImageOfNeighbors);
    colorbar('SouthOutside','FontSize',FontSize)
    title([ObjectName,' colored by number of neighbors'],'FontSize',FontSize)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE IMAGES TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the grayscale version of objects to the handles structure so
%%% they can be saved to the hard drive, if the user requests. Here, the
%%% background is -1, and the objects range from 0 (if it has no neighbors)
%%% up to the highest number of neighbors. The -1 value makes it
%%% incompatible with the Convert To Image module which expects a label
%%% matrix starting at zero.
if ~strcmpi(GrayscaleNeighborsName,'Do not save')
    handles.Pipeline.(GrayscaleNeighborsName) = ColoredImageOfNeighbors;
end

%%% Saves the grayscale version of objects to the handles structure so
%%% they can be saved to the hard drive, if the user requests. Here, the
%%% scalar value 1 is added to every pixel so that the background is zero
%%% and the objects are from 1 up to the highest number of neighbors, plus
%%% one. This makes the objects compatible with the Convert To Image
%%% module.
if ~strcmpi(ColoredNeighborsName,'Do not save')
    handles.Pipeline.(['Segmented',ColoredNeighborsName]) = ColoredImageOfNeighbors + 1;
end