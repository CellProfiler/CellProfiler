function handles = MeasureObjectNeighbors(handles)

% Help for the Measure Object Neighbors module:
% Category: Measurement
%
% SHORT DESCRIPTION:
% Calculates how many neighbors each object has.
% *************************************************************************
%
% Given an image with objects identified (e.g. nuclei or cells), this
% module determines how many neighbors each object has. The user
% selects the distance within which objects should be considered
% neighbors.
%
% How it works:
% Retrieves a segmented image of the objects, in label matrix format.
% The objects are expanded by the number of pixels the user specifies,
% and then the module counts up how many other objects the object
% is overlapping. Alternately, the module can measure the number of
% neighbors each object has if every object were expanded up until the
% point where it hits another object.  To use this option, enter 0
% (the number zero) for the pixel distance.  Please note that
% currently the image of the objects, colored by how many neighbors
% each has, cannot be saved using the SaveImages module, because it is
% actually a black and white image displayed using a particular
% colormap
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

%textVAR01 = What did you call the objects whose neighbors you want to measure?
%infotypeVAR01 = objectgroup
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = Objects are considered neighbors if they are within this distance (pixels):
%defaultVAR02 = 0
NeighborDistance = str2double(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = What do you want to call the image of the objects, colored by the number of neighbors?
%defaultVAR03 = ColoredNeighbors
%infotypeVAR03 = objectgroup indep
ColoredNeighborsName = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%%%VariableRevisionNumber = 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a variable,
%%% "OrigImage".
fieldname = ['Segmented',ObjectName];
%%% Checks whether the image exists in the handles structure.
if ~isfield(handles.Pipeline,fieldname)
    error(['Image processing was canceled in the ', ModuleName, ' module. Prior to running this module, you must have previously run a segmentation module.  You specified that the desired objects were named ', ObjectName, '. These objects were not found.']);
end
IncomingLabelMatrixImage = handles.Pipeline.(fieldname);

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

ThisModuleFigureNumber = CPwhichmodulefigurenumber(CurrentModule);
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
    %%% Sets the width of the figure window to be appropriate (half width).
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        originalsize = get(ThisModuleFigureNumber, 'position');
        newsize = originalsize;
        newsize(3) = 0.5*originalsize(3);
        set(ThisModuleFigureNumber, 'position', newsize);
    end
    drawnow

    CPfigure(handles,ThisModuleFigureNumber);
    subplot(2,1,1); CPimagesc(ColoredIncomingObjectsImage); title(ObjectName,'FontSize',FontSize)
    set(gca,'FontSize',FontSize)
    subplot(2,1,2); CPimagesc(ImageOfNeighbors);
    colorbar('SouthOutside','FontSize',FontSize)
    title(ColoredNeighborsName,'FontSize',FontSize)
    set(gca,'FontSize',FontSize)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE IMAGES TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the colored version of images to the handles structure so
%%% they can be saved to the hard drive, if the user requests.
handles.Pipeline.(ColoredNeighborsName) = ColoredImageOfNeighbors;