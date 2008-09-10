function handles = MeasureRadialDistribution(handles)

% Help for the Measure Radial Distribution module:
% Category: Measurement
%
% SHORT DESCRIPTION:
% Measures radial distribution of one or more proteins within a cell.
% *************************************************************************
%
% Given an image with objects identified, this module measures the
% intensity distribution from the center of those objects to their
% boundary within a user-controlled number of bins, for each object.
%
% The distribution can be measured within a single identified object,
% in which case it is relative to the "center" of the object (as
% defined as the point farthest from the boundary), or another object
% can be used as the center, an example of which would be using Nuclei
% for centers within Cells.
%
% Three features are measured for each object:
% - Fraction of total stain in an object at a given radius.
% - Mean fractional intensity at a given radius (Fraction of total 
%    intenstiy normalized by fraction of pixels at a given radius).
% - Coefficient of variation of intensity within a ring, calculated 
%   over 8 slices.
%
% Features measured:      Feature Number:
% FracAtD               |    1
% MeanFrac              |    2
% RadialCV              |    3
%
% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the image for which you want to measure the intentsity distribution?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What did you call the objects in which you want to measure intensity distribution?
%infotypeVAR02 = objectgroup
MainObjects = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 = What objects do you want to use as centers (use "None" to use distance-based centers)?
%choiceVAR03 = None
%infotypeVAR03 = objectgroup
CenterObjects = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 = How many bins do you want to use to store the distribution?
%defaultVAR04 = 4
BinCount = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,4}));

%%%VariableRevisionNumber = 1

%%% Set up the window for displaying the results
ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber);
    CPfigure(handles,'Text',ThisModuleFigureNumber);
    columns = 1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a variable,
%%% "OrigImage".
Image = CPretrieveimage(handles,ImageName,ModuleName);

%%% Retrieves the label matrix image that contains the segmented objects which
%%% will be measured with this module.
LabelMatrixImage = CPretrieveimage(handles,['Segmented', MainObjects],ModuleName,'MustBeGray','DontCheckScale');

if ~ strcmp(CenterObjects, 'None'),
    CenterLabels = CPretrieveimage(handles,['Segmented', CenterObjects],ModuleName,'MustBeGray','DontCheckScale');
    %%% Find the centers of the center objects (for anisotropy calculation)
    props = regionprops(CenterLabels, 'Centroid');
    Centroids = reshape(round([props(:).Centroid]), [2, max(LabelMatrixImage(:))]);
else
    %%% Find the centroids of the objects
    props = regionprops(LabelMatrixImage, 'Centroid');
    Centroids = reshape(round([props(:).Centroid]), [2, max(LabelMatrixImage(:))]);
    CenterLabels = full(sparse(centroids(1,:), centroids(2,:), 1:size(centroids, 2), size(LabelMatrixImage, 1), size(LabelMatrixImage, 2)));
end



%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%

%%% Find distance from Centers within Objects
%%% We include the center objects within the propagate mask, because
%%% otherwise distances don't propagate out of them.
CombinedLabels = max(LabelMatrixImage, CenterLabels);
NumObjects = max(CombinedLabels(:));
[IgnoreLabels, DistanceFromCenters] = IdentifySecPropagateSubfunction(CenterLabels, zeros(size(CenterLabels)), CombinedLabels > 0, 1.0);

%%% Find distance from outer boundaries to Centers.  We find the
%%% boundaries of the objects unified with their centers, to handle
%%% cases where they do not overlap (and in which case the inner &
%%% outer boundaries would be confused).
[IgnoreLabels, DistanceFromEdges] = IdentifySecPropagateSubfunction(CPlabelperim(CombinedLabels), zeros(size(CenterLabels)), CombinedLabels > 0, 1.0);

%%% Compute normalized distance.  Last term in the denominator prevents divide by zero, and also makes sure the largest value is less than 1
NormalizedDistance = DistanceFromCenters ./ (DistanceFromCenters + DistanceFromEdges + 0.001);
TotalDistance = DistanceFromCenters + DistanceFromEdges;

%%% Bin the values.  Yay for "full(sparse(...))".
BinIndexes = floor(NormalizedDistance * BinCount + 1);
Mask = (LabelMatrixImage > 0);
BinnedValues = full(sparse(LabelMatrixImage(Mask), BinIndexes(Mask), Image(Mask), NumObjects, BinCount));
% Fraction of stain at a particular radius
FractionAtDistance = BinnedValues ./ repmat(sum(BinnedValues, 2), 1, BinCount);
% Average density at a particular radius - note that to make this invariant to scale changes, we adjust the normalizer by total pixels.
NumberOfPixelsAtDistance = full(sparse(LabelMatrixImage(Mask), BinIndexes(Mask), 1, NumObjects, BinCount));
MeanPixelFraction = FractionAtDistance ./ (NumberOfPixelsAtDistance ./ repmat(sum(NumberOfPixelsAtDistance, 2), 1, BinCount) + eps);

%%% Anisotropy calculation.  Split each cell into eight wedges, then
%%% compute coefficient of variation of the wedges' mean intensities
%%% in each ring.
%%%
%%% Compute each pixel's delta from the center object's centroid
[Horiz, Vert] = meshgrid(1:size(LabelMatrixImage, 2), 1:size(LabelMatrixImage, 1));
Horiz(LabelMatrixImage == 0) = 0;
Vert(LabelMatrixImage == 0) = 0;
CentroidHoriz = zeros(size(LabelMatrixImage));
CentroidHoriz(Mask) = Centroids(1, LabelMatrixImage(LabelMatrixImage > 0));
CentroidVert = zeros(size(LabelMatrixImage));
CentroidVert(Mask) = Centroids(2, LabelMatrixImage(LabelMatrixImage > 0));
DeltaHoriz = Horiz - CentroidHoriz;
DeltaVert = Vert - CentroidVert;
%%% We now compute three single-bit images, dividing the object into eight radial slices, numbered 1 to 8.
Mask1 = (DeltaHoriz > 0);
Mask2 = (DeltaVert > 0);
Mask3 = (abs(DeltaHoriz) > abs(DeltaVert));
RadialSlice = 1 + Mask1 + 2 * Mask2 + 4 * Mask3;
%%% Now, for each (Label, Bin, RadialSlice) triplet, we need the mean
%%% intensity.  Matlab lacks 3D sparse matrices, so we'll loop over
%%% the bins
RadialCV = zeros(NumObjects, BinCount);
for Bin = 1:BinCount,
    % similar to computations above, but limited to a particular bin
    Bin_Mask = Mask & (BinIndexes == Bin);
    RadialValues = (sparse(LabelMatrixImage(Bin_Mask), RadialSlice(Bin_Mask), Image(Bin_Mask), NumObjects, 8));
    NumberOfPixelsInSlice = (sparse(LabelMatrixImage(Bin_Mask), RadialSlice(Bin_Mask), 1, NumObjects, 8));
    RadialSliceMeans = RadialValues ./ NumberOfPixelsInSlice;
    RadialCV(:, Bin) = CPnanstd(RadialSliceMeans')' ./ CPnanmean(RadialSliceMeans')';
end
RadialCV(isnan(RadialCV)) = 0;


%%% Store Measurements
for k = 1:BinCount,
    handles = CPaddmeasurements(handles, MainObjects, CPjoinstrings('RadialIntensityDist', 'FracAtD', ImageName, num2str(k)), FractionAtDistance(:, k));
    handles = CPaddmeasurements(handles, MainObjects, CPjoinstrings('RadialIntensityDist', 'MeanFrac', ImageName, num2str(k)), MeanPixelFraction(:, k));
    handles = CPaddmeasurements(handles, MainObjects, CPjoinstrings('RadialIntensityDist', 'RadialCV', ImageName, num2str(k)), RadialCV(:, k));
end


%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    %%% Activates the appropriate figure window.
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure(Image,'TwoByTwo',ThisModuleFigureNumber);
    end
    %%% A subplot of the figure window is set to display the distance image.
    subplot(2,2,1);
    CPimagesc(NormalizedDistance, handles);
    title(['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% subplot to display distance/intensity histogram
    subplot(2,2,2);
    CPimagesc(FractionAtDistance, handles);
    title('FractionAtDistance');
    subplot(2,2,3);
    CPimagesc(MeanPixelFraction, handles);
    title('MeanPixelFraction');
    subplot(2,2,4);
    CPimagesc(RadialCV, handles);
    title('RadialCV');
end
