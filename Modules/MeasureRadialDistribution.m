function handles = MeasureRadialDistribution(handles,varargin)

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

% MBray 2009_03_20: Comments on variables for pyCP upgrade
%
% Recommended variable order (setting, followed by current variable in MATLAB CP)
% (1) What did you call the image from which you want to measure the intensity distribution? (ImageName)
% (2) What did you call the objects from which you want to measure the intensity distribution? (MainObjects)
% (3) What objects do you want to use as centers? (use "Do not use" to use distance-based centers) (CenterObjects)
% (4) How many bins do you want to use to store the distribution? (BinCount)
%
% (i) User should be permitted to specify a range of values in (4) so they
% don't have to add a separate module for each
% (ii) A button should be added after (4) allowing the user to add more 
% images, associated objects, and binning to specify other object radial
% distributions.
%
% Anne 4-9-09: Can we add these features to MeasureObjectIntensity? I think
% it would only add a few options. At the very least we should rename this
% module to MeasureObjectIntensityDistributions. But that's just awkward. 
%
% We also need to re-word Variable 4. I think it's saying how many
% concentric rings do you want to divide the object into (?).


%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%

drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the image for which you want to measure the intensity distribution?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What did you call the objects in which you want to measure intensity distribution?
%infotypeVAR02 = objectgroup
MainObjects = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 = What objects do you want to use as centers (use "Do not use" to use distance-based centers)?
%choiceVAR03 = Do not use
%infotypeVAR03 = objectgroup
CenterObjects = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 = How many bins do you want to use to store the distribution?
%defaultVAR04 = 4
BinCount = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,4}));

%%%%%%%%%%%%%%%%
%%% FEATURES %%%
%%%%%%%%%%%%%%%%

if nargin > 1 
    switch varargin{1}
%feature:categories
        case 'categories'
            if nargin == 1 || strcmp(varargin{2},MainObjects)
                result = { 'RadialIntensityDist' };
            else
                result = {};
            end
%feature:measurements
        case 'measurements'
            result = {};
            if nargin >= 3 &&...
                strcmp(varargin{3},'RadialIntensityDist') &&...
                strcmp(varargin{2},MainObjects)
                result = { 'FracAtD','MeanFrac','RadialCV' };
            end
        otherwise
            error(['Unhandled category: ',varargin{1}]);
    end
    handles=result;
    return;
end

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

if ~strcmp(CenterObjects, 'Do not use'),
    CenterLabels = CPretrieveimage(handles,['Segmented', CenterObjects],ModuleName,'MustBeGray','DontCheckScale');
    %%% Find the centers of the center objects (for anisotropy calculation)
    props = regionprops(CenterLabels, 'Centroid');
    Centroids = reshape(round([props(:).Centroid]), [2, max(LabelMatrixImage(:))]);
else
    %%% Find the point per object farthest from the edge of the object
    Perimeters = CPlabelperim(LabelMatrixImage);
    LabelsWOPerimeters=LabelMatrixImage;
    LabelsWOPerimeters(Perimeters > 0)=0;
    Distances = bwdist(LabelsWOPerimeters==0)+random('unif',0,.1,size(LabelMatrixImage));
    Points = regionprops(LabelsWOPerimeters,Distances,'PixelValues','PixelList');
    Centroids = zeros(2,max(LabelMatrixImage(:)));
    for k = 1:length(Points)
        [ignore,index]=max(Points(k).PixelValues);
        Centroids(:,k)=Points(k).PixelList(index,:);
    end
    CenterLabels = full(sparse(Centroids(2,:), Centroids(1,:), 1:size(Centroids, 2), size(LabelMatrixImage, 1), size(LabelMatrixImage, 2)));
end



%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%

[Horiz, Vert] = meshgrid(1:size(LabelMatrixImage, 2), 1:size(LabelMatrixImage, 1));
Horiz(LabelMatrixImage == 0) = 0;
Vert(LabelMatrixImage == 0) = 0;
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
    if NumberOfPixelsInSlice ~= 0,
        RadialSliceMeans = RadialValues ./ NumberOfPixelsInSlice;
        RadialCV(:, Bin) = CPnanstd(RadialSliceMeans')' ./ CPnanmean(RadialSliceMeans')';
    else
        RadialCV(:, Bin) = NaN;
    end
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
    hAx=subplot(2,2,1,'Parent',ThisModuleFigureNumber);
    CPimagesc(NormalizedDistance, handles,hAx);
    title(hAx,['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% subplot to display distance/intensity histogram
    hAx=subplot(2,2,2,'Parent',ThisModuleFigureNumber);
    CPimagesc(FractionAtDistance, handles,hAx);
    title(hAx,'FractionAtDistance');
    hAx=subplot(2,2,3,'Parent',ThisModuleFigureNumber);
    CPimagesc(MeanPixelFraction, handles,hAx);
    title(hAx,'MeanPixelFraction');
    hAx=subplot(2,2,4,'Parent',ThisModuleFigureNumber);
    CPimagesc(RadialCV, handles,hAx);
    title(hAx,'RadialCV');
end
