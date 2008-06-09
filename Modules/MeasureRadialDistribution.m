function handles = MeasureRadialDistribution(handles)

% Help for the Measure Object Intensity module:
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
else
    %%% This would be easiest if we could dilate the label matrix.
    %%% However, we can't guarantee that objects don't touch.  We'll
    %%% work in two steps:
    %%%
    %%% First, separate objects by removing their boundary pixels, and then dilate.
    SeparatedCenters = LabelMatrixImage - CPlabelperim(LabelMatrixImage);
    DilatedCenters = LabelMatrixImage .* bwmorph(SeparatedCenters ~= 0, 'shrink', Inf);
    %%% Second, find any labels that lack a center, and add them back individually.
    MissingCenters = 1:max(LabelMatrixImage);
    MissingCenters(DilatedCenters(DilatedCenters > 0)) = 0;
    MissingCenters = MissingCenters(MissingCenters > 0);
    for m = MissingCenters,
        DilatedCenters = DilatedCenters + m * bwmorph(LabelMatrixImage == m, 'shrink', Inf);
    end
    CenterLabels = DilatedCenters;
end

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%

%%% Find distance from Centers within Objects
%%% We include the center objects within the propagate mask, because
%%% otherwise distances don't propagate out of them.
CombinedLabels = max(LabelMatrixImage, CenterLabels);
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
Mask = (BinIndexes>0) & (LabelMatrixImage > 0);
BinnedValues = full(sparse(LabelMatrixImage(Mask), BinIndexes(Mask), Image(Mask), max(CombinedLabels(:)), BinCount));
% Fraction of stain at a particular radius
FractionAtDistance = BinnedValues ./ repmat(sum(BinnedValues, 2), 1, BinCount);
% Average density at a particular radius - note that to make this invariant to scale changes, we adjust the normalizer by total pixels.
NumberOfPixelsAtDistance = full(sparse(LabelMatrixImage(Mask), BinIndexes(Mask), 1, max(CombinedLabels(:)), BinCount));
MeanPixelFraction = FractionAtDistance ./ (NumberOfPixelsAtDistance ./ repmat(sum(NumberOfPixelsAtDistance, 2), 1, BinCount) + eps);

%%% Store Measurements
for k = 1:BinCount,
    handles = CPaddmeasurements(handles, MainObjects, CPjoinstrings('Intensity', 'RadialDist', 'FracAtD', ImageName, num2str(k)), FractionAtDistance(:, k));
    handles = CPaddmeasurements(handles, MainObjects, CPjoinstrings('Intensity', 'RadialDist', 'MeanFrac', ImageName, num2str(k)), MeanPixelFraction(:, k));
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
end
