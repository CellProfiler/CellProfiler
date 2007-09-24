function FilledLabelMatrix = CPfill_holes(LabelMatrix)
%%% function FilledLabelMatrix = CPfill_holes(LabelMatrix)
%%%
%%% Fill holes (0-values) in a label matrix, as defined by a
%%% 4-connected region.  Holes are filled only if surrounded by a
%%% single value.

MaximaImage = ordfilt2(LabelMatrix, 5, [0 1 0; 1 1 1 ; 0 1 0]);
ZerosMaxima = MaximaImage .* (LabelMatrix == 0);
        
%%% Likewise, for every zero pixel, find its smallest adjacent
%%% label.  A little trickier, since we need to ignore zeros.
%%% replace 0s with a unique label
UniqueLabel = max(LabelMatrix(:)) + 1;
NoZeros = LabelMatrix;
NoZeros(NoZeros == 0) = UniqueLabel;
MinimaImage = ordfilt2(NoZeros, 1, [0 1 0; 1 1 1 ; 0 1 0]);
%%% replace the unique label with zeros
MinimaImage(MinimaImage == UniqueLabel) = 0;
ZerosMinima = MinimaImage .* (LabelMatrix == 0);

%%% Find the zero regions, removing any that touch the border.
ZeroRegions = CPclearborder(bwlabel(LabelMatrix == 0, 4));

%%% Boundaries of the zero regions are those with a nonzero MaximaImage
ZeroBoundaries = ((ZerosMaxima ~= 0) & (ZeroRegions ~= 0));

%%% Now, build a map from zero region labels to object labels, based
%%% on ZerosMaxima
ZeroLocations = find(ZeroBoundaries);
LocationsZerosAndMaxima = sparse(ZeroLocations, ZeroRegions(ZeroLocations), ZerosMaxima(ZeroLocations));
LZMaxSorted = sort(LocationsZerosAndMaxima);
ZeroRemapperMax = LZMaxSorted(end, :);

%%% Now the same for ZerosMinima, except to find the minimum, we have to reverse the order of labels.
Reverser = [0 max(ZerosMinima(:)):-1:1];
LocationsZerosAndMinima = sparse(ZeroLocations, ZeroRegions(ZeroLocations), Reverser(ZerosMinima(ZeroLocations) + 1));
LZMinSorted = sort(LocationsZerosAndMinima);
ZeroRemapperMin = Reverser(LZMinSorted(end, :) + 1);

%%% Create the zero remapper
ZeroRemapper = ZeroRemapperMax;

%%% Anywhere that disagrees, set the remapper to 0
ZeroRemapper(ZeroRemapperMax ~= ZeroRemapperMin) = 0;

%%% Pad for zeros in the ZeroRegions image (i.e., nonzero regions in
%%% the LabelMatrix)
ZeroRemapper = [0 ZeroRemapper];

%%% Finally, remap the ZeroRegions, and combine them with the
%%% LabelMatrix to fill the holes.
ZerosReplaced = ZeroRemapper(ZeroRegions + 1);
FilledLabelMatrix = full(max(LabelMatrix, ZerosReplaced));
