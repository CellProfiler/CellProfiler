function visualization = ...
    CPvisualizeProbabilisticSegmentation(ColoredLabelMatrixImage, ...
					 FinalLabelMatrixImage, probabilities)
%
% Compute a version of the ColoredLabelMatrixImage that attempts to
% visualize each pixel's probability of having its most likely label.
%
% $Revision$

visualization = im2double(ColoredLabelMatrixImage);
r = visualization(:,:,1);
g = visualization(:,:,2);
b = visualization(:,:,3);
for label = 1:max(max(FinalLabelMatrixImage))
    indices = find(FinalLabelMatrixImage == label);
    p = probabilities(:,:,label);
    r(indices) = r(indices) .* p(indices);
    g(indices) = g(indices) .* p(indices);
    b(indices) = b(indices) .* p(indices);
end
visualization(:,:,1) = r;
visualization(:,:,2) = g;
visualization(:,:,3) = b;

