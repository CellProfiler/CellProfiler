function [magnitude,normbox] = edgeBoxMagVect(dimg,mask)
% find std of edge magnitudes, and normalize by it
% Returns:
% magnitude: The std of the values in mbox
% normBox: the normalized box (mbox divided by magnitude)
box = ones(size(mask,1),size(mask,2));
meanB = conv2(dimg,box,'same')/sum(box(:));
sqrB = conv2(dimg.^2,box,'same');
magnitude = sqrt(sqrB/sum(box(:))-meanB.^2);
