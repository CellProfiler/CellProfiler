function [magnitude,normbox] = edgeBoxMag(mbox)
% find std of edge magnitudes, and normalize by it
% Returns:
% magnitude: The std of the values in mbox
% normBox: the normalized box (mbox divided by magnitude)
magnitude = std(mbox(:));
normbox = mbox ./ magnitude;
