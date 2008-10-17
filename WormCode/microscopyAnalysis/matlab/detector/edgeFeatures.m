function F = edgeFeatures(bins,angle,mag,mask)
% calculate edge based features for region defined by mask
% bins = number of bins in histogram
% angle = angle of edge (0-1) = (0-360 deg) 1 should never appear.
% mag = magnitude of edge (always positive)
% mask = a bit matrix defining the region of interest.

A = mag .* mask;
B = floor(angle*bins);
B(B==bins)=bins-1;

for i = 0:bins-1
    F(i+1) = sum(A(B==i));
end

F=F';

