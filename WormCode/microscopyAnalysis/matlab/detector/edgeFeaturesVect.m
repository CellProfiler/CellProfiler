function F = edgeFeaturesVect(bins,angleImg,magImg,stdMag,mask)
% calculate edge based features for region defined by mask
% bins = number of bins in histogram
% angle = angle of edge (0-1) = (0-360 deg) 1 should never appear.
% mag = magnitude of edge (always positive)
% mask = a bit matrix defining the region of interest.

B = floor(angleImg*bins);
B(B==bins)=bins-1;

F = zeros(size(angleImg,1),size(angleImg,2),bins);

for i=0:bins-1
    F(:,:,i+1) = conv2(double(B==i).*magImg,double(mask),'same')./stdMag;
end
