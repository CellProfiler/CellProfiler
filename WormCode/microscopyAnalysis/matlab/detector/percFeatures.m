function F = percFeatures(edges,box,mask)
% Finds percentile features i.e., values of different percentiles
% edges = different prctiles to be found
% box = image
% mask = a bit matrix defining the region of interest.

values = box(mask>0);
F = prctile(double(values),edges)';
