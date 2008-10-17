function F = histFeatures(edges,box,mask)
% calculate a histogram of grey values for region defined by mask
% edges = boundaries beterrn the bins (bins+1 elements) 
% box = image
% mask = a bit matrix defining the region of interest.

values = box(mask>0);

n = histc(values,edges);
n(end-1)=n(end-1)+n(end);   % the last element in n counts the values that are equal to the highest vaue in edges.

F=n(1:end-1) ./ length(values);
