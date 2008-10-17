function F = percFeaturesVect(img,edges,mask)
% Finds percentile features i.e., values of different percentiles
% edges = different prctiles to be found
% box = image
% mask = a bit matrix defining the region of interest.

F = zeros(size(img,1),size(img,2),length(edges));
num = 1;
for ee=edges
    ord = round(sum(mask(:))*ee/100);
    F(:,:,num) = ordfilt2(img,ord,mask,'symmetric');
    num=num+1;
end