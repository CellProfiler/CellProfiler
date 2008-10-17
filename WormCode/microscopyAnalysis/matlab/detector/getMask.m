function out = getMask(name,size)
% get a mask from the file masks/<name>.mat and resize it to size 
    load(['masks/' name '.mat']);
    out =imresize(mask(:,:,:),[size,size]);
    out = out>0.5;
    
    
