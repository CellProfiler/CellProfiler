function LoG(filename)
% create the laplacian of gaussians (LoG) image for a given image. File name should exclude
% the extension .tif
    
    A=double(imread([filename '.tif']));

    h=fspecial('log');
    J=imfilter(A,h,'replicate');
    J=(J-min(J(:)))/(max(J(:))-min(J(:)));
    imwrite(J,[filename '.log.tif'],'tif','Compression','none');
    %imtool(J);
