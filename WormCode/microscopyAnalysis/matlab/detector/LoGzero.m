function aLoG(size,infile,outfile)
% create the abs value of laplacian of gaussians (LoG) image for a given image. File name should exclude
% the extension .tif
    
    A=double(imread(infile));

    h=fspecial('log',6*size+1,size);
    k=zero_crossings(h);
    imwrite(J,outfile,'tif','Compression','none');
    %imtool(J);
