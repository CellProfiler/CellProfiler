function aLoG(size,infile,outfile)
% create the abs value of laplacian of gaussians (LoG) image for a given image. File name should exclude
% the extension .tif
    
    A=double(imread(infile));

    h=fspecial('log',6*size+1,size);
    J=abs(imfilter(A,h,'replicate'));
    q=quantile(J(:),0.99);
    J(J>q)=q;%clip at 99 percent
    J=J/q;
    imwrite(J,outfile,'tif','Compression','none');
    %imtool(J);
