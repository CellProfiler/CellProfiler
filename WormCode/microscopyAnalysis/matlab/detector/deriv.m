function deriv(infile,outfile)
% create the derivatibe image for a given image. File name should exclude
% the extension .tif
    
    A=imread(infile);

    A=double(A);
    A=(A-min(A(:)))/(max(A(:))-min(A(:)));
    
    h = double(fspecial('sobel'));

    hor = conv2(A,h,'same');

    ver = conv2(A,h','same');

    angle=atan2(hor,ver)/(2*pi) + .5;
    mag=hor.^2 + ver.^2;
    stddev=sqrt(mean(mag(:)));
    mag=sqrt(mag)/(5*stddev);
    %hist(mag(:),500);
    mag(mag>1)=1;

    [i,j]=size(mag);
    HSV=ones(i,j,3);
    HSV(:,:,1)=angle;
    HSV(:,:,2)=mag;
    RGB=hsv2rgb(HSV);
    
    imwrite(RGB,outfile,'tif','Compression','none');
