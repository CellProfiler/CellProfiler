function scaleImage(infile,outfile)
% Normalize a greyscale image to scale btween 0 and 1 after clipping values
% further than 3 std from mean
    
    A=double(imread(infile));

    m = mean(A(:));
    s = std(A(:));
    A=(A-(m-3*s))/(6*s);
    A(A<0)=0;
    A(A>1)=1;
    imwrite(A,outfile,'tif','Compression','none');
    %imtool(A);
