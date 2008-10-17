function averStk(infile,outfile)

im = tiffread2(infile);
for j=1:length(im)
    B = double(im(j).data);
    q99 = quantile(B(:),0.99);
    q1 = quantile(B(:),0.01);
    B = (B-q1)/(q99-q1);
    B(B>1)=1;
    B(B<0)=0;
    A(:,:,j)=B;
end

C=mean(A,3);
imwrite(C,outfile,'tif','Compression','none');