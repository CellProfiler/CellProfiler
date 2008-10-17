function normalizeImage(scale,input_image,output_image)
%Normalized the image

img = imread(input_image);

img = double(img);
img = img-min(img(:));
img = img/max(img(:));

limg = log(img+0.04);
[h w]=size(img);

smallImg = imresize(limg,1/scale);
medfiltImg = medfilt2(smallImg,[25 25]);
bigMedImg = imresize(medfiltImg,scale);

bigMedImg = bigMedImg(1:h,1:w);

normImg = limg-bigMedImg;
normImg = normImg-min(normImg(:));
normImg = uint8(255*normImg/max(normImg(:)));
imwrite(normImg,output_image,'tiff');
