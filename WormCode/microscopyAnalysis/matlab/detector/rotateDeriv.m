A=imread('../../data/JamesOrth/MCF7/MCF7 JDO76_S22_T023.deriv.tif');
angle=115;

HSV=rgb2hsv(A);
H=HSV(:,:,1);
H=mod(H-(angle/360),1);
HSV(:,:,1)=H;
RGB=imrotate(hsv2rgb(HSV),angle);
imtool(RGB);
