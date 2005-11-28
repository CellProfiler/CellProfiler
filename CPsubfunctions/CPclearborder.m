function imout = clearborder(im)
map = 0:max(im(:));
map(im(1,:)+1)=0;
map(im(end,:)+1)=0;
map(im(:,1)+1)=0;
map(im(:,end)+1)=0;
imout = map(im+1);
