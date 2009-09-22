function CPimagesc_test(Image, titletext)
% $Revision$

figure
imagesc(Image);
colormap(gray)
colorbar
if nargin > 1
    title(titletext);
end
