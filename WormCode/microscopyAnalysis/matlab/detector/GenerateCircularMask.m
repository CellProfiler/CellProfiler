% create an array of masks of the type used to detect yeast cells
radius=25;
th=10;

bigr=radius+2*th;
rs=radius/2;
rb=radius+th;

l=2*bigr+1; m=l;

circ = zeros(l,m,3);
sml = zeros(l,m);
gap=2*th+floor(radius/2);
rad = ceil(radius/2);

sml(gap+1:end-gap,gap+1:end-gap) =  fspecial('disk',rad);
circ(:,:,1)=sml>0;

med = zeros(l,m);
med(2*th+1:end-2*th,2*th+1:end-2*th) =  fspecial('disk',radius);
circ(:,:,2)=med>0 & sml==0;

big = zeros(l,m);
big(th+1:end-th,th+1:end-th) = fspecial('disk',radius+th);
circ(:,:,3) = big>0 & med==0;

[L,M] = meshgrid(-bigr:bigr,-bigr:bigr);
q = zeros(l,m,4);
q(:,:,1) = L<0 & M<0;
q(:,:,2) = L<0 & M>0;
q(:,:,3) = L>0 & M>0;
q(:,:,4) = L>0 & M<0;

mi=1;
for i=1:3   % loop on circles
    for j=1:4 % loop on quadrants
        mask(:,:,mi) = circ(:,:,i)&q(:,:,j);
        mi=mi+1;
    end
end

save('masks/circularMasks.mat','mask');
