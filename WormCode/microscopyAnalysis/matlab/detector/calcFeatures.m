function F=calcFeatures(subimage,radius,th)

fi=1;

bigr=radius+2*th;
rs=radius/2;
rb=radius+th;
HSV=rgb2hsv(subimage);
[l,m,n] = size(HSV);

H = zeros(l,m); S=H;
H(:,:) = floor(3.999 *HSV(:,:,1));
S(:,:) = HSV(:,:,2);
F(fi) = max(S(:)); fi = fi+1;
if(F(1)>0) 
    S = S ./ (max(S(:))*l*m);
end

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

for i=1:3   % loop on circles
    for j=1:4 % loop on quadrants
        F(fi:fi+3) = histFeatures(H,S,circ(:,:,i)&q(:,:,j));
        fi=fi+4;
    end
end
