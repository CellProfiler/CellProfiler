function [image,hImage]=CPplatemap(axes,data,innersize,outersize,bordersize,Minimum,Maximum)

% Create a 3-color x float image of a plate map based on an array
% of data (typically 8 rows of 12 columns or 16 rows of 24 columns)
%
% axes       - Draw the platemap on these axes
% data       - array to be graphically displayed. Set a well without
%              a value to NAN and it will be displayed in blue.
% innersize  - diameter of a well in pixels
% outersize  - size of well
% bordersize - amount of space around border of image
% Minimum    - lower bound (displayed as green)
% Maximum    - upper bound (displayed as red)

image = zeros(size(data,1)*outersize+2*bordersize,size(data,2)*outersize+2*bordersize,3);
image(:) = .5; % Make the image all gray
radius = floor(innersize/2);
isize = radius*2+1;
x=repmat(-radius:radius,isize,1);
y=x';
mask=x.*x+y.*y < radius*radius;
inside=mask & (x.*x+y.*y >= (radius-1)*(radius-1));
crossmask = mask & ((abs(x+y)<2) | inside);
iborder = outersize-innersize;
for j=1:size(data,1)
    for i=1:size(data,2)
        x0=j*outersize-isize-iborder+bordersize+1;
        x1=x0+isize-1;
        y0=i*outersize-isize-iborder+bordersize+1;
        y1=y0+isize-1;
        if isnan(data(j,i))
            image(x0:x1,y0:y1,1) = .5*(1-mask)+crossmask;
            image(x0:x1,y0:y1,2) = .5*(1-mask);
            image(x0:x1,y0:y1,3) = .5*(1-mask);
        else
            image(x0:x1,y0:y1,3)=.5*(1-mask);
            if Minimum == Maximum
                DataPoint = 0;
            else
                DataPoint=2*max([min([(data(j,i)-Minimum)/(Maximum-Minimum),1]),0])-1;
            end
            DataChannel=1;
            OtherChannel=2;
            if DataPoint < 0
                DataPoint = -DataPoint;
                DataChannel=2;
                OtherChannel=1;
            end
            image(x0:x1,y0:y1,OtherChannel)=.5*(1-mask);
            image(x0:x1,y0:y1,DataChannel)=DataPoint*mask+.5*(1-mask);
        end
    end
end

% The colormap here is pretty bogus in that this is a true color image
% Hopefully, when you show the colorbar, it should show the colormap
% and annotate it with the minimum and maximum values.
cmap = zeros(128,3);
for i=1:64
    cmap(i+64,1)=i/64;
    cmap(i,2)=(64-i)/64;
end
hImage = imshow(image,cmap,'Parent',axes);
RowLabels = [{'A'},{'B'},{'C'},{'D'},{'E'},{'F'},{'G'},{'H'},...
             {'I'},{'J'},{'K'},{'L'},{'M'},{'N'},{'O'},{'P'},...
             {'Q'},{'R'},{'S'},{'T'},{'U'},{'V'},{'W'},{'X'},...
             {'Y'},{'Z'},{'AA'},{'BB'},{'CC'},{'DD'},{'EE'},{'FF'} ];
RowLabels = RowLabels(1:size(data,1));
ColumnLabels = 1:size(data,2);
set(axes,'XTick',bordersize+outersize*(1:size(data,2))-outersize/2);
set(axes,'XTickLabel',ColumnLabels);
set(axes,'XAxisLocation','top');
set(axes,'YTick',bordersize+outersize*(1:size(data,1))-outersize/2);
set(axes,'YTickLabel',RowLabels);
set(axes,'Visible','on');
