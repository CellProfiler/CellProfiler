clear
box_size=10;

image_no='023';
name_prefix=['../../data/JamesOrth/MCF7/MCF7 JDO76_S22_T' image_no];
XMLin  = [name_prefix '.tif.bix'];
outputFile = [name_prefix '.arrows.txt'];
V = xml_parseany(fileread(XMLin));

output = fopen(outputFile,'w');

A=imread([name_prefix '.tif']);

cmap =colormap(lines); % colors for the lines

colormap(gray);
imagesc(A);
FigurePosition = get(gcf,'Position');   % increase figure size
FigurePosition(3:4) = 2.6*FigurePosition(3:4);
set(gcf,'Position',FigurePosition);

[ymax,xmax] = size(A);

l=length(V.item{11}.value{1}.bfi{1}.gobject);

color_no=0;

Lines={};

for i=1:l
    X=V.item{11}.value{1}.bfi{1}.gobject{i}.ATTRIBUTE.type;
    if(strcmp(X,'polygon'))
        vertices = V.item{11}.value{1}.bfi{1}.gobject{i}.vertex;
        no_of_vertices=length(vertices);
        Lines{color_no+1}={};

        % read polygon corners into Lines
        x(1)=-1; y(1)=-10;
        for j=1:no_of_vertices
            Attr=vertices{j}.ATTRIBUTE;
            x(2)=str2num(Attr.x); y(2)=str2num(Attr.y);
            Lines{color_no+1}{j}.x = x(2);
            Lines{color_no+1}{j}.y = y(2);
            if(x(1)>=0)
                dx=x(2)-x(1); dy=y(2)-y(1);
                x(3)=x(2)+dy/2; y(3)=y(2)-dx/2;
                line(x,y,'Color',cmap(color_no+1,:));
            end
            x(1)=x(2); y(1)=y(2);
        end
                
        %go around polygon and generate boxes
        x(1) = Lines{color_no+1}{no_of_vertices}.x;
        y(1) = Lines{color_no+1}{no_of_vertices}.y;
 
        for j=1:no_of_vertices
            x(2) = Lines{color_no+1}{j}.x;
            y(2) = Lines{color_no+1}{j}.y;
            leng = floor(sqrt((x(2)-x(1))^2 + (y(2)-y(1))^2));
            xstep=(x(2)-x(1))/leng;
            ystep=(y(2)-y(1))/leng;
            x_line=ystep*5;
            y_line=-xstep*5;
            xx=x(1); yy=y(1);
            for i=1:leng
                xx=xx+xstep; yy=yy+ystep;
                line([xx,xx-x_line],[yy,yy-y_line],'Color','r');
                line([xx,xx+x_line],[yy,yy+y_line],'Color','g');
                fprintf(output,'x1=%d, y1=%d, x2=%d, y2=%d\n', ...
                        floor(xx-x_line),floor(yy-y_line), ...
                        floor(xx+x_line),floor(yy+y_line));
            end
            x(1)=x(2); y(1)=y(2);
        end
        
        color_no=mod(color_no+1,size(cmap,1));
        
    end
end

