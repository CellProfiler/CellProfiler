function [Ibox,Dbox,labels,weights] = extractBoxes(I,D,linelist,label,weight,sizeFactor)
% Extract a list of boxes from images according to a list of lines
% I - Rotated Images
% D - Rotated Derivative images
% labels - the labels correspnding to the images (need this because
% out-of-bounds images are removed from the list)
% linelist - a table defining the lines
% sizefactor - if positive: the size of the box relative to the length of
%              the line, if negative: the (negation of) the size of the
%              box.
% Iboxes, Dboxes: the extracted subimages
    Ibox = {};
    Dbox = {};
    labels = {};
    weights = [];
    if sizeFactor <= 0
        sz = -sizeFactor;
        sz = sz + 1-mod(sz,2);  % make sz odd
    end
    
    Isize = size(I,1);
    Dsize = size(D,1);
    
    [yc,xc] = size(I{1}); yc=round(yc/2); xc=round(xc/2);

    n=1;
    for i=1:size(linelist,1)
        
        x1=linelist(i,1);
        y1=linelist(i,2);
        x2=linelist(i,3);
        y2=linelist(i,4);
        
        angle = atan2((y2-y1),(x2-x1))*(180/pi)+90;
        
        if sizeFactor>0
            sz = sizeFactor*round(sqrt((x2-x1)^2 + (y2-y1)^2));
            sz = sz + 1-mod(sz,2);  % make sz odd
        end
        
	cx=round((x1+x2)/2-xc);
	cy=round((y1+y2)/2-yc);
	% extract boxes only if they fit comfortably within image frame
	if (abs(cx)+sz<xc && abs(cy)+sz<yc)
	  for j=1:Isize
            Ibox{n,j} = getRotatedBox(I(j,:),cx,cy,sz,round(angle));
	  end
	  for j=1:Dsize
            Dbox{n,j} = getRotatedDerivBox(D(j,:),cx,cy,sz,round(angle));
	  end
	  labels(n) = label(i);
      weights(n) = weight(i);
	  n=n+1;
	end
	
% visualization code for debugging purposes        
%         for j=1:Isize
%             subplot(1,Isize+Dsize,j);
%             imshow(Ibox{i,j});
%         end
%         title([num2str(angle) ' degrees']);
% 
%         for j=1:Dsize
%             clear HSV
%             HSV(1:sz,1:sz,3)=1;
%             HSV(1:sz,1:sz,2)=Dbox{i,j}.mag;
%             HSV(1:sz,1:sz,1)=Dbox{i,j}.angle;
%             subplot(1,Isize+Dsize,Isize+j);
%             imshow(hsv2rgb(HSV));
%         end
%         
%         answer=input('step forward or end (Forward)','s');
%         if isempty(answer)
%             i
%         else
%             return;
%         end
    end





