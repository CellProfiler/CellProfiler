function wormHighScoreOutLines(scores,outputFile,scale,sizeScale)

sizeScale = -sizeScale;
common_size = size(scores{end});
offset = ceil(length(scores)/2);

count = 1;
for cur_size=1:length(scores)
  if(isempty(scores{cur_size})); continue; end
  
  cur_score = scores{cur_size};
  
  % Make everything into common size. Ignore fringes.
  [h w nAngles] = size(cur_score);
  startx = floor((w-common_size(2))/2)+1;
  endx = ceil((w-common_size(2))/2);
  starty = floor((h-common_size(1))/2)+1;
  endy = ceil((h-common_size(1))/2);

  cur_score = cur_score(starty:end-endy,startx:end-endx,:);
  cropScore(:,:,:,count)= imresize(cur_score,scale);
  edgeSize(count)=cur_size;
  count=count+1;
  
end

nSize = count-1;

% -- This one outputs only one line for each location. Also only positive
% score lines are printed.

[scoreMaxSize sz] = max(cropScore,[],4);
[scoreMax maxAng] = max(scoreMaxSize,[],3);
maxSize = zeros(size(maxAng));
for ii = 1:size(maxAng,1)
    for jj=1:size(maxAng,2)
        maxSize(ii,jj)=sz(ii,jj,maxAng(ii,jj));
    end
end
maxSize = edgeSize(maxSize);

maxScore = max(scoreMax(:));
map = get_colormap();

out = fopen(outputFile,'w');
cutOff = prctile(scoreMax(:),99);

if cutOff<0
    cutOff = 0;
end

for ii = 1:size(maxAng,1)
    for jj=1:size(maxAng,2)
        if(scoreMax(ii,jj)<cutOff);continue;end
        curScore = scoreMax(ii,jj);
        normScore = curScore/maxScore;
        curColor = uint8(map(uint8(normScore*63)+1,:)*255);

        curSize = maxSize(ii,jj);
        curAngle = maxAng(ii,jj);
        xx = jj; yy = ii;
        x1 = offset+xx-sind(curAngle*10)*curSize/sizeScale;
        y1 = offset+yy+cosd(curAngle*10)*curSize/sizeScale;
        x2 = offset+xx+sind(curAngle*10)*curSize/sizeScale;
        y2 = offset+yy-cosd(curAngle*10)*curSize/sizeScale;
        fprintf(out,'%.2f %.2f %.2f %.2f #%02x%02x%02x %.2f %.2f\n', ...
            x1,y1,x2,y2,curColor(1),curColor(2),curColor(3),curAngle,curSize);
    end
end

fclose(out);

hsvimg = zeros(size(scoreMax,1),size(scoreMax,2),3);
hsvimg(:,:,1) = maxAng/18;
hsvimg(:,:,2) = maxSize/max(edgeSize(:));
tt = scoreMax;
tt = tt/max(tt(:));
tt(tt<0.2)=0;
hsvimg(:,:,3) = tt;
rgbimg = hsv2rgb(hsvimg);
imwrite(rgbimg,[outputFile '_image.png'],'png');
        
% --- Following is to get lines that are in top 0.2 percentile of score ---
%
% cutOff=prctile(cropScore(:),99.8);
% selScore = cropScore>cutOff;
% highScore = cropScore;
% highScore(~selScore)=0;
% 
% minScore = min(cropScore(selScore));
% maxScore = max(cropScore(selScore));
% map = get_colormap();
% 
% for numSize = 1:size(highScore,4)
%   for curAngle = 1:size(highScore,3)
%     for yy = 1:size(highScore,1)
%       for xx = 1:size(highScore,2)
%         if(highScore(yy,xx,curAngle,numSize)==0);continue;end
%         curScore = highScore(yy,xx,curAngle,numSize);
%         normScore = (curScore-minScore)/(maxScore-minScore);
%         curColor = uint8(map(uint8(normScore*63)+1,:)*255);
%     
%         curSize = edgeSize(numSize);
%         x1 = offset+xx*scale-sind(curAngle*10)*curSize/sizeScale;
%         y1 = offset+yy*scale+cosd(curAngle*10)*curSize/sizeScale;
%         x2 = offset+xx*scale+sind(curAngle*10)*curSize/sizeScale;
%         y2 = offset+yy*scale-cosd(curAngle*10)*curSize/sizeScale;
%         fprintf(out,'%.2f %.2f %.2f %.2f #%02x%02x%02x %.2f %.2f\n', ...
%           x1,y1,x2,y2,curColor(1),curColor(2),curColor(3),curAngle,curSize);
%       end
%     end
%   end
% end
% 
% fclose(out);


function map = get_colormap()
map =[   0         0    0.5625 
         0         0    0.6250 
         0         0    0.6875 
         0         0    0.7500  
         0         0    0.8125
         0         0    0.8750
         0         0    0.9375
         0         0    1.0000
         0    0.0625    1.0000
         0    0.1250    1.0000
         0    0.1875    1.0000
         0    0.2500    1.0000
         0    0.3125    1.0000
         0    0.3750    1.0000
         0    0.4375    1.0000
         0    0.5000    1.0000
         0    0.5625    1.0000
         0    0.6250    1.0000
         0    0.6875    1.0000
         0    0.7500    1.0000
         0    0.8125    1.0000
         0    0.8750    1.0000
         0    0.9375    1.0000
         0    1.0000    1.0000
    0.0625    1.0000    0.9375
    0.1250    1.0000    0.8750
    0.1875    1.0000    0.8125
    0.2500    1.0000    0.7500
    0.3125    1.0000    0.6875
    0.3750    1.0000    0.6250
    0.4375    1.0000    0.5625
    0.5000    1.0000    0.5000
    0.5625    1.0000    0.4375
    0.6250    1.0000    0.3750
    0.6875    1.0000    0.3125
    0.7500    1.0000    0.2500
    0.8125    1.0000    0.1875
    0.8750    1.0000    0.1250
    0.9375    1.0000    0.0625
    1.0000    1.0000         0
    1.0000    0.9375         0
    1.0000    0.8750         0
    1.0000    0.8125         0
    1.0000    0.7500         0
    1.0000    0.6875         0
    1.0000    0.6250         0
    1.0000    0.5625         0
    1.0000    0.5000         0
    1.0000    0.4375         0
    1.0000    0.3750         0
    1.0000    0.3125         0
    1.0000    0.2500         0
    1.0000    0.1875         0
    1.0000    0.1250         0
    1.0000    0.0625         0
    1.0000         0         0
    0.9375         0         0
    0.8750         0         0
    0.8125         0         0
    0.7500         0         0
    0.6875         0         0
    0.6250         0         0
    0.5625         0         0
    0.5000         0         0];