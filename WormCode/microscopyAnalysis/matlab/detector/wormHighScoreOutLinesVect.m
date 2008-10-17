function wormHighScoreOutLinesVect(scores,outputFile,scale,sizeScale)

% This code generates lines which get high score during classification. 
% The lines are ddded to .bix file to get the feedback.

angleStep = 30;

sizeScale = -sizeScale;

[imgH imgW] = size(scores{end}{1});
nAngles = length(scores{end});
count = 1;
for cur_size = 1:length(scores)
  if(isempty(scores{cur_size})); continue; end
  count = count+1;
end
count = count-1;

cropScore = zeros(imgH,imgW,nAngles,count);

count = 1;
for cur_size=1:length(scores)
  if(isempty(scores{cur_size})); continue; end
  
  cur_score = scores{cur_size};
  
  % Make everything into common size. Ignore fringes.
  [h w] = size(cur_score);
  nAngles = length(cur_score);

  for i=1:nAngles
      cropScore(:,:,i,count)= cur_score{i};
      edgeSize(count)=cur_size;
  end
  count=count+1;
  
end

nSize = count-1;
out = fopen(outputFile,'w');

% --- Following is to get lines that are in top 0.2 percentile of score ---
%
% cutOff=prctile(cropScore(:),99.95);
cutOff = max(cropScore(:))/2.5;
selScore = cropScore>cutOff;
highScore = cropScore;
highScore(~selScore)=0;

minScore = min(cropScore(selScore));
maxScore = max(cropScore(selScore));
map = get_colormap();

for numSize = 1:size(highScore,4)
  for curAngle = 1:size(highScore,3)
      curSize = edgeSize(numSize);
      curScore = highScore(:,:,curAngle,numSize);
      normScore = (curScore-minScore)/(maxScore-minScore);
      [xx yy] = meshgrid(1:imgW,1:imgH);
      x1 = xx*scale-sind(curAngle*angleStep)*curSize/sizeScale;
      x2 = xx*scale+sind(curAngle*angleStep)*curSize/sizeScale;
      y1 = yy*scale+cosd(curAngle*angleStep)*curSize/sizeScale;
      y2 = yy*scale-cosd(curAngle*angleStep)*curSize/sizeScale;
      
      x1(x1<0)=1; y1(y1<0)=1;
      x2(x2<0)=1; y2(y2<0)=1;
      allNdx = find(curScore(:));
      for ndx = allNdx'
          curColor = uint8(map(uint8(normScore(ndx)*63)+1,:)*255);
    
          fprintf(out,'%.2f %.2f %.2f %.2f #%02x%02x%02x %.2f %.2f\n', ...          
          x1(ndx),y1(ndx),x2(ndx),y2(ndx),curColor(1),curColor(2),curColor(3),curAngle*angleStep,curSize);

      end
  end
end
      
fclose(out);
      
%       
%       
%     for yy = 1:size(highScore,1)
%       for xx = 1:size(highScore,2)
%         offset = 0;
%         if(highScore(yy,xx,curAngle,numSize)==0);continue;end
%         curScore = highScore(yy,xx,curAngle,numSize);
%         normScore = (curScore-minScore)/(maxScore-minScore);
%         curColor = uint8(map(uint8(normScore*63)+1,:)*255);
%     
%         curSize = edgeSize(numSize);
%         x1 = offset+xx*scale-sind(curAngle*angleStep)*curSize/sizeScale;
%         y1 = offset+yy*scale+cosd(curAngle*angleStep)*curSize/sizeScale;
%         x2 = offset+xx*scale+sind(curAngle*angleStep)*curSize/sizeScale;
%         y2 = offset+yy*scale-cosd(curAngle*angleStep)*curSize/sizeScale;
%         fprintf(out,'%.2f %.2f %.2f %.2f #%02x%02x%02x %.2f %.2f\n', ...
%           x1,y1,x2,y2,curColor(1),curColor(2),curColor(3),curAngle*angleStep,curSize);
%       end
%     end
%   end
% end
% 
% fclose(out);
% 

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