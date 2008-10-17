function wormPosOutLines(scores,outputFile,scale,sizeScale)

out = fopen(outputFile,'w');

sizeScale = -sizeScale;
common_size = size(scores{end});
max_score = -inf*ones(common_size(1),common_size(2));
% max_angle = zeros(common_size(1),common_size(2));
avg_score = zeros(common_size(1),common_size(2));
avg_score_weight = zeros(common_size(1),common_size(2));
offset = 10;

for cur_size=1:length(scores)
  if(isempty(scores{cur_size})); continue; end
  
  cur_score = scores{cur_size};
  
  % Make everything into common size. Ignore fringes.
  [h w a] = size(cur_score);
  startx = floor((w-common_size(2))/2)+1;
  endx = ceil((w-common_size(2))/2);
  starty = floor((h-common_size(1))/2)+1;
  endy = ceil((h-common_size(1))/2);
  cur_score = cur_score(starty:end-endy,startx:end-endx,:);
  
  %Find the maximum scores for current size
  [maxCur angleNdx] = max(cur_score,[],3);
  toReplace = maxCur>max_score;
  max_score(toReplace)=maxCur(toReplace);
%   max_angle(toReplace) = angleNdx(toReplace);
  
  %Choose the pixels with positive scores
  cur_score(cur_score<0)=0;

  %Convert them into complex numbers (Since we have symmetry 180d->360d).
  %Average the complex numbers with weights as scores. 
  %Keep track of weights in a separate matrix
  
  for curAngle=1:a
    avg_score = avg_score+cur_size*cur_score(:,:,curAngle).*...
         complex(cosd(curAngle*20),sind(curAngle*20));
    avg_score_weight = avg_score_weight + cur_score(:,:,curAngle); 
  end
  
end

avg_score = avg_score./avg_score_weight;
avg_score(isnan(avg_score))=0;
avg_score = imfilter(avg_score,fspecial('gaussian',13,2));
avg_score = imresize(avg_score,scale);

pos_max_score = max_score;
pos_max_score(pos_max_score<0)=0;
pos_max_score = imfilter(pos_max_score,fspecial('gaussian',13,2));
smooth_score = imresize(pos_max_score,scale);

avg_size = abs(avg_score);
avg_angle = angle(avg_score)*90/pi+90;

nonmaxScore = nonmaxsup(smooth_score,avg_angle,1.5);
nonmaxScore = nonmaxScore-min(nonmaxScore(:));
nonmaxScore = nonmaxScore/max(nonmaxScore(:));
nonmaxScore(nonmaxScore<0.1) = 0;

map = get_colormap();
for xx = 1:size(smooth_score,2)
  for yy=1:size(smooth_score,1)
    if(nonmaxScore(yy,xx)<=0); continue; end
    curAngle = avg_angle(yy,xx)+90;
    curSize = avg_size(yy,xx);
    curColor = uint8(map(uint8(nonmaxScore(yy,xx)*63)+1,:)*255);
    
    x1 = offset+xx-sind(curAngle)*curSize/sizeScale;
    y1 = offset+yy+cosd(curAngle)*curSize/sizeScale;
    x2 = offset+xx+sind(curAngle)*curSize/sizeScale;
    y2 = offset+yy-cosd(curAngle)*curSize/sizeScale;
    fprintf(out,'%.2f %.2f %.2f %.2f #%02x%02x%02x %.2f %.2f\n', ...
      x1,y1,x2,y2,curColor(1),curColor(2),curColor(3),curAngle,curSize);
  end
end

fclose(out);


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