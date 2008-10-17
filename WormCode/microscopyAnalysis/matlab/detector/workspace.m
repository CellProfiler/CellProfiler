%%
common_size = size(scores{end});
max_score = zeros(common_size(1),common_size(2));
max_angle = zeros(common_size(1),common_size(2));
max_size = zeros(common_size(1),common_size(2));
num=1;
for i=1:length(scores)
  if(isempty(scores{i})); continue; end
  cur_score = scores{i};
  
  [h w a] = size(cur_score);
  startx = floor((w-common_size(2))/2)+1;
  endx = ceil((w-common_size(2))/2);
  starty = floor((h-common_size(1))/2)+1;
  endy = ceil((h-common_size(1))/2);

  cur_score = cur_score(starty:end-endy,startx:end-endx,:);
  jj(:,:,:,num) = cur_score;
  num=num+1;

%   [maxCur,maxCurAngle] = max(cur_score,[],3);
%   toReplace = maxCur>max_score;
%   max_score(toReplace)=maxCur(toReplace);
%   max_angle(toReplace)=maxCurAngle(toReplace);
%   max_size(toReplace) = i;

end

%%

for image = [1 18 19 57 72]
    imageName = sprintf('/Users/mkabra/worms_run/images/norm%03d',image)
    scoreTrainingData(imageName,[imageName 'OutLines'],-2,-10,10);
end
fprintf('Done\n');
%%
    
for image = [1 18 19 22 57 72]
    imageName = sprintf('/Users/mkabra/worms_run/images/norm%03d',image);
    scoreFile = sprintf('/Users/mkabra/worms_run/matlab/norm%03dScores',image)
    eval(['load ' scoreFile])
    wormHighScoreOutLines(scores,[imageName 'OutLines'],4,-2);
end
fprintf('Done\n');
%%

jj = randperm(96);
for image = jj(1:20)
    imgName = sprintf('/Users/mkabra/worms_run/images/norm%03d.tif',image);
    img = imread(imgName);
    imgMask = wormDist(img,10);
    overlay(img,~zero_crossings(imgMask-30),1);
    pause;
end
%%
dummy_var = rand(100000,165)*1000;
tic;
for i=1:size(dummy_var,1)
    pred = calcScore(dummy_var(i,:),ones(1,165));
end
toc;
tic;
pred = calcScore_vect(dummy_var);
toc;