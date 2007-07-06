im = imread('7cap33x.pgm');
restartprob = 0.0001;
nwalks = 1000;

seeds = { [[202:210]' [465:473]'] ...
	  [[361:369]' [286:294]'] ...
	  [[605:613]' [228:236]'] ...
	  [[85:93]' [216:224]'] };

origdisp = zeros(size(im, 1), size(im, 2), 3);
origdisp(:,:,1) = im2double(im);
origdisp(:,:,2) = im2double(im);
origdisp(:,:,3) = im2double(im);
for i=1:numel(seeds)
  seed = seeds{i};
  origdisp(seed(:,2),seed(:,1),1) = 1;
  origdisp(seed(:,2),seed(:,1),2) = 0;
  origdisp(seed(:,2),seed(:,1),3) = 0;
end
subplot(2, 1, 1);
imshow(origdisp);
title('Original image with seeds');

pmasks = cell(1, numel(seeds));
for i=1:numel(seeds)
  pmasks{i} = ljosaprobseg(im, restartprob, nwalks, seeds{i});
end
disp = zeros(size(im, 1), size(im, 2), 3);
disp(:,:,1) = pmasks{1} + pmasks{4};
disp(:,:,2) = pmasks{2} + pmasks{4};
disp(:,:,3) = pmasks{3};
disp = min(disp, 1);
subplot(2, 1, 2);
imshow(disp);
title('Segmented image');

