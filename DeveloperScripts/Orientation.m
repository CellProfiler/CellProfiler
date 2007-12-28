function [out_im, out_vec, binim, ridge_enhanced_im, orientim, orient_vec, reliability] = Orientation(im)

% Identify ridge-like regions and normalise image
blksze = 4; thresh = 0.1;
[normim, mask] = ridgesegment(im, blksze, thresh);
%show(normim,1);

% Determine ridge orientations
[orientim, reliability] = ridgeorient(normim, 1, 4, 4); 
%plotridgeorient(orientim, 3, im, 2)
%show(reliability,6)
%figure(6),imagesc(reliability);

% Determine ridge frequency values across the image
blksze = 36; 
[freq, medfreq] = ridgefreq(normim, mask, orientim, blksze, 5, 5, 15);
%show(freq,3) 

% Actually I find the median frequency value used across the whole
% fingerprint gives a more satisfactory result...
freq = medfreq.*mask;

% Now apply filters to enhance the ridge pattern
ridge_enhanced_im = ridgefilter(normim, orientim, freq, 0.5, 0.5, 1);
%show(ridge_enhanced_im,4);

% Binarise, ridge/valley threshold is 0
binim = ridge_enhanced_im > 0;
%show(binim,5);

%figure(100), imagesc(ridge_enhanced_im);
%figure(101), imagesc(reliability.*binim);
%figure(102), imagesc(orientim.*binim);

% Display binary image for where the mask values are one and where
% the orientation reliability is greater than 0.5
relim = binim.*mask.*(reliability>0.7);
relim = relim > 0;
%show(binim.*mask.*(reliability>0.5), 7)
%figure(8), imagesc(relim);

out_im = relim;

%out_im = imresize(out_im, rescale);
%orientim = imresize(orientim, rescale);
%binim = imresize(binim, rescale);

%figure(9), imagesc(out_im);
%figure(10), imagesc(orientim);
%figure(11), imagesc(orientim.*out_im);
%figure(12), imagesc(binim);
image = out_im(:,:,1);
[x1, x2] = find(image);
out_vec = [x1 x2];
orient_vec = orientim(find(image));