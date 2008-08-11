function [thresh isUnimodal curvature smoothTest tcurv] = Threshold_Tsai(image,fsize)
% Extra outputs, for troubleshooting:
% isUnimodal curvature smoothTest tcurv
% 
% Here's the code. I've expanded my analysis to a few more image sets, and it 
% still seems to be doing a good job- I was having a bit of a problem with 
% false peaks (i.e. a "bumpy" background distribution), but I added a few 
% lines of code to deal with that issue. Like I said previously, for images 
%     with a really tight background distribution, and then a more spread out 
%     foreground distribution, this method appears to give much better results 
%     than the metric-based (inter-class variance, entropy, etc.). The exact 
%     paper the original algorithm was pulled from is:
% Du-Ming Tsai. A fast thresholding selection procedure for multimodal and 
% unimodal histograms. Pattern Recognition Letters 16(6): 653-666 (1995)
% 
% I'm happy to give more info on my implementation, and to take any 
% suggestions on its further improvement. Like I said, my experience is a 
% really nonexistent when it comes to this stuff, so I won't be surprised if
% the algorithm suffers from limited range of application.
% 
% When I put it in CPthreshold, I used a secondary function (Tsai.m) to call 
% Threshold_Tsai and provide the whole means of dealing with cropped images, 
% similar to the existing threshold functions.
% 
%
% This thresholding method uses feature extraction from the image histogram
% to set the threshold: it finds the valley between the foreground
% and background disributions, or maximizes a calculated "curvature" in the
% unimodal case.
% Thresholding method based on description in:
% Du-Ming Tsai: A fast thresholding selection procedure for multimodal and
% unimodal histograms. Pattern Recognition Letters 16(6): 653-666 (1995)

% Brooks Taylor, UVa
% Written 7/3/08

% Drop saturated pixels, get histogram function. Also, define parameter
% used to avoid false background peaks later on.
im = double(image(:));
im(im ==max(im)) = [];
[num x] = hist(im,256);
backLevel = 2/3;
% Define Gaussian kernel, construct filter: fsize tells how big the
% smoothing filter should be.
gaussian = [0.2261 0.5478 0.2261];
gauss = gaussian;
if fsize > 1
for i = 1:fsize-1
gauss = conv(gauss,gaussian);
end
end

% According to the algorithm, the histogram is to be smoothed until only 2
% peaks are found (my desired amount). For the sake of keeping from
% smoothing the data -that- much, I'm only going to worry about relatively
% high peaks (say within 20x of max peak, though this parameter can change.)
% I only want to do the above if there are at least 2 peaks in the slighly
% smoothed histogram (in unimodal case, use curvature method)

% Issue #1: sometimes, the background has a multi-peak shape that isn't
% smoothed out, causing a "false" valley that's way up in the background
% region. I'm going to take advantage of the fact that this distribution
% is tighter and higher to differentiate these background peaks.
% Alternatively, I could increase the minimum filter size from 3...

smoothTest = ifft(fft(num).*fft(gaussian,length(num)).*conj(fft(gaussian,length(num))));
[peakTestOrig locTestOrig] = findpeaks(smoothTest, 'minpeakheight',max(smoothTest)/20);
dropInd = find( peakTestOrig < max(peakTestOrig)& peakTestOrig > max(peakTestOrig)*backLevel);
peakTest = peakTestOrig; locTest = locTestOrig;
peakTest(dropInd) = []; locTest(dropInd) = [];

if length(peakTest) > 1
smooth = ifft(fft(num).*fft(gauss,length(num)).*conj(fft(gauss,length(num))));
[peak loc] = findpeaks(smooth, 'minpeakheight',max(smooth)/20);
dropInd = find( peak < max(peak)& peak > max(peak)*backLevel);
peak(dropInd) = [];
loc(dropInd) = [];
N = length(peak);
while N > 2
smooth = ifft(fft(smooth).*fft(gauss,length(num)).*conj(fft(gauss,length(num))));
[peak loc] = findpeaks(smooth, 'minpeakheight',max(smooth)/20);
dropInd = find( peak < max(peak)& peak > max(peak)*backLevel);
peak(dropInd) = [];
loc(dropInd) = [];
N = length(peak);
end
% Now find the minimum in between the two identified peaks. Convert that
% number, and set it as the threshold.
[min idx] = findpeaks(-smooth(loc(1):loc(2)));
thresh = x(loc(1)+idx-1);
curvature = [];
tcurv = [];
else
% Curvature- maximizing-method. The curvature looks like it's often
% going to be highest at the background peak, so I need to make sure I
% get on the other side of that. Since I'm setting the origin at the
% last background peak, I can choose the first maximum of curvature
% from that point.
isUnimodal = 1;
r = find(smoothTest == max(smoothTest));
r = max([r,locTestOrig(max(dropInd))])+1;
psi = @(t)(1/r) .* sum((smoothTest(t+1:r+t))-smoothTest(t-1:-1:t-r)./(2:2:2*r));
Kt = @(t) (1/r) *abs(sum(psi(t+1:t+r) - psi(t-1:-1:t-r)));
curvature = zeros(length(smoothTest)-2*r-4,1);
tcurv = r+2:length(smoothTest)-r-2;
for i = tcurv
curvature(i-r-1) = Kt(i);
end
[peak loc] = findpeaks(curvature);
thresh = x(loc(1) + (r+1));
end