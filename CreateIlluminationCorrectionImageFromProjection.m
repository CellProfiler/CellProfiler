function CreateIlluminationCorrectionImageFromProjection(MeanImageFileName, DesiredIllumCorrFileName)

% This code is run from the command line.
%
% Its purpose is to take a projection (mean) image from some other
% source (not a CellProfiler module) and make an illumination
% correction function from it.
%
% To use it, type:
% CreateIlluminationCorrectionImageFromProjection('OriginalFilename.
% mat','NewFilename.mat') at the command line, where the first
% filename is what you called the mean image and the second is what
% you want to call the illumination correction image.


load(MeanImageFileName);
%%% The following is used to fit a low-dimensional polynomial to the mean image.
%%% The result, IlluminationImage, is an image of the smooth illumination function.
[x,y] = meshgrid(1:size(MeanImage,2), 1:size(MeanImage,1));
x2 = x.*x;
y2 = y.*y;
xy = x.*y;
o = ones(size(MeanImage));
Ind = find(MeanImage > 0);
Coeffs = [x2(Ind) y2(Ind) xy(Ind) x(Ind) y(Ind) o(Ind)] \ double(MeanImage(Ind));
drawnow
IlluminationImage1 = reshape([x2(:) y2(:) xy(:) x(:) y(:) o(:)] * Coeffs, size(MeanImage));
%%% The final IlluminationImage is produced by dividing each
%%% pixel of the illumination image by a scalar: the minimum
%%% pixel value anywhere in the illumination image. (If the
%%% minimum value is zero, .00000001 is substituted instead.)
%%% This rescales the IlluminationImage from 1 to some number.
%%% This ensures that the final, corrected image will be in a
%%% reasonable range, from zero to 1.
drawnow
IlluminationImage = IlluminationImage1 ./ max([min(min(IlluminationImage1)); .00000001]);
%%% Note: the following "imwrite" saves the illumination
%%% correction image in TIF format, but the image is compressed
%%% so it is not as smooth as the image that is saved using the
%%% "save" function below, which is stored in matlab ".mat"
%%% format.
% imwrite(IlluminationImage, 'IlluminationImage.tif', 'tif')

%%% Saves the illumination correction image to the hard
%%% drive.
try save(DesiredIllumCorrFileName, 'IlluminationImage')
catch error(['There was a problem saving the illumination correction image to the hard drive. The attempted filename was ', DesiredIllumCorrFileName, '.'])
end

figure, subplot(1,2,1)
imagesc(MeanImage)
colormap(gray), title('Input image: MeanImage')
subplot(1,2,2)
imagesc(IlluminationImage)
colormap(gray), title('Illumination Correction Function')