function SmoothedImage = CPsmooth(OrigImage,SmoothingMethod)

%%% This subfunction is used for several illumination correction
%%% modules, including SmoothImageForIllumCorrection.

if strcmpi(SmoothingMethod,'N') == 1
elseif strcmpi(SmoothingMethod,'P') == 1
    %%% The following is used to fit a low-dimensional polynomial to the original image.
    [x,y] = meshgrid(1:size(OrigImage,2), 1:size(OrigImage,1));
    x2 = x.*x;
    y2 = y.*y;
    xy = x.*y;
    o = ones(size(OrigImage));
    Ind = find(OrigImage > 0);
    Coeffs = [x2(Ind) y2(Ind) xy(Ind) x(Ind) y(Ind) o(Ind)] \ double(OrigImage(Ind));
    drawnow
    SmoothedImage1 = reshape([x2(:) y2(:) xy(:) x(:) y(:) o(:)] * Coeffs, size(OrigImage));
    %%% The final SmoothedImage is produced by dividing each
    %%% pixel of the smoothed image by a scalar: the minimum
    %%% pixel value anywhere in the smoothed image. (If the
    %%% minimum value is zero, .00000001 is substituted instead.)
    %%% This rescales the SmoothedImage from 1 to some number.
    %%% This ensures that the final, corrected image will be in a
    %%% reasonable range, from zero to 1.
    drawnow
    SmoothedImage = SmoothedImage1 ./ max([min(min(SmoothedImage1)); .0001]);
 else try ArtifactWidth = str2num(SmoothingMethod);
        ArtifactRadius = 0.5*ArtifactWidth;
        StructuringElementLogical = getnhood(strel('disk', ArtifactRadius));
%         MsgBoxHandle = CPmsgbox('Now calculating the smoothed image, which may take a long time.');
        SmoothedImage1 = ordfilt2(OrigImage, floor(sum(sum(StructuringElementLogical))/2), StructuringElementLogical, 'symmetric');
        SmoothedImage = SmoothedImage1 ./ max([min(min(SmoothedImage1)); .0001]);
%         MsgBox = 'Calculations for smoothing are complete.';
    catch
        error(['The text you entered for the smoothing method in the Smooth Image For Illum Correction module is unrecognizable for some reason. You must enter a positive, even number or the letter P.  Your entry was ',SmoothingMethod])
    end
end