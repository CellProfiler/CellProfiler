function SmoothedImage = CPsmooth(OrigImage,SmoothingMethod)

%%% This subfunction is used for several modules, including
%%% SMOOTHIMAGE, MAKEPROJECTION_AVERAGEIMAGES,
%%% CORRECTILLUMINATION_APPLY,
%%% CORRECTILLUMINATION_CALCULATEUSINGINTENSITIES,
%%% CORRECTILLUMINATION_CALCULATEUSINGBACKGROUNDINTENSITIES.

if strcmpi(SmoothingMethod,'N') == 1
elseif strcmpi(SmoothingMethod,'P') == 1
    %%% The following is used to fit a low-dimensional polynomial to the original image.
    [x,y] = meshgrid(1:size(OrigImage,2), 1:size(OrigImage,1));
    x2 = x.*x;
    y2 = y.*y;
    xy = x.*y;
    o = ones(size(OrigImage));
    drawnow
    Ind = find(OrigImage > 0);
    Coeffs = [x2(Ind) y2(Ind) xy(Ind) x(Ind) y(Ind) o(Ind)] \ double(OrigImage(Ind));
    drawnow
    SmoothedImage = reshape([x2(:) y2(:) xy(:) x(:) y(:) o(:)] * Coeffs, size(OrigImage));
 else try ArtifactWidth = str2num(SmoothingMethod);
        ArtifactRadius = 0.5*ArtifactWidth;
        StructuringElementLogical = getnhood(strel('disk', ArtifactRadius));
%         MsgBoxHandle = CPmsgbox('Now calculating the smoothed image, which may take a long time.');
        SmoothedImage = ordfilt2(OrigImage, floor(sum(sum(StructuringElementLogical))/2), StructuringElementLogical, 'symmetric');
%         MsgBox = 'Calculations for smoothing are complete.';
    catch
        error(['The text you entered for the smoothing method is not valid for some reason. You must enter N, P, or a positive, even number.  Your entry was ',SmoothingMethod])
    end
end



