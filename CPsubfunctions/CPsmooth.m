function SmoothedImage = CPsmooth(OrigImage,SmoothingMethod,SetBeingAnalyzed)

% This subfunction is used for several modules, including SMOOTH, AVERAGE,
% CORRECTILLUMINATION_APPLY, CORRECTILLUMINATION_CALCULATE
%
% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne E. Carpenter
%   Thouis Ray Jones
%   In Han Kang
%   Ola Friman
%   Steve Lowe
%   Joo Han Chang
%   Colin Clarke
%   Mike Lamprecht
%
% Website: http://www.cellprofiler.org
%
% $Revision$


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
        ArtifactRadiusPre = 0.5*ArtifactWidth;
        ArtifactRadius = floor(ArtifactRadiusPre);
        if (SetBeingAnalyzed == 1) && (ArtifactRadiusPre ~= ArtifactRadius)
            CPmsgbox('The number you entered was odd and has been rounded down to an even number.');
        end
        StructuringElementLogical = getnhood(strel('disk', ArtifactRadius));
        SmoothedImage = ordfilt2(OrigImage, floor(sum(sum(StructuringElementLogical))/2), StructuringElementLogical, 'symmetric');
    catch
        error(['The text you entered for the smoothing method is not valid for some reason. You must enter N, P, or a positive, even number. Your entry was ',SmoothingMethod])
    end
end