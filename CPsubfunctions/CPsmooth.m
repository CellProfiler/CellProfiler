function SmoothedImage = CPsmooth(OrigImage,SmoothingMethod,SetBeingAnalyzed)

% This subfunction is used for several modules, including SMOOTH, AVERAGE,
% CORRECTILLUMINATION_APPLY, CORRECTILLUMINATION_CALCULATE,
% IDENTIFYPRIMAUTOMATIC
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


SmoothedImage = OrigImage;

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
else try SizeOfSmoothingFilter = str2num(SmoothingMethod);
        sigma = SizeOfSmoothingFilter/2.35;   % Convert between Full Width at Half Maximum (FWHM) to sigma
        if SizeOfSmoothingFilter == 0
            %%% No blurring is done.
        else
            FiltLength = min(30,max(1,ceil(2*sigma)));                            % Determine filter size, min 3 pixels, max 61
            [x,y] = meshgrid(-FiltLength:FiltLength,-FiltLength:FiltLength);      % Filter kernel grid
            f = exp(-(x.^2+y.^2)/(2*sigma^2));f = f/sum(f(:));                    % Gaussian filter kernel
            %%% The original image is blurred. Prior to this blurring, the
            %%% image is padded with values at the edges so that the values
            %%% around the edge of the image are not artificially low.  After
            %%% blurring, these extra padded rows and columns are removed.
            SmoothedImage = conv2(padarray(OrigImage, [FiltLength,FiltLength], 'replicate'),f,'same');
            SmoothedImage = SmoothedImage(FiltLength+1:end-FiltLength,FiltLength+1:end-FiltLength);

            %DIAGNOSTIC
            %figure, imagesc(f), title('within CPsmooth')

        end
    catch
        error(['The text you entered for the smoothing method is not valid for some reason. You must enter N, P, or a positive, even number. Your entry was ', SmoothingMethod])
    end
end