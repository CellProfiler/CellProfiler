function [SmoothedImage RealFilterLength] = CPsmooth(OrigImage,SmoothingMethod,SizeOfSmoothingFilter,WidthFlg)

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
%   Peter Swire
%   Rodrigo Ipince
%   Vicky Lay
%   Jun Liu
%   Chris Gang
%
% Website: http://www.cellprofiler.org
%
% $Revision$

SmoothedImage = OrigImage;
RealFilterLength = 0;

%%% For now, nothing fancy is done to calculate the size automatically. We
%%% just choose 1/40 the size of the image, with a min of 1 and max of 30.
if strcmpi(SizeOfSmoothingFilter,'A')
    if size(OrigImage,3) > 1
        error('CPSmooth only works on grayscale images.')
    end
    SizeOfSmoothingFilter = min(30,max(1,ceil(mean(size(OrigImage))/40))); % Get size of filter
    WidthFlg = 0;
end

%%% If we are NOT using the polynomial method and the user set the Size of
%%% Smoothing Filter to be 0, no smoothing will be done.
if SizeOfSmoothingFilter == 0 && ~strncmp(SmoothingMethod,'P',1)
    %%% No blurring is done.
    return;
end

switch SmoothingMethod
    case 'P'
        %%% The following is used to fit a low-dimensional polynomial to
        %%% the original image. The SizeOfSmoothingFilter is not relevant
        %%% for this method.
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
    case 'S'
        %%% The following is used for the Sum of squares method.
        FiltLength = SizeOfSmoothingFilter;
        PaddedImage = padarray(OrigImage,[FiltLength FiltLength],'replicate');
        %%% Could be good to use a disk structuring element of
        %%% floor(FiltLength/2) radius instead of a square window, or allow
        %%% user to choose.
        SmoothedImage = conv2(PaddedImage.^2,ones(FiltLength,FiltLength),'same');
        SmoothedImage = SmoothedImage(FiltLength+1:end-FiltLength,FiltLength+1:end-FiltLength);
        RealFilterLength=2*FiltLength;
    case 'Q'
        %%% The following is used for the Square of sum method.
        FiltLength = SizeOfSmoothingFilter;
        PaddedImage = padarray(OrigImage,[FiltLength FiltLength],'replicate');
        %%% Could be good to use a disk structuring element of
        %%% floor(FiltLength/2) radius instead of a square window, or allow
        %%% user to choose.
        SumImage = conv2(PaddedImage,ones(FiltLength,FiltLength),'same');
        SmoothedImage = SumImage.^2;
        SmoothedImage = SmoothedImage(FiltLength+1:end-FiltLength,FiltLength+1:end-FiltLength);
        RealFilterLength=2*FiltLength;
    case 'M'
        %%% The following is used for the Median Filtering method.
        if WidthFlg
            %%% Empirically done (from IdentifyPrimAutomatic)
            sigma = SizeOfSmoothingFilter/3.5;
        else
            sigma = SizeOfSmoothingFilter/2.35; % Convert between Full Width at Half Maximum (FWHM) to sigma
        end
        FiltLength = min(30,max(1,ceil(2*sigma))); % Determine filter size, min 3 pixel, max 61
        [x,y] = meshgrid(-FiltLength:FiltLength,-FiltLength:FiltLength);      % Filter kernel grid
        f = exp(-(x.^2+y.^2)/(2*sigma^2));f = f/sum(f(:));                    % Gaussian filter kernel
        %%% The original image is blurred. Prior to this blurring, the
        %%% image is padded with values at the edges so that the values
        %%% around the edge of the image are not artificially low.  After
        %%% blurring, these extra padded rows and columns are removed.
        SmoothedImage = conv2(padarray(OrigImage, [FiltLength,FiltLength], 'replicate'),f,'same');
        SmoothedImage = SmoothedImage(FiltLength+1:end-FiltLength,FiltLength+1:end-FiltLength);
        % I think this is wrong, but we should ask Ray.
        % RealFilterLength = 2*FiltLength+1;
        RealFilterLength = FiltLength;
    case 'A'
        %%% The following is used for the Smooth to average method.
        %%% Creates an image where every pixel has the value of the mean of the original
        %%% image.
        SmoothedImage = mean(OrigImage(:))*ones(size(OrigImage));
    otherwise
        if ~strcmp(SmoothingMethod,'N');
            error('The smoothing method you specified is not valid. This error should not have occurred. Check the code in the module or tool you are using or let the CellProfiler team know.');
        end
end