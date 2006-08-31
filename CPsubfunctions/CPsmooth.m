function [SmoothedImage FiltLength] = CPsmooth(OrigImage,SmoothingMethod,SizeOfSmoothingFilter,WidthFlg)

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
FiltLength = 0;

%%% For now, nothing fancy is done to calculate the size automatically. We
%%% just choose 1/40 the size of the image, with a min of 1 and max of 30. 
%%%
%%% TODO: shouldn't size(OrigImage) be limited to the first two dimensions
%%% (length and width) in case it's a color image? I suppose in general
%%% only grayscale images will make it to this module anyway, but perhaps
%%% we should explicitly confirm that. -Anne
if strcmpi(SizeOfSmoothingFilter,'A')
    SizeOfSmoothingFilter = min(30,max(1,ceil(mean(size(OrigImage))/40))); % Get size of filter
    WidthFlg = 0;
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
        if SizeOfSmoothingFilter == 0
            %%% No blurring is done.
            return;
        elseif WidthFlg
            %%% The way we choose the filter size was taken from what was done in IdentifyPrimAutomatic
            SizeOfSmoothingFilter = 4*SizeOfSmoothingFilter/3.5;
        end
        FiltLength = SizeOfSmoothingFilter;
        PaddedImage = padarray(OrigImage,[FiltLength FiltLength],'replicate');
        %%% Could be good to use a disk structuring element of
        %%% floor(FiltLength/2) radius instead of a square window, or allow
        %%% user to choose.
        SmoothedImage = conv2(PaddedImage.^2,ones(FiltLength,FiltLength),'same');
        SmoothedImage = SmoothedImage(FiltLength+1:end-FiltLength,FiltLength+1:end-FiltLength);
    case 'Q'
        %%% The following is used for the Square of sum method.
        if SizeOfSmoothingFilter == 0
            %%% No blurring is done.
            return;
        elseif WidthFlg
            %%% The way we choose the filter size was taken from what was done in IdentifyPrimAutomatic
            SizeOfSmoothingFilter = 4*SizeOfSmoothingFilter/3.5;
        end
        FiltLength = SizeOfSmoothingFilter;
        PaddedImage = padarray(OrigImage,[FiltLength FiltLength],'replicate');
        %%% Could be good to use a disk structuring element of
        %%% floor(FiltLength/2) radius instead of a square window, or allow
        %%% user to choose.
        SumImage = conv2(PaddedImage,ones(FiltLength,FiltLength),'same');
        SmoothedImage = SumImage.^2;
        SmoothedImage = SmoothedImage(FiltLength+1:end-FiltLength,FiltLength+1:end-FiltLength);
    case 'M'
        %%% The following is used for the Median Filtering method.
        if SizeOfSmoothingFilter == 0;
            %%% No blurring is done.
            return;
        end
%%% Old versions of the code, prior to Rodrigo, used this:
% sigma = SizeOfSmoothingFilter/2.35;   % Convert between Full Width at Half Maximum (FWHM) to sigma
%%% Why suddenly all this mess about WidthFlag and why are we using 3.5 and
%%% 4 rather than 2.35 as before?? Why do we multiply the FiltLength by two at the end??
%%% Why is FiltLength exported at all? Asked Rodrigo 8-31-06
%%% I am hoping that we can remove WidthFlag altogether, because I think
%%% width is simply calculated from SizeOfSmoothingFilter in the modules
%%% which require it. -Anne

        if WidthFlg
            %%% Empirically done (from IdentifyPrimAutomatic)
            sigma = SizeOfSmoothingFilter/3.5;         % Convert between Full Width at Half Maximum (FWHM) to sigma
            FiltLength = min(30,max(1,ceil(2*sigma))); % Determine filter size, min 3 pixels, max 61
        else
            sigma = SizeOfSmoothingFilter/4;           % Select sigma to be roughly the same as above (relatively)
            FiltLength = min(30,max(1,ceil(2*sigma)));
        end
        [x,y] = meshgrid(-FiltLength:FiltLength,-FiltLength:FiltLength);      % Filter kernel grid
        f = exp(-(x.^2+y.^2)/(2*sigma^2));f = f/sum(f(:));                    % Gaussian filter kernel
        %%% The original image is blurred. Prior to this blurring, the
        %%% image is padded with values at the edges so that the values
        %%% around the edge of the image are not artificially low.  After
        %%% blurring, these extra padded rows and columns are removed.
        SmoothedImage = conv2(padarray(OrigImage, [FiltLength,FiltLength], 'replicate'),f,'same');
        SmoothedImage = SmoothedImage(FiltLength+1:end-FiltLength,FiltLength+1:end-FiltLength);
        FiltLength = 2*FiltLength;
    otherwise
        if ~strcmp(SmoothingMethod,'N');
            error('The smoothing method you specified is not valid. This error should not have occurred. Check the code in the module or tool you are using or let the CellProfiler team know.');
        end
end