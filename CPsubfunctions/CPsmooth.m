function [SmoothedImage RealFilterLength] = CPsmooth(OrigImage,SmoothingMethod,SizeOfSmoothingFilter,WidthFlg)

% This subfunction is used for several modules, including SMOOTH, AVERAGE,
% CORRECTILLUMINATION_APPLY, CORRECTILLUMINATION_CALCULATE,
% IDENTIFYPRIMAUTOMATIC
%
% SizeOfSmoothingFilter = Diameter of the Filter Window (Box).
%                       ~ roughly equal to object diameter
%
% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Developed by the Broad Institute of MIT and Harvard
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
%   Kyungnam Kim
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%% If SizeOfSmoothingFilter(S) >= LARGESIZE_OF_SMOOTHINGFILTER (L), 
%%% then rescale the original image by L/S, and rescale S to L.
%%% It is a predefined effective maximum filter size (diameter).
LARGESIZE_OF_SMOOTHINGFILTER = 50; 

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

%%% If the incoming image is binary (logical), we convert it to grayscale.
if islogical(OrigImage)
    OrigImage = im2double(OrigImage);
    %OrigImage = im2single(OrigImage);
end

%%% For faster smoothing with a large filter size:
%%% If the SizeOfSmoothingFilter is greather than
%%% LARGESIZE_OF_SMOOTHINGFILTER, then we resize the original image
%%% Tip: Smoothing with filter size LARGESIZE_OF_SMOOTHINGFILTER is the slowest.
if (SizeOfSmoothingFilter >= LARGESIZE_OF_SMOOTHINGFILTER)
    ResizingFactor = LARGESIZE_OF_SMOOTHINGFILTER/SizeOfSmoothingFilter;
    OrigImage = imresize(OrigImage, ResizingFactor);    
    SizeOfSmoothingFilter = LARGESIZE_OF_SMOOTHINGFILTER; % equal to SizeOfSmoothingFilter * ResizingFactor;
end

switch lower(SmoothingMethod)
    case {'fit polynomial','p'}
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
    case {'sum of squares','s'}
        %%% The following is used for the Sum of squares method.
        FiltLength = SizeOfSmoothingFilter;
        PaddedImage = padarray(OrigImage,[FiltLength FiltLength],'replicate');
        %%% Could be good to use a disk structuring element of
        %%% floor(FiltLength/2) radius instead of a square window, or allow
        %%% user to choose.
        SmoothedImage = conv2(PaddedImage.^2,ones(FiltLength,FiltLength),'same');
        SmoothedImage = SmoothedImage(FiltLength+1:end-FiltLength,FiltLength+1:end-FiltLength);
        RealFilterLength=2*FiltLength;
    case {'square of sum','q'}
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
    case 'median filter'
        %%% The following is used for the Median Filtering smoothing method
        %%% [Kyungnam 2007-Aug-3] 
        %%% 'medfilt2' pads the image with 0's on the edges, so the median
        %%% values for the points within [m n]/2 of the edges might appear distorted (to 0).
        %%% So, pad the image with 'replicate' values, and then median-filters.
        FiltLength = SizeOfSmoothingFilter;
        PaddedImage = padarray(OrigImage,[FiltLength FiltLength],'replicate');
        SmoothedImage = medfilt2(PaddedImage,[SizeOfSmoothingFilter SizeOfSmoothingFilter]);
        SmoothedImage = SmoothedImage(FiltLength+1:end-FiltLength,FiltLength+1:end-FiltLength);
    case {'median filtering','m'}
        %%% We leave this SmoothingMethod to be compatible with previous
        %%% pipelines that used 'median filtering'
        CPwarndlg('The smoothing method ''Median Filtering'' is not valid any more. Please replace it with ''Gaussian Filtering'' if you still want to make your pipeline working as it was. Or use ''Median Filter'' which was re-implemented.');
    case 'gaussian filter'
        %%% The following is used for the Gaussian lowpas filtering method.
        if WidthFlg
            %%% Empirically done (from IdentifyPrimAutomatic)
            sigma = SizeOfSmoothingFilter/3.5;
        else
            sigma = SizeOfSmoothingFilter/2.35; % Convert between Full Width at Half Maximum (FWHM) to sigma
        end
        h = fspecial('gaussian', [round(SizeOfSmoothingFilter) round(SizeOfSmoothingFilter)], sigma);
        SmoothedImage = imfilter(OrigImage, h, 'replicate');
%       [Kyungnam Jul-30-2007: The following old code that was replaced with the above code has been left for reference]        
%         FiltLength = min(30,max(1,ceil(2*sigma))); % Determine filter size, min 3 pixel, max 61
%         [x,y] = meshgrid(-FiltLength:FiltLength,-FiltLength:FiltLength);      % Filter kernel grid
%         f = exp(-(x.^2+y.^2)/(2*sigma^2));f = f/sum(f(:));                    % Gaussian filter kernel
%         %%% The original image is blurred. Prior to this blurring, the
%         %%% image is padded with values at the edges so that the values
%         %%% around the edge of the image are not artificially low.  After
%         %%% blurring, these extra padded rows and columns are removed.
%         SmoothedImage = conv2(padarray(OrigImage, [FiltLength,FiltLength], 'replicate'),f,'same');
%         SmoothedImage = SmoothedImage(FiltLength+1:end-FiltLength,FiltLength+1:end-FiltLength);
%         % I think this is wrong, but we should ask Ray.
%         % RealFilterLength = 2*FiltLength+1;
%         RealFilterLength = FiltLength;
    case {'smooth to average','a'}
        %%% The following is used for the Smooth to average method.
        %%% Creates an image where every pixel has the value of the mean of the original
        %%% image.
        SmoothedImage = mean(OrigImage(:))*ones(size(OrigImage));        
%       [Kyungnam Jul-30-2007: If you want to use the traditional averaging filter, use the following]
%        h = fspecial('average', [SizeOfSmoothingFilter SizeOfSmoothingFilter]);
%        SmoothedImage = imfilter(OrigImage, h, 'replicate');
    case 'remove brightroundspeckles'
        %%% It does a grayscle open morphological operation. Effectively, 
        %%% it removes speckles of SizeOfSmoothingFilter brighter than its
        %%% surroundings. If comebined with the 'Subtract' module, it
        %%% behaves like a tophat filter        
        SPECKLE_RADIUS = round(SizeOfSmoothingFilter/2);
        disk_radus = round(SPECKLE_RADIUS);
        SE = strel('disk', disk_radus);
        SmoothedImage = imopen(OrigImage, SE);
    otherwise
        if ~strcmp(SmoothingMethod,'N');
            error('The smoothing method you specified is not valid. This error should not have occurred. Check the code in the module or tool you are using or let the CellProfiler team know.');
        end       
end

%%% Resize back to original if resized earlier due to the large filter size
if (SizeOfSmoothingFilter >= LARGESIZE_OF_SMOOTHINGFILTER)
    SmoothedImage = imresize(SmoothedImage, 1/ResizingFactor);  
    RealFilterLength = RealFilterLength * ResizingFactor;
end
