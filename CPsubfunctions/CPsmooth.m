function SmoothedImage = CPsmooth(OrigImage,SmoothingMethod,SizeOfSmoothingFilter,WidthFlg)

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

%%% For now, nothing fancy is done to calculate the size automatically. We
%%% just choose a fraction of the size of the image, as long as its always
%%% between 1 and 30.
if strcmpi(SizeOfSmoothingFilter,'A')
    SizeOfSmoothingFilter = min(30,max(1,ceil(mean(size(OrigImage))/40))); % Get size of filter
    WidthFlg = 0;
end

switch SmoothingMethod
    case 'P'
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
    case 'S'
        if SizeOfSmoothingFilter == 0
            %%% No blurring is done.
            return;
        elseif WidthFlg
            %%% We should implement what IdentifyPrimAutomatic does here,
            %%% for now lets use userdefined filter sizes.
            SizeOfSmoothingFilter = 2*SizeOfSmoothingFilter;
        end
        %%% temporarily hard-coded;
%         CPfigure,imagesc(OrigImage);title('Original Image');
        FiltLength = SizeOfSmoothingFilter;
%         handles.Preferences.IntensityColorMap = 'gray';
%         handles.Preferences.FontSize = 8;
        PaddedImage = padarray(OrigImage,[FiltLength FiltLength],'replicate');
        SmoothedImage = conv2(PaddedImage.^2,ones(FiltLength,FiltLength),'same');
        SmoothedImage = SmoothedImage(FiltLength+1:end-FiltLength,FiltLength+1:end-FiltLength);
%         UnpaddedSQImage = conv2(OrigImage.^2,ones(FiltLength,FiltLength),'same');
%         PaddedSumImage = conv2(PaddedImage,ones(FiltLength,FiltLength),'same');
%         PaddedSumImage = PaddedSumImage.^2;
%         PaddedSumImage = PaddedSumImage(FiltLength+1:end-FiltLength,FiltLength+1:end-FiltLength);
%         UnpaddedSumImage = conv2(OrigImage,ones(FiltLength,FiltLength),'same');
%         UnpaddedSumImage = UnpaddedSumImage.^2;
%         figureNo = CPfigure('Name','for testing only');
%         CPresizefigure(OrigImage,'TwoByTwo',figureNo);
%         subplot(2,2,1);
%         CPimagesc(PaddedSQImage,handles);title('Padded - Sum of squared pixels');
%         subplot(2,2,2);
%         CPimagesc(PaddedSumImage,handles);title('Padded - Square of summed pixels');
%         subplot(2,2,3);
%         CPimagesc(UnpaddedSQImage,handles);title('Unpadded - Sum of squared pixels');
%         subplot(2,2,4);
%         CPimagesc(UnpaddedSumImage,handles);title('Unpadded - Square of summed pixels');

%         PaddedRatio1 = PaddedSQImage./PaddedSumImage;
%         PaddedRatio2 = PaddedSumImage./PaddedSQImage;
%         UnpaddedRatio1 = UnpaddedSQImage./UnpaddedSumImage;
%         UnpaddedRatio2 = UnpaddedSumImage./UnpaddedSQImage;
%         figureNo2 = CPfigure('Name','for testing only too');
%         CPresizefigure(OrigImage,'TwoByTwo',figureNo2);
%         subplot(2,2,1);
%         CPimagesc(PaddedRatio1,handles);title('Padded Ratio - SQ over Sum');
%         subplot(2,2,2);
%         CPimagesc(PaddedRatio2,handles);title('Padded Ratio - Sum over SQ');
%         subplot(2,2,3);
%         CPimagesc(UnpaddedRatio1,handles);title('Unpadded Ratop - SQ over Sum');
%         subplot(2,2,4);
%         CPimagesc(UnpaddedRatio2,handles);title('Unpadded Ratio - Sum over SQ');

%         CPthresh_tool(SmoothedImage,'gray');
%         disp('ok1');
    case 'Q'
        if SizeOfSmoothingFilter == 0
            %%% No blurring is done.
            return;
        elseif WidthFlg
            SizeOfSmoothingFilter = 2*SizeOfSmoothingFilter;
        end
        FiltLength = SizeOfSmoothingFilter;
        PaddedImage = padarray(OrigImage,[FiltLength FiltLength],'replicate');
        %%% Could be good to use a disk structuring element of
        %%% floor(FiltLength/2) size instead of a square window.
        SumImage = conv2(PaddedImage,ones(FiltLength,FiltLength),'same');
        SmoothedImage = SumImage.^2;
        SmoothedImage = SmoothedImage(FiltLength+1:end-FiltLength,FiltLength+1:end-FiltLength);
%         CPthresh_tool(SmoothedImage,'gray');
    case 'M'
        if SizeOfSmoothingFilter == 0;
            %%% No blurring is done.
            return;
        end
        if WidthFlg
            sigma = SizeOfSmoothingFilter/2.35;        % Convert between Full Width at Half Maximum (FWHM) to sigma
            FiltLength = min(30,max(1,ceil(2*sigma))); % Determine filter size, min 3 pixels, max 61
        else
            sigma = SizeOfSmoothingFilter/2;
            FiltLength = min(30,max(1,ceil(SizeOfSmoothingFilter/2)));
        end
        [x,y] = meshgrid(-FiltLength:FiltLength,-FiltLength:FiltLength);      % Filter kernel grid
        f = exp(-(x.^2+y.^2)/(2*sigma^2));f = f/sum(f(:));                    % Gaussian filter kernel
        %%% The original image is blurred. Prior to this blurring, the
        %%% image is padded with values at the edges so that the values
        %%% around the edge of the image are not artificially low.  After
        %%% blurring, these extra padded rows and columns are removed.
        SmoothedImage = conv2(padarray(OrigImage, [FiltLength,FiltLength], 'replicate'),f,'same');
        SmoothedImage = SmoothedImage(FiltLength+1:end-FiltLength,FiltLength+1:end-FiltLength);
%         CPthresh_tool(SmoothedImage,'gray');
    otherwise
        if ~strcmp(SmoothingMethod,'N');
            error('The smoothing method you specified is not valid. This error should not have occurred. Check the code in the module or tool you are using or let the CellProfiler team know.');
        end
end