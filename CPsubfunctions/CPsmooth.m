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
%%% we should explicitly confirm that by checking the dimensionality here. -Anne
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
%%% which require it. I am VERY reluctant to add a new variable to one of
%%% our standard modules (IdentifyPrimAutomatic) unless absolutely
%%% necessary, because we want that module to be as simple to use as
%%% possible.
%%% Whatever we do to straighten this out, be sure that the Example PIPEs
%%% still give reasonably nice results; preferably identical results!
%%% See Rodrigo's response below...

% Quoting Anne Carpenter <carpenter@wi.mit.edu>:
% 
% Hi Rodrigo,
% I am looking through the smoothing code and trying to figure it  out...
% In the old code for median filtering, the  SizeOfSmoothingFilter is
% divided by 2.35. In the new code, If  widthFlag exists, we use 3.5 and if
% not, 4. First of all, why not use  2.35 as before, and secondly, why the
% difference between widthflag  and not? Also, why the new code where we
% multiply FiltLength by two  at the end? Lastly, why is FiltLength
% exported by the function anyway?
% 
% Thanks for helping me figure this out!
% Anne
% 
% 
% 
% ******** OLD CODE **********
%         sigma = SizeOfSmoothingFilter/2.35;   % Convert between Full  Width at Half Maximum (FWHM) to sigma
%              FiltLength = min(30,max(1,ceil (2*sigma)));                            % Determine filter size, min  3 pixels, max 61
%             [x,y] = meshgrid(-FiltLength:FiltLength,- FiltLength:FiltLength);      % Filter kernel grid
%             f = exp(-(x.^2+y.^2)/(2*sigma^2));f = f/sum(f (:));                    % Gaussian filter kernel
%             %%% The original image is blurred. Prior to this  blurring, the
%             %%% image is padded with values at the edges so that the  values
%             %%% around the edge of the image are not artificially  low.  After
%             %%% blurring, these extra padded rows and columns are  removed.
%             SmoothedImage = conv2(padarray(OrigImage,  [FiltLength,FiltLength], 'replicate'),f,'same');
%             SmoothedImage = SmoothedImage(FiltLength+1:end- FiltLength,FiltLength+1:end-FiltLength);
%         end
% ******** OLD CODE **********
% 
% 
% ******** NEW CODE **********
%         if WidthFlg
%             %%% Empirically done (from IdentifyPrimAutomatic)
%             sigma = SizeOfSmoothingFilter/3.5;         % Convert  between Full Width at Half Maximum (FWHM) to sigma
%             FiltLength = min(30,max(1,ceil(2*sigma))); % Determine  filter size, min 3 pixels, max 61
%         else
%             sigma = SizeOfSmoothingFilter/4;           % Select  sigma to be roughly the same as above (relatively)
%             FiltLength = min(30,max(1,ceil(2*sigma)));
%         end
%         [x,y] = meshgrid(-FiltLength:FiltLength,- FiltLength:FiltLength);      % Filter kernel grid
%         f = exp(-(x.^2+y.^2)/(2*sigma^2));f = f/sum(f (:));                    % Gaussian filter kernel
%         %%% The original image is blurred. Prior to this blurring, the
%         %%% image is padded with values at the edges so that the values
%         %%% around the edge of the image are not artificially low.   After
%         %%% blurring, these extra padded rows and columns are removed.
%         SmoothedImage = conv2(padarray(OrigImage,  [FiltLength,FiltLength], 'replicate'),f,'same');
%         SmoothedImage = SmoothedImage(FiltLength+1:end- FiltLength,FiltLength+1:end-FiltLength);
%         FiltLength = 2*FiltLength;
% ******** NEW CODE **********
% 
% 
% 	From: 	  reipince@MIT.EDU
% 	Subject: 	Re: Smoothing methods
% 	Date: 	August 31, 2006 2:55:25 PM EDT
% 	To: 	  carpenter@wi.MIT.EDU
% 
% Hi Anne,
% 
% This is why I made those changes: First of all, I added the WidthFlg input
% because sometimes the user wants to pick a specific filter size, but sometimes
% they want it to be picked automatically. When it was picked automatically, I
% saw that IdentifyPrimAutomatic made use of the object width to pick a filter.
% So the WidthFlg is just there to tell whether the user picked one or wants one
% picked automatically.
% 
% Now, it seems that the code for filtering used in IdentifyPrimAutomatic was
% developed before CPsmooth was created. And when this happened, some
% calculations still remained in IdentifyPrimAutomatic. I don't quite remember
% the exact code in IdentifyPrimAutomatic, but I think it set sigma to be
% SizeOfSmoothingFilter/3.5, and then multiplied by 2.35, and then passed it in
% to CPsmooth, where it got divided by 2.35... It was very weird. What I did was
% just change it so that all calculations would be done in CPsmooth, and such
% that it would turn out to be the same as before. I may be mistaken, but I think
% that this way the sizes are actually the same as they were before and nothing
% should change. I also checked if any other function used CPsmooth in the same
% way IdentifyPrimAutomatic did, and I think everything is compatible/consistent.
% 
% And the reason for using a 4 if WidthFlg is 0 is the following. If WidthFlg is
% 0, the user specified a filter size, and we want to choose sigma. Before, when
% WidthFlg was 1 and after choosing FiltLength, we chose FiltLength to be 2*sigma
% (if that number was between 1 and 30). But, the FiltLength we actually use is
% FiltLength*2 (or FiltLength*2+1 to be exact), because we create the filter
% using meshgrid from -FiltLength to +FiltLength (Hence the comment "min 3
% pixels, max 61" when the possible values of FiltLength were between 1 and 30).
% So, if the user is specifying the filter size, and FiltLength is actually half
% the filter size (roughly), and sigma is half the FiltLength, then sigma should
% be 1 fourth of the user-specified filter size SizeOfSmoothingFilter.
% 
% The real filter size is, from what I said above, 2*FiltLength + 1, which is why
% I added the line "FiltLength = 2*FiltLength", which I now notice is erroneous.
% It should be "FiltLength = 2*FiltLength+1".
% 
% Finally, I added FiltLength as an output because IdentifyPrimAutomatic (and
% maybe some other functions in the future) displays it. We need to pass it out
% of CPsmoooth when the filter size is chosen automatically, so that
% IdentifyPrimAutomatic can display the correct filter size used. I remember that
% when using the ExampleFlyImages sample pipeline, IdentifyPrimAutomatic said it
% used 6.7 as it filter size, which was wrong because (1) the filter size needs
% to be an integer (the code uses "ceil" in calculating it), and because (2) it
% wasn't the actual filter size used, but roughly half of it.
% 
% I guess the code is confusing because sometimes the variables do not represent
% what they seem to represent (e.g. SizeOfSmoothingFilter is actually the
% object's width when WidthFlg is 1, and FiltLength is actually half the real
% filter size), but I hope this answered all your questions. Let me know if you
% have other doubts.
% 
% Thanks,
% 
% Rodrigo 

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