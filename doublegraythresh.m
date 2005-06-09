function [level1 level2] = doublegraythresh(I)
%DOUBLEGRAYTHRESH Compute two global image threshold using an extension of Otsu's method.
%   [LEVEL1 LEVEL2] = GRAYTHRESH(I) computes two gloabl image thresholds (LEVEL1 and LEVEL2)
%   that can be used to identify two different groups. LEVEL1 and LEVEL2
%   is a normalized intensity value that lies in the range [0, 1].
%   GRAYTHRESH uses an extension of Otsu's method, which chooses the threshold to minimize
%   the intraclass variance of the thresholded pixels.
%
%   Class Support
%   -------------
%   The input image I can be uint8, uint16, int16, single, or double, and it
%   must be nonsparse.  LEVEL and EM are double scalars. 
%
%   Example
%   -------
%       I = imread('coins.png');
%       [level1 level2] = doublegraythresh(I);
%       BW = im2bw(I,level);
%       figure, imshow(BW)
%
%   See also IM2BW.

%   Copyright 1993-2004 The MathWorks, Inc.
%   $Revision$  $Date$

% Reference:
% N. Otsu, "A Threshold Selection Method from Gray-Level Histograms,"
% IEEE Transactions on Systems, Man, and Cybernetics, vol. 9, no. 1,
% pp. 62-66, 1979.

% One input argument required.
iptchecknargin(1,1,nargin,mfilename);
iptcheckinput(I,{'uint8','uint16','double','single','int16'},{'nonsparse'}, ...
              mfilename,'I',1);

if ~isempty(I)
  % Convert all N-D arrays into a single column.  Convert to uint8 for
  % fastest histogram computation.
  I = im2uint8(I(:));
  num_bins = 256;
  counts = imhist(I,num_bins);
  
  % Variables names are chosen to be similar to the formulas in
  % the Otsu paper.
  p = counts / sum(counts);
  omega = cumsum(p);
  mu = cumsum(p .* (1:num_bins)');
  mu_t = mu(end);
  sigma_b_squared=zeros(num_bins,num_bins);
  previous_state = warning('off', 'MATLAB:divideByZero');
  for k1 = 1:(num_bins-1)
      for k2 = (k1+1):num_bins
          sigma_b_squared(k1,k2) = (mu(k1)-omega(k1)*mu_t)^2/omega(k1)+...
              (mu(k2)-mu(k1)-(omega(k2)-omega(k1))*mu_t)^2/(omega(k2)-omega(k1))+...
              (omega(k2)*mu_t-mu(k2))^2/(1-omega(k2));
      end
  end
  warning(previous_state);

  % Find the location of the maximum value of sigma_b_squared.
  % The maximum may extend over several bins, so average together the
  % locations.  If maxval is NaN, meaning that sigma_b_squared is all NaN,
  % then return 0.
  maxval = max(max(sigma_b_squared));
  isfinite_maxval = isfinite(maxval);
  if isfinite_maxval
    [idx1 idx2] = find(sigma_b_squared == maxval,1,'first');
    % Normalize the threshold to the range [0, 1].
    level1 = (idx1 - 1) / (num_bins - 1);
    level2 = (idx2 - 1) / (num_bins - 1);
  else
    level1 = 0.0;
    level2 = 0.0;
  end
else
  level1 = 0.0;
  level2 = 0.0;
  end