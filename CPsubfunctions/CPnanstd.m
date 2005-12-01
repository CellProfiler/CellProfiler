function y = CPnanstd(varargin)
%NANSTD Standard deviation, ignoring NaNs.
%   Y = NANSTD(X) returns the sample standard deviation of the values in X,
%   treating NaNs as mssing values.  For a vector input, Y is the standard
%   deviation of the non-NaN elements of X.  For a matrix input, Y is a row
%   vector containing the standard deviation of the non-NaN elements in
%   each column of X. For N-D arrays, NANSTD operates along the first
%   non-singleton dimension of X.
%
%   NANSTD normalizes Y by (N-1), where N is the sample size.  This is the
%   square root of an unbiased estimator of the variance of the population
%   from which X is drawn, as long as X consists of independent, identically
%   distributed samples and data are missing at random.
%
%   Y = NANSTD(X,1) normalizes by N and produces the square root of the
%   second moment of the sample about its mean.  NANSTD(X,0) is the same as
%   NANSTD(X).
%
%   Y = NANSTD(X,FLAG,DIM) takes the standard deviation along dimension
%   DIM of X.
%
%   See also STD, NANVAR, NANMEAN, NANMEDIAN, NANMIN, NANMAX, NANSUM.

%   Copyright 1993-2004 The MathWorks, Inc.
%   $Revision: 2.10.2.5 $  $Date: 2004/07/28 04:38:43 $

% Call nanvar(x,flag,dim) with as many inputs as needed
y = sqrt(nanvar(varargin{:}));

function y = CPnanvar(x,w,dim)
%NANVAR Variance, ignoring NaNs.
%   Y = NANVAR(X) returns the sample variance of the values in X, treating
%   NaNs as missing values.  For a vector input, Y is the variance of the
%   non-NaN elements of X.  For a matrix input, Y is a row vector
%   containing the variance of the non-NaN elements in each column of X.
%   For N-D arrays, NANVAR operates along the first non-singleton dimension
%   of X.
%
%   NANVAR normalizes Y by N-1 if N>1, where N is the sample size of the 
%   non-NaN elements.  This is an unbiased estimator of the variance of the
%   population from which X is drawn, as long as X consists of independent,
%   identically distributed samples, and data are missing at random.  For
%   N=1, Y is normalized by N. 
%
%   Y = NANVAR(X,1) normalizes by N and produces the second moment of the
%   sample about its mean.  NANVAR(X,0) is the same as NANVAR(X).
%
%   Y = NANVAR(X,W) computes the variance using the weight vector W.  The
%   length of W must equal the length of the dimension over which NANVAR
%   operates, and its non-NaN elements must be nonnegative.  Elements of X
%   corresponding to NaN elements of W are ignored.
%
%   Y = NANVAR(X,W,DIM) takes the variance along dimension DIM of X.
%
%   See also VAR, NANSTD, NANMEAN, NANMEDIAN, NANMIN, NANMAX, NANSUM.

%   Copyright 1984-2005 The MathWorks, Inc.
%   $Revision: 1.1.6.4 $  $Date: 2005/03/23 20:25:41 $

if nargin < 2 || isempty(w), w = 0; end

sz = size(x);
if nargin < 3 || isempty(dim)
    % The output size for [] is a special case when DIM is not given.
    if isequal(x,[]), y = NaN(class(x)); return; end

    % Figure out which dimension sum will work along.
    dim = find(sz ~= 1, 1);
    if isempty(dim), dim = 1; end
elseif dim > length(sz)
    sz(end+1:dim) = 1;
end

% Need to tile the mean of X to center it.
tile = ones(size(sz));
tile(dim) = sz(dim);

if isequal(w,0) || isequal(w,1)
    % Count up non-NaNs.
    n = sum(~isnan(x),dim);

    if w == 0
        % The unbiased estimator: divide by (n-1).  Can't do this when
        % n == 0 or 1, so n==1 => we'll return zeros
        denom = max(n-1, 1);
    else
        % The biased estimator: divide by n.
        denom = n; % n==1 => we'll return zeros
    end
    denom(n==0) = NaN; % Make all NaNs return NaN, without a divideByZero warning

    x0 = x - repmat(CPnanmean(x, dim), tile);
    y = CPnansum(abs(x0).^2, dim) ./ denom; % abs guarantees a real result

% Weighted variance
elseif numel(w) ~= sz(dim)
    error('MATLAB:nanvar:InvalidSizeWgts','The length of W must be compatible with X.');
elseif ~(isvector(w) && all(w(~isnan(w)) >= 0))
    error('MATLAB:nanvar:InvalidWgts','W must be a vector of nonnegative weights, or a scalar 0 or 1.');
else
    % Embed W in the right number of dims.  Then replicate it out along the
    % non-working dims to match X's size.
    wresize = ones(size(sz)); wresize(dim) = sz(dim);
    wtile = sz; wtile(dim) = 1;
    w = repmat(reshape(w, wresize), wtile);

    % Count up non-NaNs.
    n = CPnansum(~isnan(x).*w,dim);

    x0 = x - repmat(CPnansum(w.*x, dim) ./ n, tile);
    y = CPnansum(w .* abs(x0).^2, dim) ./ n; % abs guarantees a real result
end

function y = CPnansum(x,dim)
%NANSUM Sum, ignoring NaNs.
%   Y = NANSUM(X) returns the sum of X, treating NaNs as missing values.
%   For vector input, Y is the sum of the non-NaN elements in X.  For
%   matrix input, Y is a row vector containing the sum of non-NaN elements
%   in each column.  For N-D arrays, NANSUM operates along the first
%   non-singleton dimension.
%
%   Y = NANSUM(X,DIM) takes the sum along dimension DIM of X.
%
%   See also SUM, NANMEAN, NANVAR, NANSTD, NANMIN, NANMAX, NANMEDIAN.

%   Copyright 1993-2004 The MathWorks, Inc.
%   $Revision: 2.10.2.4 $  $Date: 2004/07/28 04:38:44 $

% Find NaNs and set them to zero.  Then sum up non-NaNs.  Cols of all NaNs
% will return zero.
x(isnan(x)) = 0;
if nargin == 1 % let sum figure out which dimension to work along
    y = sum(x);
else           % work along the explicitly given dimension
    y = sum(x,dim);
end