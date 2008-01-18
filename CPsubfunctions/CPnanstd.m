function m = CPnanstd(x)
%CPNANSTD Standard deviation, ignoring NaNs.  
%   M = CPNANSTD(X) returns the sample standard deviation of X, treating
%   NaNs as missing values.  For vector input, M is the standard
%   deviation of the non-NaN elements in X.  For matrix input, M is a
%   row vector containing the standard deviation of non-NaN elements in
%   each column.  For N-D arrays, CPNANSTD operates along the first
%   non-singleton dimension.
%
%   This function will need rewriting if it needs to take stds
%   along any dimension other than the first.  Also, it only accepts
%   vectors and 2D matrices at this time.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% $Revision$


assert(length(size(x)) <= 2, 'CPnanstd can only operate on vectors and 2D matrices.');
    
if ~ any(isnan(x(:))),
    m = std(x);
else
    % If it's a row vector, just return the std of that vector
    if size(x, 1) == 1,
        m = std(x(~ isnan(x)));
    else
        % 2D matrix 

        % preallocate
        m = zeros(1, size(x, 2));
        
        % work by columns
        for i = 1:size(x, 2),
            col = x(:, i);
            m(i) = std(col(~ isnan(col)));
        end
    end
end
