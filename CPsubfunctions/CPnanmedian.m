function m = CPnanmedian(x)
%CPNANMEDIAN Median value, ignoring NaNs.
%   M = CPNANMEDIAN(X) returns the sample median of X, treating NaNs as
%   missing values.  For vector input, M is the median value of the non-NaN
%   elements in X.  For matrix input, M is a row vector containing the
%   median value of non-NaN elements in each column.
%
%   This function will need rewriting if it needs to take medians
%   along any dimension other than the first.  Also, it only accepts
%   vectors and 2D matrices at this time.

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


assert(length(size(x)) <= 2, 'CPnanmedian can only operate on vectors and 2D matrices.');
    
if ~ any(isnan(x(:))),
    m = median(x);
else
    % If it's a row vector, just return the median of that vector
    if size(x, 1) == 1,
        m = median(x(~ isnan(x)));
    else
        % 2D matrix 

        % preallocate
        m = zeros(1, size(x, 2));
        
        % work by columns
        for i = 1:size(x, 2),
            col = x(:, i);
            m(i) = median(col(~ isnan(col)));
        end
    end
end
