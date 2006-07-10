function CurrentDir = CPcd(NewDir)

% This function will check to make sure the directory specified exist
% before performing the cd function.
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
% $Revision: 2802 $

if nargin == 0
    if isdir(cd)
        CurrentDir = cd;
    else
        CPwarndlg('This directory no longer exists! This function will default to the Matlab root directory.');
        CurrentDir = matlabroot;
    end
elseif nargin == 1
    if isdir(NewDir)
        CurrentDir = cd(NewDir);
    else
        CPwarndlg('This directory no longer exists! This function will default to the Matlab root directory.');
        CurrentDir = matlabroot;
    end
end