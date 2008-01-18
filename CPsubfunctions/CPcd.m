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
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% $Revision$

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