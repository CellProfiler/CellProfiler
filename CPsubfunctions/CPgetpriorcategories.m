function categories=CPgetpriorcategories(handles, CurrentModuleNum)

% Get the measurement categories created by modules prior to
% CurrentModuleNum, returning them as cells.
%
% Note: The initial release of this returns all possible categories.
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
% $Revision: 5025 $
categories = sort({ ...
    'AreaOccupied','AreaShape','Children','Parent','Correlation',...
    'Intensity','Neighbors','Ratio','Texture','RadialDistribution',...
    'Granularity'});
end
