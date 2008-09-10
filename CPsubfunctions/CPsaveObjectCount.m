% CPSAVEOBJECTCOUNT Save the count of segmented objects.
%   The function returns a new version of the handles structure, in which
%   the number of segmented objects has been saved.
%
%   Example:
%      handles = CPsaveObjectCount(handles, 'Cells', labelMatrix)
%      creates handles.Measurements.Cells{i}.Count_Cells.
function handles = CPsaveObjectCount(handles, objectName, labels)
%
% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2008.
%
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% $Revision$
handles = CPaddmeasurements(handles, 'Image', ...
                            CPjoinstrings('Count', objectName), ...
			    max(labels(:)));
