function handles = CPsaveObjectLocations(handles, objectName, labels)
% CPSAVEOBJECTLOCATIONS Save the location of each segmented object.
%   The function returns a new version of the handles structure, in
%   which the location of each segmented object has been saved.
%
%   Example:
%      handles = CPsaveObjectLocations(handles, 'Cells', cellLabelMatrix)
%      creates handles.Measurements.Cells{1}.Location_Center_X and
%      handles.Measurements.Cells{1}.Location_Center_Y.
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
tmp = regionprops(labels, 'Centroid');
centroids = cat(1,tmp.Centroid);
if isempty(centroids)
  centroids = zeros(0,2);
end
handles = CPaddmeasurements(handles, objectName, 'Location_Center_X', ...
                            centroids(:,1));
handles = CPaddmeasurements(handles, objectName, 'Location_Center_Y', ...
                            centroids(:,2));
