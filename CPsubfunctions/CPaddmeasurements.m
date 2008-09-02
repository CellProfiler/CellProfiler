function handles = CPaddmeasurements(handles, ObjectName, FeatureName, Data, ImageSetNumber)
% Add measurements of a feature to the handles.Measurements structure.
% Location will be "handles.Measurements.ObjectName.FeatureName".
% ObjectName can be "Image".  
%
% Data can be multiple doubles, or a single string (only if ObjectName is "Image").

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


if nargin < 5,
    ImageSetNumber = handles.Current.SetBeingAnalyzed;
end

% Check that either this is a new measurement being added in the first
% set, or an old measurement being appended to in a later set.
FirstSet = (ImageSetNumber == 1);
OldMeasurement = ...
    isfield(handles.Measurements, ObjectName) && ...
    isfield(handles.Measurements.(ObjectName), FeatureName);
BatchProcessing = isfield(handles.Current, 'BatchInfo');

CPvalidfieldname(FeatureName)

%%% Don't allow overwriting of measurements, *except* in batch processing.
if (FirstSet && OldMeasurement && ~BatchProcessing),
    error(['Image processing was canceled because you are attempting to recreate the same measurements, please remove redundant module (#', handles.Current.CurrentModuleNumber, ').']);
end

if (~FirstSet) && (~OldMeasurement) && (~ strcmp(ObjectName, 'Experiment')),
    error(['This should not happen.  CellProfiler Coding Error.  Attempting to add new measurement ', ObjectName, '.',  FeatureName, ' in set ', int2str(ImageSetNumber) ' that was not added in first set.']);
end

%%% Verify we can add this type of Measurement to this type of object
if ischar(Data) && (~ strcmp(ObjectName, 'Image')),
    error(['This should not happen.  CellProfiler Coding Error.  Attempting to add string measurement to non-image ', ObjectName, '.', FeatureName]);
elseif ~strcmp(ObjectName, 'Image') && ~isvector(Data) && ~isempty(Data)
    error(['This should not happen.  CellProfiler Coding Error.  Attempting to add multidimensional (', int2str(size(Data)), ') measurement ', ObjectName, '.', FeatureName]);
elseif strcmp(ObjectName, 'Image') && isnumeric(Data) && ~isscalar(Data),
    error(['This should not happen.  CellProfiler Coding Error.  Attempting to add non-scalar (', int2str(size(Data)), ') measurement to ', ObjectName, '.', FeatureName]);
end


%%% Checks have passed, add the data.
if strcmp(ObjectName, 'Experiment'),
    handles.Measurements.(ObjectName).(FeatureName) = Data;
else
    handles.Measurements.(ObjectName).(FeatureName){ImageSetNumber} = Data;
end
