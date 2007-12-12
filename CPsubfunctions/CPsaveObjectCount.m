%%% Saves the ObjectCount, i.e. the number of segmented objects.
function handles = CPsaveObjectCount(handles, objectName, labels)
if ~isfield(handles.Measurements.Image,'ObjectCountFeatures')
  handles.Measurements.Image.ObjectCountFeatures = {};
  handles.Measurements.Image.ObjectCount = {};
end
column = find(~cellfun('isempty',strfind(handles.Measurements.Image.ObjectCountFeatures, objectName)));
if isempty(column)
  handles.Measurements.Image.ObjectCountFeatures(end+1) = { ['ObjectCount ' objectName] };
  column = length(handles.Measurements.Image.ObjectCountFeatures);
end
handles.Measurements.Image.ObjectCount{handles.Current.SetBeingAnalyzed}(1, column) = max(labels(:));
