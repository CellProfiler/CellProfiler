%%% Saves the location of each segmented object
function handles = CPsaveObjectLocations(handles, objectName, labels)
handles.Measurements.(objectName).LocationFeatures = {'CenterX', 'CenterY'};
tmp = regionprops(labels, 'Centroid');
Centroid = cat(1,tmp.Centroid);
if isempty(Centroid)
  Centroid = [0 0];
end
handles.Measurements.(objectName).Location(handles.Current.SetBeingAnalyzed) = { Centroid };
