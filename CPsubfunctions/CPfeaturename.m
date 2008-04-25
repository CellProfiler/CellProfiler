function FeatureName = CPfeaturename(handles,ObjectName,MeasurementCategory,FeatureNumber)

MeasureName = [MeasurementCategory 'Features']; %%OLD way, previous to Measurements overhaul
% MeasureName = MeasurementCategory; %%NEW way, after Measurements overhaul
FeatureName = handles.Measurements.(ObjectName).(MeasureName){FeatureNumber};