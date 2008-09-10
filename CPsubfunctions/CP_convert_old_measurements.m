function handles = CP_convert_old_measurements(handles)

% $Revision$

%%% Get all fieldnames in Measurements
ObjectFields = fieldnames(handles.Measurements);

HaveQueried = false;

for i = 1:length(ObjectFields)
    ObjectName = char(ObjectFields(i));
    %%% Ignore Neighbors - too hard to deal with
    if any(strcmpi(ObjectName,{'Neighbors'}))
        continue;
    end

    try
        %%% Get all fieldnames in Measurements.(ObjectName)
        MeasureFields = fieldnames(handles.Measurements.(ObjectName));
    catch %%% Must have been text field and ObjectName is class 'cell'
        CPwarndlg(['CP programming weirdness: Non-struct in Measurements.', ObjectName]);
        continue
    end

    % Are there any old-style measurements?
    if ~ any(cell2mat(regexp(MeasureFields, 'Features$'))),
        continue;
    end

    for j = 1:length(MeasureFields)
        Offset = 0;
        
        if regexp(MeasureFields{j}, 'Features$'),
            Offset = 8;
        end
        if regexp(MeasureFields{j}, 'Text$'),
            Offset = 4;
        end

        % If neither matched, go on
        if Offset == 0
            continue;
        end

        if ~ HaveQueried,
            HaveQueried = true;
            Proceed = CPquestdlg('Measurements from a previous version of CellProfiler detected.  Processing of this data may require using the same version of CellProfiler that created them.  Should an attempt be made to read them anyway?', 'Out-of-date Measurements Detected', 'Yes', 'No', 'Yes');
            if strcmp(Proceed, 'No'),
                return;
            end
        end
        
        % Extract the values
        SubFeatureNames = handles.Measurements.(ObjectName).(MeasureFields{j});
        BaseFeatureName = MeasureFields{j}(1:end-Offset);
        Values = handles.Measurements.(ObjectName).(BaseFeatureName);
        
        % Drop the names and values from the handles
        handles.Measurements.(ObjectName) = rmfield(handles.Measurements.(ObjectName), MeasureFields{j});
        handles.Measurements.(ObjectName) = rmfield(handles.Measurements.(ObjectName), BaseFeatureName);
        
        % Write them back with new names
        for k = 1:length(SubFeatureNames),
            NewName = CPjoinstrings(BaseFeatureName, SubFeatureNames{k});
            NewName(findstr(NewName, ' ')) = '_';
            handles.Measurements.(ObjectName).(NewName) = ...
                cellfun(@(x) x(:, k), Values, 'UniformOutput', false);
        end
    end
end
