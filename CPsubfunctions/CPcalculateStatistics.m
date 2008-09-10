function handles = CPcalculateStatistics(handles,DataName,Logarithmic,FigureName,ModuleName,LicenseStats)

% $Revision$

FigureIncrement = 1;

%%% Get all fieldnames in Measurements
ObjectFields = fieldnames(handles.Measurements);
GroupingStrings = handles.Measurements.Image.(DataName);
%%% Need column vector
GroupingValues = str2num(char(GroupingStrings')); %#ok Ignore MLint

%%% Get the handle to the waitbar and update the text in the waitbar
waitbarhandle = CPwaitbar(0,'');
CPwaitbar(0,waitbarhandle,'CPcalculateStatistics Progress');

%%% Check for old-style measurements and try to recover them.
handles = CP_convert_old_measurements(handles);

for i = 1:length(ObjectFields)
    ObjectName = char(ObjectFields(i));
    %%% Filter out Experiment and Neighbor fields
    if any(strcmpi(ObjectName,{'Experiment', 'Neighbors'}))
        continue;
    end

    try
        %%% Get all fieldnames in Measurements.(ObjectName)
        MeasureFields = fieldnames(handles.Measurements.(ObjectName));
    catch %%% Must have been text field and ObjectName is class 'cell'
        CPwarndlg(['CP programming weirdness: Non-struct in Measurements.', ObjectName]);
        continue
    end

    for j = 1:length(MeasureFields)
        MeasureFeatureName = MeasureFields{j};
        if (length(MeasureFeatureName) > 7) && (strcmpi(MeasureFeatureName(end-7:end),'Features')),
            CPwarndlg(['CP programming weirdness: feature ending in ''Feature'': Measurements.' ObjectName '.' MeasureFeatureName]);
            continue;
        end
        
        if any(findstr(MeasureFeatureName,'Location')) || ...
                any(findstr(MeasureFeatureName, 'FileName')) ||...
                any(findstr(MeasureFeatureName, 'PathName')) ||...
                any(findstr(MeasureFeatureName, 'LoadedText')) ||...
                any(findstr(MeasureFeatureName, 'ModuleError')), 
            continue;
        end

        %%% Get Features
        Measurements = handles.Measurements.(ObjectName).(MeasureFeatureName);

        %%% Compute means (on per-image measurements returns just the measurement)
        Ymatrix = cellfun(@mean, Measurements)';
        Ymatrix(isnan(Ymatrix)) = 0;

        GroupingValueRows = size(GroupingValues,1);
        YmatrixRows = size(Ymatrix,1);
        if GroupingValueRows ~= YmatrixRows
            CPwarndlg(['There was an error in the Calculate Statistics module: number of measurements (' num2str(YmatrixRows) ') for feature ' ObjectName '.' MeasureFeatureName ' was not the same as the number of lines of dosage information (' num2str(GroupingValueRows) ').  CellProfiler will proceed but this module will be skipped.']);
            close(waitbarhandle)
            return;
        else
            [v, z, z_one_tailed, OrderedUniqueDoses, OrderedAverageValues] = CP_VZfactors(GroupingValues,Ymatrix);
            if LicenseStats == 1
                if ~strcmpi(FigureName,'Do not use')
                    PartialFigureName = fullfile(handles.Current.DefaultOutputDirectory,FigureName);
                else PartialFigureName = FigureName;
                end
                try
                    [FigureIncrement, ec50stats] = CPec50(OrderedUniqueDoses',OrderedAverageValues,Logarithmic,PartialFigureName,ModuleName,DataName,FigureIncrement);
                catch
                    ec50stats = zeros(size(OrderedAverageValues,2),4);
                end
                ec = ec50stats(:,3);
                if strcmpi(Logarithmic,'Yes')
                    ec = exp(ec);
                end
            end
        end

        %%% Write measurements.  We don't use CPaddmeasurements,
        %%% because this function writes out measurements for the
        %%% entire experiment at once.
        handles.Measurements.Experiment.(CPtruncatefeaturename(CPjoinstrings('Zfactor', ObjectName, MeasureFeatureName))) = z;
        handles.Measurements.Experiment.(CPtruncatefeaturename(CPjoinstrings('Vfactor', ObjectName, MeasureFeatureName))) = v;
        handles.Measurements.Experiment.(CPtruncatefeaturename(CPjoinstrings('EC50', ObjectName, MeasureFeatureName))) = ec;
        handles.Measurements.Experiment.(CPtruncatefeaturename(CPjoinstrings('OneTailedZfactor', ObjectName, MeasureFeatureName))) = z_one_tailed;
    end
    %%% Update waitbar
    CPwaitbar(i./length(ObjectFields),waitbarhandle);
end
close(waitbarhandle)
