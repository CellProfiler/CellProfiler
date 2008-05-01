function CPwritemeasurements(handles,ExportInfo,RawPathname)
%%% This function exports full and summary measurement reports

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org

%%% Get the handle to the waitbar and update the text in the waitbar
waitbarhandle = CPwaitbar(0,'');

%%% Do we need to compute means/medians/stddevs?
if any(strcmp('Image', ExportInfo.ObjectNames)) && any(strcmp(ExportInfo.DataParameter, {'mean', 'std', 'median'})),
    CompositeValues = {};
    CompositeNames = {};
    CPwaitbar(0,waitbarhandle,['Export Status - computing ', ExportInfo.DataParameter, 's']);
    
    % find the number of measurements we need to compute (for the waitbar).
    FieldCount = 0;
    AllFields = fieldnames(handles.Measurements);
    for i = 1:length(AllFields)
        ObjectName = AllFields{i};
        if any(strcmp(ObjectName, {'Image', 'Experiment', 'Neighbors'})), 
            continue;
        end
        FieldCount = FieldCount + length(fieldnames(handles.Measurements.(ObjectName)));
    end

    % find the function to reduce values with
    if ExportInfo.IgnoreNaN == 1
        switch ExportInfo.DataParameter,
            case 'mean'
                reducer = @CPnanmean;
            case 'std'
                reducer = @CPnanstd;
            case 'median'
                reducer = @CPnanmedian;
        end
    else
        switch ExportInfo.DataParameter,
            case 'mean'
                reducer = @mean;
            case 'std'
                reducer = @std;
            case 'median'
                reducer = @median;
        end
    end
    
    % Reduce per-object to per-image
    FieldsCompleted = 0;
    for i = 1:length(AllFields)
        ObjectName = AllFields{i};
        if any(strcmp(ObjectName, {'Image', 'Experiment', 'Neighbors'})), 
            continue;
        end

        fields = fieldnames(handles.Measurements.(ObjectName));
        for k = 1:length(fields)
            fieldname = fields{k};

            CPwaitbar(FieldsCompleted / FieldCount,waitbarhandle,['Export Status - computing ', ExportInfo.DataParameter, 's']);
            FieldsCompleted = FieldsCompleted + 1;

            % Name this measurement
            CompositeNames{FieldsCompleted} = ['(', ExportInfo.DataParameter, ', ', ObjectName, ', ', fieldname, ')'];

            Values = handles.Measurements.(ObjectName).(fieldname);
            % loop over images
            for v = 1:length(Values),
                CompositeValues{v, FieldsCompleted} = reducer(Values{v});
            end
        end
    end
end

CPwaitbar(0,waitbarhandle,'Export Status - writing data');

for Object = 1:length(ExportInfo.ObjectNames)
    ObjectName = ExportInfo.ObjectNames{Object};

    %%% This should warn
    if strcmp(ObjectName,'Neighbors')
        continue
    end

    if ~isfield(handles.Measurements,ObjectName)
        CPwarndlg(['The ',ObjectName,' results cannot be exported because those measurements were not made.'],'Warning')
        continue
    end

    %%% Update waitbar
    CPwaitbar((Object-1)/length(ExportInfo.ObjectNames),waitbarhandle,sprintf('Exporting %s',ObjectName));

    %%% Find number of objects for each image
    if strcmp(ObjectName, 'Image'),
        fields = fieldnames(handles.Measurements.Image);
        NumObjects = ones(length(handles.Measurements.Image.(fields{1})), 1);
    else
        fields = fieldnames(handles.Measurements.(ObjectName));
        for k = 1:length(fields)
            fieldname = fields{k};
            FieldValues = handles.Measurements.(ObjectName).(fieldname);
            for l = 1:length(FieldValues),
                Lengths(l) = length(FieldValues{l});
            end
            if k == 1,
                NumObjects = Lengths;
            else
                NumObjects = max(Lengths, NumObjects);
            end
        end
    end

    %%% Find offset for data of each image
    ImageOffsets = (cumsum(NumObjects) - NumObjects + 1);

    %%% Gather Data
    Values = {};
    ValueNames = {};
    fields = fieldnames(handles.Measurements.(ObjectName));
    
    %%% Bring all fields to full size
    for k = 1:length(fields)
        fieldname = fields{k};
        FieldValues = handles.Measurements.(ObjectName).(fieldname);

        %%% Give a meaningful name
        ValueNames{k} = fieldname;

        %%% Some measurements might not exist for all objects, so we need to bring them to full size
        for l = 1:length(FieldValues),
            if ischar(FieldValues{l}),
                Values{ImageOffsets(l), k} = FieldValues{l};
            else
                numvals = length(FieldValues{l});
                destination = ImageOffsets(l):(ImageOffsets(l)+numvals-1);
                for d = 1:numvals,
                    Values{destination(d), k} = FieldValues{l}(d);
                end
            end
        end
    end

    %%% If Image data, and computing composite data, join the two for export
    if strcmp(ObjectName, 'Image') && any(strcmp('Image', ExportInfo.ObjectNames)) && any(strcmp(ExportInfo.DataParameter, {'mean', 'std', 'median'})),
        Values = [Values CompositeValues];
        ValueNames = [ValueNames CompositeNames];
    end


    %%% Add in image # indicators
    Prefix = {};
    AllImageFields = fieldnames(handles.Measurements.Image);
    ImageFilenameFields = AllImageFields(strmatch('FileName', AllImageFields));
    FirstFilename = ImageFilenameFields{1};
    imageidx = 1;
    for k = ImageOffsets(:)',
        Prefix{k, 1} = sprintf('Set #%d, %s',imageidx,handles.Measurements.Image.(FirstFilename){imageidx});
        imageidx = imageidx + 1;
    end
    if size(Prefix, 1) < size(Values, 1),
        Prefix{size(Values, 1), 1} = '';
    end
    Values = [Prefix Values];

    % Open a file for exporting the measurements
    % Add dot in extension if it's not there
    if ExportInfo.MeasurementExtension(1) ~= '.';
        ExportInfo.MeasurementExtension = ['.',ExportInfo.MeasurementExtension];
    end
    filename = [ExportInfo.MeasurementFilename,'_',ObjectName,ExportInfo.MeasurementExtension];
    if exist(fullfile(RawPathname,filename),'file')
        Answer=CPquestdlg(['Do you want to overwrite ',fullfile(RawPathname,filename),'?'],'Overwrite File','Yes','No','Yes');
        if strcmp(Answer,'No')
            continue
        end
    end
    fid = fopen(fullfile(RawPathname,filename),'w');
    if fid == -1
        error('Cannot create the output file %s. Check permissions and that the file is not locked by another program.',filename);
    end

    %%% Write tab-separated file that can be imported into Excel
    % Header part
    fprintf(fid,'%s\n\n', ObjectName);

    %%% row or columns major?
    if strcmp(ExportInfo.SwapRowsColumnInfo,'No'),
        %%% Write feature names
        % lead with an extra tab to offset from image set info, which was prefixed into Values
        fprintf(fid, '\t%s', ValueNames{:});
        fprintf(fid, '\n');

        % loop over rows of the data
        for row = 1:size(Values, 1),
            for col = 1:size(Values, 2),
                if col ~= 1,
                    fprintf(fid, '\t');
                end
                val = Values{row, col};
                if isempty(val),
                    fprintf(fid, '');
                elseif ischar(val),
                    fprintf(fid, '%s', val);
                else
                    fprintf(fid, '%d', val);        
                end
            end
            fprintf(fid, '\n');
        end
    else
        %%% Add an empty Feature header (corresponding to the set # and fileinfo)
        ValueNames = {'', ValueNames{:}};

        %%% loop over columns
        for col = 1:size(Values, 2),
            %%% Feature name
            fprintf(fid, '%s', ValueNames{col});

            % loop over rows of the data
            for row = 1:size(Values, 1),
                val = Values{row, col};
                if isempty(val),
                    fprintf(fid, '\t');
                elseif ischar(val),
                    fprintf(fid, '\t%s', val);
                else
                    fprintf(fid, '\t%d', val);        
                end
            end
            fprintf(fid, '\n');
        end
    end
end

close(waitbarhandle);
