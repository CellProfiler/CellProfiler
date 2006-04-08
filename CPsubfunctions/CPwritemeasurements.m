function CPwritemeasurements(handles,ExportInfo,RawPathname)
%%% This function exports full and summary measurement reports

%%% Get the handle to the waitbar and update the text in the waitbar
waitbarhandle = CPwaitbar(0,'');
CPwaitbar(0,waitbarhandle,'Export Status');

%%% Step 1: Create a cell array containing matrices with all measurements for each object type
%%% concatenated.
SuperMeasurements = cell(length(ExportInfo.ObjectNames),1);
SuperFeatureNames = cell(length(ExportInfo.ObjectNames),1);
for Object = 1:length(ExportInfo.ObjectNames)
    ObjectName = ExportInfo.ObjectNames{Object};

    %%% Get fields in handles.Measurements
    fields = fieldnames(handles.Measurements.(ObjectName));

    %%% Organize numerical features and text in a format suitable for exportation. This
    %%% piece of code creates a super measurement matrix, where
    %%% all features for all objects are stored. There will be one such
    %%% matrix for each image set, and each column in such a matrix
    %%% corresponds to one feature, e.g. Area or IntegratedIntensity.
    MeasurementNames = {};
    Measurements = {};
    TextNames = {};
    Text = {};

    for k = 1:length(fields)

        % Suffix 'Features' indicates that we have found a cell array with measurements, i.e.,
        % where each cell contains a matrix of size
        % [(Nbr of objects in image) x (Number of features of this feature type)]
        if length(fields{k}) > 8 & strcmp(fields{k}(end-7:end),'Features')
            % Get the associated cell array of measurements
            try
                CellArray = handles.Measurements.(ObjectName).(fields{k}(1:end-8));
            catch
                error(['Error in handles.Measurements structure. The field ',fields{k},' does not have an associated measurement field.']);

            end
            if length(Measurements) == 0
                Measurements = CellArray;
            else
                % Loop over the image sets
                for j = 1:length(CellArray)
                    Measurements{j} = cat(2,Measurements{j},CellArray{j});
                end
            end

            % Construct informative feature names
            tmp = handles.Measurements.(ObjectName).(fields{k});     % Get the feature names
            for j = 1:length(tmp)
                tmp{j} = [tmp{j} ' (', ObjectName,', ',fields{k}(1:end-8),')'];
            end
            MeasurementNames = cat(2,MeasurementNames,tmp);

            % Suffix 'Text' indicates that we have found a cell array with text information, i.e.,
            % where each cell contains a cell array of strings
        elseif length(fields{k}) > 4 & strcmp(fields{k}(end-3:end),'Text')

            % Get the associated cell array of measurements
            try
                CellArray = handles.Measurements.(ObjectName).(fields{k}(1:end-4));
            catch
                error(['Error in handles.Measurements structure. The field ',fields{k},' does not have an associated text field.']);
            end

            %%% If this is the first measurement structure encounterered we have to initialize instead of concatenate
            if length(Text) == 0
                Text = CellArray;
                %%% else concatenate
            else
                % Loop over the image sets
                for j = 1:length(CellArray)
                    Text{j} = cat(2,Text{j},CellArray{j});
                end
            end
            TextNames = cat(2,TextNames,handles.Measurements.(ObjectName).(fields{k}));
        elseif length(fields{k}) > 11 & strcmp(fields{k}(end-10:end),'Description')
            % Get the associated cell array of measurements
            try
                TempCellArray = handles.Measurements.(ObjectName).(fields{k}(1:end-11));
            catch
                error(['Error in handles.Measurements structure. The field ',fields{k},' does not have an associated text field.']);
            end
            if length(TempCellArray) == handles.Current.NumberOfImageSets
                for j = 1:length(TempCellArray)
                    CellArray{j} = TempCellArray(j);
                end
                %%% If this is the first measurement structure encounterered we have to initialize instead of concatenate
                if length(Text) == 0
                    Text = CellArray;
                    %%% else concatenate
                else
                    % Loop over the image sets
                    for j = 1:length(CellArray)
                        Text{j} = cat(2,Text{j},CellArray{j});
                    end
                end
                TextNames = cat(2,TextNames,handles.Measurements.(ObjectName).(fields{k}));
            end
        end
    end % end loop over the fields in the current object type

    %%% Create the super measurement structure
    SuperMeasurements{Object} = Measurements;
    SuperMeasurementNames{Object} = MeasurementNames;
    SuperText{Object} = Text;
    SuperTextNames{Object} = TextNames;
end % end loop over object types, i.e., Cells, Nuclei, Cytoplasm, Image

%%% Step 2: Write the measurements to file
for Object = 1:length(ExportInfo.ObjectNames)

    ObjectName = ExportInfo.ObjectNames{Object};

    %%% Update waitbar
    CPwaitbar((Object-1)/length(ExportInfo.ObjectNames),waitbarhandle,sprintf('Exporting %s',ObjectName));

    % Open a file for exporting the measurements
    % Add dot in extension if it's not there
    if ExportInfo.MeasurementExtension(1) ~= '.';
        ExportInfo.MeasurementExtension = ['.',ExportInfo.MeasurementExtension];
    end
    filename = [ExportInfo.MeasurementFilename,'_',ObjectName,ExportInfo.MeasurementExtension];
    fid = fopen(fullfile(RawPathname,filename),'w');
    if fid == -1
        error(sprintf('Cannot create the output file %s. There might be another program using a file with the same name.',filename));
    end

    % Get the measurements and feature names to export
    Measurements     = SuperMeasurements{Object};
    MeasurementNames = SuperMeasurementNames{Object};
    Text             = SuperText{Object};
    TextNames        = SuperTextNames{Object};

    % The 'Image' object type gets some special treatement.
    % Add the average values for all other measurements
    if strcmp(ObjectName,'Image')
        AllFields = fieldnames(handles.Measurements);
        ObjectNumber = 0;
        AllSuperMeasurements = {};
        AllSuperMeasurementNames = {};
        for i = 1:length(AllFields)
            AllMeasurementNames = {};
            AllMeasurements = {};
            AllObjectName = AllFields{i};
            if ~strcmpi(AllObjectName,'Image') && ~strcmpi(AllObjectName,'Experiment')
                ObjectNumber = ObjectNumber + 1;
                fields = fieldnames(handles.Measurements.(AllObjectName));
                for k = 1:length(fields)
                    if length(fields{k}) > 8 & strcmp(fields{k}(end-7:end),'Features')
                        % Get the associated cell array of measurements
                        try
                            CellArray = handles.Measurements.(AllObjectName).(fields{k}(1:end-8));
                        catch
                            error(['Error in handles.Measurements structure. The field ',fields{k},' does not have an associated measurement field.']);
                        end
                        if length(AllMeasurements) == 0
                            AllMeasurements = CellArray;
                        else
                            % Loop over the image sets
                            for j = 1:length(CellArray)
                                AllMeasurements{j} = cat(2,AllMeasurements{j},CellArray{j});
                            end
                        end
                        % Construct informative feature names
                        tmp = handles.Measurements.(AllObjectName).(fields{k});     % Get the feature names
                        for j = 1:length(tmp)
                            tmp{j} = [tmp{j} ' (', AllObjectName,', ',fields{k}(1:end-8),')'];
                        end
                        AllMeasurementNames = cat(2,AllMeasurementNames,tmp);
                    end %%% End of if length(fields{k}) > 8 & strcmp(fields{k}(end-7:end),'Features')
                end %%% End of k = 1:length(fields)
                %%% Create the super measurement structure
                AllSuperMeasurements{ObjectNumber} = AllMeasurements;
                AllSuperMeasurementNames{ObjectNumber} = AllMeasurementNames;
            end %%% End of if ~strcmpi(AllObjectName,'Image') && ~strcmpi(AllObjectName,'Experiment')
        end %%% End of for i = 1:length(AllFields)

        Measurements     = SuperMeasurements{Object};
        MeasurementNames = SuperMeasurementNames{Object};

        for k = 1:ObjectNumber
            MeasurementNames = cat(2,MeasurementNames,AllSuperMeasurementNames{k});
            tmpMeasurements = AllSuperMeasurements{k};
            if ExportInfo.IgnoreNaN == 1
                for imageset = 1:length(Measurements)
                    if strcmp(ExportInfo.DataParameter,'std')
                        Measurements{imageset} = cat(2,Measurements{imageset},CPnanstd(tmpMeasurements{imageset},1));
                    else
                        if strcmp(ExportInfo.DataParameter,'mean')
                            Measurements{imageset} = cat(2,Measurements{imageset},CPnanmean(tmpMeasurements{imageset},1));                      
                        else
                            if strcmp(ExportInfo.DataParameter,'median')
                                Measurements{imageset} = cat(2,Measurements{imageset},CPnanmedian(tmpMeasurements{imageset},1));
                            end;
                        end;
                    end;                            
                end
            else
                for imageset = 1:length(Measurements)
                    if strcmp(ExportInfo.DataParameter,'std')
                        Measurements{imageset} = cat(2,Measurements{imageset},std(tmpMeasurements{imageset},1));
                    else
                        if strcmp(ExportInfo.DataParameter,'mean')
                            Measurements{imageset} = cat(2,Measurements{imageset},mean(tmpMeasurements{imageset},1));
                        else
                            if strcmp(ExportInfo.DataParameter,'median')
                                Measurements{imageset} = cat(2,Measurements{imageset},median(tmpMeasurements{imageset},1));
                            end;
                        end;
                    end;
                end
            end
        end
    else %%% If not the 'Image' field, add a Object Nbr feature instead
        MeasurementNames = cat(2,{'Object Nbr'},MeasurementNames);
    end


    %%% Write tab-separated file that can be imported into Excel
    % Header part
    fprintf(fid,'%s\n\n', ObjectName);

    % Write data in columns or rows?
    if strcmp(ExportInfo.SwapRowsColumnInfo,'No')
        % Write feature names in one row
        % Interleave feature names with commas and write to file
        strMeasurement = cell(2*length(MeasurementNames),1);
        strMeasurement(1:2:end) = {'\t'};
        strMeasurement(2:2:end) = MeasurementNames;
        strText = cell(2*length(TextNames),1);
        strText(1:2:end) = {'\t'};
        strText(2:2:end) = TextNames;
        fprintf(fid,sprintf('%s%s\n',char(cat(2,strText{:})),char(cat(2,strMeasurement{:}))));

        % Loop over the images sets
        for imageset = 1:max(length(Measurements),length(Text))

            % Write info about the image set (some unnecessary code here)
            fprintf(fid,'Set #%d, %s',imageset,handles.Measurements.Image.FileNames{imageset}{1});

            %%% Write measurements and text row by row
            %%% First, determine number of rows to write. Have to do this to protect
            %%% for the cases of no Measurements or no Text.
            if ~isempty(Measurements)
                NbrOfRows = size(Measurements{imageset},1);
                if NbrOfRows == 0
                    fprintf(fid,sprintf('\n'));
                end
            elseif ~isempty(Text)
                NbrOfRows = size(Text{imageset},1);
            else
                NbrOfRows = 0;
            end

            for row = 1:NbrOfRows    % Loop over the rows
                % If not the 'Image' field, write an object number
                if ~strcmp(ObjectName,'Image')
                    fprintf(fid,'\t%d',row);
                end
                % Write text
                strText = {};
                if ~isempty(TextNames)
                    strText = cell(2*length(TextNames),1);
                    strText(1:2:end) = {'\t'};                 % 'Text' is a cell array where each cell contains a cell array
                    tmp = Text{imageset}(row,:);               % Get the right row in the right image set
                    index = strfind(tmp,'\');                  % To use sprintf(), we need to duplicate any '\' characters
                    for k = 1:length(index)
                        for l = 1:length(index{k})
                            tmp{k} = [tmp{k}(1:index{k}(l)+l-1),'\',tmp{k}(index{k}(l)+l:end)];   % Duplicate '\':s
                        end
                    end
                    strText(2:2:end) = tmp;                    % Interleave with tabs
                end
                % Write measurements
                strMeasurement = {};
                if ~isempty(MeasurementNames)
                    tmp = cellstr(num2str(Measurements{imageset}(row,:)','%g'));  % Create cell array with measurements
                    strMeasurement = cell(2*length(tmp),1);
                    strMeasurement(1:2:end) = {'\t'};                             % Interleave with tabs
                    strMeasurement(2:2:end) = tmp;
                end
                fprintf(fid,sprintf('%s%s\n',char(cat(2,strText{:})),char(cat(2,strMeasurement{:}))));            % Write to file
            end
        end
        %%% Write each measurement as a row, with each object as a column
    else
        %%% Write first row where the image set starting points are indicated
        fprintf(fid,'\t%d',[]);
        for imageset = 1:length(Measurements)
            str = cell(size(Measurements{imageset},1)+1,1);
            str(1) = {sprintf('Set #%d, %s',imageset,handles.Measurements.Image.FileNames{imageset}{1})};
            str(2:end) = {'\t'};
            fprintf(fid,sprintf('%s',cat(2,str{:})));
        end
        fprintf(fid,'\n');

        %%% If the current object type isn't 'Image'
        %%% add the 'Object count' to the Measurement matrix
        if ~strcmp(ObjectName,'Image')
            for imageset = 1:length(Measurements)
                Measurements{imageset} = cat(2,[1:size(Measurements{imageset},1)]',Measurements{imageset});
            end
        end

        %%% Start by writing text
        %%% Loop over rows, writing one image set's text features at the time
        for row = 1:length(TextNames)
            fprintf(fid,'%s',TextNames{row});
            for imageset = 1:length(Text)
                strText = cell(2*size(Text{imageset},1),1);
                strText(1:2:end) = {'\t'};                 % 'Text' is a cell array where each cell contains a cell array
                tmp = Text{imageset}(:,row)';              % Get the right row in the right image set
                index = strfind(tmp,'\');                  % To use sprintf(), we need to duplicate any '\' characters
                for k = 1:length(index)
                    for l = 1:length(index{k})
                        tmp{k} = [tmp{k}(1:index{k}(l)+l-1),'\',tmp{k}(index{k}(l)+l:end)];   % Duplicate '\':s
                    end
                end
                strText(2:2:end) = tmp;                    % Interleave with tabs
                fprintf(fid,sprintf('%s',char(cat(2,strText{:}))));
            end
            fprintf(fid,'\n');
        end

        %%% Next, write numerical measurements
        %%% Loop over rows, writing one image set's measurements at the time
        for row = 1:length(MeasurementNames)
            fprintf(fid,'%s',MeasurementNames{row});
            for imageset = 1:length(Measurements)
                if size(Measurements{imageset},2) >= row
                    tmp = cellstr(num2str(Measurements{imageset}(:,row),'%g'));  % Create cell array with measurements
                else
                    tmp = {' '};
                end
                strMeasurement = cell(2*size(Measurements{imageset},1),1);
                strMeasurement(1:2:end) = {'\t'};
                strMeasurement(2:2:end) = tmp;                    % Interleave with tabs
                fprintf(fid,sprintf('%s',char(cat(2,strMeasurement{:}))));
            end
            fprintf(fid,'\n');
        end
    end % Ends 'if' row/column flip
    fclose(fid);
end % Ends 'for'-loop over object types

close(waitbarhandle)
