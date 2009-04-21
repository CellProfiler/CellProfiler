function CPconvertsql(handles,OutDir,OutfilePrefix,DBname,TablePrefix,FirstSet,LastSet,SQLchoice,StatisticsCalculated,ObjectsToBeExported)

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

FeaturesNotToBeExported = {'Description_', 'ModuleError_', 'TimeElapsed_'};
ObjectFeaturesNotToBeAveraged = {}; %{'Mean_'};

wantMeanCalculated = any(strcmpi(StatisticsCalculated,'mean'));
wantStdDevCalculated = any(strcmpi(StatisticsCalculated,'standard deviation'));
wantMedianCalculated = any(strcmpi(StatisticsCalculated,'median'));

per_image_names = {};
per_object_names = {};

if ~isfield(handles,'Measurements')
    error('There are no measurements to be converted to SQL.')
end

if strcmp(TablePrefix,'Do not use')
    TablePrefix = '';
else
    TablePrefix = [TablePrefix,'_'];
end

basename = [OutfilePrefix,int2str(FirstSet),'_',int2str(LastSet)];

%%% Extract the object types to write (e.g., Image, Nuclei, ...).  The transpose allows looping below.
if any(strcmpi(ObjectsToBeExported,'all objects'))
    ObjectsToBeExported = fieldnames(handles.Measurements);
else
    ObjectsToBeExported = ObjectsToBeExported(isfield(handles.Measurements,ObjectsToBeExported));
end
ObjectsToOmitFromPerImageTable = [];

for ObjectCell = ObjectsToBeExported,
    % why matlab, why?
    ObjectName = ObjectCell{1};

    %%% Some objects are not exported: experiments, subobjects, neighbors
    if any(strcmp(ObjectName, {'Experiment', 'Neighbors'})) || isfield(handles.Measurements.(ObjectName), 'SubObjectFlag'),
        continue;
    end
    
    %%% Find the features for this object
    Features = fieldnames(handles.Measurements.(ObjectName))';
    for FeatureCell = Features,
        FeatureName = FeatureCell{1};
        
        %%% Certain features are not exported
        if any(cell2mat(cellfun(@(k)strmatch(k, FeatureName), ...
                FeaturesNotToBeExported, ...
                'UniformOutput', false)))
            continue
        end

        if strcmp(ObjectName, 'Image')
            per_image_names{end+1} = cleanup(CPtruncatefeaturename(CPjoinstrings('Image', FeatureName)));
        else
            per_object_names{end+1} = cleanup(CPtruncatefeaturename(CPjoinstrings(ObjectName, FeatureName)));
            ObjectsToOmitFromPerImageTable(end+1) = any(strmatch(FeatureName,ObjectFeaturesNotToBeAveraged));
        end
    end %end of loop over feature names
end %end of loop over object names

%%% We need to find the maximum width of the paths and image
%%% filenames.  Unfortunately, this function can be called with an
%%% incomplete measurement set, usually because of being split up into
%%% cluster jobs.  For this reason, we have to go to handles.Pipeline
%%% to find these widths, and rely on the modules not changing their
%%% output values too far from what's in handles.Pipeline.

% set a reasonable minimum
FileNameWidth = 128;
for fld=fieldnames(handles.Pipeline)',
    if strfind(fld{1}, 'FileList'),
        for str=handles.Pipeline.(fld{1}),
            FileNameWidth = max(FileNameWidth, length(str{1}));
        end
    end
end
% Pad length since we don't know the true max length will be across all cycles 
PadLength = 20;
FileNameWidth = FileNameWidth + PadLength;

% set a reasonable minimum
PathNameWidth = 128;
for fld=fieldnames(handles.Pipeline)',
    if strfind(fld{1}, 'Pathname'),
        PathNameWidth = max(PathNameWidth, length(handles.Pipeline.(fld{1})));
    end
end
% Pad length since we don't know the true max length will be across all cycles 
PathNameWidth = PathNameWidth + PadLength;

MetadataNameWidth = PathNameWidth;

%%% Write the SQL table description and data loader.
if (FirstSet == 1)
    if strcmp(SQLchoice,'MySQL')

        fmain = fopen(fullfile(OutDir, [DBname '_SETUP.SQL']), 'W');

        fprintf(fmain, 'CREATE DATABASE IF NOT EXISTS %s;\n', DBname);
        fprintf(fmain, 'USE %s;\n\n', DBname);

        fprintf(fmain, 'CREATE TABLE %sPer_Image (ImageNumber INTEGER PRIMARY KEY',TablePrefix);

        for i = per_image_names,
            if strfind(i{1}, 'FileName')
                fprintf(fmain, ',\n%s VARCHAR(%d)', i{1}, FileNameWidth);
            elseif  strfind(i{1}, 'Path'),
                fprintf(fmain, ',\n%s VARCHAR(%d)', i{1}, PathNameWidth);
            elseif  strfind(i{1}, 'Metadata_'),
                fprintf(fmain, ',\n%s VARCHAR(%d)', i{1}, MetadataNameWidth);
            else
                fprintf(fmain, ',\n%s FLOAT NOT NULL', i{1});
            end
        end

        %add columns for mean, median and stddev for per_object_names
        if wantMeanCalculated
            for j = 1:length(per_object_names),
                if ~ObjectsToOmitFromPerImageTable(j)
                    fprintf(fmain, ',\n%s FLOAT NOT NULL', CPtruncatefeaturename(['Mean_', per_object_names{j}]));
                end
            end
        end
        
        if wantMedianCalculated
            for k = 1:length(per_object_names),
                if ~ObjectsToOmitFromPerImageTable(k)
                    fprintf(fmain, ',\n%s FLOAT NOT NULL', CPtruncatefeaturename(['Median_', per_object_names{k}]));
                end
            end
        end
        
        if wantStdDevCalculated
            for l = 1:length(per_object_names),
                if ~ObjectsToOmitFromPerImageTable(l)
                    fprintf(fmain, ',\n%s FLOAT NOT NULL', CPtruncatefeaturename(['StDev_', per_object_names{l}]));
                end
            end
        end

        fprintf(fmain, ');\n\n');

        fprintf(fmain, 'CREATE TABLE %sPer_Object(ImageNumber INTEGER,ObjectNumber INTEGER',TablePrefix);
        for i = per_object_names
            fprintf(fmain, ',\n%s FLOAT NOT NULL', i{1});
        end

        fprintf(fmain, ',\nPRIMARY KEY (ImageNumber, ObjectNumber));\n\n');

        if strcmp(handles.Settings.ModuleNames{handles.Current.NumberOfModules},'CreateBatchFiles')
            msg = 'Please note that you will have to manually edit the "LOAD DATA" line of the SQL file to include each of the per-image and per-object .CSV files that the cluster job will create.';
            if isfield(handles.Current,'BatchInfo')
                warning(msg);
            else
                CPwarndlg(msg);
            end
            fprintf(fmain, 'LOAD DATA LOCAL INFILE ''%s1_1_image.CSV'' REPLACE INTO TABLE %sPer_Image FIELDS TERMINATED BY '','' OPTIONALLY ENCLOSED BY ''"'' ESCAPED BY '''';\n',OutfilePrefix,TablePrefix);
            fprintf(fmain, 'SHOW WARNINGS;\n');
            fprintf(fmain, 'LOAD DATA LOCAL INFILE ''%s1_1_object.CSV'' REPLACE INTO TABLE %sPer_Object FIELDS TERMINATED BY '','';\n',OutfilePrefix,TablePrefix);
            fprintf(fmain, 'SHOW WARNINGS;\n');
        else
            fprintf(fmain, 'LOAD DATA LOCAL INFILE ''%s_image.CSV'' REPLACE INTO TABLE %sPer_Image FIELDS TERMINATED BY '','' OPTIONALLY ENCLOSED BY ''"'' ESCAPED BY '''';\n',basename,TablePrefix);
            fprintf(fmain, 'SHOW WARNINGS;\n');
            fprintf(fmain, 'LOAD DATA LOCAL INFILE ''%s_object.CSV'' REPLACE INTO TABLE %sPer_Object FIELDS TERMINATED BY '','';\n',basename,TablePrefix);
            fprintf(fmain, 'SHOW WARNINGS;\n');
        end
        fclose(fmain);
    elseif strcmp(SQLchoice,'Oracle')
        %%%%%%%%%%%%%%%%%%
        %%% SETUP FILE %%%
        %%%%%%%%%%%%%%%%%%

        fsetup = fopen(fullfile(OutDir, [DBname,'_SETUP.SQL']), 'W');

        fprintf (fsetup, 'CREATE TABLE %sColumn_Names (SHORTNAME VARCHAR2(8), LONGNAME VARCHAR2(250));\n',TablePrefix);

        fprintf(fsetup, 'CREATE TABLE %sPer_Image (col1 NUMBER',TablePrefix);

        p=1;
        for i = per_image_names,
            p = p+1;
            if strfind(i{1}, 'Filename')
                fprintf(fsetup, ',\n%s VARCHAR2(%d)', ['col',num2str(p)], FileNameWidth);
            elseif  strfind(i{1}, 'Path'),
                fprintf(fsetup, ',\n%s VARCHAR2(%d)', ['col',num2str(p)], PathNameWidth);
            else
                fprintf(fsetup, ',\n%s FLOAT', ['col',num2str(p)]);
            end
        end

        %add columns for mean, median and stddev for per_object_names
        if wantMeanCalculated
            for j = per_object_names,
                if ~any(strmatch(j{1},ObjectFeaturesNotToBeAveraged))
                    p = p+1;
                    fprintf(fsetup, ',\n%s FLOAT', ['col',num2str(p)]);
                end
            end
        end
        
        if wantMedianCalculated
            for k = per_object_names,
                if ~any(strmatch(k{1},ObjectFeaturesNotToBeAveraged))
                    p = p+1;
                    fprintf(fsetup, ',\n%s FLOAT', ['col',num2str(p)]);
                end
            end
        end

        if wantStdDevCalculated
            for l = per_object_names,
                if ~any(strmatch(l{1},ObjectFeaturesNotToBeAveraged))
                    p = p+1;
                    fprintf(fsetup, ',\n%s FLOAT', ['col',num2str(p)]);
                end
            end
        end

        fprintf(fsetup, ');\n');
        p = p+1;
        PrimKeyPosition = p;
        fprintf(fsetup, 'CREATE TABLE %sPer_Object (col1 NUMBER, %s NUMBER',TablePrefix,['col',num2str(p)]);
        for i = per_object_names
            p=p+1;
            fprintf(fsetup, ',\n%s FLOAT', ['col',num2str(p)]);
        end

        fprintf(fsetup, ');\n');
        fclose(fsetup);

        %%%%%%%%%%%%%%%%%%%
        %%% FINISH FILE %%%
        %%%%%%%%%%%%%%%%%%%

        ffinish = fopen(fullfile(OutDir, [DBname,'_FINISH.SQL']), 'W');
        fprintf(ffinish, 'ALTER TABLE %sPer_Image ADD PRIMARY KEY (col1);\n',TablePrefix);
        fprintf(ffinish, 'ALTER TABLE %sPer_Object ADD PRIMARY KEY (col1, %s);',TablePrefix,['col',num2str(PrimKeyPosition)]);
        fclose(ffinish);

        %%%%%%%%%%%%%%%%%%%
        %%% COLUMN FILE %%%
        %%%%%%%%%%%%%%%%%%%

        fcol = fopen(fullfile(OutDir, [DBname,'_columnnames.CSV']), 'W');
        fprintf(fcol,'%s','col1');
        fprintf(fcol,',%s\n','ImageNumber');

        p = 1;
        for k = per_image_names
            p = p+1;
            fprintf(fcol, '%s', ['col', num2str(p)] );
            fprintf(fcol, ',%s\n', k{1} );
        end
        if wantMeanCalculated
            for l = per_object_names
                if ~any(strmatch(l{1},ObjectFeaturesNotToBeAveraged))
                    p = p+1;
                    fprintf(fcol, '%s', ['col', num2str(p)]);
                    fprintf(fcol, ',%s\n', ['Mean_', l{1}]);
                end
            end
        end
        if wantMedianCalculated
            for m = per_object_names
                if ~any(strmatch(m{1},ObjectFeaturesNotToBeAveraged))
                    p = p+1;
                    fprintf(fcol, '%s', ['col', num2str(p)]);
                    fprintf(fcol, ',%s\n', ['Median_', m{1}]);
                end
            end
        end
        if wantStdDevCalculated
                for n = per_object_names
                if ~any(strmatch(n{1},ObjectFeaturesNotToBeAveraged))
                    p = p+1;
                    fprintf(fcol, '%s', ['col', num2str(p)]);
                    fprintf(fcol, ',%s\n', ['Stdev_', n{1}]);
                end
                end
        end

        p = p+1;
        if PrimKeyPosition ~= p
            error('STOP!');
        end

        % Per_Object table's colnames
        fprintf(fcol, '%s', ['col', num2str(p)]);
        fprintf(fcol, ',%s\n','ObjectNumber');

        for n=per_object_names
            p=p+1;
            fprintf(fcol, '%s', ['col', num2str(p)]);
            fprintf(fcol, ',%s\n', n{1} );
        end
        fclose(fcol);

        FinalColumnPosition = p;

        %%%%%%%%%%%%%%%%%%%%%
        %%% COLUMN LOADER %%%
        %%%%%%%%%%%%%%%%%%%%%

        fcolload = fopen(fullfile(OutDir, [DBname,'_LOADCOLUMNS.CTL']), 'W');
        fprintf(fcolload, 'LOAD DATA INFILE ''%s'' INTO TABLE  %sColumn_Names FIELDS TERMINATED BY '','' OPTIONALLY ENCLOSED BY ''"'' ESCAPED BY '''' (shortname, longname)',[DBname, '_columnnames.CSV'],TablePrefix);
        fclose(fcolload);

        %%%%%%%%%%%%%%%%%%%%
        %%% IMAGE LOADER %%%
        %%%%%%%%%%%%%%%%%%%%

        fimageloader = fopen(fullfile(OutDir, [DBname, '_LOADIMAGE.CTL']), 'W');
        fprintf(fimageloader, 'LOAD DATA\n');
        if strcmp(handles.Settings.ModuleNames{handles.Current.NumberOfModules},'CreateBatchFiles')
            msg = 'Please note that you will have to manually edit the "LOAD DATA" line of the SQL file to include each of the per-image .CSV files that the cluster job will create.';
            if isfield(handles.Current,'BatchInfo')
                warning(msg);
            else
                CPwarndlg(msg);
            end
            fprintf(fimageloader, 'INFILE %s1_1_image.CSV\n', OutfilePrefix);
        else
            fprintf(fimageloader, 'INFILE %s\n', [basename, '_image.CSV']);
        end

        fprintf(fimageloader, 'INTO TABLE  %sPer_Image FIELDS TERMINATED BY '','' OPTIONALLY ENCLOSED BY ''"'' ESCAPED BY '''' (col1',TablePrefix);
        for i = 2:(PrimKeyPosition-1)
            fprintf(fimageloader, ',\n%s', ['col',num2str(i)]);
        end
        fprintf(fimageloader, ')');

        fclose(fimageloader);

        %%%%%%%%%%%%%%%%%%%%%
        %%% OBJECT LOADER %%%
        %%%%%%%%%%%%%%%%%%%%%

        fobjectloader = fopen(fullfile(OutDir, [DBname, '_LOADOBJECT.CTL']), 'W');
        fprintf(fobjectloader, 'LOAD DATA\n');
        if strcmp(handles.Settings.ModuleNames{handles.Current.NumberOfModules},'CreateBatchFiles')
            msg = 'Please note that you will have to manually edit the "LOAD DATA" line of the SQL file to include each of the per-object .CSV files that the cluster job will create.';
            if isfield(handles.Current,'BatchInfo')
                warning(msg);
            else
                CPwarndlg(msg);
            end
            fprintf(fobjectloader, 'INFILE %s1_1_object.CSV\n', OutfilePrefix);
        else
            fprintf(fobjectloader, 'INFILE %s\n', [basename, '_object.CSV']);
        end

        fprintf(fobjectloader, 'INTO TABLE  %sPer_Object FIELDS TERMINATED BY '','' (col1',TablePrefix);
        for i = PrimKeyPosition:FinalColumnPosition
            fprintf(fobjectloader, ',\n%s', ['col',num2str(i)]);
        end
        fprintf(fobjectloader, ')');
        fclose(fobjectloader);
    end
end

%start to write data file

[fimage,msg] = fopen(fullfile(OutDir, [basename '_image.CSV']), 'W');
if fimage == -1,
    error(msg);
end
[fobject,msg]= fopen(fullfile(OutDir, [basename '_object.CSV']), 'W');
if fobject == -1,
    error(msg);
end

perobjectvals = [];

for img_idx = FirstSet:LastSet  
    % perobjectvals_aggregate: Holds the values that are going to be aggregated per-image
    perobjectvals_aggregate = []; 
   % perobjectvals_aggregate_isobj: Indexes which values are valid from
   %    an image; non-valid values are excluded from stats later
    perobjectvals_aggregate_isobj = logical([]); 
    fprintf(fimage, '%d', img_idx);
    objectbaserow = size(perobjectvals, 1);
    objectbasecol = 2;
    objectbasemeancol = 2;
    numobj = 0;
    maxnumobj = 0;

    feature_idx = 1;

    for ObjectCell = ObjectsToBeExported,
        % why matlab, why?
        ObjectName = ObjectCell{1};

        %%% Some objects are not exported: experiments, subobjects, neighbors
        if any(strcmp(ObjectName, {'Experiment', 'Neighbors'})) || isfield(handles.Measurements.(ObjectName), 'SubObjectFlag'),
            continue;
        end

        %%% Find the features for this object
        Features = fieldnames(handles.Measurements.(ObjectName))';
        for FeatureCell = Features,
            FeatureName = FeatureCell{1};
            
            %%% Certain features are not exported
            if any(cell2mat(cellfun(@(k)strmatch(k, FeatureName), ...
                        FeaturesNotToBeExported, ...
                        'UniformOutput', false)))
                continue
            end

            %%% Old code checked if data for img_idx existed, but this one always should (entry should be [] if no objects).
            try
                vals = handles.Measurements.(ObjectName).(FeatureName){img_idx};
            catch
                error(['Measurements missing for image #' int2str(img_idx) ' and feature handles.Measurements.' ObjectName '.' FeatureName '.  (Max size is ' int2str(length(handles.Measurements.(ObjectName).(FeatureName))) '.)']);
            end

            %%% write image data, gather object data for later writing
            if strcmp(ObjectName, 'Image'),
                if ischar(vals)
                    fprintf(fimage, ',%s', vals);
                elseif isnumeric(vals)
                    if (~isscalar(vals)),
                        error(['Attempt to write non-scalar numeric value in per_image data, feature handles.Measurements.' ObjectName '.' FeatureName ', value ', num2str(vals)]);
                    end
                    vals(~isfinite(vals)) = 0;
                    fprintf(fimage, ',%g', vals);
                    %%% Test that counts are integers
                    if strcmp(FeatureName(max(findstr(FeatureName, 'Count')):end), 'Count') && (floor(vals) ~= vals),
                        warning(['Attempt to write non-integer "Count" feature in per_image data, feature handles.Measurements.', ObjectName, '.', FeatureName ', value ', num2str(vals)]);
                        CPwarndlg(['Attempt to write non-integer "Count" feature in per_image data, feature handles.Measurements.' ObjectName '.' FeatureName ', value ', num2str(vals)]);
                    end
                else
                    CPwarndlg(['Non-string, non-numeric data, feature handles.Measurements.' ObjectName '.' FeatureName ', type ', class(vals)]);
                    % error(['Non-string, non-numeric data, feature handles.Measurements.' ObjectName '.' FeatureName ', type ', class(vals)]);
                end
            else
                %%% Sanity check
                if ~ strcmp(per_object_names{feature_idx}, cleanup(CPtruncatefeaturename(CPjoinstrings(ObjectName, FeatureName)))),
                    error(['Mismatched feature names #', int2str(feature_idx), ' ', per_object_names{feature_idx}, '!=', cleanup(CPtruncatefeaturename(CPjoinstrings(ObjectName, FeatureName)))])
                end
                feature_idx = feature_idx + 1;

                if ~isa(vals,'numeric')
                    error(['Non-numeric data not currently supported in per-object SQL data feature handles.Measurements.' ObjectName '.' FeatureName ', type ', class(vals)]);
                end

                if ~isvector(vals) && ~isempty(vals)
                    error(['This should not happen.  CellProfiler Coding Error.  Attempting to export multidimensional (', int2str(size(vals)), ') measurement ', ObjectName, '.', FeatureName]);
                end

                numobj = length(vals);
                % put in nx1 orientation
                vals = vals(:);

                %%% There might be different numbers of different types of objects, unfortunately.  These will be filled with zeros, later.
                if maxnumobj < numobj
                    maxnumobj = numobj;
                end

                %%% Add the values into the per-object output and shift
                %%% right
                if numobj > 0,
                    perobjectvals((objectbaserow+1):(objectbaserow+numobj), (objectbasecol+1)) = vals;
                end
                objectbasecol = objectbasecol + 1;
                
                %%% Add the values into the per-image output and shift
                %%% right
                if ~any(strmatch(FeatureName,ObjectFeaturesNotToBeAveraged)),
                    perobjectvals_aggregate(1:numobj, (objectbasemeancol-2+1)) = vals;
                    perobjectvals_aggregate_isobj(1:numobj, (objectbasemeancol-2+1)) = true;
                    objectbasemeancol = objectbasemeancol + 1;
                end
                
                
            end
        end %%% loop over features

        %%% put in image and object numbers
        if numobj > 0
            perobjectvals((objectbaserow+1):end, 1) = img_idx;
            perobjectvals((objectbaserow+1):end, 2) = 1:maxnumobj;
        end

    end %%% loop over object types

    % Print mean, median, stdev for all measurements per image
    if (wantMeanCalculated || wantMedianCalculated || wantStdDevCalculated)
        formatstr = ['%g' repmat(',%g',1,size(perobjectvals_aggregate,2)-1)];
        if size(perobjectvals_aggregate,1)==1
            if wantMeanCalculated
                fprintf(fimage,',');    %% MEAN
                fprintf(fimage,formatstr,perobjectvals_aggregate); % ignore NaN
            end
            if wantMedianCalculated
                fprintf(fimage,',');    %% MEDIAN
                fprintf(fimage,formatstr,perobjectvals_aggregate); % ignore NaN
            end
            if wantStdDevCalculated
                for i = 1:size(perobjectvals_aggregate,2),
                    fprintf(fimage,',0'); %ignore NaN
                end
            end
        elseif size(perobjectvals_aggregate, 1) > 0,  
            % Ignore NaNs in aggregate measurements
            warning('off','MATLAB:divideByZero');
            perobjectvals_aggregate(perobjectvals_aggregate_isobj == 0) = NaN;
            if wantMeanCalculated
                fprintf(fimage,',');
                aggregate_vals = CPnanmean(perobjectvals_aggregate);
                aggregate_vals(isnan(aggregate_vals)) = 0;
                fprintf(fimage,formatstr,aggregate_vals);
            end
            if wantMedianCalculated
                fprintf(fimage,',');
                aggregate_vals = CPnanmedian(perobjectvals_aggregate);
                aggregate_vals(isnan(aggregate_vals)) = 0;
                fprintf(fimage,formatstr,aggregate_vals);
            end
            if wantStdDevCalculated
                fprintf(fimage,',');
                aggregate_vals = CPnanstd(perobjectvals_aggregate);
                aggregate_vals(isnan(aggregate_vals)) = 0;
                fprintf(fimage,formatstr,aggregate_vals);
            end
            warning('on','MATLAB:divideByZero');
        else % Write zeros if there are no measurements
            if wantStdDevCalculated
                fprintf(fimage,',');
                fprintf(fimage,formatstr,zeros(1, size(perobjectvals_aggregate, 2)));
            end
            if wantMedianCalculated
                fprintf(fimage,',');
                fprintf(fimage,formatstr,zeros(1, size(perobjectvals_aggregate, 2)));
            end
            if wantStdDevCalculated
                fprintf(fimage,',');
                fprintf(fimage,formatstr,zeros(1, size(perobjectvals_aggregate, 2)));
            end
        end
    end
    fprintf(fimage,'\n');
end

% The number of per-object value columns should be the number of per-object
% column headers + 2 (since the ImageNumber and ObjectNumber headers
% aren't included). If this is not the case, such as when the last column(s)
% of measurements are empty, pad with zeros
if ~isempty(perobjectvals) && size(perobjectvals,2) < length(per_object_names) + 2,
    perobjectvals(end,length(per_object_names)+2) = 0;
end

formatstr = ['%g' repmat(',%g',1,size(perobjectvals, 2)-1) '\n'];
%%% THIS LINE WRITES ENTIRE OBJECT VALS FILE
%%% if vals{1} is empty skip writting into object file
if ~iscell(vals) || (iscell(vals) && ~isempty(vals{1}))
    perobjectvals(~ isfinite(perobjectvals)) = 0;
    fprintf(fobject, formatstr, perobjectvals');
end

fclose(fimage);
fclose(fobject);

function sc=cleanup(s)
sc = s;
sc(strfind(s,' ')) = '_';
if (length(sc) >= 64)
    warning(['Column name ' sc ' too long in CPconvertsql.']) %#ok Ignore MLint
end