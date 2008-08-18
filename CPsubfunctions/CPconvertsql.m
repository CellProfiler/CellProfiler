function CPconvertsql(handles,OutDir,OutfilePrefix,DBname,TablePrefix,FirstSet,LastSet,SQLchoice)

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

features_not_to_be_exported = {'Description_', 'ModuleError_', 'TimeElapsed_'};

per_image_names = {};
per_object_names = {};

if ~isfield(handles,'Measurements')
    error('There are no measurements to be converted to SQL.')
end

if strcmp(TablePrefix,'/')
    TablePrefix = '';
else
    TablePrefix = [TablePrefix,'_'];
end

basename = [OutfilePrefix,int2str(FirstSet),'_',int2str(LastSet)];

%%% Extract the object types to write (e.g., Image, Nuclei, ...).  The transpose allows looping below.
ObjectNames = fieldnames(handles.Measurements)';

for ObjectCell = ObjectNames,
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
				features_not_to_be_exported, ...
				'UniformOutput', false)))
	    continue
	end

        if strcmp(ObjectName, 'Image')
            per_image_names{end+1} = cleanup(CPjoinstrings('Image', FeatureName));
        else
            per_object_names{end+1} = cleanup(CPjoinstrings(ObjectName, FeatureName));
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

% set a reasonable minimum
PathNameWidth = 128;
for fld=fieldnames(handles.Pipeline)',
    if strfind(fld{1}, 'Pathname'),
        PathNameWidth = max(PathNameWidth, length(handles.Pipeline.(fld{1})));
    end
end


%%% Write the SQL table description and data loader.
if (FirstSet == 1)
    if strcmp(SQLchoice,'MySQL')

        fmain = fopen(fullfile(OutDir, [DBname '_SETUP.SQL']), 'W');

        fprintf(fmain, 'CREATE DATABASE %s;\n', DBname);
        fprintf(fmain, 'USE %s;\n\n', DBname);

        fprintf(fmain, 'CREATE TABLE %sPer_Image (ImageNumber INTEGER PRIMARY KEY',TablePrefix);

        for i = per_image_names,
            if strfind(i{1}, 'FileName')
                fprintf(fmain, ',\n%s VARCHAR(%d)', i{1}, FileNameWidth);
            elseif  strfind(i{1}, 'Path'),
                fprintf(fmain, ',\n%s VARCHAR(%d)', i{1}, PathNameWidth);
            else
                fprintf(fmain, ',\n%s FLOAT NOT NULL', i{1});
            end
        end

        %add columns for mean and stddev for per_object_names
        for j=per_object_names,
            fprintf(fmain, ',\n%s FLOAT NOT NULL', ['Mean_', j{1}]);
        end

        for k=per_object_names,
            fprintf(fmain, ',\n%s FLOAT NOT NULL', ['StDev_', k{1}]);
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
            p=p+1;
            if strfind(i{1}, 'Filename')
                fprintf(fsetup, ',\n%s VARCHAR2(%d)', ['col',num2str(p)], FileNameWidth);
            elseif  strfind(i{1}, 'Path'),
                fprintf(fsetup, ',\n%s VARCHAR2(%d)', ['col',num2str(p)], PathNameWidth);
            else
                fprintf(fsetup, ',\n%s FLOAT', ['col',num2str(p)]);
            end
        end

        %add columns for mean and stddev for per_object_names
        for j=per_object_names,
            p=p+1;
            fprintf(fsetup, ',\n%s FLOAT', ['col',num2str(p)]);
        end

        for h=per_object_names,
            p=p+1;
            fprintf(fsetup, ',\n%s FLOAT', ['col',num2str(p)]);
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

        p=1;
        for k=per_image_names
            p=p+1;
            fprintf(fcol, '%s', ['col', num2str(p)] );
            fprintf(fcol, ',%s\n', k{1} );
        end
        for l=per_object_names
            p=p+1;
            fprintf(fcol, '%s', ['col', num2str(p)]);
            fprintf(fcol, ',%s\n', ['Mean_', l{1}]);
        end
        for m=per_object_names
            p=p+1;
            fprintf(fcol, '%s', ['col', num2str(p)]);
            fprintf(fcol, ',%s\n', ['Stdev_', m{1}]);
        end

        p=p+1;
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
    perobjectvals_mean=[];
    fprintf(fimage, '%d', img_idx);
    objectbaserow = size(perobjectvals, 1);
    objectbasecol = 2;
    numobj = 0;
    maxnumobj=0;

    feature_idx = 1;

    for ObjectCell = ObjectNames,
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
                        features_not_to_be_exported, ...
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
                if ~ strcmp(per_object_names{feature_idx}, cleanup(CPjoinstrings(ObjectName, FeatureName))),
                    error(['Mismatched feature names #', int2str(feature_idx), ' ', per_object_names{feature_idx}, '!=', cleanup(CPjoinstrings(ObjectName, FeatureName))])
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

                %%% Add the values into the output 
                perobjectvals((objectbaserow+1):(objectbaserow+numobj), (objectbasecol+1)) = vals;
                perobjectvals_mean(1:numobj, (objectbasecol-2+1)) = vals;

                %%% shift right
                objectbasecol = objectbasecol + 1;
            end
        end %%% loop over features

        %%% put in image and object numbers
        if numobj > 0
            perobjectvals((objectbaserow+1):end, 1) = img_idx;
            perobjectvals((objectbaserow+1):end, 2) = 1:maxnumobj;
        end

    end %%% loop over object types

    %print mean, stdev for all measurements per image
    formatstr = ['%g' repmat(',%g',1,size(perobjectvals_mean,2)-1)];
    if size(perobjectvals_mean,1)==1
        fprintf(fimage,',');
        fprintf(fimage,formatstr,perobjectvals_mean); % ignore NaN
        for i= 1:size(perobjectvals_mean,2),
            fprintf(fimage,',0'); %ignore NaN
        end
    elseif size(perobjectvals_mean, 1) > 0,  % don't write anything if there are no measurements
        fprintf(fimage,',');
        fprintf(fimage,formatstr,(CPnanmean(perobjectvals_mean))); % ignore NaN
        fprintf(fimage,',');
        fprintf(fimage,formatstr,(CPnanstd(perobjectvals_mean)));%ignore NaN
    end
    fprintf(fimage,'\n');
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