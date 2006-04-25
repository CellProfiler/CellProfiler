function CPconvertsql(handles,OutDir,OutfilePrefix,DBname,TablePrefix,FirstSet,LastSet,SQLchoice)

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne E. Carpenter
%   Thouis Ray Jones
%   In Han Kang
%   Ola Friman
%   Steve Lowe
%   Joo Han Chang
%   Colin Clarke
%   Mike Lamprecht
%   Susan Ma
%
% Website: http://www.cellprofiler.org
%
% $Revision$

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

%%% SubMeasurementFieldnames usually includes 'Image' and objects like 'Nuclei'
SubMeasurementFieldnames = fieldnames(handles.Measurements)';

for RemainingSubMeasurementFieldnames = SubMeasurementFieldnames,
    %%%SubFieldname is the first fieldname in SubMeasurementFieldnames.
    SubFieldname = RemainingSubMeasurementFieldnames{1};
    substruct = handles.Measurements.(SubFieldname);
    substructfields = fieldnames(substruct)';
    for ssfc = substructfields
        ssf = ssfc{1};

        if strfind(ssf, 'Features')
            continue;
        end

        if strfind(ssf, 'PathnameOrig')
            continue;
        end

        if strfind(ssf, 'Text'),
            if (strfind(ssf, 'Text') + 3) == length(ssf)
                continue;
            end
        end

        if strfind(ssf, 'ModuleError')
            continue;
        end

        if strfind(ssf, 'TimeElapsed')
            continue;
        end

        if strfind(ssf, 'Description')
            continue;
        end

        if isfield(substruct, [ssf 'Features']),
            names = handles.Measurements.(SubFieldname).([ssf 'Features']);
        elseif isfield(substruct, [ssf 'Text']),
            names = handles.Measurements.(SubFieldname).([ssf 'Text']);
        elseif isfield(substruct, [ssf 'Description'])
            if length(handles.Measurements.(SubFieldname).(ssf)) ~= handles.Current.NumberOfImageSets
                continue;
            else
                names = handles.Measurements.(SubFieldname).([ssf 'Description']);
            end
        else
            names = {ssf};
        end

        vals = handles.Measurements.(SubFieldname).(ssf);

        if ~ischar(vals{1})
            if size(vals{1},2) ~= length(names) % make change here vals{1},2
                if ~isempty(vals{1})
                    error([SubFieldname ' ' ssf ' does not have right number of names ']);
                end
            end
        end

        if strcmp(SubFieldname, 'Image')
            for n = 1:length(names)
                per_image_names{end+1} = cleanup([SubFieldname '_' ssf '_' names{n}]);
            end
        else
            for n = 1:length(names)
                per_object_names{end+1} = cleanup([SubFieldname '_' ssf '_' names{n}]);
            end
        end
    end %end of substrucfield
end %end of remainfield

if handles.Current.SetBeingAnalyzed == 1

    if strcmp(SQLchoice,'MySQL')

        fmain = fopen(fullfile(OutDir, [DBname '_SETUP.SQL']), 'W');

        fprintf(fmain, 'CREATE DATABASE %s;\n', DBname);
        fprintf(fmain, 'USE %s;\n\n', DBname);

        fprintf(fmain, 'CREATE TABLE %sPer_Image (ImageNumber INTEGER PRIMARY KEY',TablePrefix);

        for i = per_image_names,
            if strfind(i{1}, 'Filename')
                fprintf(fmain, ',\n%s VARCHAR(128)', i{1});
            elseif  strfind(i{1}, 'Path'),
                fprintf(fmain, ',\n%s VARCHAR(128)', i{1});
            else
                fprintf(fmain, ',\n%s FLOAT', i{1});
            end
        end

        %add columns for mean and stddev for per_object_names
        for j=per_object_names,
            fprintf(fmain, ',\n%s FLOAT', ['Mean_', j{1}]);
        end

        for k=per_object_names,
            fprintf(fmain, ',\n%s FLOAT', ['StDev_', k{1}]);
        end

        fprintf(fmain, ');\n\n');

        fprintf(fmain, 'CREATE TABLE %sPer_Object(ImageNumber INTEGER,ObjectNumber INTEGER',TablePrefix);
        for i = per_object_names
            fprintf(fmain, ',\n%s FLOAT', i{1});
        end

        fprintf(fmain, ',\nPRIMARY KEY (ImageNumber, ObjectNumber));\n\n');

        if strcmp(handles.Settings.ModuleNames{handles.Current.NumberOfModules},'CreateBatchFiles')
            BatchSize = str2double(char(handles.Settings.VariableValues{handles.Current.NumberOfModules,2}));
            if isnan(BatchSize)
                errordlg('STOP!');
            end
            fprintf(fmain, 'LOAD DATA LOCAL INFILE ''%s1_1_image.CSV'' REPLACE INTO TABLE %sPer_Image FIELDS TERMINATED BY '','';\n',OutfilePrefix,TablePrefix);
            fprintf(fmain, 'LOAD DATA LOCAL INFILE ''%s1_1_object.CSV'' REPLACE INTO TABLE %sPer_Object FIELDS TERMINATED BY '','';\n',OutfilePrefix,TablePrefix);
            for n = 2:BatchSize:handles.Current.NumberOfImageSets
                StartImage = n;
                EndImage = min(StartImage + BatchSize - 1, handles.Current.NumberOfImageSets);
                ImageSQLFileName = sprintf('%s%d_%d_image.CSV', OutfilePrefix, StartImage, EndImage);
                ObjectSQLFileName = sprintf('%s%d_%d_object.CSV', OutfilePrefix, StartImage, EndImage);
                fprintf(fmain, 'LOAD DATA LOCAL INFILE ''%s'' REPLACE INTO TABLE %sPer_Image FIELDS TERMINATED BY '','';\n',ImageSQLFileName,TablePrefix);
                fprintf(fmain, 'LOAD DATA LOCAL INFILE ''%s'' REPLACE INTO TABLE %sPer_Object FIELDS TERMINATED BY '','';\n',ObjectSQLFileName,TablePrefix);
            end
        else
            fprintf(fmain, 'LOAD DATA LOCAL INFILE ''%s_image.CSV'' REPLACE INTO TABLE %sPer_Image FIELDS TERMINATED BY '','';\n',basename,TablePrefix);
            fprintf(fmain, 'LOAD DATA LOCAL INFILE ''%s_object.CSV'' REPLACE INTO TABLE %sPer_Object FIELDS TERMINATED BY '','';\n',basename,TablePrefix);
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
                fprintf(fsetup, ',\n%s VARCHAR2(128)', ['col',num2str(p)]);
            elseif  strfind(i{1}, 'Path'),
                fprintf(fsetup, ',\n%s VARCHAR2(128)', ['col',num2str(p)]);
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
            errordlg('STOP!');
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
        fprintf(fcolload, 'LOAD DATA INFILE ''%s'' INTO TABLE  %sColumn_Names FIELDS TERMINATED BY '','' OPTIONALLY ENCLOSED BY ''"'' (shortname, longname)',[DBname, '_columnnames.CSV'],TablePrefix);
        fclose(fcolload);

        %%%%%%%%%%%%%%%%%%%%
        %%% IMAGE LOADER %%%
        %%%%%%%%%%%%%%%%%%%%

        fimageloader = fopen(fullfile(OutDir, [DBname, '_LOADIMAGE.CTL']), 'W');
        fprintf(fimageloader, 'LOAD DATA\n');
        if strcmp(handles.Settings.ModuleNames{handles.Current.NumberOfModules},'CreateBatchFiles')
            BatchSize = str2double(char(handles.Settings.VariableValues{handles.Current.NumberOfModules,2}));
            if isnan(BatchSize)
                errordlg('STOP!');
            end
            fprintf(fimageloader, 'INFILE %s1_1_image.CSV\n', OutfilePrefix);
            for n = 2:BatchSize:handles.Current.NumberOfImageSets
                StartImage = n;
                EndImage = min(StartImage + BatchSize - 1, handles.Current.NumberOfImageSets);
                SQLFileName = sprintf('%s%d_%d_image.CSV', OutfilePrefix, StartImage, EndImage);
                fprintf(fimageloader, 'INFILE %s\n', SQLFileName);
            end
        else
            fprintf(fimageloader, 'INFILE %s\n', [basename, '_image.CSV']);
        end

        fprintf(fimageloader, 'INTO TABLE  %sPer_Image FIELDS TERMINATED BY '','' OPTIONALLY ENCLOSED BY ''"'' (col1',TablePrefix);
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
            BatchSize = str2double(char(handles.Settings.VariableValues{handles.Current.NumberOfModules,2}));
            if isnan(BatchSize)
                errordlg('STOP!');
            end
            fprintf(fobjectloader, 'INFILE %s1_1_object.CSV\n', OutfilePrefix);
            for n = 2:BatchSize:handles.Current.NumberOfImageSets
                StartImage = n;
                EndImage = min(StartImage + BatchSize - 1, handles.Current.NumberOfImageSets);
                SQLFileName = sprintf('%s%d_%d_object.CSV', OutfilePrefix, StartImage, EndImage);
                fprintf(fobjectloader, 'INFILE %s\n', SQLFileName);
            end
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

fimage = fopen(fullfile(OutDir, [basename '_image.CSV']), 'W');
fobject = fopen(fullfile(OutDir, [basename '_object.CSV']), 'W');

perobjectvals = [];

for img_idx = FirstSet:LastSet
    perobjectvals_mean=[];
    fprintf(fimage, '%d', img_idx);
    objectbaserow = size(perobjectvals, 1);
    objectbasecol = 2;
    objectbaserow_mean=size(perobjectvals_mean,1);
    numobj = 0;
    maxnumobj=0;
    for RemainingSubMeasurementFieldnames = SubMeasurementFieldnames,
        SubFieldname = RemainingSubMeasurementFieldnames{1};
        substruct = handles.Measurements.(SubFieldname);
        substructfields = fieldnames(substruct)';
        for ssfc = substructfields,
            ssf = ssfc{1};

            if strfind(ssf, 'Features'),
                continue;
            end

            if strfind(ssf, 'PathnameOrig'),
                continue;
            end

            if strfind(ssf, 'ModuleError'),
                continue;
            end

            if strfind(ssf, 'TimeElapsed'),
                continue;
            end

            if strfind(ssf,'Text'),
                if (strfind(ssf,'Text') + 3) == length(ssf),
                    continue;
                end
            end

            if strfind(ssf,'Description'),
                continue;
            end

            if size(handles.Measurements.(SubFieldname).(ssf),2) >= img_idx
                vals = handles.Measurements.(SubFieldname).(ssf){img_idx};
            else
                vals = [];
            end

            if strcmp(SubFieldname, 'Image'),
                if ischar(vals)
                    fprintf(fimage, ',%s', vals);
                    %vals{} is cellarray, need loop through to get all elements value
                elseif iscell(vals)
                    if ischar(vals{1}) %is char
                        for cellindex = 1:size(vals,2),
                            fprintf(fimage, ',%s', vals{cellindex});
                        end
                    else %vals{cellindex} is not char
                        fprintf(fimage, ',%g', cell2mat(vals));
                    end
                else %vals is number
                    fprintf(fimage, ',%g', vals);
                end
            else
                if ~isa(vals,'numeric')
                    error('Non-numeric data not currently supported in per-object SQL data.');
                end
                numcols = size(vals,2);
                numobj = size(vals,1);
                if maxnumobj <numobj  % different measurement have different object count
                    maxnumobj=numobj;
                end
                perobjectvals((objectbaserow+1):(objectbaserow+numobj), (objectbasecol+1):(objectbasecol+numcols)) = vals;
                perobjectvals_mean((objectbaserow_mean+1):(objectbaserow_mean+numobj), (objectbasecol-2+1):(objectbasecol-2+numcols)) = vals;
                objectbasecol = objectbasecol + numcols;
            end
        end
        if numobj > 0
            perobjectvals((objectbaserow+1):end, 1) = img_idx;
            perobjectvals((objectbaserow+1):end, 2) = 1:maxnumobj;
        end

    end
    %print mean, stdev for all measurements per image

    fprintf(fimage,',');
    formatstr = ['%g' repmat(',%g',1,size(perobjectvals_mean, 2)-1)];
    if size(perobjectvals_mean,1)==1
        fprintf(fimage,formatstr,perobjectvals_mean); % ignore NaN
        fprintf(fimage,',');
        for i= 1:size(perobjectvals_mean,2),
            fprintf(fimage,',',''); %ignore NaN
        end
        fprintf(fimage, '\n');
    else
        fprintf(fimage,formatstr,(CPnanmean(perobjectvals_mean))); % ignore NaN
        fprintf(fimage,',');
        fprintf(fimage,formatstr,(CPnanstd(perobjectvals_mean)));%ignore NaN
        fprintf(fimage, '\n');
    end
end

formatstr = ['%g' repmat(',%g',1,size(perobjectvals, 2)-1) '\n'];
%%% THIS LINE WRITES ENTIRE OBJECT VALS FILE
%%% if vals{1} is empty skip writting into object file
if ~iscell(vals) || (iscell(vals) && ~isempty(vals{1}))
    perobjectvals(isnan(perobjectvals)) = 0;
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