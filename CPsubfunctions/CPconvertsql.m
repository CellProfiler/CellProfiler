function CPconvertsql(handles,OutDir,OutfilePrefix,DBname,TablePrefix,FirstSet,LastSet)

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
%   Wyman Li
%
% Website: http://www.cellprofiler.org
%
% $Revision$

per_image_names = {};
per_object_names = {};

if ~isfield(handles,'Measurements')
    error('There are no measurements to be converted to SQL.')
end

Measurements = handles.Measurements;
basename = [OutfilePrefix int2str(FirstSet) '_' int2str(LastSet)];

%%% SubMeasurementFieldnames usually includes 'Image' and objects like 'Nuclei'
SubMeasurementFieldnames = fieldnames(Measurements)';

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
            names = handles.Measurements.(SubFieldname).([ssf 'Description']);
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

%full .sql file name
fmain = fopen(fullfile(OutDir, [basename '.SQL']), 'W');

fprintf(fmain, 'USE %s;\n', DBname);

%begin writing the sql script for creating tables
%first create look up table for colmun names
fprintf (fmain, 'CREATE TABLE IF NOT EXISTS FIELDNAMES (TABLENAME CHAR(100), COLUMNNUMBER CHAR(25), ORGFIELDNAME VARCHAR(250));\n');

%create table perimage
fprintf(fmain, 'CREATE TABLE IF NOT EXISTS %sPerImage (col1 INTEGER PRIMARY KEY', TablePrefix);

p=1;% col1 alread added
for i = per_image_names,
    p=p+1;
    if strfind(i{1}, 'Filename')
        fprintf(fmain, ', %s CHAR(128)', ['col',num2str(p)]);
    elseif  strfind(i{1}, 'Path'),
        fprintf(fmain, ', %s CHAR(128)', ['col',num2str(p)]);
    else
        fprintf(fmain, ', %s FLOAT', ['col',num2str(p)]);
    end
end

%add columns for mean and stddev for per_object_names
for j=per_object_names,
    p=p+1;
    fprintf(fmain, ', %s FLOAT', ['col',num2str(p)]);
end

for h=per_object_names,
    p=p+1;
    fprintf(fmain, ', %s FLOAT', ['col',num2str(p)]);
end

fprintf(fmain, ');\n');

fprintf(fmain, 'CREATE TABLE IF NOT EXISTS %sPerObject (col1 INTEGER, col2 INTEGER', TablePrefix);

p=2;
for i = per_object_names
    p=p+1;
    fprintf(fmain, ', %s FLOAT', ['col',num2str(p)]);
end
fprintf(fmain, ', PRIMARY KEY (col1, col2));\n');

%start to write data file
fcol = fopen(fullfile(OutDir, [basename '_colname.CSV']), 'W');
fimage = fopen(fullfile(OutDir, [basename '_image.CSV']), 'W');
fobject = fopen(fullfile(OutDir, [basename '_object.CSV']), 'W');

fprintf(fmain, 'LOAD DATA LOCAL INFILE ''%s'' REPLACE INTO TABLE  %sPerImage FIELDS TERMINATED BY ''    '';\n', [basename '_image.CSV'], TablePrefix);
fprintf(fmain, 'LOAD DATA LOCAL INFILE ''%s'' REPLACE INTO TABLE  %sPerObject FIELDS TERMINATED BY ''   '';\n', [basename '_object.CSV'], TablePrefix);
fprintf (fmain, 'LOAD DATA LOCAL INFIFLE ''%s'' REPLACE INTO TABLE %sFIEDNAMES FIELDS TERMINATED BY ''  '';\n', [basename '_colname.CSV']);
%insert columnname into look up table, using tab as delimiter

fprintf(fcol,'%sPerImage',TablePrefix);
fprintf(fcol, '\t%s', 'col1');
fprintf(fcol, '\t%s','ImageNumber');
fprintf(fcol, '\n');

p=1;
for k=per_image_names
    p=p+1;
    fprintf(fcol, '%sPerImage',TablePrefix);
    fprintf(fcol, '\t%s', ['col', num2str(p)] );
    fprintf(fcol, '\t%s', k{1} );
    fprintf(fcol, '\n');

end
for l=per_object_names
    p=p+1;
    fprintf(fcol, '%sPerImage',TablePrefix);
    fprintf(fcol, '\t%s', ['col', num2str(p)]);
    fprintf(fcol, '\t%s', ['Mean_', l{1}]);
    fprintf(fcol, '\n');

end
for m=per_object_names
    p=p+1;
    fprintf(fcol, '%sPerImage',TablePrefix);
    fprintf(fcol, '\t%s', ['col', num2str(p)]);
    fprintf(fcol, '\t%s', ['Stdev_', m{1}]);
    fprintf(fcol, '\n');
end

p=2;% reset for perobject table

%perobject table's colnames
fprintf(fcol,'%sPerObject',TablePrefix);
fprintf(fcol, '\t%s', 'col1');
fprintf(fcol, '\t%s','ImageNumber');
fprintf(fcol, '\n');

fprintf(fcol,'%sPerObject',TablePrefix);
fprintf(fcol, '\t%s', 'col2');
fprintf(fcol, '\t%s','ObjectNumber');
fprintf(fcol, '\n');

for n=per_object_names
    p=p+1;
    fprintf(fcol, '%sPerObject',TablePrefix);
    fprintf(fcol, '\t%s', ['col', num2str(p)]);
    fprintf(fcol, '\t%s', n{1} );
    fprintf(fcol, '\n');
end
%end for colname file

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

            vals = handles.Measurements.(SubFieldname).(ssf){img_idx};

            if strcmp(SubFieldname, 'Image'),
                if ischar(vals)
                    fprintf(fimage, '\t%s', vals);
                    %vals{} is cellarray, need loop through to get all elements value
                elseif iscell(vals)
                    if ischar(vals{1}) %is char
                        for cellindex = 1:size(vals,2),
                            fprintf(fimage, '\t%s', vals{cellindex});
                        end
                    else %vals{cellindex} is not char
                        fprintf(fimage, '\t%g', cell2mat(vals));
                    end
                else %vals is number
                    fprintf(fimage, '\t%g', vals);
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

    fprintf(fimage,'\t');
    formatstr = ['%g' repmat('\t%g',1,size(perobjectvals_mean, 2)-1)];
    if size(perobjectvals_mean,1)==1,
        fprintf(fimage,formatstr,perobjectvals_mean); % ignore NaN
        fprintf(fimage,'\t');
        for i= 1:size(perobjectvals_mean,2),
            fprintf(fimage,'\t','');%ignore NaN
        end
        %fprintf(fimage, 'atest');
        fprintf(fimage, '\n');

    else

        fprintf(fimage,formatstr,(CPnanmean(perobjectvals_mean))); % ignore NaN
        fprintf(fimage,'\t');
        fprintf(fimage,formatstr,(CPnanstd(perobjectvals_mean)));%ignore NaN
        fprintf(fimage, '\n');

    end
end

formatstr = ['%g' repmat('\t%g',1,size(perobjectvals, 2)-1) '\n'];
%if vals{1} is empty skip writting into object file
if ~iscell(vals) ||( iscell(vals) && (~isempty(vals{1}))  )
    fprintf(fobject, formatstr, perobjectvals');
end

fclose(fimage);
fclose(fobject);
fclose(fmain);
fclose(fcol);

function sc=cleanup(s)
sc = s;
sc(strfind(s,' ')) = '_';
if (length(sc) >= 64)
    warning(['Column name ' sc ' too long in CPconvertsql.'])
end