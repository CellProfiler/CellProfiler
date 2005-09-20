function CPConvertSQL(handles, OutDir, OutfilePrefix, DBname, TablePrefix, FirstSet, LastSet)

per_image_names = {};
per_object_names = {};

if ~isfield(handles,'Measurements')
    error('There are no measurements to be converted to SQL.')
end

Measurements = handles.Measurements;

%%% SubMeasurementFieldnames usually includes 'Image' and objects like 'Nuclei',
%%% 'Cells', etc.
SubMeasurementFieldnames = fieldnames(Measurements)';
for RemainingSubMeasurementFieldnames = SubMeasurementFieldnames,
    %%%SubFieldname is the first fieldname in SubMeasurementFieldnames.
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
        
        if strfind(ssf, 'Filename'),
             continue;
        end
        
        if isfield(substruct, [ssf 'Features']),
            names = handles.Measurements.(SubFieldname).([ssf 'Features']);
        else
            names = {ssf};
        end

        vals = handles.Measurements.(SubFieldname).(ssf);
        if (~ ischar(vals{1})) 
            if (size(vals{1},2) ~= length(names)), % make change here vals{1},2
                error([SubFieldname ' ' ssf ' does not have right number of names ']);
            
            end
        end
        
        if (size(vals{1},1) == 1),
            for n = 1:length(names),
                per_image_names{end+1} = cleanup([SubFieldname '_' ssf '_' names{n}]);
            end
        else
            for n = 1:length(names),
                per_object_names{end+1} = cleanup([SubFieldname '_' ssf '_' names{n}]);
            end
        end
    end
end

basename = [OutfilePrefix int2str(FirstSet) '_' int2str(LastSet)];
fmain = fopen(fullfile(OutDir, [basename '.SQL']), 'W');
fprintf(fmain, 'USE %s;\n', DBname);
fprintf(fmain, 'CREATE TABLE IF NOT EXISTS %sPerImage (ImageNumber INTEGER PRIMARY KEY', TablePrefix);
for i = per_image_names,
    if strfind(i{1}, 'Filename'),
        fprintf(fmain, ', %s CHAR(128)', i{1});
    else
        fprintf(fmain, ', %s FLOAT', i{1});
    end
end
fprintf(fmain, ');\n');

fprintf(fmain, 'CREATE TABLE IF NOT EXISTS %sPerObject (ImageNumber INTEGER, ObjectNumber INTEGER', TablePrefix);
for i = per_object_names,
    fprintf(fmain, ', %s FLOAT', i{1});
end
fprintf(fmain, ', PRIMARY KEY (ImageNumber, ObjectNumber));\n');



fimage = fopen(fullfile(OutDir, [basename '_image.SQL']), 'W');
fobject = fopen(fullfile(OutDir, [basename '_object.SQL']), 'W');

fprintf(fmain, 'LOAD DATA LOCAL INFILE ''%s'' REPLACE INTO TABLE  %sPerImage FIELDS TERMINATED BY ''|'';\n', [basename '_image.SQL'], TablePrefix);
fprintf(fmain, 'LOAD DATA LOCAL INFILE ''%s'' REPLACE INTO TABLE  %sPerObject FIELDS TERMINATED BY ''|'';\n', [basename '_object.SQL'], TablePrefix);

perobjectvals = [];

for img_idx = FirstSet:LastSet,
    fprintf(fimage, '%d', img_idx);
    objectbaserow = size(perobjectvals, 1);
    objectbasecol = 2;
    numobj = 0;

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
            
            % img_idx
            % handles.Measurements.(SubFieldname).(ssf)
            vals = handles.Measurements.(SubFieldname).(ssf){img_idx};
            if (size(vals,1) == 1),
                if ischar(vals),
                    fprintf(fimage, '|%s', vals);
                elseif iscell(vals)
                   fprintf(fimage, '|%s', char(vals));
                else
                    fprintf(fimage, '|%g', vals);
                end
            else
                if (~ isa(vals, 'numeric')),
                    error('Non-numeric data not currently supported in per-object SQL data');
                end
                numcols = size(vals,2);
                numobj = size(vals,1);
                perobjectvals((objectbaserow+1):(objectbaserow+numobj), (objectbasecol+1):(objectbasecol+numcols)) = vals;
                objectbasecol = objectbasecol + numcols;
            end
        end

        if numobj > 0,
            perobjectvals((objectbaserow+1):end, 1) = img_idx;
            perobjectvals((objectbaserow+1):end, 2) = 1:numobj;
        end
    end
    fprintf(fimage, '\n');
end

formatstr = ['%g' repmat('|%g',1,size(perobjectvals, 2)-1) '\n'];
fprintf(fobject, formatstr, perobjectvals');


fclose(fimage);
fclose(fobject);
fclose(fmain);

function sc=cleanup(s)
sc = s;
sc(strfind(s,' ')) = '_';
if (length(sc) >= 64),
    warning(['Column name ' sc ' too long in CPConvertSQL'])
end