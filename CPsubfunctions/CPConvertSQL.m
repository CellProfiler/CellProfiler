function CPConvertSQL(handles, OutDir, OutfilePrefix, DBname, TablePrefix, FirstSet, LastSet,ExportInfo)

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

ExportType='';
FileExtension='.csv';%default

if ~isempty(ExportInfo),%exporting to excel
    ExportType=ExportInfo.ExportType;
    FileExtension=ExportInfo.MeasurementExtension;
    Swap=ExportInfo.SwapRowsColumnInfo;
end

per_image_names = {};
per_object_names = {};

if ~isfield(handles,'Measurements')
    error('There are no measurements to be converted to SQL.')
end

Measurements = handles.Measurements;
basename = [OutfilePrefix int2str(FirstSet) '_' int2str(LastSet)];

%%% SubMeasurementFieldnames usually includes 'Image' and objects like 'Nuclei'

SubMeasurementFieldnames = fieldnames(Measurements)';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% start to for excle object data %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(ExportType,'Excel')
    global waitbarhandle
    CPwaitbar(0,waitbarhandle,'Export Status');
    objectCount=0;
    for excelRemainingSubMeasurementFieldnames = SubMeasurementFieldnames

        excelSubFieldname = excelRemainingSubMeasurementFieldnames{1};

        excelsubstruct = handles.Measurements.(excelSubFieldname);
        excelsubstructfields = fieldnames(excelsubstruct)';
        ExportObject=0;

        %check which objects have been choosen
        for a=1:size(ExportInfo.ObjectNames,1),
            if strcmp(ExportInfo.ObjectNames{a},excelSubFieldname);
                ExportObject='1';
                continue;
            end
        end % end for

        if ExportObject %first decide whether the object is choosen by user
            objectCount=objectCount+1;
            %%% Update waitbar
            CPwaitbar(objectCount/length(ExportInfo.ObjectNames),waitbarhandle,sprintf('Exporting %s',excelSubFieldname));

            fexcelobject=fopen(fullfile(OutDir,[basename '_' excelSubFieldname FileExtension]), 'W');

            excel_object_names={};

            if strcmp (excelSubFieldname,'Image') & strcmp(Swap, 'Yes') %for swapped img file only
                %fexcelimg_swapped=fopen(fullfile(OutDir,[basename '_swapped_'  excelSubFieldname FileExtension]), 'W');
                fprintf(fexcelobject, '%s', 'ImageNumber');
                for I=FirstSet:LastSet,
                    fprintf(fexcelobject, '\t%g', I);
                end
                fprintf(fexcelobject,'\n');
            end

            %first print out column names
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            for ssfc = excelsubstructfields
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

                if isfield(excelsubstruct, [ssf 'Features']),
                    names = handles.Measurements.(excelSubFieldname).([ssf 'Features']);
                elseif isfield(excelsubstruct, [ssf 'Text']),
                    names = handles.Measurements.(excelSubFieldname).([ssf 'Text']);
                elseif isfield(excelsubstruct, [ssf 'Description'])
                    names = handles.Measurements.(excelSubFieldname).([ssf 'Description']);
                else
                    names = {ssf};
                end
                dim_names=size(names,2);

                vals = handles.Measurements.(excelSubFieldname).(ssf);

                if (~ ischar(vals{1}))
                    if (size(vals{1},2) ~= length(names)),
                        if ~isempty(vals{1})
                            error([excelSubFieldname ' ' ssf ' does not have right number of names ']);
                        end
                    end
                end

                %get columnames for excel
                for n=1:length(names),
                    excel_object_names{end+1} = cleanup([excelSubFieldname '_' ssf '_' names{n}]);

                end
                
                %%%%%%%%%%%%%%%%%%%%% swap col/row for image file only %%%%%%%%%%%%%%%%%%%%%%%%%%%
                if strcmp (excelSubFieldname,'Image') & strcmp(Swap, 'Yes')
                    if dim_names >1,
                        for i=1:dim_names
                            fprintf(fexcelobject,'%s',excel_object_names{end-dim_names+i});

                            for img_no=FirstSet:LastSet
                                imgvals = handles.Measurements.(excelSubFieldname).(ssf){img_no};

                                if ischar(imgvals),
                                    fprintf(fexcelobject, '\t%s', imgvals);

                                    %vals{} is cellarray, need loop through to get all elements value
                                elseif iscell(imgvals)
                                    if (ischar(imgvals{1})), %is char
                                        %for cellindex = 1:size(imgvals,2),
                                        %fprintf(fexcelobject, '\t%s', imgvals{cellindex});
                                        fprintf(fexcelobject, '\t%s', imgvals{i});
                                        %image_val(end+1,:)=char(excelvals{cellindex});
                                        %end
                                    else, %vals{cellindex} is not char
                                        fprintf(fexcelobject, '\t%g', cell2mat(imgvals{i}));
                                    end
                                else %vals is number
                                    fprintf(fexcelobject, '\t%g', imgvals(i));
                                end
                            end
                            fprintf(fexcelobject, '\n');
                        end  %end for dim_names
                    else
                        fprintf(fexcelobject,'%s',excel_object_names{end});
                        for img_no=FirstSet:LastSet
                            imgvals = handles.Measurements.(excelSubFieldname).(ssf){img_no};

                            %if (size(vals,1) == 1),
                            if ischar(imgvals),
                                fprintf(fexcelobject, '\t%s', imgvals);

                                %vals{} is cellarray, need loop through to get all elements value
                            elseif iscell(imgvals)
                                if (ischar(imgvals{1})), %is char
                                    for cellindex = 1:size(imgvals,2),
                                        fprintf(fexcelobject, '\t%s', imgvals{cellindex});
                                        %image_val(end+1,:)=char(excelvals{cellindex});
                                    end

                                else, %vals{cellindex} is not char
                                    fprintf(fexcelobject, '\t%g', cell2mat(imgvals));
                                end
                            else %vals is number
                                fprintf(fexcelobject, '\t%g', imgvals);
                            end
                        end
                        fprintf(fexcelobject, '\n');
                    end %end of if dim_names>1
                end % end of if image and swap
            end %end of ssfc

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%start to write column name
            if ~strcmp (excelSubFieldname,'Image');
                if strcmp(Swap, 'Yes'),

                    fexcelmean_swap=fopen(fullfile(OutDir,[basename '_' excelSubFieldname '_mean_'  FileExtension]), 'W');
                    fexcelstd_swap=fopen(fullfile(OutDir,[basename '_' excelSubFieldname '_std_'  FileExtension]), 'W');
                else
                    fexcelmean=fopen(fullfile(OutDir,[basename '_' excelSubFieldname '_mean_'  FileExtension]), 'W');
                    fexcelstd =fopen(fullfile(OutDir,[basename '_' excelSubFieldname '_std_' FileExtension]), 'W');
                end

            else % for all object files
                if ~strcmp(Swap, 'Yes'),
                    fprintf(fexcelobject,'%s','ImageNumber');
                    for e=excel_object_names,
                        fprintf(fexcelobject,'\t%s', e{1});
                    end
                    fprintf(fexcelobject,'\n');
                else
                    if ~strcmp (excelSubFieldname,'Image');
                        fprintf(fexcelobject, '%s', 'ImageNumber');
                        for I=FirstSet:LastSet,
                            fprintf(fexcelobject, '\t%g', I);
                        end
                        fprintf(fexcelobject,'\n');
                    end
                end
            end

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% writing data
            perobjectvals=[];
            excel_means=[];
            excel_stdevs=[];
            for img_idx = FirstSet:LastSet

                %write img number for image file
                if strcmp (excelSubFieldname,'Image') & ~strcmp(Swap, 'Yes');
                    fprintf(fexcelobject,'%d',img_idx);
                end
                perobjectvals_mean=[];
                %fprintf(fexcelobject, '%d', img_idx); %img number first
                objectbaserow = size(perobjectvals, 1);
                objectbasecol = 2;
                objectbaserow_mean=size(perobjectvals_mean,1);
                numobj = 0;
                maxnumobj=0;

                for excelssfc = excelsubstructfields,
                    %img_colcount=img_colcount+1;
                    excelssf = excelssfc{1};
                    %fprintf (ssf);
                    if strfind(excelssf, 'Features'),
                        continue;
                    end

                    if strfind(excelssf, 'PathnameOrig'),
                        continue;
                    end
                    if strfind(excelssf, 'ModuleError'),
                        continue;
                    end
                    if strfind(excelssf, 'TimeElapsed'),
                        continue;
                    end
                    if strfind(excelssf, 'Description'),
                        continue;
                    end

                    if strfind(excelssf, 'Text'),
                        if (strfind(excelssf, 'Text') + 3) == length(excelssf),
                            continue;
                        end
                    end

                    excelvals = handles.Measurements.(excelSubFieldname).(excelssf){img_idx};
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                    if strcmp(excelSubFieldname, 'Image')
                        if ~strcmp(Swap, 'Yes'),

                            %if (size(vals,1) == 1),
                            if ischar(excelvals),
                                fprintf(fexcelobject, '\t%s', excelvals);

                                %vals{} is cellarray, need loop through to get all elements value
                            elseif iscell(excelvals)
                                if (ischar(excelvals{1})), %is char
                                    for cellindex = 1:size(excelvals,2),
                                        fprintf(fexcelobject, '\t%s', excelvals{cellindex});

                                    end


                                else, %vals{cellindex} is not char
                                    fprintf(fexcelobject, '\t%g', cell2mat(excelvals));
                                end

                            else %vals is number

                                fprintf(fexcelobject, '\t%g', excelvals);
                            end
                        end

                    else % perobject data
                        if (~ isa(excelvals, 'numeric')),
                            error('Non-numeric data not currently supported in per-object SQL data');
                        end

                        numcols = size(excelvals,2);
                        numobj = size(excelvals,1);
                        if maxnumobj <numobj,  % different measurement have different object count
                            maxnumobj=numobj;
                        end

                        perobjectvals((objectbaserow+1):(objectbaserow+numobj), (objectbasecol+1):(objectbasecol+numcols)) = excelvals;

                        perobjectvals_mean((objectbaserow_mean+1):(objectbaserow_mean+numobj), (objectbasecol):(objectbasecol-1+numcols)) = excelvals;

                        objectbasecol = objectbasecol + numcols;
                    end


                end % end of for ssfc


                if numobj > 0,

                    perobjectvals((objectbaserow+1):end, 1) = img_idx;
                    perobjectvals((objectbaserow+1):end, 2) = 1:maxnumobj;
                    perobjectvals_mean((objectbaserow_mean+1):end, 1) = img_idx;

                end

                if strcmp(excelSubFieldname, 'Image') & ~strcmp(Swap, 'Yes'),
                    fprintf(fexcelobject, '\n');
                else

                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%% write mean and stdev data
                    premeans=[];
                    prestdev=[];
                    formatstr = ['%g' repmat('\t%g',1,size(perobjectvals_mean, 2)-1)];
                    if  size(perobjectvals_mean,1)==1, %if only one object, mean is same , std is empty

                        premeans=perobjectvals_mean;
                        premeans(:,1)=img_idx;

                    else

                        premeans=CPnanmean(perobjectvals_mean);
                        premeans(:,1)=img_idx;
                        prestdev=CPnanstd(perobjectvals_mean);
                    end


                    excel_means( end+1, 1:size(premeans,2))=premeans; % add one more row
                    if isempty(prestdev),
                        excel_stdevs(end+1, :)=0;
                    else
                        excel_stdevs( end+1, 1:size(prestdev,2))=prestdev ;
                    end
                    excel_stdevs(end,1)=img_idx;
                end % end of if= image

            end % end of image index

            if ~strcmp(excelSubFieldname, 'Image')

                if strcmp(Swap, 'Yes') ,
                    %swapped mean
                    swapmean=excel_means';
                    formatstr_swap = ['\t%g' repmat('\t%g',1,size(swapmean, 2)-1) '\n'];
                    fprintf(fexcelmean_swap,'%s','ImageNumber');
                    fprintf(fexcelmean_swap,formatstr_swap,swapmean(1,:));
                    for i=1:length(excel_object_names)
                        fprintf(fexcelmean_swap,'%s', 'mean_',excel_object_names{i});
                        fprintf(fexcelmean_swap,formatstr_swap,swapmean(i+1,:));
                    end
                    %swapped stdev
                    swapstd=excel_stdevs';
                    formatstr_swap = ['\t%g' repmat('\t%g',1,size(swapstd, 2)-1) '\n'];
                    fprintf(fexcelstd_swap,'%s','ImageNumber');
                    fprintf(fexcelstd_swap,formatstr_swap,swapstd(1,:));
                    for w=1:length(excel_object_names)
                        fprintf(fexcelstd_swap,'%s', 'std_',excel_object_names{w});
                        %fprintf(fexcelstd_swap, '%g', size(swapstd,1), '\n');
                        if w<size(swapstd,1),
                            fprintf(fexcelstd_swap,formatstr_swap,swapstd(w+1,:));
                        else
                            fprintf(fexcelstd_swap, '\n');
                        end
                    end

                    %%%per object
                    a= perobjectvals';
                    formatstr = ['\t%g' repmat('\t%g',1,size(perobjectvals', 2)-1) '\n'];
                    fprintf(fexcelobject,'%s','ImageNumber');

                    fprintf(fexcelobject, formatstr, a(1,:));

                    fprintf(fexcelobject,'%s','ObjectNumber');
                    fprintf(fexcelobject, formatstr, a(2,:));


                    for e=1:length(excel_object_names),
                        fprintf(fexcelobject,'%s', excel_object_names{e});
                        if e <size(perobjectvals,2)

                            fprintf(fexcelobject, formatstr, a(e+2,:));
                        end
                    end
                else
                    %normal,features as row
                    fprintf(fexcelmean,'%s','ImageNumber');
                    for e=excel_object_names,
                        fprintf(fexcelmean,'\t%s', e{1});
                    end
                    fprintf(fexcelmean, '\n');
                    for j=1:size(excel_means,1)
                        formatstr2 = ['%g' repmat('\t%g',1,size(excel_means, 2)-1) '\n'];
                        fprintf(fexcelmean,formatstr2,excel_means(j,:));
                    end
                    %%%stdev
                    fprintf(fexcelstd,'%s','ImageNumber');
                    for e=excel_object_names,
                        fprintf(fexcelstd,'\t%s', e{1});
                    end
                    fprintf(fexcelstd, '\n');
                    for j=1:size(excel_stdevs,1)
                        formatstr2 = ['%g' repmat('\t%g',1,size(excel_stdevs, 2)-1) '\n'];

                        fprintf(fexcelstd,formatstr2,excel_stdevs(j,:));

                    end
                    %%%per object not include image file
                    fprintf(fexcelobject,'%s','ImageNumber');
                    fprintf(fexcelobject,'\t%s','ObjectNumber');
                    for e=excel_object_names,
                        fprintf(fexcelobject,'\t%s', e{1});
                    end
                    fprintf(fexcelobject, '\n');
                    formatstr = ['%g' repmat('\t%g',1,size(perobjectvals, 2)-1) '\n'];
                    %if vals{1} is empty skip writting into object file
                    if ~iscell(vals) ||( iscell(vals) && (~isempty(vals{1}))  )
                        fprintf(fexcelobject, formatstr, perobjectvals');
                    end
                end
                if strcmp(Swap, 'Yes')
                    fclose(fexcelstd_swap);
                    fclose(fexcelmean_swap);
                else
                    fclose (fexcelstd);
                    fclose (fexcelmean);
                end
            end
            fclose (fexcelobject);
        end % end of if ExportObject=1
    end % end of subfield, object type
    return;
end  % end of if excel

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%rest for SQL %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

        if strfind(ssf, 'Text'),
            if (strfind(ssf, 'Text') + 3) == length(ssf),
                continue;
            end
        end

        if strfind(ssf, 'ModuleError'),
            continue;
        end

        if strfind(ssf, 'TimeElapsed'),
            continue;
        end
        if strfind(ssf, 'Description'),
            continue;
        end

        if isfield(substruct, [ssf 'Features']),
            names = handles.Measurements.(SubFieldname).([ssf 'Features']);
        elseif isfield(substruct, [ssf 'Text']),
            names = handles.Measurements.(SubFieldname).([ssf 'Text']);
        elseif isfield(excelsubstruct, [ssf 'Description'])
            names = handles.Measurements.(excelSubFieldname).([ssf 'Description']);
        else
            names = {ssf};
        end

        vals = handles.Measurements.(SubFieldname).(ssf);

        if (~ ischar(vals{1}))
            if (size(vals{1},2) ~= length(names)), % make change here vals{1},2
                if ~isempty(vals{1})
                    error([SubFieldname ' ' ssf ' does not have right number of names ']);
                end
            end
        end

        if strcmp(SubFieldname, 'Image'),
            %if (size(vals{1},1) == 1),
            for n = 1:length(names),
                per_image_names{end+1} = cleanup([SubFieldname '_' ssf '_' names{n}]);
            end
        else
            for n = 1:length(names),
                per_object_names{end+1} = cleanup([SubFieldname '_' ssf '_' names{n}]);
            end
        end

    end %end of substrucfield
end %end of remainfield

%full .sql file name
%basename = [OutfilePrefix int2str(FirstSet) '_' int2str(LastSet)];
fmain = fopen(fullfile(OutDir, [basename '.SQL']), 'W');

fprintf(fmain, 'USE %s;\n', DBname);

%begin writing the sql script for creating tables
%first create look up table for colmun names
fprintf (fmain, 'CREATE TABLE IF NOT EXISTS FIELDNAMES (TABLENAME CHAR(100), COLUMNNUMBER CHAR(25), ORGFIELDNAME VARCHAR(250));\n');

%create table perimage
%fprintf(fmain, 'CREATE TABLE IF NOT EXISTS %sPerImage (ImageNumber INTEGER PRIMARY KEY', TablePrefix);
fprintf(fmain, 'CREATE TABLE IF NOT EXISTS %sPerImage (col1 INTEGER PRIMARY KEY', TablePrefix);

p=1;% col1 alread added
for i = per_image_names,
    p=p+1;
    if strfind(i{1}, 'Filename')
        %fprintf(fmain, ', %s CHAR(128)', i{1});
        fprintf(fmain, ', %s CHAR(128)', ['col',num2str(p)]);
    elseif  strfind(i{1}, 'Path'),
        %fprintf(fmain, ', %s CHAR(128)', i{1});
        fprintf(fmain, ', %s CHAR(128)', ['col',num2str(p)]);
    else
        %fprintf(fmain, ', %s FLOAT', i{1});
        fprintf(fmain, ', %s FLOAT', ['col',num2str(p)]);
    end
end

%add columns for mean and stddev for per_object_names
for j=per_object_names,
    p=p+1;
    fprintf(fmain, ', %s FLOAT', ['col',num2str(p)]);
    %fprintf(fmain, ', %s FLOAT', ['Mean_',j{1}]);
end

for h=per_object_names,
    p=p+1;
    fprintf(fmain, ', %s FLOAT', ['col',num2str(p)]);
    %fprintf(fmain, ', %s FLOAT', ['Std_',h{1}]);
end

fprintf(fmain, ');\n');

%fprintf(fmain, 'CREATE TABLE IF NOT EXISTS %sPerObject (ImageNumber INTEGER, ObjectNumber INTEGER', TablePrefix);
fprintf(fmain, 'CREATE TABLE IF NOT EXISTS %sPerObject (col1 INTEGER, col2 INTEGER', TablePrefix);

p=2;
for i = per_object_names,
    p=p+1;
    %fprintf(fmain, ', %s FLOAT', i{1});
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
for k=per_image_names;
    p=p+1;
    fprintf(fcol, '%sPerImage',TablePrefix);
    fprintf(fcol, '\t%s', ['col', num2str(p)] );
    fprintf(fcol, '\t%s', k{1} );
    fprintf(fcol, '\n');

end
for l=per_object_names,
    p=p+1;
    fprintf(fcol, '%sPerImage',TablePrefix);
    fprintf(fcol, '\t%s', ['col', num2str(p)]);
    fprintf(fcol, '\t%s', ['Mean_', l{1}]);
    fprintf(fcol, '\n');

end
for m=per_object_names,
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

for n=per_object_names,
    p=p+1;
    fprintf(fcol, '%sPerObject',TablePrefix);
    fprintf(fcol, '\t%s', ['col', num2str(p)]);
    fprintf(fcol, '\t%s', n{1} );
    fprintf(fcol, '\n');
end
%end for colname file

perobjectvals = [];

for img_idx = FirstSet:LastSet,
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
            %fprintf (ssf);
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

            % img_idx
            % handles.Measurements.(Image).(FileNames){index}

            vals = handles.Measurements.(SubFieldname).(ssf){img_idx};

            if strcmp(SubFieldname, 'Image'),
                %if (size(vals,1) == 1),
                if ischar(vals),
                    fprintf(fimage, '\t%s', vals);

                    %vals{} is cellarray, need loop through to get all elements value
                elseif iscell(vals)
                    if (ischar(vals{1})), %is char
                        for cellindex = 1:size(vals,2),
                            fprintf(fimage, '\t%s', vals{cellindex});
                        end
                    else, %vals{cellindex} is not char
                        fprintf(fimage, '\t%g', cell2mat(vals));
                    end
                    %elseif isempty(vals)
                    %   fprintf(fimage, '\t%g',vals);
                else %vals is number

                    fprintf(fimage, '\t%g', vals);
                end
            else
                if (~ isa(vals, 'numeric')),
                    error('Non-numeric data not currently supported in per-object SQL data');
                end

                numcols = size(vals,2);
                numobj = size(vals,1);
                if maxnumobj <numobj,  % different measurement have different object count
                    maxnumobj=numobj;
                end
                perobjectvals((objectbaserow+1):(objectbaserow+numobj), (objectbasecol+1):(objectbasecol+numcols)) = vals;
                perobjectvals_mean((objectbaserow_mean+1):(objectbaserow_mean+numobj), (objectbasecol-2+1):(objectbasecol-2+numcols)) = vals;

                objectbasecol = objectbasecol + numcols;
            end
        end

        if numobj > 0,

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
if (length(sc) >= 64),
    warning(['Column name ' sc ' too long in CPConvertSQL'])
end