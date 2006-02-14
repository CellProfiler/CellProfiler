function CPconvertsql2(handles,OutDir,OutfilePrefix,TablePrefix,FirstSet,LastSet)

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
% $Revision: 2981 $

per_image_names = {};
per_object_names = {};

if ~isfield(handles,'Measurements')
    error('There are no measurements to be converted to SQL.')
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

    %%%%%%%%%%%%%%%%%%
    %%% SETUP FILE %%%
    %%%%%%%%%%%%%%%%%%

    %full .sql file name
    fsetup = fopen(fullfile(OutDir, [TablePrefix '_SETUP.SQL']), 'W');

    fprintf (fsetup, 'CREATE TABLE %s_Column_Names (SHORTNAME VARCHAR2(8), LONGNAME VARCHAR2(250));\n',TablePrefix);

    fprintf(fsetup, 'CREATE TABLE %s_Per_Image (col1 NUMBER', TablePrefix);

    p=1;
    for i = per_image_names,
        p=p+1;
        if strfind(i{1}, 'Filename')
            fprintf(fsetup, ', %s VARCHAR2(128)', ['col',num2str(p)]);
        elseif  strfind(i{1}, 'Path'),
            fprintf(fsetup, ', %s VARCHAR2(128)', ['col',num2str(p)]);
        else
            fprintf(fsetup, ', %s FLOAT', ['col',num2str(p)]);
        end
    end

    %add columns for mean and stddev for per_object_names
    for j=per_object_names,
        p=p+1;
        fprintf(fsetup, ', %s FLOAT', ['col',num2str(p)]);
    end

    for h=per_object_names,
        p=p+1;
        fprintf(fsetup, ', %s FLOAT', ['col',num2str(p)]);
    end

    fprintf(fsetup, ');\n');
    p = p+1;
    PrimKeyPosition = p;
    fprintf(fsetup, 'CREATE TABLE IF NOT EXISTS %s_Per_Object (col1 NUMBER, %s NUMBER', TablePrefix,['col',num2str(p)]);
    for i = per_object_names
        p=p+1;
        fprintf(fsetup, ', %s FLOAT', ['col',num2str(p)]);
    end

    fprintf(fsetup, ');\n');
    fclose(fsetup);

    %%%%%%%%%%%%%%%%%%%
    %%% FINISH FILE %%%
    %%%%%%%%%%%%%%%%%%%

    ffinish = fopen(fullfile(OutDir, [TablePrefix, '_FINISH.SQL']), 'W');
    fprintf(ffinish, 'ALTER TABLE %s_Per_Image ADD PRIMARY KEY (col1);',TablePrefix);
    fprintf(ffinish, 'ALTER TABLE %s_Per_Object ADD PRIMARY KEY (col1, %s);',TablePrefix,['col',num2str(PrimKeyPosition)]);
    fclose(ffinish);

    %%%%%%%%%%%%%%%%%%%
    %%% COLUMN FILE %%%
    %%%%%%%%%%%%%%%%%%%

    fcol = fopen(fullfile(OutDir, [TablePrefix, '_columnnames.CSV']), 'W');
    fprintf(fcol, '%s', 'col1');
    fprintf(fcol, '\t%s','ImageNumber');
    fprintf(fcol, '\n');

    p=1;
    for k=per_image_names
        p=p+1;
        fprintf(fcol, '%s', ['col', num2str(p)] );
        fprintf(fcol, '\t%s', k{1} );
        fprintf(fcol, '\n');

    end
    for l=per_object_names
        p=p+1;
        fprintf(fcol, '%s', ['col', num2str(p)]);
        fprintf(fcol, '\t%s', ['Mean_', l{1}]);
        fprintf(fcol, '\n');

    end
    for m=per_object_names
        p=p+1;
        fprintf(fcol, '%s', ['col', num2str(p)]);
        fprintf(fcol, '\t%s', ['Stdev_', m{1}]);
        fprintf(fcol, '\n');
    end

    p=p+1;
    if PrimKeyPosition ~= p
        errordlg('STOP!');
    end

    % Per_Object table's colnames
    fprintf(fcol, '%s', ['col', num2str(p)]);
    fprintf(fcol, '\t%s','ObjectNumber');
    fprintf(fcol, '\n');

    for n=per_object_names
        p=p+1;
        fprintf(fcol, '%s', ['col', num2str(p)]);
        fprintf(fcol, '\t%s', n{1} );
        fprintf(fcol, '\n');
    end
    fclose(fcol);

    FinalColumnPosition = p;

    %%%%%%%%%%%%%%%%%%%%%
    %%% COLUMN LOADER %%%
    %%%%%%%%%%%%%%%%%%%%%

    fcolload = fopen(fullfile(OutDir, [TablePrefix, '_LOADCOLUMNS.CTL']), 'W');
    fprintf(fcolload, 'LOAD DATA INFILE ''%s'' INTO TABLE  %s_Column_Names FIELDS TERMINATED '' '' (shortname, longname)',[TablePrefix, '_columnnames.CSV'],TablePrefix);
    fclose(fcolload);

    %%%%%%%%%%%%%%%%%%%%
    %%% IMAGE LOADER %%%
    %%%%%%%%%%%%%%%%%%%%

    fimageloader = fopen(fullfile(OutDir, [TablePrefix, '_LOADIMAGE.CTL']), 'W');
    fprintf(fimageloader, 'LOAD DATA\n');
    if strcmp(handles.Settings.ModuleNames{handles.Current.NumberOfModules},'CreateBatchScripts')
        BatchSize = str2double(char(handles.Settings.VariableValues{handles.Current.NumberOfModules,1}));
        if isnan(BatchSize)
            errordlg('STOP!');
        end
        for n = 2:BatchSize:handles.Current.NumberOfImageSets
            StartImage = n;
            EndImage = min(StartImage + BatchSize - 1, handles.Current.NumberOfImageSets);
            SQLFileName = sprintf('%s%d_to_%d_image.CSV', OutfilePrefix, StartImage, EndImage);
            fprintf(fimageloader, 'INFILE %s\n', SQLFileName);
        end
    else
        fprintf(fimageloader, 'INFILE %s\n', [basename, '_image.CSV']);
    end

    fprintf(fimageloader, 'INTO TABLE  %s_Per_Image FIELDS TERMINATED BY '' '' (col1,',TablePrefix);
    for i = 2:(PrimKeyPosition-1)
        fprintf(fimageloader, '\n%s', ['col',num2str(i),',']);
    end
    fprintf(fimageloader, ')');

    fclose(fimageloader);

    %%%%%%%%%%%%%%%%%%%%%
    %%% OBJECT LOADER %%%
    %%%%%%%%%%%%%%%%%%%%%

    fobjectloader = fopen(fullfile(OutDir, [TablePrefix, '_LOADOBJECT.CTL']), 'W');
    fprintf(fobjectloader, 'LOAD DATA\n');
    if strcmp(handles.Settings.ModuleNames{handles.Current.NumberOfModules},'CreateBatchScripts')
        BatchSize = str2double(char(handles.Settings.VariableValues{handles.Current.NumberOfModules,1}));
        if isnan(BatchSize)
            errordlg('STOP!');
        end
        for n = 2:BatchSize:handles.Current.NumberOfImageSets
            StartImage = n;
            EndImage = min(StartImage + BatchSize - 1, handles.Current.NumberOfImageSets);
            SQLFileName = sprintf('%s%d_to_%d_object.CSV', OutfilePrefix, StartImage, EndImage);
            fprintf(fobjectloader, 'INFILE %s\n', SQLFileName);
        end
    else
        fprintf(fobjectloader, 'INFILE %s\n', [basename, '_object.CSV']);
    end

    fprintf(fobjectloader, 'INTO TABLE  %s_Per_Object FIELDS TERMINATED BY '' '' (col1,',TablePrefix);
    for i = PrimKeyPosition:FinalColumnPosition
        fprintf(fobjectloader, '\n%s', ['col',num2str(i),',']);
    end
    fprintf(fobjectloader, ')');

    fclose(fobjectloader);
end

%start to write data file

fimage = fopen(fullfile(OutDir, [basename '_image.CSV']), 'W');
fobject = fopen(fullfile(OutDir, [basename '_object.CSV']), 'W');

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
    if size(perobjectvals_mean,1)==1
        fprintf(fimage,formatstr,perobjectvals_mean); % ignore NaN
        fprintf(fimage,'\t');
        for i= 1:size(perobjectvals_mean,2),
            fprintf(fimage,'\t',''); %ignore NaN
        end
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

function sc=cleanup(s)
sc = s;
sc(strfind(s,' ')) = '_';
if (length(sc) >= 64)
    warning(['Column name ' sc ' too long in CPconvertsql.']) %#ok Ignore MLint
end