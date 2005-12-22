function ManualTableWizard

fid = fopen('ManualTableWizardText.xls','w');

Modulefilelist = dir('Modules/*.m');
FileProcessingFiles ={};
PreProcessingFiles={};
ObjectProcessingFiles={};
MeasurementFiles={};
OtherFiles={};
for i=1:length(Modulefilelist)
    name = Modulefilelist(i).name;
    if file_in_category(Modulefilelist(i).name, 'File Processing')
        FileProcessingFiles(length(FileProcessingFiles)+1)=cellstr(name);
    elseif file_in_category(Modulefilelist(i).name, 'Image Processing')
        PreProcessingFiles(length(PreProcessingFiles)+1)=cellstr(name);
    elseif file_in_category(Modulefilelist(i).name, 'Object Processing')
        ObjectProcessingFiles(length(ObjectProcessingFiles)+1)=cellstr(name);
    elseif file_in_category(Modulefilelist(i).name, 'Measurement')
        MeasurementFiles(length(MeasurementFiles)+1)=cellstr(name);
    else
        OtherFiles(length(OtherFiles)+1)=cellstr(name);
    end
end

fprintf(fid,'MODULES\n');
fprintf(fid,'File Processing Files\n');
for i=1:length(FileProcessingFiles)
    name = FileProcessingFiles{i};
    name = name(1:end-2);
    fprintf(fid,[name,'\t']);
    body = char(strread(help(FileProcessingFiles{i}),'%s','delimiter','','whitespace',''));
    flag = 0;
    for j = 1:size(body,1)
        if strcmp(body(j,1:75),'  *************************************************************************')
            flag = 0;
            fprintf(fid,'\n');
        end
        if flag
            fixedtext = fixthistext2(body(j,:));
            fprintf(fid,fixedtext(3:end));
        end
        if strcmp(body(j,1:20),'  SHORT DESCRIPTION:')
            flag = 1;
        end
    end
end

fprintf(fid,'\nImage Processing Files\n');
for i=1:length(PreProcessingFiles)
    name = PreProcessingFiles{i};
    name = name(1:end-2);
    fprintf(fid,[name,'\t']);
    body = char(strread(help(PreProcessingFiles{i}),'%s','delimiter','','whitespace',''));
    flag = 0;
    for j = 1:size(body,1)
        if strcmp(body(j,1:75),'  *************************************************************************')
            flag = 0;
            fprintf(fid,'\n');
        end
        if flag
            fixedtext = fixthistext2(body(j,:));
            fprintf(fid,fixedtext(3:end));
        end
        if strcmp(body(j,1:20),'  SHORT DESCRIPTION:')
            flag = 1;
        end
    end
end

fprintf(fid,'\nObject Processing Files\n');
for i=1:length(ObjectProcessingFiles)
    name = ObjectProcessingFiles{i};
    name = name(1:end-2);
    fprintf(fid,[name,'\t']);
    body = char(strread(help(ObjectProcessingFiles{i}),'%s','delimiter','','whitespace',''));
    flag = 0;
    for j = 1:size(body,1)
        if strcmp(body(j,1:75),'  *************************************************************************')
            flag = 0;
            fprintf(fid,'\n');
        end
        if flag
            fixedtext = fixthistext2(body(j,:));
            fprintf(fid,fixedtext(3:end));
        end
        if strcmp(body(j,1:20),'  SHORT DESCRIPTION:')
            flag = 1;
        end
    end
end

fprintf(fid,'\nMeasurement Files\n');
for i=1:length(MeasurementFiles)
    name = MeasurementFiles{i};
    name = name(1:end-2);
    fprintf(fid,[name,'\t']);
    body = char(strread(help(MeasurementFiles{i}),'%s','delimiter','','whitespace',''));
    flag = 0;
    for j = 1:size(body,1)
        if strcmp(body(j,1:75),'  *************************************************************************')
            flag = 0;
            fprintf(fid,'\n');
        end
        if flag
            fixedtext = fixthistext2(body(j,:));
            fprintf(fid,fixedtext(3:end));
        end
        if strcmp(body(j,1:20),'  SHORT DESCRIPTION:')
            flag = 1;
        end
    end
end

fprintf(fid,'\nOther Files\n');
for i=1:length(OtherFiles)
    name = OtherFiles{i};
    name = name(1:end-2);
    fprintf(fid,[name,'\t']);
    body = char(strread(help(OtherFiles{i}),'%s','delimiter','','whitespace',''));
    flag = 0;
    for j = 1:size(body,1)
        if strcmp(body(j,1:75),'  *************************************************************************')
            flag = 0;
            fprintf(fid,'\n');
        end
        if flag
            fixedtext = fixthistext2(body(j,:));
            fprintf(fid,fixedtext(3:end));
        end
        if strcmp(body(j,1:20),'  SHORT DESCRIPTION:')
            flag = 1;
        end
    end
end

fprintf(fid,'\n\nDATA TOOLS:\n');

Datatoolsfilelist = dir('DataTools/*.m');
for i=1:length(Datatoolsfilelist)
    name=Datatoolsfilelist(i).name;
    name=name(1:end-2);
    fprintf(fid,[name,'\t']);
    body = char(strread(help(Datatoolsfilelist(i).name),'%s','delimiter','','whitespace',''));
    flag = 0;
    for j = 1:size(body,1)
        if strcmp(body(j,1:75),'  *************************************************************************')
            flag = 0;
            fprintf(fid,'\n');
        end
        if flag
            fixedtext = fixthistext2(body(j,:));
            fprintf(fid,fixedtext(3:end));
        end
        if strcmp(body(j,1:20),'  SHORT DESCRIPTION:')
            flag = 1;
        end
    end
end

fprintf(fid,'\n\nIMAGE TOOLS:\n');

Imagetoolsfilelist = dir('ImageTools/*.m');
for i=1:length(Imagetoolsfilelist)
    name=Imagetoolsfilelist(i).name;
    name=name(1:end-2);
    fprintf(fid,[name,'\t']);
    body = char(strread(help(Imagetoolsfilelist(i).name),'%s','delimiter','','whitespace',''));
    flag = 0;
    for j = 1:size(body,1)
        if strcmp(body(j,1:75),'  *************************************************************************')
            flag = 0;
            fprintf(fid,'\n');
        end
        if flag
            fixedtext = fixthistext2(body(j,:));
            fprintf(fid,fixedtext(3:end));
        end
        if strcmp(body(j,1:20),'  SHORT DESCRIPTION:')
            flag = 1;
        end
    end
end

fclose(fid);

function fixedtext = fixthistext2(text)
fixedtext = strrep(text,'''','''''');
fixedtext = strrep(fixedtext,'\','\\');
fixedtext = strrep(fixedtext,'%','%%');
while 1
    if strcmp(fixedtext(end),' ')
        fixedtext = fixedtext(1:end-1);
    else
        fixedtext = [fixedtext,' '];
        break;
    end
end

function c = file_in_category(filename, category)
h = help(filename);
c = strfind(h, ['Category: ' category]);