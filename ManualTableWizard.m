function ManualTableWizard

fid = fopen('ManualTableWizardText.xls','w');

Modulefilelist = dir('Modules/*.m');
for i=1:length(Modulefilelist)
    name=Modulefilelist(i).name;
    name=name(1:end-2);
    fprintf(fid,[name,'\t']);
    body = char(strread(help(Modulefilelist(i).name),'%s','delimiter','','whitespace',''));
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