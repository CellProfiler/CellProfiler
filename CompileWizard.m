function CompileWizard
% CompileWizard
% This function, when run in the default CellProfiler path, will produce a
% file with help information stored in variables to be used in the compiled
% version of CellProfiler.

fid = fopen('CompileWizardText.m','w');

ImageToolfilelist = dir('ImageTools/*.m');
fprintf(fid,'%%%%%% IMAGE TOOL HELP\n');
fprintf(fid,'ToolHelpInfo = ''Help information from individual image tool files, which are Matlab m-files located within the ImageTools directory:'';\n\n');
for i=1:length(ImageToolfilelist)
    ToolName = ImageToolfilelist(i).name;
    fprintf(fid,[ToolName(1:end-2),'Help = sprintf([...\n']);
    body = char(strread(help(ImageToolfilelist(i).name),'%s','delimiter','','whitespace',''));
    for j = 1:size(body,1)
        fixedtext = fixthistext(body(j,:));
        newtext = ['''',fixedtext,'\\n''...\n'];
        fprintf(fid,newtext);
    end
    fprintf(fid,']);\n\n');
    fprintf(fid,['ToolHelp{',num2str(i),'} = [ToolHelpInfo, ''-----------'' 10 ',[ToolName(1:end-2),'Help'],'];\n\n']);
    if exist('ToolList','var')
        ToolList = [ToolList, ' ''',ToolName(1:end-2),''''];
        ToolListNoQuotes = [ToolListNoQuotes,' ',ToolName(1:end-2)];
    else
        ToolList = ['''',ToolName(1:end-2),''''];
        ToolListNoQuotes = ToolName(1:end-2);
    end
end
fprintf(fid,['handles.Current.ImageToolsFilenames = {''Image tools'' ',ToolList,'};\n']);
fprintf(fid,'handles.Current.ImageToolHelp = ToolHelp;\n');
fprintf(fid,['%%#function ',ToolListNoQuotes,'\n\n']);

clear ToolList ToolListNoQuotes

DataToolfilelist = dir('DataTools/*.m');
fprintf(fid,'%%%%%% DATA TOOL HELP\n');
fprintf(fid,'ToolHelpInfo = ''Help information from individual data tool files, which are Matlab m-files located within the DataTools directory:'';\n\n');
for i=1:length(DataToolfilelist)
    ToolName = DataToolfilelist(i).name;
    fprintf(fid,[ToolName(1:end-2),'Help = sprintf([...\n']);
    body = char(strread(help(DataToolfilelist(i).name),'%s','delimiter','','whitespace',''));
    for j = 1:size(body,1)
        fixedtext = fixthistext(body(j,:));
        newtext = ['''',fixedtext,'\\n''...\n'];
        fprintf(fid,newtext);
    end
    fprintf(fid,']);\n\n');
    fprintf(fid,['ToolHelp{',num2str(i),'} = [ToolHelpInfo, ''-----------'' 10 ',[ToolName(1:end-2),'Help'],'];\n\n']);
    if exist('ToolList','var')
        ToolList = [ToolList, ' ''',ToolName(1:end-2),''''];
        ToolListNoQuotes = [ToolListNoQuotes,' ',ToolName(1:end-2)];
    else
        ToolList = ['''',ToolName(1:end-2),''''];
        ToolListNoQuotes = ToolName(1:end-2);
    end
end
fprintf(fid,['handles.Current.DataToolsFilenames = {''Data tools'' ',ToolList,'};\n']);
fprintf(fid,'handles.Current.DataToolHelp = ToolHelp;\n');
fprintf(fid,['%%#function ',ToolListNoQuotes,'\n\n']);

clear ToolList ToolListNoQuotes

DataToolfilelist = dir('Modules/*.m');
fprintf(fid,'%%%%%% MODULES HELP\n');
for i=1:length(DataToolfilelist)
    ToolName = DataToolfilelist(i).name;
    fprintf(fid,[ToolName(1:end-2),'Help = sprintf([...\n']);
    body = char(strread(help(DataToolfilelist(i).name),'%s','delimiter','','whitespace',''));
    for j = 1:size(body,1)
        fixedtext = fixthistext(body(j,:));
        newtext = ['''',fixedtext,'\\n''...\n'];
        fprintf(fid,newtext);
    end
    fprintf(fid,']);\n\n');
    fprintf(fid,['ToolHelp{',num2str(i),'} = ',ToolName(1:end-2),'Help;\n\n']);
    if exist('ToolList','var')
        ToolList = [ToolList, ' ''',ToolName(1:end-2),''''];
    else
        ToolList = ['''',ToolName(1:end-2),''''];
    end
end
fprintf(fid,['handles.Current.ModuleFilenames = {',ToolList,'};\n']);
fprintf(fid,'handles.Current.ModuleHelp = ToolHelp;\n\n');

clear ToolList

Helpfilelist = dir('Help/*.m');
fprintf(fid,'%%%%%% HELP\n');
for i=1:length(Helpfilelist)
    ToolName = Helpfilelist(i).name;
    fprintf(fid,['ToolHelp{',num2str(i),'} = sprintf([...\n']);
    body = char(strread(help(Helpfilelist(i).name),'%s','delimiter','','whitespace',''));
    for j = 1:size(body,1)
        fixedtext = strrep(body(j,:),'''','''''');
        fixedtext = strrep(fixedtext,'\','\\\\');
        fixedtext = strrep(fixedtext,'%','%%%%');
        newtext = ['''',fixedtext,'\\n''...\n'];
        fprintf(fid,newtext);
    end
    fprintf(fid,']);\n\n');
    if exist('ToolList','var')
        ToolList = [ToolList, ' ''',ToolName(1:end-2),''''];
    else
        ToolList = ['''',ToolName(1:end-2),''''];
    end
end
fprintf(fid,['handles.Current.HelpFilenames = {''Help'' ',ToolList,'};\n']);
fprintf(fid,'handles.Current.Help = ToolHelp;\n\n');

clear ToolList

Modulefilelist = dir('Modules/*.m');
fprintf(fid,'%%%%%% Module List\n');
fprintf(fid,'%%#function');
FileProcessingFiles ={};
PreProcessingFiles={};
ObjectProcessingFiles={};
MeasurementFiles={};
OtherFiles={};
for i=1:length(Modulefilelist)
    name=Modulefilelist(i).name;
    name=name(1:end-2);
    fprintf(fid,[' ',name]);
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

    %%% CODE TO WRITE TEXT FILES OF MODULES
    fid2=fopen(fullfile(pwd,'Modules',Modulefilelist(i).name));
    fid3=fopen(fullfile(pwd,'Modules',[name,'.txt']),'w');
    while 1;
        output = fgetl(fid2); if ~ischar(output); break; end;
        if strncmp(output,'%Start VariableSet',18)
            fprintf(fid3,[fixthistext2(output),'\n']);
        elseif strncmp(output,'%End VariableSet',16)
            fprintf(fid3,[fixthistext2(output),'\n']);
        elseif strncmp(output,'%defaultVAR',11)
            fprintf(fid3,[fixthistext2(output),'\n']);
        elseif strncmp(output,'%choiceVAR',10)
            fprintf(fid3,[fixthistext2(output),'\n']);
        elseif strncmp(output,'%textVAR',8)
            fprintf(fid3,[fixthistext2(output),'\n']);
        elseif strncmp(output,'%pathnametextVAR',16)
            fprintf(fid3,[fixthistext2(output),'\n']);
        elseif strncmp(output,'%filenametextVAR',16)
            fprintf(fid3,[fixthistext2(output),'\n']);
        elseif strncmp(output,'%infotypeVAR',12)
            fprintf(fid3,[fixthistext2(output),'\n']);
        elseif strncmp(output,'%inputtypeVAR',13)
            fprintf(fid3,[fixthistext2(output),'\n']);
        elseif strncmp(output,'%%%VariableRevisionNumber',25)
            fprintf(fid3,[fixthistext2(output),'\n']);
        end
    end
    fclose(fid2);
    fclose(fid3);
    %%% END CODE TO WRITE TEXT FILES OF MODULES
end
fprintf(fid,'\n\nCategoryList = {''File Processing'' ''Image Processing'' ''Object Processing'' ''Measurement'' ''Other''};\n');

fprintf(fid,'FileProcessingFiles = {');
for i=1:length(FileProcessingFiles)
    fprintf(fid,['''',FileProcessingFiles{i},''' ']);
end
fprintf(fid,'};\n');

fprintf(fid,'PreProcessingFiles = {');
for i=1:length(PreProcessingFiles)
    fprintf(fid,['''',PreProcessingFiles{i},''' ']);
end
fprintf(fid,'};\n');

fprintf(fid,'ObjectProcessingFiles = {');
for i=1:length(ObjectProcessingFiles)
    fprintf(fid,['''',ObjectProcessingFiles{i},''' ']);
end
fprintf(fid,'};\n');

fprintf(fid,'MeasurementFiles = {');
for i=1:length(MeasurementFiles)
    fprintf(fid,['''',MeasurementFiles{i},''' ']);
end
fprintf(fid,'};\n');

fprintf(fid,'OtherFiles = {');
for i=1:length(OtherFiles)
    fprintf(fid,['''',OtherFiles{i},''' ']);
end
fprintf(fid,'};\n');

fprintf(fid,'set(AddModuleWindowHandles.ModuleCategoryListBox,''String'',CategoryList,''Value'',[])\n');
fprintf(fid,'set(AddModuleWindowHandles.ModulesListBox,''String'',FileProcessingFiles,''Value'',[])\n');
fprintf(fid,'AddModuleWindowHandles.ModuleStrings{1} = FileProcessingFiles;\n');
fprintf(fid,'AddModuleWindowHandles.ModuleStrings{2} = PreProcessingFiles;\n');
fprintf(fid,'AddModuleWindowHandles.ModuleStrings{3} = ObjectProcessingFiles;\n');
fprintf(fid,'AddModuleWindowHandles.ModuleStrings{4} = MeasurementFiles;\n');
fprintf(fid,'AddModuleWindowHandles.ModuleStrings{5} = OtherFiles;\n');
fprintf(fid,'guidata(AddModuleWindowHandles.AddModuleWindow,AddModuleWindowHandles);\n');

fclose(fid);

%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTIONS %%%
%%%%%%%%%%%%%%%%%%%%

function fixedtext = fixthistext(text)

fixedtext = strrep(text,'''','''''''''');
fixedtext = strrep(fixedtext,'\','\\\\');
fixedtext = strrep(fixedtext,'%','%%%%');

function c = file_in_category(filename, category)
h = help(filename);
c = strfind(h, ['Category: ' category]);

function fixedtext = fixthistext2(text)

fixedtext = strrep(text,'''','''''');
fixedtext = strrep(fixedtext,'\','\\');
fixedtext = strrep(fixedtext,'%','%%');