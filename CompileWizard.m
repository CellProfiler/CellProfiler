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
    else
        ToolList = ['''',ToolName(1:end-2),''''];
    end
end
fprintf(fid,['handles.Current.ImageToolsFilenames = {',ToolList,'};\n']);
fprintf(fid,'handles.Current.ImageToolHelp = ToolHelp;\n\n');

clear ToolList

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
    else
        ToolList = ['''',ToolName(1:end-2),''''];
    end
end
fprintf(fid,['handles.Current.DataToolsFilenames = {',ToolList,'};\n']);
fprintf(fid,'handles.Current.DataToolHelp = ToolHelp;\n\n');

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
fprintf(fid,['handles.Current.HelpFilenames = {',ToolList,'};\n']);
fprintf(fid,'handles.Current.Help = ToolHelp;\n');

fclose(fid);

%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTIONS %%%
%%%%%%%%%%%%%%%%%%%%

function fixedtext = fixthistext(text)

fixedtext = strrep(text,'''','''''');
fixedtext = strrep(fixedtext,'\','\\\\');
fixedtext = strrep(fixedtext,'%','%%%%');