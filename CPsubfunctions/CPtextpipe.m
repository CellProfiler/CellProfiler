function CPtextpipe(handles)

if ~isfield(handles.Settings,'VariableValues') || ~isfield(handles.Settings,'VariableInfoTypes') || ~isfield(handles.Settings,'ModuleNames')
    CPmsgbox('You do not have a pipeline loaded!');
    return
end

VariableValues = handles.Settings.VariableValues;
VariableInfoTypes = handles.Settings.VariableInfoTypes;
ModuleNames = handles.Settings.ModuleNames;
ModuleNamedotm = [char(ModuleNames(1)) '.m'];
%Prompt what to save file as, and where to save it.
if exist(ModuleNamedotm,'file')
    FullPathname = which(ModuleNamedotm);
    [PathnameModules, filename, ext, versn] = fileparts(FullPathname);
else
    %%% If the module.m file is not on the path, it won't be
    %%% found, so ask the user where the modules are.
    PathnameModules = uigetdir('','Please select directory where modules are located');
    pause(.1);
    figure(handles.figure1);
    if PathnameModules == 0
        return
    end
end
[filename,SavePathname] = uiputfile(fullfile(handles.Current.DefaultOutputDirectory,'*.txt'), 'Save Settings As...');
if filename == 0
    CPmsgbox('You have canceled the option to save the pipeline as a text file, but your pipeline will still be saved in .mat format.');
    return
end
% make sure # of modules equals number of variable rows.
VariableSize = size(VariableValues);
if VariableSize(1) ~= max(size(ModuleNames))
    error('Your settings are not valid.')
end
display = ['Saved Pipeline, in file ' filename ', Saved on ' date];
% Loop for each module loaded.
for p = 1:VariableSize(1)
    Module = [char(ModuleNames(p))];
    display = strvcat(display, ['Module #' num2str(p) ': ' Module]);
    ModuleNamedotm = [Module '.m'];
    fid=fopen(fullfile(PathnameModules,ModuleNamedotm));
    while 1
        output = fgetl(fid);
        if ~ischar(output), break, end
        if strncmp(output,'%textVAR',8)
            displayval = output(13:end);
            istr = output(9:10);
            i = str2num(istr);
            VariableDescriptions(i) = {displayval};
        end
        if strncmp(output,'%pathnametextVAR',16)
            displayval = output(21:end);
            istr = output(17:18);
            i = str2num(istr);
            VariableDescriptions(i) = {displayval};
        end
    end
    fclose(fid);
    % Loop for each variable in the module.
    for q = 1:length(handles.VariableBox{p})
        VariableDescrip = char(VariableDescriptions(q));
        try
            VariableVal = char(VariableValues(p,q));
        catch
            VariableVal = '  ';

        end
        display =strvcat(display, ['    ' VariableDescrip '    ' VariableVal]);
    end
end
%% tack on rest of Settings information.
PixelSizeDisplay = ['Pixel Size: ' handles.Settings.PixelSize];
RevisionNumbersDisplay = ['Variable Revision Numbers: ' num2str(handles.Settings.VariableRevisionNumbers)];
display = strvcat(display, PixelSizeDisplay, RevisionNumbersDisplay);
%% Save to a .txt file.
dlmwrite(fullfile(SavePathname,filename), display, 'delimiter', '');
helpdlg('The pipeline .txt file has been written.');