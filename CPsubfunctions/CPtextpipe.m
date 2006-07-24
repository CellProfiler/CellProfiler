function CPtextpipe(handles,ExportInfo,RawFilename,RawPathname)

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
%
% Website: http://www.cellprofiler.org
%
% $Revision: 2802 $

if ~isfield(handles.Settings,'VariableValues') || ~isfield(handles.Settings,'VariableInfoTypes') || ~isfield(handles.Settings,'ModuleNames')
    CPmsgbox('You do not have a pipeline loaded!');
    return
end

VariableValues = handles.Settings.VariableValues;
VariableInfoTypes = handles.Settings.VariableInfoTypes;
ModuleNames = handles.Settings.ModuleNames;
ModuleNamedotm = [char(ModuleNames(1)) '.m'];
%%% Check for location of m-files
if exist(ModuleNamedotm,'file')
    if ~isdeployed
        FullPathname = which(ModuleNamedotm);
        [PathnameModules, filename, ext, versn] = fileparts(FullPathname); % Why is filename, ext, and versn taken out? They aren't used later on...
    else
        PathnameModules = handles.Preferences.DefaultModuleDirectory;
    end
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

if isstruct(ExportInfo)
    if ExportInfo.ProcessInfoExtension(1) ~= '.';
        ExportInfo.ProcessInfoExtension = ['.',ExportInfo.ProcessInfoExtension];
    end
    filename = [ExportInfo.ProcessInfoFilename ExportInfo.ProcessInfoExtension];
    fid = fopen(fullfile(RawPathname,filename),'w');
    if fid == -1
        error(sprintf('Cannot create the output file %s. There might be another program using a file with the same name, or you do not have permission to write this file.',filename));
    end
else
    %Prompt what to save file as, and where to save it.
    [filename,SavePathname] = uiputfile(fullfile(handles.Current.DefaultOutputDirectory,'*.txt'), 'Save Settings As...');
    if filename == 0
        CPmsgbox('You have canceled the option to save the pipeline as a text file, but your pipeline will still be saved in .mat format.');
        return
    end
    fid = fopen(fullfile(SavePathname,filename),'w');
    if fid == -1
        error(sprintf('Cannot create the output file %s. There might be another program using a file with the same name.',filename));
    end
end

% make sure # of modules equals number of variable rows.
VariableSize = size(VariableValues);
if VariableSize(1) ~= max(size(ModuleNames))
    error('Your settings are not valid.')
end

if ~isstr(RawPathname) || ~isstr(RawFilename) || ~isstruct(ExportInfo)
    fprintf(fid,['Saved Pipeline, in file ' filename ', Saved on ' date '\n']);
else
    fprintf(fid,'Processing info for file: %s\n',fullfile(RawPathname, RawFilename));
    fprintf(fid,'Processed (start time): %s\n\n',handles.Current.TimeStarted);
    NbrOfProcessedSets = length(handles.Measurements.Image.FileNames);
    fprintf(fid,'Number of processed image sets: %d\n',NbrOfProcessedSets);
end

fprintf(fid,['\nPixel Size: ' handles.Settings.PixelSize '\n']);
fprintf(fid,'\nPipeline:\n');
for module = 1:length(handles.Settings.ModuleNames)
    fprintf(fid,'\t%s\n',handles.Settings.ModuleNames{module});
end
RevNums = handles.Settings.VariableRevisionNumbers;
% Loop for each module loaded.
for p = 1:VariableSize(1)
    Module = [char(ModuleNames(p))];
    fprintf(fid,['\nModule #' num2str(p) ': ' Module ' revision - ' num2str(RevNums(p)) '\n']);
    ModuleNamedotm = [Module '.m'];
    fid2=fopen(fullfile(PathnameModules,ModuleNamedotm));
    while 1
        output = fgetl(fid2);
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
        if strncmp(output,'%filenametextVAR',16)
            displayval = output(21:end);
            istr = output(17:18);
            i = str2num(istr);
            VariableDescriptions(i) = {displayval};
        end
    end
    fclose(fid2);
    % Loop for each variable in the module.
    for q = 1:length(handles.VariableBox{p})
        VariableDescrip = char(VariableDescriptions(q));
        try
            VariableVal = char(VariableValues(p,q));
        catch
            VariableVal = '  ';

        end
        fprintf(fid,['    ' VariableDescrip '    ' VariableVal '\n']);
    end
end
%% Save to a .txt file.
fclose(fid);
if ~isstr(RawPathname) || ~isstr(RawFilename)
    CPhelpdlg('The pipeline .txt file has been written.');
end