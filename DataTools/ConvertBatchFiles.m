function ConvertBatchFiles(handles)
%
% ConvertBatchFiles is a temporary tool that converts
% batch files to regular CellProfiler output files.
% It does so by removing empty entries in the
% handles.Measurements structure. It saves new files
% with 'Converted' as a prefix in the filename.
%

%%% Let the user select one output file to indicate the directory
[ExampleFile, Pathname] = uigetfile('*.mat','Select one Batch output file');
if ~Pathname,return,end

%%% Get all files with .mat extension in the chosen directory.
%%% If the selected file name contains an 'OUT', it is assumed
%%% that all interesting files contain an 'OUT'.
AllFiles = dir(Pathname);                                                        % Get all file names in the chosen directory
AllFiles = {AllFiles.name};                                                      % Cell array with file names
files = AllFiles(~cellfun('isempty',strfind(AllFiles,'.mat')));                  % Keep files that has a .mat extension
if strfind(ExampleFile,'OUT')
    files = files(~cellfun('isempty',strfind(files,'OUT')));                     % Keep files with an 'OUT' in the name
end

%%% Let the user select the files to be converted
[selection,ok] = listdlg('liststring',files,'name','Convert Batch Files',...
    'PromptString','Select files to convert. Use Ctrl+Click or Shift+Click.','listsize',[300 500]);
if ~ok, return, end
files = files(selection);

%%% Open the files, remove empty entries in the handles.Measurements structure
%%% and store the files.
waitbarhandle = waitbar(0,'');
for fileno = 1:length(files)
    waitbar(fileno/length(files),waitbarhandle,sprintf('Converting %s.',files{fileno}));drawnow
    load(fullfile(Pathname, files{fileno}));
    firstfields = fieldnames(handles.Measurements);
    for i = 1:length(firstfields)
        secondfields = fieldnames(handles.Measurements.(firstfields{i}));
        for j = 1:length(secondfields)
            if iscell(handles.Measurements.(firstfields{i}).(secondfields{j}))
                index = ~cellfun('isempty',handles.Measurements.(firstfields{i}).(secondfields{j}));
                if sum(index==0) > 0       % There exist empty cells, remove them
                    index(1) = 0;          % First set is a dummy set
                    handles.Measurements.(firstfields{i}).(secondfields{j}) = ...
                        handles.Measurements.(firstfields{i}).(secondfields{j})(index);
                end
            end
        end
    end
    [ignore,Attributes] = fileattrib(fullfile(Pathname,['Converted',files{fileno}]));
    if Attributes.UserWrite == 0
        error(['You do not have permission to write ',fullfile(Pathname,['Converted',files{fileno}]),'!']);
    else
        save(fullfile(Pathname,['Converted',files{fileno}]),'handles');
    end
end
close(waitbarhandle)
CPmsgbox('Converting is completed.')                   
        
    
