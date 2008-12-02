function CPCluster(batchfile,StartingSet,EndingSet,OutputFolder,BatchFilePrefix,WriteMatFiles,KeepAlive)
% $Revision$

%%% Must list all CellProfiler modules here
%% DO NOT CHANGE THE LINE BELOW
%%% BuildCellProfiler: INSERT FUNCTIONS HERE

tic

try
    state = warning('off', 'all'); % necessary to get around pipelines that complain about missing functions.
    load(batchfile);
    warning(state);
catch
    reportBatchError(['Batch Error: Loading batch file (' batchfile ')']);
end

try
    % r2008b saves the preferences as it exits and complains
    % so we put the preferences somewhere random
    prefdir = OutputFolder;
catch
    warning('Failed to set the preferences directory')
end

% arguments come in as strings, convert to integer
StartingSet = str2num(StartingSet);
EndingSet = str2num(EndingSet);

% this is necessary for some modules (e.g., ExportToDatabase) to work correctly.
handles.Current.BatchInfo.Start = StartingSet;
handles.Current.BatchInfo.End = EndingSet;

for BatchSetBeingAnalyzed = StartingSet:EndingSet,
    t_set_start = toc;
    disp(sprintf('Analyzing set %d.', BatchSetBeingAnalyzed));
    handles.Current.SetBeingAnalyzed = BatchSetBeingAnalyzed;

    if (BatchSetBeingAnalyzed == StartingSet),
        disp('Pipeline:')
        for SlotNumber = 1:handles.Current.NumberOfModules,
            ModuleNumberAsString = sprintf('%02d', SlotNumber);
            ModuleName = char(handles.Settings.ModuleNames(SlotNumber));
            disp(sprintf('     module %d - %s', SlotNumber, ModuleName));
        end
    end


    for SlotNumber = 1:handles.Current.NumberOfModules,
        % Signal that we're alive
        system(KeepAlive);

        t_start = toc;
        ModuleNumberAsString = sprintf('%02d', SlotNumber);
        ModuleName = char(handles.Settings.ModuleNames(SlotNumber));
        disp(sprintf('  executing module %d - %s', SlotNumber, ModuleName));
        handles.Current.CurrentModuleNumber = ModuleNumberAsString;
        try
            handles = feval(ModuleName,handles);
        catch
            reportBatchError(['Batch Error: ' ModuleName]);
        end
        t_end = toc;
        disp(sprintf('    %f seconds', t_end - t_start));
     end
     disp(sprintf('  %f seconds for image set %d.', toc - t_set_start, BatchSetBeingAnalyzed));
end

t_tot = toc;
disp(sprintf('All sets analyzed in %f seconds (%f per image set)', t_tot, t_tot / (EndingSet - StartingSet + 1)));

if strcmp(WriteMatFiles, 'yes'),
    handles.Pipeline = [];
    OutputFileName = sprintf('%s/%s%d_to_%d_OUT.mat',OutputFolder,BatchFilePrefix,StartingSet,EndingSet);
    save(OutputFileName,'handles');
end

OutputFileName = sprintf('%s/%s%d_to_%d_DONE.mat',OutputFolder,BatchFilePrefix,StartingSet,EndingSet);
save(OutputFileName,'BatchSetBeingAnalyzed');
disp(sprintf('Created %s',OutputFileName));
quit;

function reportBatchError(errorstring)
errorinfo = lasterror;
if isfield(errorinfo, 'stack'),
    try
        stackinfo = errorinfo.stack(1,1);
        ExtraInfo = [' (file: ', stackinfo.file, ' function: ', stackinfo.name, ' line: ', num2str(stackinfo.line), ')'];
    catch
        %%% The line stackinfo = errorinfo.stack(1,1); will fail if the
        %%% errorinfo.stack is empty, which sometimes happens during
        %%% debugging, I think. So we catch it here.
        ExtraInfo = '';
    end
end
disp([errorstring ': ' lasterr]);
disp(ExtraInfo);
quit;
