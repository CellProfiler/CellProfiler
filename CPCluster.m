function CPCluster(batchfile,StartingSet,EndingSet,OutputFolder,BatchFilePrefix,WriteMatFiles,KeepAlive)
% $Revision$

%%% Must list all CellProfiler modules here
%#function Align ApplyThreshold Average CalculateImageOverlap CalculateMath CalculateRatios CalculateStatistics ClassifyObjects ClassifyObjectsByTwoMeasurements ColorToGray Combine ConvertToImage CorrectIllumination_Apply CorrectIllumination_Calculate CreateBatchFiles CreateWebPage Crop DefineGrid DisplayDataOnImage DisplayGridInfo DisplayHistogram DisplayImageHistogram DisplayMeasurement EditObjectsManually Exclude ExpandOrShrink ExportToDatabase ExportToExcel FilterByObjectMeasurement FindEdges Flip GrayToColor GroupMovieFrames IdentifyObjectsInGrid IdentifyPrimAutomatic IdentifyPrimLoG IdentifyPrimManual IdentifySecondary IdentifyTertiarySubregion ImageMath InvertForPrinting InvertIntensity KeepLargestObject LabelImages LoadImages LoadSingleImage LoadText MaskImage MeasureCorrelation MeasureImageAreaOccupied MeasureImageGranularity MeasureImageIntensity MeasureImageQuality MeasureObjectAreaShape MeasureObjectIntensity MeasureObjectNeighbors MeasureRadialDistribution MeasureTexture Morph Multiply OverlayOutlines PauseCellProfiler PlaceAdjacent Relate RenameOrRenumberFiles RescaleIntensity Resize Restart Rotate SaveImages SendEmail SmoothKeepingEdges SmoothOrEnhance SpeedUpCellProfiler SplitIntoContiguousObjects SplitOrSpliceMovie Subtract SubtractBackground Tile TrackObjects UnifyObjects 

tic

try
    state = warning('off', 'all'); % necessary to get around pipelines that complain about missing functions.
    load(batchfile);
    warning(state);
catch
    reportBatchError(['Batch Error: Loading batch file (' batchfile ')']);
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
