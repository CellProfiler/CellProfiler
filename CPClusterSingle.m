function CPClusterSingle(batchfile,StartingSet,EndingSet,OutputFolder,BatchFilePrefix,WriteMatFiles,KeepAlive)

%%% Must list all CellProfiler modules here
%#function CPwritemeasurements CPlogo CPconvertsql CPmsgbox CPretrievemediafilenames CPrelateobjects CPselectmodules CPaddmeasurements CPtextdisplaybox CPthresh_tool CPselectoutputfiles CPnanmean CPfigure CPrescale CPinputdlg CPimagesc CPclearborder CPlabel2rgb CPimagetool CPwarndlg CPhistbins CPcontrolhistogram CPtextpipe CPdilatebinaryobjects CPsigmoid CPimread CPwhichmodule CPthreshold CPmakegrid CPlistdlg CPcompilesubfunction CPsmooth CPresizefigure CPgetfeature CPwaitbar CPrgsmartdilate CPretrieveimage CPnanstd CPerrordlg CPaverageimages CPcd CPblkproc CPnanmedian CPplotmeasurement CPhelpdlg CPquestdlg CPnlintool Tile SaveImages SplitOrSpliceMovie Morph CalculateStatistics SpeedUpCellProfiler FilterByObjectMeasurement PlaceAdjacent LoadImages ConvertToImage ApplyThreshold MeasureTexture GrayToColor Align MeasureObjectAreaShape MeasureObjectNeighbors CreateBatchFiles MeasureImageGranularity ExportToDatabase DistinguishPixelLabels ClassifyObjects Crop InvertIntensity RescaleIntensity ExpandOrShrink IdentifyTertiarySubregion CorrectIllumination_Apply CalculateRatios CreateWebPage Restart SendEmail CorrectIllumination_New Combine ExportToExcel LoadText MeasureCorrelation ClassifyObjectsByTwoMeasurements DisplayImageHistogram CalculateMath DefineGrid Smooth MaskImage Relate Subtract CorrectIllumination_Calculate RenameOrRenumberFiles Resize IdentifySecondary OverlayOutlines MeasureObjectIntensity DisplayHistogram ColorToGray Average IdentifyObjectsInGrid SmoothKeepingEdges LoadSingleImage IdentifyPrimManual DisplayMeasurement MeasureImageIntensity SubtractBackground Flip MeasureImageSaturationBlur IdentifyPrimAutomatic Exclude DisplayDataOnImage Rotate FindEdges DisplayGridInfo MeasureImageAreaOccupied

tic

try
    state = warning('off', 'all'); % necessary to get around pipelines that complain about missing functions.
    load(batchfile);
    warning(state);
catch
    reportBatchError(['Batch Error: Loading batch file (' batchfile ')']);
end


% If we get the argument 'all', use EndingSet as a step size and print
% out the imageset numbers for each set that still needs to run.
if strcmp(StartingSet, 'all'),
    if exist(OutputFolder) ~= 7, % directory
        error(sprintf('Output folder %s does not exist.', OutputFolder));
        quit;
    end
    StepSize = str2num(EndingSet);
    Starts = 2:StepSize:handles.Current.NumberOfImageSets;
    Ends = Starts + StepSize - 1;
    Ends(end) = handles.Current.NumberOfImageSets;
    for imagesets = [Starts ; Ends],
        if ~ exist(sprintf('%s/%s%d_to_%d_DONE.mat',OutputFolder,BatchFilePrefix,imagesets(1),imagesets(2))),
            disp(sprintf('%d %d', imagesets(1), imagesets(2)));
        end
    end
    quit;
end

% arguments come in as strings, convert to integer
StartingSet = str2num(StartingSet);
EndingSet = str2num(EndingSet);

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
    OutputFileName = sprintf('%s/%s%d_to_%d_DATA.mat',OutputFolder,BatchFilePrefix,StartingSet,EndingSet);
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
