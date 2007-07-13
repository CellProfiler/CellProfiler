function CPClusterSingle(batchfile,BatchSetBeingAnalyzed,OutputFolder,BatchFilePrefix)

%%% Must list all CellProfiler modules here
%#function CPwritemeasurements CPlogo CPconvertsql CPmsgbox CPretrievemediafilenames CPrelateobjects CPselectmodules CPaddmeasurements CPtextdisplaybox CPthresh_tool CPselectoutputfiles CPnanmean CPfigure CPrescale CPinputdlg CPimagesc CPclearborder CPlabel2rgb CPimagetool CPwarndlg CPhistbins CPcontrolhistogram CPtextpipe CPdilatebinaryobjects CPsigmoid CPimread CPwhichmodule CPthreshold CPmakegrid CPlistdlg CPcompilesubfunction CPsmooth CPresizefigure CPgetfeature CPwaitbar CPrgsmartdilate CPretrieveimage CPnanstd CPerrordlg CPaverageimages CPcd CPblkproc CPnanmedian CPplotmeasurement CPhelpdlg CPquestdlg CPnlintool Tile SaveImages SplitOrSpliceMovie Morph CalculateStatistics SpeedUpCellProfiler FilterByObjectMeasurement PlaceAdjacent LoadImages ConvertToImage ApplyThreshold MeasureTexture GrayToColor Align MeasureObjectAreaShape MeasureObjectNeighbors CreateBatchFiles MeasureImageGranularity ExportToDatabase DistinguishPixelLabels ClassifyObjects Crop InvertIntensity RescaleIntensity ExpandOrShrink IdentifyTertiarySubregion CorrectIllumination_Apply CalculateRatios CreateWebPage Restart SendEmail CorrectIllumination_New Combine ExportToExcel LoadText MeasureCorrelation ClassifyObjectsByTwoMeasurements DisplayImageHistogram CalculateMath DefineGrid Smooth MaskImage Relate Subtract CorrectIllumination_Calculate RenameOrRenumberFiles Resize IdentifySecondary OverlayOutlines MeasureObjectIntensity DisplayHistogram ColorToGray Average IdentifyObjectsInGrid SmoothKeepingEdges LoadSingleImage IdentifyPrimManual DisplayMeasurement MeasureImageIntensity SubtractBackground Flip MeasureImageSaturationBlur IdentifyPrimAutomatic Exclude DisplayDataOnImage Rotate FindEdges DisplayGridInfo MeasureImageAreaOccupied strel


try
    state = warning('off', 'all');
    load(batchfile);
    warning(state);
catch
    reportBatchError(['Batch Error: Loading batch file (' batchfile ')']);
end

% If we get the argument 'all', print out the imageset numbers for
% each set that still needs to run.
if strcmp(BatchSetBeingAnalyzed, 'all'),
    if exist(OutputFolder) ~= 7, % directory
        error(sprintf('Output folder %s does not exist.', OutputFolder));
        quit;
    end
    for imageset = 2:handles.Current.NumberOfImageSets,
        if ~ exist(sprintf('%s/%s%d_DONE',OutputFolder,BatchFilePrefix,imageset)),
            disp(imageset);
        end
    end
    quit;
end

disp('past loop');

foo = strel('disk', 7, 0)

% arguments come in as strings, convert to integer
BatchSetBeingAnalyzed = str2num(BatchSetBeingAnalyzed);

tic
handles.Current.BatchInfo.Start = BatchSetBeingAnalyzed;
handles.Current.BatchInfo.End = BatchSetBeingAnalyzed;

disp(sprintf('Analyzing set %d.', BatchSetBeingAnalyzed));
handles.Current.SetBeingAnalyzed = BatchSetBeingAnalyzed;

disp('Pipeline:')
for SlotNumber = 1:handles.Current.NumberOfModules,
    ModuleNumberAsString = sprintf('%02d', SlotNumber);
    ModuleName = char(handles.Settings.ModuleNames(SlotNumber));
    disp(sprintf('     module %d - %s', SlotNumber, ModuleName));
end

for SlotNumber = 1:handles.Current.NumberOfModules,
    ModuleNumberAsString = sprintf('%02d', SlotNumber);
    ModuleName = char(handles.Settings.ModuleNames(SlotNumber));
    disp(sprintf('executing module %d - %s', SlotNumber, ModuleName));
    handles.Current.CurrentModuleNumber = ModuleNumberAsString;
    try
        handles = feval(ModuleName,handles);
    catch
        reportBatchError(['Batch Error: ' ModuleName]);
    end
    disp('done');
    toc
    if SlotNumber == BatchSetBeingAnalyzed,
        quit;
    end
end
toc

% handles.Pipeline = [];
% save(sprintf('%s%d_OUT',BatchFilePrefix,BatchSetBeingAnalyzed),'handles');
OutputFileName = sprintf('%s/%s%d_DONE.mat',OutputFolder,BatchFilePrefix,BatchSetBeingAnalyzed);
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
rethrow(errorinfo);
quit;
