function CPCluster(batchfile,clusterfile)

%%% Must list all CellProfiler modules here
%#function Align ApplyThreshold Average CalculateMath CalculateRatios CalculateStatistics ClassifyObjects ClassifyObjectsByTwoMeasurements ColorToGray Combine ConvertToImage CorrectIllumination_Apply CorrectIllumination_Calculate CreateBatchFiles CreateWebPage Crop DefineGrid DisplayDataOnImage DisplayGridInfo DisplayHistogram DisplayImageHistogram DisplayMeasurement DistinguishPixelLabels Exclude ExpandOrShrink ExportToDatabase ExportToExcel FilterByObjectMeasurement FindEdges Flip GrayToColor IdentifyObjectsInGrid IdentifyPrimAutomatic IdentifyPrimManual IdentifySecondary IdentifyTertiarySubregion InvertIntensity LoadImages LoadSingleImage LoadText MaskImage MeasureCorrelation MeasureImageAreaOccupied MeasureImageGranularity MeasureImageIntensity MeasureImageSaturationBlur MeasureObjectAreaShape MeasureObjectIntensity MeasureObjectNeighbors MeasureTexture Morph OverlayOutlines PlaceAdjacent Relate RenameOrRenumberFiles RescaleIntensity Resize Restart Rotate SaveImages SendEmail Smooth SpeedUpCellProfiler SplitOrSpliceMovie Subtract SubtractBackground Tile CPaddmeasurements CPaverageimages CPblkproc CPcd CPclearborder CPcompilesubfunction CPcontrolhistogram CPconvertsql CPdilatebinaryobjects CPerrordlg CPfigure CPgetfeature CPhelpdlg CPhistbins CPimagesc CPimagetool CPimread CPinputdlg CPlabel2rgb CPlistdlg CPlogo CPmakegrid CPmsgbox CPnanmean CPnanmedian CPnanstd CPnlintool CPplotmeasurement CPquestdlg CPrelateobjects CPrescale CPresizefigure CPretrieveimage CPretrievemediafilenames CPrgsmartdilate CPselectmodules CPselectoutputfiles CPsigmoid CPsmooth CPtextdisplaybox CPtextpipe CPthresh_tool CPthreshold CPwaitbar CPwarndlg CPwhichmodule CPwritemeasurements

try
    load(batchfile);
catch
    reportBatchError(['Batch Error: Loading batch file (' batchfile ')']);
end

try
    load(clusterfile);
catch
    reportBatchError(['Batch Error: Loading cluster file (' clusterfile ')']);
end


tic
handles.Current.BatchInfo.Start = cluster.StartImage;
handles.Current.BatchInfo.End = cluster.EndImage;
for BatchSetBeingAnalyzed = cluster.StartImage:cluster.EndImage,
    disp(sprintf('Analyzing set %d', BatchSetBeingAnalyzed));
    toc;
    handles.Current.SetBeingAnalyzed = BatchSetBeingAnalyzed;
    for SlotNumber = 1:handles.Current.NumberOfModules,
        ModuleNumberAsString = sprintf('%02d', SlotNumber);
        ModuleName = char(handles.Settings.ModuleNames(SlotNumber));
        handles.Current.CurrentModuleNumber = ModuleNumberAsString;
        try
            handles = feval(ModuleName,handles);
        catch
            reportBatchError(['Batch Error: ' ModuleName]);
        end
    end
end
cd(cluster.OutputFolder);
handles.Pipeline = [];
save(sprintf('%s%d_to_%d_OUT',cluster.BatchFilePrefix,cluster.StartImage,cluster.EndImage),'handles');

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
