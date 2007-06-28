function CPCluster(batchfile,clusterfile)

%%% Must list all CellProfiler modules here
%#function Align ApplyThreshold Average CalculateMath CalculateRatios CalculateStatistics ClassifyObjects ClassifyObjectsByTwoMeasurements ColorToGray Combine ConvertToImage CorrectIllumination_Apply CorrectIllumination_Calculate CreateBatchFiles CreateWebPage Crop DefineGrid DisplayDataOnImage DisplayGridInfo DisplayHistogram DisplayImageHistogram DisplayMeasurement DistinguishPixelLabels Exclude ExpandOrShrink ExportToDatabase ExportToExcel FilterByObjectMeasurement FindEdges Flip GrayToColor IdentifyObjectsInGrid IdentifyPrimAutomatic IdentifyPrimManual IdentifySecondary IdentifyTertiarySubregion InvertIntensity LoadImages LoadSingleImage LoadText MaskImage MeasureCorrelation MeasureImageAreaOccupied MeasureImageGranularity MeasureImageIntensity MeasureImageSaturationBlur MeasureObjectAreaShape MeasureObjectIntensity MeasureObjectNeighbors MeasureTexture Morph OverlayOutlines PlaceAdjacent Relate RenameOrRenumberFiles RescaleIntensity Resize Restart Rotate SaveImages SendEmail Smooth SpeedUpCellProfiler SplitOrSpliceMovie Subtract SubtractBackground Tile CPaddmeasurements CPaverageimages CPblkproc CPcd CPclearborder CPcompilesubfunction CPcontrolhistogram CPconvertsql CPdilatebinaryobjects CPerrordlg CPfigure CPgetfeature CPhelpdlg CPhistbins CPimagesc CPimagetool CPimread CPinputdlg CPlabel2rgb CPlistdlg CPlogo CPmakegrid CPmsgbox CPnanmean CPnanmedian CPnanstd CPnlintool CPplotmeasurement CPquestdlg CPrelateobjects CPrescale CPresizefigure CPretrieveimage CPretrievemediafilenames CPrgsmartdilate CPselectmodules CPselectoutputfiles CPsigmoid CPsmooth CPtextdisplaybox CPtextpipe CPthresh_tool CPthreshold CPwaitbar CPwarndlg CPwhichmodule CPwritemeasurements

load(batchfile);
load(clusterfile);
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
            handles.BatchError = [ModuleName ' ' lasterr];
            disp(['Batch Error: ' ModuleName ' ' lasterr]);
            rethrow(lasterror);
            quit;
        end
    end
end
cd(cluster.OutputFolder);
handles.Pipeline = [];
save(sprintf('%s%d_to_%d_OUT',cluster.BatchFilePrefix,cluster.StartImage,cluster.EndImage),'handles');
