cd ..
rm CellProfiler.zip
rm ExampleFlyImages.zip

zip CellProfiler.zip CellProfiler/CellProfiler.m
zip CellProfiler.zip CellProfiler/CellProfiler.fig
zip CellProfiler.zip CellProfiler/Alg*.m
zip CellProfiler.zip CellProfiler/Help*.m
zip CellProfiler.zip CellProfiler/Interactive*.m
zip CellProfiler.zip CellProfiler/LICENSE
zip CellProfiler.zip CellProfiler/ProgrammingNotes.m
zip CellProfiler.zip CellProfiler/ReplaceTextInModules.m
zip CellProfiler.zip CellProfiler/AlgIdentifySecPropagateSubfunction.cpp
zip CellProfiler.zip CellProfiler/AlgIdentifySecPropagateSubfunction.m
zip CellProfiler.zip CellProfiler/AlgIdentifySecPropagateSubfunction.dll
zip CellProfiler.zip CellProfiler/AlgIdentifySecPropagateSubfunction.mexglx
zip CellProfiler.zip CellProfiler/AlgIdentifySecPropagateSubfunction.mexmac

cd CellProfiler
zip ../ExampleFlyImages.zip ExampleFlyImages/*.TIF
zip ../ExampleFlyImages.zip ExampleFlyImages/ExampleFlyOUT.mat
zip ../ExampleFlyImages.zip ExampleFlyImages/ExampleFlySettings.mat
