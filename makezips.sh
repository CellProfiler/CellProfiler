cd ..
rm CellProfiler.zip
rm ExampleFlyImages.zip

zip CellProfiler.zip CellProfiler/CellProfiler.m
zip CellProfiler.zip CellProfiler/CellProfiler.fig
zip CellProfiler.zip CellProfiler/Modules/*.m
zip CellProfiler.zip CellProfiler/ImageTools/*.m
zip CellProfiler.zip CellProfiler/DataTools/*.m
zip CellProfiler.zip CellProfiler/Help/Help*.m
zip CellProfiler.zip CellProfiler/LICENSE
zip CellProfiler.zip CellProfiler/Modules/IdentifySecPropagateSubfunction.cpp
zip CellProfiler.zip CellProfiler/Modules/IdentifySecPropagateSubfunction.dll
zip CellProfiler.zip CellProfiler/Modules/IdentifySecPropagateSubfunction.mexglx
zip CellProfiler.zip CellProfiler/Modules/IdentifySecPropagateSubfunction.mexmac

cd CellProfiler
zip ../ExampleFlyImages.zip ExampleFlyImages/*.TIF
zip ../ExampleFlyImages.zip ExampleFlyImages/ExampleFlyOUT.mat
zip ../ExampleFlyImages.zip ExampleFlyImages/ExampleFlySettings.mat
