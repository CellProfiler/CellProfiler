function handles = AreaImage(handles)

%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

%textVAR01 = What did you call the segmented objects?
%defaultVAR01 = Cells
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = Enter the largest size (in micrometers) for the small bin
%defaultVAR02 = 0.6724
SmallBinMax = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,2}));

%textVAR03 = Enter the smallest size (in micrometers) for the large bin
%defaultVAR03 = 1.519625
LargeBinMin = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,3}));

%%%VariableRevisionNumber = 1

%%% Retrieves the label matrix image that contains the segmented objects
fieldname = ['Segmented', ObjectName];

%%% Checks whether the image exists in the handles structure.
if isfield(handles.Pipeline, fieldname) == 0,
    error(['Image processing has been canceled. Prior to running the Area Image module, you must have previously run a module that generates an image with the objects identified.  You specified in the Area Image module that the primary objects were named ',ObjectName,' which should have produced an image in the handles structure called ', fieldname, '. The Area Image module cannot locate this image.']);
end
LabelMatrixImage = handles.Pipeline.(fieldname);

if SmallBinMax > LargeBinMin
    error('Image processing has been canceled because the value you entered in the Area Image module for the largest size for the small bin is greater than the smallest size for the large bin.')
end

%%% Retrieves the pixel size that the user entered (micrometers per pixel).
PixelSize = str2double(handles.Settings.PixelSize);

% Get areas
props = regionprops(LabelMatrixImage,'Area');              % Area in pixels
area  = cat(1,props.Area)*PixelSize^2;                                 

% Quantize areas
index1 = find(area < SmallBinMax);
index3 = find(area > LargeBinMin);
index2 = setdiff(1:length(area),[index1;index3]);
qarea = zeros(size(area));
qarea(index1) = 1;qarea(index2) = 2;qarea(index3) = 3;
qarea = [0;qarea];

% Generate area map
areamap = qarea(LabelMatrixImage+1);

% Display result
fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
figure(ThisModuleFigureNumber);
image(areamap+1)
cmap = [0 0 0;              % RGB color for background
        1 0 0;              % RGB color for objects with area < SmallBinMax
        0 1 0;              % RGB color for objects with SmallBinMax < area < LargeBinMin
        0 0 1];             % RGB color for objects with area > LargeBinMin
colormap(cmap)
