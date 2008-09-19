function handles = ClassifyObjects(handles)

% Help for the Classify Objects module:
% Category: Object Processing
%
% SHORT DESCRIPTION:
% Classifies objects into different classes according to the value of a
% measurement you choose.
% *************************************************************************
%
% This module classifies objects into a number of different classes
% according to the value of a measurement (e.g. by size, intensity, shape).
% Choose the measurement feature to be used to classify your objects and
% specify what bins to use. This module requires that you run a measurement
% module previous to this module in the pipeline so that the measurement
% values can be used to classify the objects. If you are classifying by the
% ratio of two measurements, you must put a CalculateRatios module previous
% to this module in the pipeline.
%
% Settings:
%
% Feature Number:
% The feature number specifies which feature from the Measure module will
% be used for classifying. See each Measure module's help for the numbered
% list of the features measured by that module.
%
% If you are selecting Ratio, this is the order of ratio measurements that
% you calculated, i.e. module order in pipeline. For instance, if you previously
% calculated the ratio of Area to Perimeter for nuclei, MajorAxisLength to
% MinorAxisLength for cells, and MeanIntensity to MaxIntensity for nuclei,
% the value for the Area to Perimeter for nuclei would be 1, the value for
% MajorAxisLength to MinorAxisLength for cells would be 2, and the value
% for MeanIntensity to MaxIntensity for nuclei would be 3.
%
% Saving:
%
% Category = 'ClassifyObjects'
% Features measured:                         Feature Number:
% (As named in module's last setting)     |       1
%
% See also ClassifyObjectsByTwoMeasurements, FilterByObjectMeasurement.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%%%%%%%%%%%%%%%%
% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the identified objects (for Ratio, enter the numerator object)?
%infotypeVAR01 = objectgroup
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu custom

%textVAR02 = Enter the feature type:
Category = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu category

%textVAR03 = Enter feature number or name (see help):
%defaultVAR03 = 1
%inputtypeVAR03 = popupmenu measurement
FeatureNbr = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = For INTENSITY, AREAOCCUPIED or TEXTURE features, which image's measurements would you like to use?
%infotypeVAR04 = imagegroup
%inputtypeVAR04 = popupmenu
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = For TEXTURE, RADIAL DISTRIBUTION, OR NEIGHBORS features, what previously measured size scale (TEXTURE OR NEIGHBORS) or previously used number of bins (RADIALDISTRIBUTION) do you want to use?
%defaultVAR05 = 1
%inputtypeVAR05 = popupmenu scale
SizeScale = str2double(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = To create evenly spaced bins, enter number of bins, or type "C:1 2 3 4" to custom define the bin edges, or type P:XXX, where XXX is the numerical threshold to measure the percent of objects that are above a threshold.
%defaultVAR06 = 3
NbrOfBins = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = If you want to label your bins, enter the bin labels separated by commas (e.g. bin1,bin2,bin3), if the number of bins does not equal the number of labels, this step will be ignored. Leave "Do not use" for default labels.
%defaultVAR07 = Do not use
Labels = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = To create evenly spaced bins, enter the lower limit for the lower bin
%defaultVAR08 = 0
LowerBinMin = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,8}));

%textVAR09 = To create evenly spaced bins, enter the upper limit for the upper bin
%defaultVAR09 = 100
UpperBinMax = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,9}));

%textVAR10 = What do you want to call the resulting color-coded image?
%defaultVAR10 = Do not use
%infotypeVAR10 = imagegroup indep
SaveColoredObjects = char(handles.Settings.VariableValues{CurrentModuleNum,10});

%textVAR11 = Do you want the absolute number of objects or percentage of object?
%choiceVAR11 = Absolute
%choiceVAR11 = Percentage
%inputtypeVAR11 = popupmenu
AbsoluteOrPercentage = char(handles.Settings.VariableValues{CurrentModuleNum,11});

%%%VariableRevisionNumber = 7

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow
SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;

try
    FeatureName = CPgetfeaturenamesfromnumbers(handles,ObjectName,Category,...
        FeatureNbr,ImageName,SizeScale);
catch
    error([lasterr '  Image processing was canceled in the ', ModuleName, ...
        ' module (#' num2str(CurrentModuleNum) ...
        ') because an error ocurred when retrieving the data.  '...
        'Likely the category of measurement you chose, ',...
        Category, ', was not available for ', ...
        ObjectName,' with feature number ' num2str(FeatureNbr) ...
        ', possibly specific to image ''' ImageName ''' and/or ' ...
        'Texture Scale = ' num2str(SizeScale) '.']);
end

% Retrieves the label matrix image that contains the segmented objects
SegmentedField = ['Segmented', ObjectName];

% Checks whether the image exists in the handles structure.
if isfield(handles.Pipeline, SegmentedField)
    LabelMatrixImage = CPretrieveimage(handles,SegmentedField,ModuleName);
    % If we are using a user defined field, there is no corresponding
    % image.
elseif strcmpi(Category,'Ratio')
    LabelMatrixImage = zeros(100);
else
    error(['Image processing was canceled in the ', ModuleName, ...
        ' module. Prior to running the ', ModuleName, ' module, you must' ...
        'have previously run a module that generates an image with the ' ...
        'objects identified.  You specified in the ', ModuleName,  ...
        'module that the primary objects were named ',Category, ...
        'which should have produced an image in the handles structure ' ...
        'called ', SegmentedField, '. The ', ModuleName, ' module cannot ' ...
        'locate this image.']);
end

if ~strcmp(Category,'Ratio')
    % Checks whether the feature type exists in the handles structure.
    if ~isfield(handles.Measurements.(ObjectName),FeatureName)
        error(['Image processing was canceled in the ', ModuleName, ...
            ' module because the feature type entered does not exist.']);
    end
end

if isempty(LowerBinMin) || isempty(UpperBinMax) || LowerBinMin > UpperBinMax
    error(['Image processing was canceled in the ', ModuleName, ...
        ' module because an error in the specification of the lower and upper limits was found.']);
end

%%%%%%%%%%%%%%%%%%%%%%
% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

% Get Measurements
Measurements = handles.Measurements.(ObjectName).(FeatureName){SetBeingAnalyzed};

PercentFlag = 0;
CustomFlag = 0;
try
    if strncmpi(NbrOfBins,'P',1)
        LowerBinMin = min(Measurements(~isinf(Measurements)))-eps;
        UpperBinMax = max(Measurements(~isinf(Measurements)))+eps;
        MidPointToUse = str2double(NbrOfBins(3:end));
        PercentFlag = 1;
    elseif strncmpi(NbrOfBins,'C',1)
        NbrOfBins = str2num(NbrOfBins(3:end)); %#ok Ignore MLint
        if length(NbrOfBins) >= 2
            CustomFlag = 1;
        end
    else
        NbrOfBins = str2double(NbrOfBins);
        if isempty(NbrOfBins) || NbrOfBins < 1
            error(['Image processing was canceled in the ', ModuleName, ...
                ' module because an error was found in the number of bins specification.']);
        end
    end
catch
    error(['Image processing was canceled in the ', ModuleName, ...
        ' module because you must enter a number, the letter P, or the letter C for the number of bins.'])
end

if PercentFlag == 1
    edges = [LowerBinMin,MidPointToUse,UpperBinMax];
    NbrOfBins = 2;
elseif CustomFlag == 1
    edges = NbrOfBins;
    NbrOfBins = length(NbrOfBins)-1;
else
    % Quantize measurements
    edges = linspace(LowerBinMin,UpperBinMax,NbrOfBins+1);
end

if any(isinf(edges)) == 1
    error(['Image processing was canceled in the ', ModuleName, ...
        ' module because you cannot enter Infinity as one of the bin ' ...
        'edges because it affects the plotting function. This could be ' ...
        'changed in the code if we set the first and last bin edges to ' ...
        'equal the min and max actual data values but in the meantime ' ...
        'please enter an actual numerical value.']);
end

edges(1) = edges(1) - sqrt(eps);    % Just a fix so that objects with a 
                                    % measurement that equals the lower 
                                    % bin edge of the lowest bin are counted
QuantizedMeasurements = zeros(size(Measurements));
bins = zeros(1,NbrOfBins);
RemainingLabels = Labels;
ClassifyFeatureNames = cell(1,NbrOfBins);
for BinNum = 1:NbrOfBins
    bin_index{BinNum} = find(Measurements > edges(BinNum) & Measurements <= edges(BinNum+1));
    QuantizedMeasurements(bin_index{BinNum}) = BinNum;
    bins(BinNum) = length(bin_index{BinNum});
    if length(strfind(Labels,',')) == (NbrOfBins - 1)
        [BinLabel,RemainingLabels] = strtok(RemainingLabels,',');
        if ~isvarname(BinLabel)
            BinLabel = ['Module',CurrentModule,'Bin',num2str(BinNum),];
        end
    else
        % Assign bins with generic names
        BinLabel = ['Module',CurrentModule,'Bin',num2str(BinNum),];
    end
    ClassifyFeatureNames{BinNum} = ['ClassifyObjects_' BinLabel];
end

NbrOfObjects = length(Measurements);

if ~strcmpi(Category,'Ratio')
    % Produce image where the the objects are colored according to the original
    % measurements and the quantized measurements (though this does not apply to 'Ratio')
    NonQuantizedImage = zeros(size(LabelMatrixImage));
    props = regionprops(LabelMatrixImage,'PixelIdxList');              % Pixel indexes for objects fast
    if ~isempty(props)
        for BinNum = 1:NbrOfObjects
            NonQuantizedImage(props(BinNum).PixelIdxList) = Measurements(BinNum);
        end
        QuantizedMeasurementsWithBackground = [0;QuantizedMeasurements];                 % Add a background class
        QuantizedImage = QuantizedMeasurementsWithBackground(LabelMatrixImage+1);
        handlescmap = handles.Preferences.LabelColorMap;
        cmap = [0 0 0;feval(handlescmap,length(bins))];
        QuantizedRGBimage = ind2rgb(QuantizedImage+1,cmap);
    else
        QuantizedRGBimage = NonQuantizedImage;
    end
end

%%%%%%%%%%%%%%%%%%%
% DISPLAY RESULTS %
%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    % Activates the appropriate figure window.
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    % If we are using a user defined field, there is no corresponding
    % image.
    if ~strcmpi(Category,'Ratio')
        if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
            CPresizefigure(NonQuantizedImage,'TwoByTwo',ThisModuleFigureNumber);
        end
    end
 
    % If we are using a user defined field, there is no corresponding
    % image.
    if ~strcmpi(Category,'Ratio')
        % A subplot of the figure window is set to display the original image.
        hAx = subplot(2,2,1,'Parent',ThisModuleFigureNumber);
        CPimagesc(NonQuantizedImage,handles,hAx);
        IntensityColormap = handles.Preferences.IntensityColorMap;
        if max(Measurements) > length(colormap)
            colormap(hAx,feval(IntensityColormap,max(Measurements)))
        end
        title([ObjectName,' shaded according to ',FeatureName],'Parent',hAx)
    end

    % Produce and plot histogram of original data
    hAx = subplot(2,2,2,'Parent',ThisModuleFigureNumber);
    Nbins = min(round(NbrOfObjects),40);
    hist(hAx,Measurements,Nbins);
    % Took this out: don't want to use misleading colors.
    %    set(get(gca,'Children'),'FaceVertexCData',hot(Nbins));
    xlabel(hAx,FeatureName,'fontsize',handles.Preferences.FontSize);
    ylabel(hAx,['# of ',ObjectName],'fontsize',handles.Preferences.FontSize);
    title(hAx,['Histogram of ',Category],'fontsize',handles.Preferences.FontSize);
    % Using "axis tight" here is ok, I think, because we are displaying
    % data, not images.
    ylimits = ylim;
    axis(hAx,'tight')
    xlimits = xlim;
    axis(hAx,[xlimits ylimits]);

    % If we are using a user defined field, there is no corresponding
    % image.
    if ~strcmpi(Category,'Ratio')
        % A subplot of the figure window is set to display the quantized image.
        hAx = subplot(2,2,3,'Parent',ThisModuleFigureNumber)
        CPimagesc(QuantizedRGBimage,handles,hAx);
        title(hAx,['Classified ', ObjectName]);
        % TODO add legend
    end
    % Produce and plot histogram
    hAx=subplot(2,2,4,'Parent',ThisModuleFigureNumber);
    bar_ctr = edges(1:end-1) + (edges(2)-edges(1))/2;
    h = bar(hAx,bar_ctr,bins,1);
    xlabel(hAx,FeatureName,'fontsize',handles.Preferences.FontSize);
    ylabel(hAx,['# of ',ObjectName],'fontsize',handles.Preferences.FontSize);
    title(hAx,['Classified by ',Category],'fontsize',handles.Preferences.FontSize);
    % Using "axis tight" here is ok, I think, because we are displaying
    % data, not images.
    axis(hAx,'tight');
    xlimits(1) = min(xlimits(1),LowerBinMin);   % Extend limits if necessary and save them
    xlimits(2) = max(UpperBinMax,edges(end));   % so they can be used for the second histogram
    axis(hAx,[xlimits ylim]);
    % Took this out: don't want to use misleading colors.
    %     handlescmap = handles.Preferences.LabelColorMap;
    %     set(get(h,'Children'),'FaceVertexCData',feval(handlescmap,max(2,NbrOfBins)));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SAVE DATA TO HANDLES STRUCTURE %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

% If we are using a user defined field, there is no corresponding
% image.
if strcmpi(Category,'Ratio') && ~strcmpi(SaveColoredObjects,'Do not use')
    error(['Image processing was canceled in the ', ModuleName, ' module ' ...
        'because you have requested to save the resulting color-coded image' ...
        'called ',SaveColoredObjects,' but that image cannot be produced by' ...
        'the Classify module. The color-coded image can only be produced ' ...
        'when using measurements straight from a Measure module, not when ' ...
        'using measurements from a CalculateRatios module. Sorry for the inconvenience.']);
end
if ~strcmpi(SaveColoredObjects,'Do not use')
    % Saves images to the handles structure so they can be saved to the hard
    % drive, if the user requests.
    handles.Pipeline.(SaveColoredObjects) = QuantizedRGBimage;
end

% Calculate Objects per Bin (Absolute or Percentage)
if strcmp(AbsoluteOrPercentage,'Percentage')
    ObjsPerBin = bins/length(Measurements);
else
    ObjsPerBin = bins;
end

% Save FeatureNames and the indices of the objects that fall into each
% bin, as well as ObjsPerBin
for BinNum = 1:NbrOfBins
    handles = CPaddmeasurements(handles, ObjectName, [ClassifyFeatureNames{BinNum} 'indices'], bin_index{BinNum});
    handles = CPaddmeasurements(handles, ObjectName, [ClassifyFeatureNames{BinNum} 'ObjsPerBin'], ObjsPerBin(:,BinNum));
end

% Save Bin Edges
handles = CPaddmeasurements(handles,ObjectName,[ClassifyFeatureNames{BinNum} 'BinEdges'],edges');