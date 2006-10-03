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
% you calculated for the numerator. For instance, if you previously
% calculated the ratio of Area to Perimeter for nuclei, MajorAxisLength to
% MinorAxisLength for cells, and MeanIntensity to MaxIntensity for nuclei,
% the value for the Area to Perimeter for nuclei would be 1, the value for
% MajorAxisLength to MinorAxisLength for cells would be 2, and the value
% for MeanIntensity to MaxIntensity for nuclei would be 3.
%
% See also ClassifyObjectsByTwoMeasurements, FilterByObjectMeasurement.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne E. Carpenter
%   Thouis Ray Jones
%   In Han Kang
%   Ola Friman
%   Steve Lowe
%   Joo Han Chang
%   Colin Clarke
%   Mike Lamprecht
%   Peter Swire
%   Rodrigo Ipince
%   Vicky Lay
%   Jun Liu
%   Chris Gang
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the identified objects (for Ratio, enter the numerator object)?
%infotypeVAR01 = objectgroup
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu custom

%textVAR02 = Enter the feature type:
%choiceVAR02 = AreaShape
%choiceVAR02 = Correlation
%choiceVAR02 = Texture
%choiceVAR02 = Intensity
%choiceVAR02 = Neighbors
%choiceVAR02 = Ratio
FeatureType = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu custom

%textVAR03 = Enter feature number (see help):
%defaultVAR03 = 1
FeatureNbr = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,3}));

%textVAR04 = For INTENSITY or TEXTURE features, which image's measurements would you like to use?
%infotypeVAR04 = imagegroup
%inputtypeVAR04 = popupmenu
Image = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = To create evenly spaced bins, enter number of bins, or type "C:1 2 3 4" to custom define the bin edges, or type P:XXX, where XXX is the numerical threshold to measure the percent of objects that are above a threshold.
%defaultVAR05 = 3
NbrOfBins = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = If you want to label your bins, enter the bin labels separated by commas (e.g. bin1,bin2,bin3), if the number of bins does not equal the number of labels, this step will be ignored. Leave "/" for default labels.
%defaultVAR06 = /
Labels = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = To create evenly spaced bins, enter the lower limit for the lower bin
%defaultVAR07 = 0
LowerBinMin = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,7}));

%textVAR08 = To create evenly spaced bins, enter the upper limit for the upper bin
%defaultVAR08 = 100
UpperBinMax = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,8}));

%textVAR09 = What do you want to call the resulting color-coded image?
%choiceVAR09 = Do not save
%choiceVAR09 = ColorClassifiedNuclei
%inputtypeVAR09 = popupmenu custom
%infotypeVAR09 = imagegroup indep
SaveColoredObjects = char(handles.Settings.VariableValues{CurrentModuleNum,9});

%textVAR10 = Do you want the absolute number of objects or percentage of object?
%choiceVAR10 = Absolute
%choiceVAR10 = Percentage
%inputtypeVAR10 = popupmenu
AbsoluteOrPercentage = char(handles.Settings.VariableValues{CurrentModuleNum,10});

%%%VariableRevisionNumber = 6

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

PercentFlag = 0;
CustomFlag = 0;
try
    if strncmpi(NbrOfBins,'P',1)
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
            error(['Image processing was canceled in the ', ModuleName, ' module because an error was found in the number of bins specification.']);
        end
    end
catch
    error(['Image processing was canceled in the ', ModuleName, ' module because you must enter a number, the letter P, or the letter C for the number of bins.'])
end

%%% Retrieves the label matrix image that contains the segmented objects
fieldname = ['Segmented', ObjectName];
%%% Checks whether the image exists in the handles structure.
if isfield(handles.Pipeline, fieldname)
    LabelMatrixImage = CPretrieveimage(handles,fieldname,ModuleName);
    %%% If we are using a user defined field, there is no corresponding
    %%% image.
elseif strcmpi(FeatureType,'Ratio')
    LabelMatrixImage = zeros(100);
else
    error(['Image processing was canceled in the ', ModuleName, ' module. Prior to running the ', ModuleName, ' module, you must have previously run a module that generates an image with the objects identified.  You specified in the ', ModuleName, ' module that the primary objects were named ',FeatureType,' which should have produced an image in the handles structure called ', fieldname, '. The ', ModuleName, ' module cannot locate this image.']);
end

if strcmp(FeatureType,'Intensity') || strcmp(FeatureType,'Texture')
    FeatureType = [FeatureType, '_', Image];
end

if ~strcmp(FeatureType,'Ratio')
    %%% Checks whether the feature type exists in the handles structure.
    if ~isfield(handles.Measurements.(ObjectName),FeatureType)
        error(['Image processing was canceled in the ', ModuleName, ' module because the feature type entered does not exist.']);
    end
end

if isempty(LowerBinMin) || isempty(UpperBinMax) || LowerBinMin > UpperBinMax
    error(['Image processing was canceled in the ', ModuleName, ' module because an error in the specification of the lower and upper limits was found.']);
end

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

% Get Measurements
Measurements = handles.Measurements.(ObjectName).(FeatureType){handles.Current.SetBeingAnalyzed}(:,FeatureNbr);


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
    error(['Image processing was canceled in the ', ModuleName, ' module because you cannot enter Infinity as one of the bin edges because it affects the plotting function. This could be changed in the code if we set the first and last bin edges to equal the min and max actual data values but in the meantime please enter an actual numerical value.']);
end

edges(1) = edges(1) - sqrt(eps);   % Just a fix so that objects with a measurement that equals the lower bin edge of the lowest bin are counted
QuantizedMeasurements = zeros(size(Measurements));
bins = zeros(1,NbrOfBins);
RemainingLabels = Labels;
ListOfLabels = {};
EmptyIndex = 1:size(Measurements);
for k = 1:NbrOfBins
    index = find(Measurements > edges(k) & Measurements <= edges(k+1));
    QuantizedMeasurements(index) = k;
    bins(k) = length(index);
    if length(strfind(Labels,',')) == (NbrOfBins - 1)
        [BinLabel,RemainingLabels]=strtok(RemainingLabels,',');
        ListOfLabels(k)={BinLabel};
        EmptyIndex(index) = 0;
    else
        ListOfLabels{k} = ['Bin', num2str(k)];
    end
end

NbrOfObjects = length(Measurements);

%%% If we are using a user defined field, there is no corresponding
%%% image.
if ~strcmpi(FeatureType,'Ratio')
    % Produce image where the the objects are colored according to the original
    % measurements and the quantized measurements
    NonQuantizedImage = zeros(size(LabelMatrixImage));
    props = regionprops(LabelMatrixImage,'PixelIdxList');              % Pixel indexes for objects fast
    if ~isempty(props)
        for k = 1:NbrOfObjects
            NonQuantizedImage(props(k).PixelIdxList) = Measurements(k);
        end
        QuantizedMeasurements = [0;QuantizedMeasurements];                 % Add a background class
        QuantizedImage = QuantizedMeasurements(LabelMatrixImage+1);
        handlescmap = handles.Preferences.LabelColorMap;
        cmap = [0 0 0;feval(handlescmap,length(bins))];
        QuantizedRGBimage = ind2rgb(QuantizedImage+1,cmap);
    else
        QuantizedRGBimage = NonQuantizedImage;
    end
    FeatureName = handles.Measurements.(ObjectName).([FeatureType,'Features']){FeatureNbr};
else
    FeatureName = ObjectName;
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    %%% Activates the appropriate figure window.
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    %%% If we are using a user defined field, there is no corresponding
    %%% image.
    if ~strcmpi(FeatureType,'Ratio')
        if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
            CPresizefigure(NonQuantizedImage,'TwoByTwo',ThisModuleFigureNumber);
        end
    end
    AdjustedObjectName = strrep(ObjectName,'_','\_');
    AdjustedFeatureName = strrep(FeatureName,'_','\_');
    AdjustedFeatureType = strrep(FeatureType,'_','\_');

    %%% If we are using a user defined field, there is no corresponding
    %%% image.
    if ~strcmpi(FeatureType,'Ratio')
        %%% A subplot of the figure window is set to display the original image.
        subplot(2,2,1)
        CPimagesc(NonQuantizedImage,handles);
        IntensityColormap = handles.Preferences.IntensityColorMap;
        colormap(feval(IntensityColormap,max(Measurements)))
        title([AdjustedObjectName,' colored according to ',AdjustedFeatureName])
    end

    %%% Produce and plot histogram of original data
    subplot(2,2,2)
    Nbins = min(round(NbrOfObjects),40);
    hist(Measurements,Nbins);
    %%% Took this out: don't want to use misleading colors.
    %%%    set(get(gca,'Children'),'FaceVertexCData',hot(Nbins));
    xlabel(AdjustedFeatureName,'fontsize',handles.Preferences.FontSize);
    ylabel(['# of ',AdjustedObjectName],'fontsize',handles.Preferences.FontSize);
    title(['Histogram of ',AdjustedFeatureType],'fontsize',handles.Preferences.FontSize);
    %%% Using "axis tight" here is ok, I think, because we are displaying
    %%% data, not images.
    ylimits = ylim;
    axis tight
    xlimits = xlim;
    axis([xlimits ylimits]);

    %%% If we are using a user defined field, there is no corresponding
    %%% image.
    if ~strcmpi(FeatureType,'Ratio')
        %%% A subplot of the figure window is set to display the quantized image.
        subplot(2,2,3)
        CPimagesc(QuantizedRGBimage,handles);
        title(['Classified ', AdjustedObjectName]);
    end
    %%% Produce and plot histogram
    subplot(2,2,4)
    x = edges(1:end-1) + (edges(2)-edges(1))/2;
    h = bar(x,bins,1);
    xlabel(AdjustedFeatureName,'fontsize',handles.Preferences.FontSize);
    ylabel(['# of ',AdjustedObjectName],'fontsize',handles.Preferences.FontSize);
    title(['Classified by ',AdjustedFeatureType],'fontsize',handles.Preferences.FontSize);
    %%% Using "axis tight" here is ok, I think, because we are displaying
    %%% data, not images.
    axis tight
    xlimits(1) = min(xlimits(1),LowerBinMin);                          % Extend limits if necessary and save them
    xlimits(2) = max(UpperBinMax,edges(end));                          % so they can be used for the second histogram
    axis([xlimits ylim]);
    %%% Took this out: don't want to use misleading colors.
    %     handlescmap = handles.Preferences.LabelColorMap;
    %     set(get(h,'Children'),'FaceVertexCData',feval(handlescmap,max(2,NbrOfBins)));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% If we are using a user defined field, there is no corresponding
%%% image.
if ~strcmpi(FeatureType,'Ratio') && ~strcmpi(SaveColoredObjects,'Do not save')
    %%% Saves images to the handles structure so they can be saved to the hard
    %%% drive, if the user requests.
    handles.Pipeline.(SaveColoredObjects) = QuantizedRGBimage;
end

ClassifyFeatureNames = cell(1,NbrOfBins);

for k = 1:NbrOfBins
    ClassifyFeatureNames{k} = ['Bin', num2str(k)];
end
FeatureName = FeatureName(~isspace(FeatureName));                    % Remove spaces in the feature name

if length(strfind(Labels,',')) == (NbrOfBins - 1)
    EmptyIndex(EmptyIndex==0)=[];
    handles.Measurements.(ObjectName).(['Classify_',FeatureName,'Description']) = {[ObjectName,'_',FeatureName]};
    handles.Measurements.(ObjectName).(['Classify_',FeatureName])(handles.Current.SetBeingAnalyzed) = {ListOfLabels};
end

handles.Measurements.Image.(['Classify_',ObjectName,'_',FeatureName,'Features']) = ListOfLabels;
if strcmp(AbsoluteOrPercentage,'Percentage')
    handles.Measurements.Image.(['Classify_',ObjectName,'_',FeatureName])(handles.Current.SetBeingAnalyzed) = {bins/length(Measurements)};
else
    handles.Measurements.Image.(['Classify_',ObjectName,'_',FeatureName])(handles.Current.SetBeingAnalyzed) = {bins};
end