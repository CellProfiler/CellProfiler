function handles = ClassifyObjects(handles)

% Help for the Classify Objects module:
% Category: Other
%
% SHORT DESCRIPTION:
% Classifies objects into a number of different classes according to the
% value of a measurement you choose.
% *************************************************************************
%
% This module classifies objects into a number of different classes
% according to the value of a measurement (e.g. by size, intensity, shape,
% etc.). Choose the measurement feature to be used to classify your objects
% and specify what bins to use. This module requires that you run a
% measurement module previous to this module in the pipeline so that the
% measurement values can be used to classify the objects. If you are
% classifying by the ratio of two measurements, you must put a MeasureRatio
% module previous to this module in the pipeline.
%
% Settings:
%
% Feature Number:
% The feature number is the parameter from the chosen module (AreaShape,
% Intensity, Texture) which will be used for the ratio. Please see
% individual measurement module help for list of measurements and their
% feature numbers.
%
% If you are selecting Ratio, this is the order of ratio measurements that
% you calculated for the numerator.  For instance, if you previously
% calculated the ratio of Area to Perimeter for nuclei, MajorAxisLength to
% MinorAxisLength for cells, and MeanIntensity to MaxIntensity for nuclei,
% the value for the Area to Perimeter for nuclei would be 1, the value for
% MajorAxisLength to MinorAxisLength for cells would be 2, and the value
% for MeanIntensity to MaxIntensity for nuclei would be 3.
%
% See also <nothing relevant>.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne Carpenter
%   Thouis Jones
%   In Han Kang
%   Ola Friman
%   Steve Lowe
%   Joo Han Chang
%   Colin Clarke
%   Mike Lamprecht
%   Susan Ma
%   Wyman Li
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

%textVAR04 = To create evenly spaced bins, enter number of bins, or type "C:1 2 3 4" to custom define the bin edges, or type P:XXX, where XXX is the numerical threshold to measure the percent of objects that are above a threshold.
%defaultVAR04 = 3
NbrOfBins = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = To create evenly spaced bins, enter the lower limit for the lower bin
%defaultVAR05 = 0
LowerBinMin = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,5}));

%textVAR06 = To create evenly spaced bins, enter the upper limit for the upper bin
%defaultVAR06 = 100
UpperBinMax = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,6}));

%textVAR07 = What do you want to call the resulting color-coded image?
%choiceVAR07 = Do not save
%choiceVAR07 = ColorClassifiedNuclei
%inputtypeVAR07 = popupmenu custom
%infotypeVAR07 = imagegroup indep
SaveColoredObjects = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%%%VariableRevisionNumber = 3

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
    error(['Image processing was canceled in the ', ModuleName, ' module because you must enter a number, or the letter P for the number of bins.'])
end

%%% Retrieves the label matrix image that contains the segmented objects
fieldname = ['Segmented', ObjectName];
%%% Checks whether the image exists in the handles structure.
if isfield(handles.Pipeline, fieldname)
    LabelMatrixImage = handles.Pipeline.(fieldname);
    %%% If we are using a user defined field, there is no corresponding
    %%% image.
elseif strcmpi(FeatureType,'Ratio')
    LabelMatrixImage = zeros(100);
else
    error(['Image processing was canceled in the ', ModuleName, ' module. Prior to running the ', ModuleName, ' module, you must have previously run a module that generates an image with the objects identified.  You specified in the ', ModuleName, ' module that the primary objects were named ',FeatureType,' which should have produced an image in the handles structure called ', fieldname, '. The ', ModuleName, ' module cannot locate this image.']);
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

if strcmp(FeatureType,'Ratio')
    Measurements = handles.Measurements.(ObjectName).Ratios{handles.Current.SetBeingAnalyzed}(:,FeatureNbr);
else
    % Get Measurements
    Measurements = handles.Measurements.(ObjectName).(FeatureType){handles.Current.SetBeingAnalyzed}(:,FeatureNbr);
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

edges(1) = edges(1) - sqrt(eps);   % Just a fix so that objects with a measurement that equals the lower bin edge of the lowest bin are counted
QuantizedMeasurements = zeros(size(Measurements));
bins = zeros(1,NbrOfBins);
for k = 1:NbrOfBins
    index = find(Measurements > edges(k) & Measurements <= edges(k+1));
    QuantizedMeasurements(index) = k;
    bins(k) = length(index);
end

NbrOfObjects = length(Measurements);

%%% If we are using a user defined field, there is no corresponding
%%% image.
if ~strcmpi(FeatureType,'Ratio')
    % Produce image where the the objects are colored according to the original
    % measurements and the quantized measurements
    NonQuantizedImage = zeros(size(LabelMatrixImage));
    props = regionprops(LabelMatrixImage,'PixelIdxList');              % Pixel indexes for objects fast
    for k = 1:NbrOfObjects
        NonQuantizedImage(props(k).PixelIdxList) = Measurements(k);
    end
    QuantizedMeasurements = [0;QuantizedMeasurements];                 % Add a background class
    QuantizedImage = QuantizedMeasurements(LabelMatrixImage+1);
    handlescmap = handles.Preferences.LabelColorMap;
    cmap = [0 0 0;feval(handlescmap,max(64,length(bins)))];
    QuantizedRGBimage = ind2rgb(QuantizedImage+1,cmap);
    FeatureName = handles.Measurements.(ObjectName).([FeatureType,'Features']){FeatureNbr};
else
    FeatureName = ObjectName;
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber) == 1;
    drawnow
    %%% Activates the appropriate figure window.
    CPfigure(handles,ThisModuleFigureNumber);
    AdjustedObjectName = strrep(ObjectName,'_','\_');
    AdjustedFeatureName = strrep(FeatureName,'_','\_');
    %%% If we are using a user defined field, there is no corresponding
    %%% image.
    if ~strcmpi(FeatureType,'Ratio')
        %%% A subplot of the figure window is set to display the original image.
        subplot(2,2,1)
        ImageHandle = CPimagesc(NonQuantizedImage,[min(Measurements) max(Measurements)]);
        set(ImageHandle,'Tag',sprintf('%s colored according to %s',AdjustedObjectName,AdjustedFeatureName))
        axis image
        set(gca,'Fontsize',handles.Preferences.FontSize)
        title(sprintf('%s colored according to %s',AdjustedObjectName,AdjustedFeatureName))
    end
    %%% Produce and plot histogram of original data
    subplot(2,2,2)
    Nbins = min(round(NbrOfObjects/5),40);
    hist(Measurements,Nbins)
    set(get(gca,'Children'),'FaceVertexCData',hot(Nbins));
    set(gca,'Fontsize',handles.Preferences.FontSize);
    xlabel(AdjustedFeatureName),ylabel(['#',AdjustedObjectName]);
    title(sprintf('Histogram of %s',AdjustedFeatureName));
    ylimits = ylim;
    axis tight
    xlimits = xlim;
    axis([xlimits ylimits])
    %%% If we are using a user defined field, there is no corresponding
    %%% image.
    if ~strcmpi(FeatureType,'Ratio')
        %%% A subplot of the figure window is set to display the quantized image.
        subplot(2,2,3)
        ImageHandle = CPimagesc(QuantizedRGBimage);axis image
        set(ImageHandle,'Tag',['Classified ', AdjustedObjectName])
        set(gca,'Fontsize',handles.Preferences.FontSize)
        title(['Classified ', AdjustedObjectName],'fontsize',handles.Preferences.FontSize);
    end
    %%% Produce and plot histogram
    subplot(2,2,4)
    x = edges(1:end-1) + (edges(2)-edges(1))/2;
    h = bar(x,bins,1);
    set(gca,'Fontsize',handles.Preferences.FontSize);
    xlabel(AdjustedFeatureName),ylabel(['#',AdjustedObjectName])
    title(sprintf('Histogram of %s',AdjustedFeatureName))
    axis tight
    xlimits(1) = min(xlimits(1),LowerBinMin);                          % Extend limits if necessary and save them
    xlimits(2) = max(UpperBinMax,edges(end));                          % so they can be used for the second histogram
    axis([xlimits ylim])
    handlescmap = handles.Preferences.LabelColorMap;
    set(get(h,'Children'),'FaceVertexCData',feval(handlescmap,max(64,NbrOfBins)));

    %%% If we are using a user defined field, there is no corresponding
    %%% image.
    if ~strcmpi(FeatureType,'Ratio')
        CPFixAspectRatio(NonQuantizedImage);
    end
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
    ClassifyFeatureNames{k} = ['Bin ',num2str(k)];
end
FeatureName = FeatureName(~isspace(FeatureName));                    % Remove spaces in the feature name
%%% We are truncating the ObjectName in case it's really long.
MaxLengthOfFieldname = min(20,length(FeatureName));
handles.Measurements.Image.(['ClassifyObjects_',ObjectName,'_',FeatureName(1:MaxLengthOfFieldname),'Features']) = ClassifyFeatureNames;
handles.Measurements.Image.(['ClassifyObjects_',ObjectName,'_',FeatureName(1:MaxLengthOfFieldname)])(handles.Current.SetBeingAnalyzed) = {bins};