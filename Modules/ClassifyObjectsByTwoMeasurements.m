function handles = ClassifyObjectsByTwoMeasurements(handles)

% Help for the Classify Objects By Two Measurements module:
% Category: Object Processing
%
% SHORT DESCRIPTION:
% Classifies objects into different classes according to the value of two
% measurements of your choice.
% *************************************************************************
%
% This module classifies objects into four different classes according to
% the value of two measurements of your choice (e.g. size, intensity,
% shape). Choose the measurement features you want to use, and select a
% threshold for each set of data (measurements). The objects will then be
% separated in four classes: (1) objects whose first and second
% measurements are both below the specified thresholds, (2) objects whose
% first measurement is below the first threshold and whose second
% measurement is above the second threshold, (3) the opposite of class 2,
% and (4) objects whose first and second measurements are both above the
% specified thresholds. You can give names to the class/bins, or leave the
% default names of LowLow, LowHigh, HighLow, HighHigh. This module requires
% that you run measurement modules previous to this module in the pipeline
% so that the measurement values can be used to classify the objects.
% Currently, classifying by the ratio of two measurements is unavailable.
%
% Settings:
%
% Feature Number:
% The feature number specifies which feature from the Measure module will
% be used for classifying. See each Measure module's help for the numbered
% list of the features measured by that module.
%
% See also ClassifyObjects, FilterByObjectMeasurement.

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

%textVAR01 = What did you call the objects that you want to classify into bins? 
%infotypeVAR01 = objectgroup
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu custom

%textVAR02 = Enter the first feature type to use:
Category{1} = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu category

%textVAR03 = Enter the first feature number to use (see help):
%defaultVAR03 = 1
%inputtypeVAR03 = popupmenu measurement
FeatureNbr{1} = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = For INTENSITY, AREAOCCUPIED or TEXTURE features, which image's measurements would you like to use for the first feature?
%infotypeVAR04 = imagegroup
%inputtypeVAR04 = popupmenu
ImageName{1} = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = For TEXTURE, RADIAL DISTRIBUTION, OR NEIGHBORS features, what previously measured size scale (TEXTURE OR NEIGHBORS) or previously used number of bins (RADIALDISTRIBUTION) do you want to use?
%defaultVAR05 = 1
%inputtypeVAR05 = popupmenu scale
SizeScale{1} = str2double(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = Enter the second feature type to use:
Category{2} = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%inputtypeVAR06 = popupmenu category

%textVAR07 = Enter the second feature number to use (see help):
%defaultVAR07 = 1
%inputtypeVAR07 = popupmenu measurement
FeatureNbr{2} = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = For INTENSITY, AREAOCCUPIED or TEXTURE features, which image's measurements would you like to use for the second feature?
%infotypeVAR08 = imagegroup
%inputtypeVAR08 = popupmenu
ImageName{2} = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%textVAR09 = For TEXTURE, RADIAL DISTRIBUTION, OR NEIGHBORS features, what previously measured size scale (TEXTURE OR NEIGHBORS) or previously used number of bins (RADIALDISTRIBUTION) do you want to use?
%defaultVAR09 = 1
%inputtypeVAR09 = popupmenu scale
SizeScale{2} = str2double(handles.Settings.VariableValues{CurrentModuleNum,9});

%textVAR10 = Enter threshold for the first set of measurements. Enter a number, "Mean", or "Median"
%defaultVAR10 = Mean
Separator{1} = handles.Settings.VariableValues{CurrentModuleNum,10};

%textVAR11 = Enter threshold for the second set of measurements. Enter a number, "Mean", or "Median"
%defaultVAR11 = Mean
Separator{2} = handles.Settings.VariableValues{CurrentModuleNum,11};

%textVAR12 = If you want to label your bins, enter the bin labels separated by commas (e.g. bin1,bin2,bin3,bin4). If the number of labels does not equal the number of bins (which is 4), this step will be ignored. Leave "Do not use" for default labels.
%%%TODO: Instead of "Do not use", this should just list the actual default
%%%names that will be used (which are what? bin1,bin2,bin3,bin4?)
%defaultVAR12 = Do not use
Labels = handles.Settings.VariableValues{CurrentModuleNum,12};

%textVAR13 = What do you want to call the resulting color-coded image?
%defaultVAR13 = Do not use
%infotypeVAR13 = imagegroup indep
SaveColoredObjects = char(handles.Settings.VariableValues{CurrentModuleNum,13});

%%%VariableRevisionNumber = 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PRELIMINARY CALCULATIONS & FILE HANDLING %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;

% Get the correct fieldnames where measurements are located
for FeatNum=2:-1:1
    try
        FeatureName{FeatNum} = CPgetfeaturenamesfromnumbers(handles,ObjectName,Category{FeatNum},...
            FeatureNbr{FeatNum},ImageName{FeatNum},SizeScale{FeatNum}); %#ok<AGROW>
    catch
        error([lasterr '  Image processing was canceled in the ', ModuleName, ...
            ' module (#' num2str(CurrentModuleNum) ...
            ') because an error ocurred when retrieving the data.  '...
            'Likely the category of measurement you chose, ',...
            Category{FeatNum}, ', was not available for ', ...
            ObjectName,' with feature number ' num2str(FeatureNbr{FeatNum}) ...
            ', possibly specific to image ''' ImageName{FeatNum} ''' and/or ' ...
            'Texture Scale = ' num2str(SizeScale{FeatNum}) '.']);
    end
end

% Retrieves the label matrix image that contains the segmented objects
fieldname = ['Segmented', ObjectName];
% Checks whether the image exists in the handles structure.
if isfield(handles.Pipeline, fieldname)
    LabelMatrixImage = CPretrieveimage(handles,fieldname,ModuleName);
else
    error(['Image processing was canceled in the ', ModuleName, ' module. Prior to running the ', ModuleName, ' module, you must have run a module that generates an image with the objects identified. You specified in the ', ModuleName, ' module that the objects were named ',ObjectName,', which should have produced an image in the handles structure called ', fieldname, '. The ', ModuleName, ' module cannot locate this image.']);
end

% Check labels
if strcmp(Labels,'Do not use')
    BinLabels = {'LowLow' 'LowHigh' 'HighLow' 'HighHigh'};
else
    BinLabels = cell(1,4);
    [BinLabels{1}, Labels] = strtok(Labels,',');
    [BinLabels{2}, Labels] = strtok(Labels,',');
    [BinLabels{3}, Labels] = strtok(Labels,',');
    [BinLabels{4}, Labels] = strtok(Labels,',');
    if ~isempty(Labels)
        error(['Image processing was canceled in the ' ModuleName ' module because you provided more than four labels.']);
    end
end

%%%%%%%%%%%%%%%%%%
% IMAGE ANALYSIS %
%%%%%%%%%%%%%%%%%%
drawnow

% Get Measurements, and limits for bins
SetNumber = {'first' 'second'};
Measurements = cell(1,2); % Preallocate
LowerBinMin = zeros(1,2);
UpperBinMax = zeros(1,2);
for FeatNum=2:-1:1
    % Measurements
    try
        Measurements{FeatNum} = handles.Measurements.(ObjectName).(FeatureName{FeatNum}){SetBeingAnalyzed};
    catch
        error(['Image processing was canceled in the ', ModuleName, ' module because an error ocurred when retrieving the ' SetNumber{FeatNum} ' set of data. Either the category of measurement you chose, ', FeatureName{1},', was not available for ', ObjectName,', or the feature number, ', num2str(FeatureNbr{1}), ', exceeded the amount of measurements.']);
    end
    % Limits
    if strcmpi(Separator{FeatNum},'Mean')
        Separator{FeatNum} = mean(Measurements{FeatNum}); %#ok<AGROW>
    elseif strcmpi(Separator{FeatNum},'Median')
        Separator{FeatNum} = median(Measurements{FeatNum}); %#ok<AGROW>
    else
        Separator{FeatNum} = str2double(Separator{FeatNum}); %#ok<AGROW>
    end
    if isnan(Separator{FeatNum})
        error(['Image processing was canceled in the ' ModuleName ' module because the threshold ' num2str(Separator{FeatNum}) ' you selected for the ' SetNumber{FeatNum} ' set of data is invalid. Please enter a number, or the word ''Mean'', ''Median'', or ''Mode''']);
    end
    LowerBinMin(FeatNum) = min(Measurements{FeatNum}) - sqrt(eps); % Just a fix so that objects with a measurement that equals the lower bin edge of the lowest bin are counted
    UpperBinMax(FeatNum) = max(Measurements{FeatNum});
%     if Separator{FeatNum}<LowerBinMin(FeatNum) || Separator{FeatNum}>UpperBinMax(FeatNum)
%         error(['Image processing was canceled in the ' ModuleName ' module because the threshold ' num2str(Separator{FeatNum}) ' you selected for the ' SetNumber{FeatNum} ' set of data is invalid. The threshold should be a number in between the minimum and maximum measurements of the data, which are ' num2str(LowerBinMin(FeatNum)) ' and ' num2str(UpperBinMax(FeatNum)) ', respectively.']);
%     end
end

if length(Measurements{1}) ~= length(Measurements{2})
    error(['Image processing was canceled in the ' ModuleName ' module because the number of measurements in each set was not the same. This means some objects may be missing measurements.']);
else
    NbrOfObjects = length(Measurements{1});
end

% Separate objects into bins
QuantizedMeasurements = zeros(NbrOfObjects,1);
bins = zeros(1,4);
BinFlag{1} = Measurements{1} <= Separator{1} & Measurements{2} <= Separator{2};
bin_index{1} = find(BinFlag{1});  %LowLow
BinFlag{2} = Measurements{1} <= Separator{1} & Measurements{2} > Separator{2};
bin_index{2} = find(BinFlag{2});   %LowHigh
BinFlag{3} = Measurements{1} > Separator{1} & Measurements{2} <= Separator{2};
bin_index{3} = find(BinFlag{3});   %HighLow
BinFlag{4} = Measurements{1} > Separator{1} & Measurements{2} > Separator{2};
bin_index{4} = find(BinFlag{4});    %HighHigh

for BinNum=4:-1:1
    QuantizedMeasurements(bin_index{BinNum}) = BinNum;
    bins(BinNum) = length(bin_index{BinNum});
end

% Produce color image
QuantizedMeasurements = [0;QuantizedMeasurements];                 % Add a background class
QuantizedImage = QuantizedMeasurements(LabelMatrixImage+1);
handlescmap = str2func(handles.Preferences.LabelColorMap);
cmap = [0 0 0; handlescmap(length(bins))];
QuantizedRGBimage = ind2rgb(QuantizedImage+1,cmap);

%%%%%%%%%%%%%%%%%%%
% DISPLAY RESULTS %
%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    % Activates the appropriate figure window.
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    if SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure(QuantizedRGBimage,'TwoByTwo',ThisModuleFigureNumber);
    end

    % Produce and plot histograms of first and second sets of data
    FontSize = handles.Preferences.FontSize;
    for FeatNum=1:2
        hAx = subplot(2,2,FeatNum,'Parent',ThisModuleFigureNumber);
        Nbins = min(round(NbrOfObjects),40);
        hist(hAx,Measurements{FeatNum},Nbins);
        xlabel(hAx,FeatureName{FeatNum},'Fontsize',FontSize);
        ylabel(hAx,['# of ' ObjectName],'Fontsize',FontSize);
        title(hAx,['Histogram of ' FeatureName{FeatNum}],'Fontsize',FontSize);
        % Using "axis tight" here is ok, I think, because we are displaying
        % data, not images.
        ylimits = ylim;
        axis(hAx,'tight');
        xlimits = xlim;
        axis(hAx,[xlimits ylimits]);
    end

    % A subplot of the figure window is set to display the quantized image.
    hAx = subplot(2,2,3,'Parent',ThisModuleFigureNumber);
    CPimagesc(QuantizedRGBimage,handles,hAx);
    title(hAx,['Classified ', ObjectName]);

    % Produce and plot histogram
    hAx = subplot(2,2,4,'Parent',ThisModuleFigureNumber);
    x = 1:4;
    h = bar(hAx,x,bins,1);
    warning('off','MATLAB:hg:patch:RGBColorDataNotSupported');
    set(get(h,'children'),'facevertexcdata',cmap(2:end,:)); % Color code acording to adjacent plot
    warning('on','MATLAB:hg:patch:RGBColorDataNotSupported');
    xlabel(hAx,['Labels: ' BinLabels{1} ', ' BinLabels{2} ', ' BinLabels{3} ', ' BinLabels{4}],'fontsize',FontSize);
    ylabel(hAx,['# of ',ObjectName],'fontsize',FontSize);
    title(hAx,['Classified by ' FeatureName{1} ', ' FeatureName{2}],'fontsize',FontSize);
    % Using "axis tight" here is ok, I think, because we are displaying
    % data, not images.
    axis(hAx,'tight');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SAVE DATA TO HANDLES STRUCTURE %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

% Save QuantizedRGBimage or not?
if ~strcmpi(SaveColoredObjects,'Do not use')
    handles.Pipeline.(SaveColoredObjects) = QuantizedRGBimage;
end

% Calculate Objects per Bin (Absolute and Percentage)
ObjectsPerBin = bins;
PercentageOfObjectsPerBin = bins/sum(bins);

% Save FeatureNames and the indices of the objects that fall into each
% bin, as well as ObjsPerBin
for BinNum = 1:4
    ClassifyFeatureNames = ['ClassifyObjsByTwoMeas_Module',CurrentModule,'Bin',num2str(BinNum)];
    handles = CPaddmeasurements(handles, ObjectName, [ClassifyFeatureNames 'Flag'], BinFlag{BinNum});
    handles = CPaddmeasurements(handles, 'Image', [ClassifyFeatureNames 'ObjectsPerBin'], ObjectsPerBin(BinNum));
    handles = CPaddmeasurements(handles, 'Image', [ClassifyFeatureNames 'PercentageOfObjectsPerBin'], PercentageOfObjectsPerBin(BinNum));
end

% Note: no need to save 'edges' here, as in ClassifyObjects, since the bins here
% are simply categories 1:4