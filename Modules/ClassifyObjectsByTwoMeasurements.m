function handles = ClassifyObjectsByTwoMeasurements(handles)

% Help for the Classify Objects module:
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
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the identified objects?
%infotypeVAR01 = objectgroup
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu custom

%textVAR02 = Enter the first feature type to use:
%choiceVAR02 = AreaShape
%choiceVAR02 = Correlation
%choiceVAR02 = Texture
%choiceVAR02 = Intensity
%choiceVAR02 = Neighbors
Category{1} = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu custom

%textVAR03 = Enter the first feature number to use (see help):
%defaultVAR03 = 1
FeatureNbr{1} = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,3}));

%textVAR04 = For INTENSITY or TEXTURE features, which image's measurements would you like to use for the first feature?
%infotypeVAR04 = imagegroup
%inputtypeVAR04 = popupmenu
Image{1} = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = For TEXTURE features, what previously measured texture scale do you want to use?
%defaultVAR05 = 1
TextureScale{1} = str2double(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = Enter the second feature type to use:
%choiceVAR06 = AreaShape
%choiceVAR06 = Children
%choiceVAR06 = Correlation
%choiceVAR06 = Texture
%choiceVAR06 = Intensity
%choiceVAR06 = Neighbors
%choiceVAR06 = Ratio
Category{2} = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%inputtypeVAR06 = popupmenu custom

%textVAR07 = Enter the second feature number to use (see help):
%defaultVAR07 = 1
FeatureNbr{2} = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,7}));

%textVAR08 = For INTENSITY or TEXTURE features, which image's measurements would you like to use for the second feature?
%infotypeVAR08 = imagegroup
%inputtypeVAR08 = popupmenu
Image{2} = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%textVAR09 = For TEXTURE features, what previously measured texture scale do you want to use?
%defaultVAR09 = 1
TextureScale{2} = str2double(handles.Settings.VariableValues{CurrentModuleNum,9});

%textVAR10 = Enter threshold for the first set of measurements. Enter a number, "Mean", or "Median"
%defaultVAR10 = Mean
Separator{1} = handles.Settings.VariableValues{CurrentModuleNum,10};

%textVAR11 = Enter threshold for the second set of measurements. Enter a number, "Mean", or "Median"
%defaultVAR11 = Mean
Separator{2} = handles.Settings.VariableValues{CurrentModuleNum,11};

%textVAR12 = If you want to label your bins, enter the bin labels separated by commas (e.g. bin1,bin2,bin3,bin4). If the number of labels does not equal the number of bins (which is 4), this step will be ignored. Leave "/" for default labels.
%defaultVAR12 = /
Labels = handles.Settings.VariableValues{CurrentModuleNum,12};

%textVAR13 = What do you want to call the resulting color-coded image?
%choiceVAR13 = Do not save
%choiceVAR13 = ColorClassifiedNuclei
%inputtypeVAR13 = popupmenu custom
%infotypeVAR13 = imagegroup indep
SaveColoredObjects = char(handles.Settings.VariableValues{CurrentModuleNum,13});

%textVAR14 = Do you want the absolute number of objects or percentage of object?
%choiceVAR14 = Absolute
%choiceVAR14 = Percentage
%inputtypeVAR14 = popupmenu
AbsoluteOrPercentage = char(handles.Settings.VariableValues{CurrentModuleNum,14});

%%%VariableRevisionNumber = 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;

%%% Get the correct fieldnames where measurements are located
for FeatNum=2:-1:1
    try
        FeatureName{FeatNum} = CPgetfeaturenamesfromnumbers(handles,ObjectName,Category{FeatNum},...
            FeatureNbr{FeatNum},Image{FeatNum},TextureScale{FeatNum}); %#ok<AGROW>
    catch
        error(['Image processing was canceled in the ', ModuleName, ...
            ' module (#' num2str(CurrentModuleNum) ...
            ') because an error ocurred when retrieving the data.  '...
            'Likely the category of measurement you chose, ',...
            Category{FeatNum}, ', was not available for ', ...
            ObjectName,' with feature number ' num2str(FeatureNbr{FeatNum}) ...
            ', possibly specific to image ''' Image{FeatNum} ''' and/or ' ...
            'Texture Scale = ' num2str(TextureScale{FeatNum}) '.']);
    end
end

%%% Retrieves the label matrix image that contains the segmented objects
fieldname = ['Segmented', ObjectName];
%%% Checks whether the image exists in the handles structure.
if isfield(handles.Pipeline, fieldname)
    LabelMatrixImage = CPretrieveimage(handles,fieldname,ModuleName);
else
    error(['Image processing was canceled in the ', ModuleName, ' module. Prior to running the ', ModuleName, ' module, you must have run a module that generates an image with the objects identified. You specified in the ', ModuleName, ' module that the objects were named ',ObjectName,', which should have produced an image in the handles structure called ', fieldname, '. The ', ModuleName, ' module cannot locate this image.']);
end

%%% Check labels
if strcmp(Labels,'/')
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


%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Get Measurements, and limits for bins
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
    if Separator{FeatNum}<LowerBinMin(FeatNum) || Separator{FeatNum}>UpperBinMax(FeatNum)
        error(['Image processing was canceled in the ' ModuleName ' module because the threshold ' num2str(Separator{FeatNum}) ' you selected for the ' SetNumber{FeatNum} ' set of data is invalid. The threshold should be a number in between the minimum and maximum measurements of the data, which are ' num2str(LowerBinMin(FeatNum)) ' and ' num2str(UpperBinMax(FeatNum)) ', respectively.']);
    end
end

if length(Measurements{1}) ~= length(Measurements{2})
    error(['Image processing was canceled in the ' ModuleName ' module because the number of measurements in each set was not the same. This means some objects may be missing measurements.']);
else
    NbrOfObjects = length(Measurements{1});
end

%%% Separate objects into bins
QuantizedMeasurements = zeros(NbrOfObjects,1);
bins = zeros(1,4);
bin_index{1} = find(Measurements{1} <= Separator{1} & Measurements{2} <= Separator{2});  %LowLow
bin_index{2} = find(Measurements{1} <= Separator{1} & Measurements{2} > Separator{2});   %LowHigh
bin_index{3} = find(Measurements{1} > Separator{1} & Measurements{2} <= Separator{2});   %HighLow
bin_index{4} = find(Measurements{1} > Separator{1} & Measurements{2} > Separator{2});    %HighHigh
for BinNum=4:-1:1
    QuantizedMeasurements(bin_index{BinNum}) = BinNum;
    bins(BinNum) = length(bin_index{BinNum});
end

%%% Produce color image
QuantizedMeasurements = [0;QuantizedMeasurements];                 % Add a background class
QuantizedImage = QuantizedMeasurements(LabelMatrixImage+1);
handlescmap = handles.Preferences.LabelColorMap;
cmap = [0 0 0;feval(handlescmap,length(bins))];
QuantizedRGBimage = ind2rgb(QuantizedImage+1,cmap);

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    %%% Activates the appropriate figure window.
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    if SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure(QuantizedRGBimage,'TwoByTwo',ThisModuleFigureNumber);
    end
    AdjustedObjectName = strrep(ObjectName,'_','\_');
    AdjustedFeatureName = strrep(FeatureName,'_','\_');

    %%% Produce and plot histograms of first and second sets of data
    FontSize = handles.Preferences.FontSize;
    for FeatNum=1:2
        subplot(2,2,FeatNum)
        Nbins = min(round(NbrOfObjects),40);
        hist(Measurements{FeatNum},Nbins);
        xlabel(AdjustedFeatureName{FeatNum},'Fontsize',FontSize);
        ylabel(['# of ' AdjustedObjectName],'Fontsize',FontSize);
        title(['Histogram of ' AdjustedFeatureName{FeatNum}],'Fontsize',FontSize);
        %%% Using "axis tight" here is ok, I think, because we are displaying
        %%% data, not images.
        ylimits = ylim;
        axis tight
        xlimits = xlim;
        axis([xlimits ylimits]);
    end

    %%% A subplot of the figure window is set to display the quantized image.
    subplot(2,2,3)
    CPimagesc(QuantizedRGBimage,handles);
    title(['Classified ', AdjustedObjectName]);

    %%% Produce and plot histogram
    subplot(2,2,4)
    x = 1:4;
    bar(x,bins,1);
    xlabel(['Labels: ' BinLabels{1} ', ' BinLabels{2} ', ' BinLabels{3} ', ' BinLabels{4}],'fontsize',FontSize);
    ylabel(['# of ',AdjustedObjectName],'fontsize',FontSize);
    title(['Classified by ' AdjustedFeatureName{1} ', ' AdjustedFeatureName{2}],'fontsize',FontSize);
    %%% Using "axis tight" here is ok, I think, because we are displaying
    %%% data, not images.
    axis tight
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Save QuantizedRGBimage or not?
if ~strcmpi(SaveColoredObjects,'Do not save')
    handles.Pipeline.(SaveColoredObjects) = QuantizedRGBimage;
end

FeatureName = FeatureName(~isspace(FeatureName));                    % Remove spaces in the feature name

%%% Save to handles
% fieldname = ['Classify_',ObjectName,'_',FeatureName{1},'_',FeatureName{2}];
% handles.Measurements.Image.([fieldname,'Features']) = BinLabels;
% if strcmp(AbsoluteOrPercentage,'Percentage')
%     handles.Measurements.Image.(fieldname)(SetBeingAnalyzed) = {bins/length(Measurements)};
% else
%     handles.Measurements.Image.(fieldname)(SetBeingAnalyzed) = {bins};
% end

%% Calculate Objects per Bin (Absolute or Percentage)
for FeatNum = 2:-1:1
    if strcmp(AbsoluteOrPercentage,'Percentage')
        ObjsPerBin{FeatNum} = bins/length(Measurements{FeatNum}); %#ok<AGROW>
    else
        ObjsPerBin{FeatNum} = bins; %#ok<AGROW>
    end
end

%% Save FeatureNames and the indices of the objects that fall into each
%% bin, as well as ObjsPerBin
for BinNum = 1:4
    ClassifyFeatureNames = ['ClassifyObjsByTwoMeas_Module',CurrentModule,'Bin',num2str(BinNum)];
    handles = CPaddmeasurements(handles, ObjectName, [ClassifyFeatureNames '_indices'], bin_index{BinNum});
    handles = CPaddmeasurements(handles, ObjectName, [ClassifyFeatureNames '_ObjsPerBin'], ObjsPerBin{FeatNum}(:,BinNum));
end

%% Note: no need to save 'edges' here, as in ClassifyObjects, since the bins here
%% are simply categories 1:4