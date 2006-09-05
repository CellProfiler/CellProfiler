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
% $Revision: 4076 $

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
Measure{1} = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu custom

%textVAR03 = Enter the first feature number to use (see help):
%defaultVAR03 = 1
FeatureNumber{1} = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,3}));

%textVAR04 = For INTENSITY or TEXTURE features, which image's measurements would you like to use for the first feature?
%infotypeVAR04 = imagegroup
%inputtypeVAR04 = popupmenu
Image{1} = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = Enter the second feature type to use:
%choiceVAR05 = AreaShape
%choiceVAR05 = Children
%choiceVAR05 = Correlation
%choiceVAR05 = Texture
%choiceVAR05 = Intensity
%choiceVAR05 = Neighbors
%choiceVAR05 = Ratio
Measure{2} = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu custom

%textVAR06 = Enter the second feature number to use (see help):
%defaultVAR06 = 1
FeatureNumber{2} = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,6}));

%textVAR07 = For INTENSITY or TEXTURE features, which image's measurements would you like to use for the second feature?
%infotypeVAR07 = imagegroup
%inputtypeVAR07 = popupmenu
Image{2} = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = Enter threshold for the first set of measurements. Enter a number, "Mean", or "Median"
%defaultVAR08 = Mean
Separator{1} = handles.Settings.VariableValues{CurrentModuleNum,8};

%textVAR09 = Enter threshold for the second set of measurements. Enter a number, "Mean", or "Median"
%defaultVAR09 = Mean
Separator{2} = handles.Settings.VariableValues{CurrentModuleNum,9};

%textVAR10 = If you want to label your bins, enter the bin labels separated by commas (e.g. bin1,bin2,bin3,bin4). If the number of labels does not equal the number of bins (which is 4), this step will be ignored. Leave "/" for default labels.
%defaultVAR10 = /
Labels = handles.Settings.VariableValues{CurrentModuleNum,10};

%textVAR11 = What do you want to call the resulting color-coded image?
%choiceVAR11 = Do not save
%choiceVAR11 = ColorClassifiedNuclei
%inputtypeVAR11 = popupmenu custom
%infotypeVAR11 = imagegroup indep
SaveColoredObjects = char(handles.Settings.VariableValues{CurrentModuleNum,11});

%textVAR12 = Do you want the absolute number of objects or percentage of object?
%choiceVAR12 = Absolute
%choiceVAR12 = Percentage
%inputtypeVAR12 = popupmenu
AbsoluteOrPercentage = char(handles.Settings.VariableValues{CurrentModuleNum,12});

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;

%%% Get the correct fieldnames where measurements are located
for i=1:2
    CurrentMeasure = Measure{i};
    CurrentImage = Image{i};
    switch CurrentMeasure
        case 'Intensity'
            CurrentMeasure = ['Intensity_' CurrentImage];
        case 'Texture'
            CurrentMeasure = ['Texture_[0-9]*[_]?' CurrentImage '$'];
            Fields = fieldnames(handles.Measurements.(CurrentObjectName));
            TextComp = regexp(Fields,CurrentMeasure);
            A = cellfun('isempty',TextComp);
            try
                CurrentMeasure = Fields{A==0};
            catch
                error(['Image processing was canceled in the ', ModuleName, ' module because the category of measurement you chose, ', Measure{i}, ', was not available for ', ObjectName{i}]);
            end
    end
    Measure{i} = CurrentMeasure;
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
for i=1:2
    % Measurements
    try
        Measurements{i} = handles.Measurements.(ObjectName).(Measure{i}){SetBeingAnalyzed}(:,FeatureNumber{i});
    catch
        error(['Image processing was canceled in the ', ModuleName, ' module because an error ocurred when retrieving the ' SetNumber{i} ' set of data. Either the category of measurement you chose, ', Measure{1},', was not available for ', ObjectName,', or the feature number, ', num2str(FeatureNumber{1}), ', exceeded the amount of measurements.']);
    end
    % Limits
    if strcmpi(Separator{i},'Mean')
        Separator{i} = mean(Measurements{i});
    elseif strcmpi(Separator{i},'Median')
        Separator{i} = median(Measurements{i});
    else
        Separator{i} = str2double(Separator{i});
    end
    if isnan(Separator{i})
        error(['Image processing was canceled in the ' ModuleName ' module because the threshold ' num2str(Separator{i}) ' you selected for the ' SetNumber{i} ' set of data is invalid. Please enter a number, or the word ''Mean'', ''Median'', or ''Mode''']);
    end
    LowerBinMin(i) = min(Measurements{i}) - sqrt(eps); % Just a fix so that objects with a measurement that equals the lower bin edge of the lowest bin are counted
    UpperBinMax(i) = max(Measurements{i});
    if Separator{i}<LowerBinMin(i) || Separator{i}>UpperBinMax(i)
        error(['Image processing was canceled in the ' ModuleName ' module because the threshold ' num2str(Separator{i}) ' you selected for the ' SetNumber{i} ' set of data is invalid. The threshold should be a number in between the minimum and maximum measurements of the data, which are ' num2str(LowerBinMin(i)) ' and ' num2str(UpperBinMax(i)) ', respectively.']);
    end
end

if length(Measurements{1}) ~= length(Measurements{2})
    error(['Image processing was canceled in the ' ModuleName ' module because the number of measurements in each set was not the same. This means some objects may be missing measurements.']);
else
    NbrOfObjects = length(Measurements{1});
end

%%% Separate objects into bins
QuantizedMeasurements = zeros(NbrOfObjects,1);
Bins = zeros(1,4);
index{1} = find(Measurements{1} <= Separator{1} & Measurements{2} <= Separator{2});  %LowLow
index{2} = find(Measurements{1} <= Separator{1} & Measurements{2} > Separator{2});   %LowHigh
index{3} = find(Measurements{1} > Separator{1} & Measurements{2} <= Separator{2});   %HighLow
index{4} = find(Measurements{1} > Separator{1} & Measurements{2} > Separator{2});    %HighHigh
for i=1:4
    QuantizedMeasurements(index{i}) = i;
    Bins(i) = length(index{i});
end

%%% Produce color image
QuantizedMeasurements = [0;QuantizedMeasurements];                 % Add a background class
QuantizedImage = QuantizedMeasurements(LabelMatrixImage+1);
handlescmap = handles.Preferences.LabelColorMap;
cmap = [0 0 0;feval(handlescmap,length(Bins))];
QuantizedRGBimage = ind2rgb(QuantizedImage+1,cmap);

%%% Get feature names
FeatureName{1} = handles.Measurements.(ObjectName).([Measure{1},'Features']){FeatureNumber{1}};
FeatureName{2} = handles.Measurements.(ObjectName).([Measure{2},'Features']){FeatureNumber{2}};


%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    %%% Activates the appropriate figure window.
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure(QuantizedRGBimage,'TwoByTwo',ThisModuleFigureNumber);
    end
    AdjustedObjectName = strrep(ObjectName,'_','\_');
    AdjustedFeatureName{1} = strrep(FeatureName{1},'_','\_');
    AdjustedFeatureName{2} = strrep(FeatureName{2},'_','\_');

    %%% Produce and plot histograms of first and second sets of data
    FontSize = handles.Preferences.FontSize;
    for i=1:2
        subplot(2,2,i)
        Nbins = min(round(NbrOfObjects),40);
        hist(Measurements{i},Nbins);
        xlabel(AdjustedFeatureName{i},'Fontsize',FontSize);
        ylabel(['# of ' AdjustedObjectName],'Fontsize',FontSize);
        title(['Histogram of ' Measure{i}],'Fontsize',FontSize);
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
    bar(x,Bins,1);
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

%%% Save or not?
if ~strcmpi(SaveColoredObjects,'Do not save')
    handles.Pipeline.(SaveColoredObjects) = QuantizedRGBimage;
end

FeatureName{1} = FeatureName{1}(~isspace(FeatureName{1}));                    % Remove spaces in the feature name
FeatureName{2} = FeatureName{2}(~isspace(FeatureName{2}));
%%% We are truncating the ObjectName in case it's really long.
MaxLengthOfFieldname{1} = min(20,length(FeatureName{1}));
MaxLengthOfFieldname{2} = min(20,length(FeatureName{2}));

%%% Save to handles
handles.Measurements.(ObjectName).(['Classified',ObjectName,'_',FeatureName{1}(1:MaxLengthOfFieldname{1}),'_',FeatureName{2}(1:MaxLengthOfFieldname{2}),'Features']) = BinLabels;
if strcmp(AbsoluteOrPercentage,'Percentage')
    handles.Measurements.(ObjectName).(['Classified',ObjectName,'_',FeatureName{1}(1:MaxLengthOfFieldname{1}),'_',FeatureName{2}(1:MaxLengthOfFieldname{2})])(SetBeingAnalyzed) = {Bins/length(Measurements)};
else
    handles.Measurements.(ObjectName).(['Classified',ObjectName,'_',FeatureName{1}(1:MaxLengthOfFieldname{1}),'_',FeatureName{2}(1:MaxLengthOfFieldname{2})])(SetBeingAnalyzed) = {Bins};
end