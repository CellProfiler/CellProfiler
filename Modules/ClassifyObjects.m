function handles = ClassifyObjects(handles)

% Help for the Classify Objects module:
% Category: Object Processing
%
% SHORT DESCRIPTION:
% Classifies objects into different classes according to the value of a
% measurement you choose.
% *************************************************************************
%
% This module classifies objects into a number of different bins
% according to the value of a measurement (e.g. by size, intensity, shape).
% It reports how many objects fall into each class as well as the
% percentage of objects that fall into each class. The module requests that
% you select the measurement feature to be used to classify your objects and
% specify the bins to use. This module requires that you run a measurement
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
% TODO: IS THE FOLLOWING STILL TRUE?
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
%% TODO: What does that mean "As named in module's last setting"?
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
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the objects that you want to classify into bins? If you are classifying based on a ratio, enter the numerator object
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

%textVAR06 = Into what kind of bins do you want to classify objects?
%choiceVAR06 = Evenly spaced bins
%choiceVAR06 = Custom-defined bins
BinType = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%inputtypeVAR06 = popupmenu

%textVAR07 = For EVENLY SPACED BINS, type "NumberOfBins,LowerLimit,UpperLimit" (example: 5,0.2,0.8). For CUSTOM-DEFINED BINS, type "LowerLimit,MiddleBoundary,...,MiddleBoundary,UpperLimit" (example: 0,0.2,0.3,0.5,0.7,1,2), or to classify objects as being below or above a threshold value, just type the single threshold value (example: 0.2).
%defaultVAR07 = 5,0.2,0.8
BinSpecifications = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = To custom-name each class, enter the class names separated by commas (example: small,medium,large,supersized). To have names created for you, type "Do not use".
%defaultVAR08 = Do not use
Labels = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%textVAR09 = What do you want to call the resulting color-coded image?
%defaultVAR09 = Do not use
%infotypeVAR09 = imagegroup indep
SaveColoredObjects = char(handles.Settings.VariableValues{CurrentModuleNum,9});

%%%VariableRevisionNumber = 8

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
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
if ~strcmp(Category,'Ratio')
    %%TODO: if the category IS Ratio, shouldn't we check something here
    %%also?
    % Checks whether the feature type exists in the handles structure.
    if ~isfield(handles.Measurements.(ObjectName),FeatureName)
        error(['Image processing was canceled in the ', ModuleName, ...
            ' module because the feature type entered does not exist.']);
    end
end

% Retrieve measurements
Measurements = handles.Measurements.(ObjectName).(FeatureName){SetBeingAnalyzed};

% Checks whether the image exists in the handles structure.
SegmentedField = ['Segmented', ObjectName];
if isfield(handles.Pipeline, SegmentedField)
    % Retrieves the label matrix image that contains the segmented objects
    LabelMatrixImage = CPretrieveimage(handles,SegmentedField,ModuleName);
elseif strcmpi(Category,'Ratio')
    % If we are using a user defined field, there is no corresponding
    % image.
    %%% TODO: REALLY? IT SEEMS LIKE WE COULD USE THE NUMERATOR OBJECT IMAGE HERE?
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

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

try NumericalBinSpecifications = str2num(BinSpecifications);
catch error(['Image processing was canceled in the ', ModuleName, ...
        ' module because the bin specifications must be numerical values',...
        ' separated by commas. Please check your bin specifications.'])
end

NumberOfObjects = length(Measurements);

if strcmpi(BinType,'Evenly spaced bins')
    %%% Check that the proper number of arguments has been
    %%% provided:
    if length(NumericalBinSpecifications) ~= 3;
        error(['Image processing was canceled in the ', ModuleName, ...
            ' module because if you are specifying evenly spaced bins,', ...
            ' you must enter three numerical values separated by commas.', ...
            ' Please check your bin specifications.'])
    end
    NumberOfBins = NumericalBinSpecifications(1);
    LowerBinMin = NumericalBinSpecifications(2);
    UpperBinMax = NumericalBinSpecifications(3);
    % Check that the upper bin's value is higher than the lower bin's
    % value:
    if LowerBinMin > UpperBinMax
        error(['Image processing was canceled in the ', ModuleName, ...
            ' module because the value you entered for the Lower Limit is', ...
            'higher than the value you entered for the Upper Limit.', ...
            ' Please check your bin specifications.']);
    end
    % Quantize measurements to create bin edges
    Edges = linspace(LowerBinMin,UpperBinMax,NumberOfBins+1);
else % In this case, BinType must be 'Custom-defined bins'
    if length(NumericalBinSpecifications) == 1
        % In this case, the user entered a single value so they want two
        % bins: one above and one below a specified threshold.
        NumberOfBins = 2;
        if NumberOfObjects == 0
            LowerBinMin = -Inf;
            UpperBinMax = Inf;
        else
            LowerBinMin = min(Measurements(~isinf(Measurements)));
            UpperBinMax = max(Measurements(~isinf(Measurements)));
        end
        MidPointToUse = NumericalBinSpecifications;
        Edges = [LowerBinMin,MidPointToUse,UpperBinMax];
    else
        % In this case, the user entered several values so they want bins
        % with custom-entered edges.
        NumberOfBins = length(NumericalBinSpecifications)-1;
        Edges = NumericalBinSpecifications;
        LowerBinMin = NumericalBinSpecifications(1);
        UpperBinMax = NumericalBinSpecifications(end);
    end
end

RemainingLabels = Labels;
for BinNum = 1:NumberOfBins
    if strcmpi(Labels,'Do not use')
        % Assign bins with generic names
        BinLabels = ['Module',CurrentModule,'Bin',num2str(BinNum),];
    elseif length(strfind(Labels,',')) == (NumberOfBins - 1)
        [BinLabels,RemainingLabels] = strtok(RemainingLabels,',');
        if ~isvarname(BinLabels)
            error(['Image processing was canceled in the ', ModuleName, ...
                ' module because the number of bin names you entered does',...
                ' not match the number of bins you specified', ...
                ' Please check your bin specifications.']);
        end
    else error(['Image processing was canceled in the ', ModuleName, ...
            ' module because the number of bin names you entered does',...
            ' not match the number of bins you specified', ...
            ' Please check your bin specifications.']);
    end
    ClassifyFeatureNames{BinNum} = ['ClassifyObjects_' BinLabels];
end

% Initialize variables, in case there are no objects in this cycle.
BinFlag = cell(NumberOfBins);
ObjectsPerBin = zeros(1,NumberOfBins);
PercentageOfObjectsPerBin = zeros(1,NumberOfBins);

if NumberOfObjects>0
    QuantizedMeasurements = zeros(size(Measurements));
    for BinNum = 1:NumberOfBins
        bin_index{BinNum} = find(Measurements > Edges(BinNum) & Measurements <= Edges(BinNum+1));
        BinFlag{BinNum} = Measurements > Edges(BinNum) & Measurements <= Edges(BinNum+1);
        QuantizedMeasurements(bin_index{BinNum}) = BinNum;
        %TODO: Here we should add the number of objects EQUAL TO the bin edge
        %to the count:
        ObjectsPerBin(BinNum) = length(bin_index{BinNum});
    end

    %%% TODO: Producing this image should be conditional on a figure window open or
    %%% on the image being saved to the handles structure for downstream.
    if ~strcmpi(Category,'Ratio')
        % Produce image where the the objects are colored according to the original
        % measurements and the quantized measurements (though this does not apply to 'Ratio')
        NonQuantizedImage = zeros(size(LabelMatrixImage));
        props = regionprops(LabelMatrixImage,'PixelIdxList');              % Pixel indexes for objects fast
        if ~isempty(props)
            for BinNum = 1:NumberOfObjects
                NonQuantizedImage(props(BinNum).PixelIdxList) = Measurements(BinNum);
            end
            QuantizedMeasurementsWithBackground = [0;QuantizedMeasurements];                 % Add a background class
            QuantizedImage = QuantizedMeasurementsWithBackground(LabelMatrixImage+1);
            handlescmap = handles.Preferences.LabelColorMap;
            cmap = [0 0 0;feval(handlescmap,length(ObjectsPerBin))];
            QuantizedRGBimage = ind2rgb(QuantizedImage+1,cmap);
        else
            QuantizedRGBimage = NonQuantizedImage;
        end
    end

    % Calculate the percentage of objects per bin
    PercentageOfObjectsPerBin = ObjectsPerBin/length(Measurements);

    %%%%%%%%%%%%%%%%%%%%%%%
    %%% DISPLAY RESULTS %%%
    %%%%%%%%%%%%%%%%%%%%%%%
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
        Nbins = min(round(NumberOfObjects),40);
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
            hAx = subplot(2,2,3,'Parent',ThisModuleFigureNumber);
            CPimagesc(QuantizedRGBimage,handles,hAx);
            title(hAx,['Classified ', ObjectName]);
            % TODO add legend
        end
        
        if isinf(Edges(1))
            Edges(1) = min(Measurements(~isinf(Measurements)));
            % TODO: what happens here, and in the next "if", when there is
            % just one object, so min = max?
            % just for plotting purposes
        end
        if isinf(Edges(end))
            Edges(end) = max(Measurements(~isinf(Measurements)));
            % just for plotting purposes
        end
        
        % Plot histogram
        hAx=subplot(2,2,4,'Parent',ThisModuleFigureNumber);
        bar_ctr = Edges(1:end-1) + (Edges(2)-Edges(1))/2;
        h = bar(hAx,bar_ctr,ObjectsPerBin,1);
        xlabel(hAx,FeatureName,'fontsize',handles.Preferences.FontSize);
        ylabel(hAx,['# of ',ObjectName],'fontsize',handles.Preferences.FontSize);
        title(hAx,['Classified by ',Category],'fontsize',handles.Preferences.FontSize);
        % Using "axis tight" here is ok, I think, because we are displaying
        % data, not images.
        axis(hAx,'tight');
        xlimits(1) = min(xlimits(1),LowerBinMin);   % Extend limits if necessary and save them
        xlimits(2) = max(UpperBinMax,Edges(end));   % so they can be used for the second histogram
        axis(hAx,[xlimits ylim]);
        % Took this out: don't want to use misleading colors.
        %     handlescmap = handles.Preferences.LabelColorMap;
        %     set(get(h,'Children'),'FaceVertexCData',feval(handlescmap,max(2,NumberOfBins)));
    end

else
    %%TODO: Alternative display for no objects
    %% We might display some parts of the above, though. not sure yet.

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

% If we are using a user defined field, there is no corresponding
% image.
% TODO: see if this is really true that the image can't be saved (see
% comment above about NUMERATOR).
if strcmpi(Category,'Ratio') && ~strcmpi(SaveColoredObjects,'Do not use')
    error(['Image processing was canceled in the ', ModuleName, ' module ' ...
        'because you have requested to save the resulting color-coded image' ...
        'called ',SaveColoredObjects,' but that image cannot be produced by' ...
        'the Classify module. The color-coded image can only be produced ' ...
        'when using measurements straight from a Measure module, not when ' ...
        'using measurements from a CalculateRatios module. Sorry for the inconvenience.']);
end

if ~strcmpi(SaveColoredObjects,'Do not use')
    if NumberOfObjects == 0
        QuantizedRGBimage = zeros(size(LabelMatrixImage));
    end
    % Saves images to the handles structure so they can be saved to the hard
    % drive, if the user requests.
    handles.Pipeline.(SaveColoredObjects) = QuantizedRGBimage;
end

% Save FeatureNames and the indices of the objects that fall into each
% bin, as well as ObjectsPerBin
for BinNum = 1:NumberOfBins
    handles = CPaddmeasurements(handles, ObjectName, [ClassifyFeatureNames{BinNum} 'Flag'], BinFlag{BinNum});
    handles = CPaddmeasurements(handles, 'Image', [ClassifyFeatureNames{BinNum} 'ObjectsPerBin'], ObjectsPerBin(:,BinNum));
    handles = CPaddmeasurements(handles, 'Image', [ClassifyFeatureNames{BinNum} 'PercentageOfObjectsPerBin'], PercentageOfObjectsPerBin(:,BinNum));
end

% We decided to comment out the following; it could be saved as an image
% measurement, but the bin edges will not change from one cycle to the
% next.
% Save Bin Edges
% handles = CPaddmeasurements(handles,'Image',[ClassifyFeatureNames{BinNum} 'BinEdges'],Edges');