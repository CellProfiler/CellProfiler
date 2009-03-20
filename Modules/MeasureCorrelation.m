function handles = MeasureCorrelation(handles,varargin)

% Help for the Measure Correlation module:
% Category: Measurement
%
% SHORT DESCRIPTION:
% Measures the correlation between intensities in different images (e.g.
% different color channels) on a pixel by pixel basis, within identified
% objects or across an entire image.
% *************************************************************************
%
% Given two or more images, calculates the correlation between the
% pixel intensities. The correlation can be measured for the entire
% images, or individual correlation measurements can be made within each
% individual object. For example:
%                                     Image overall:  In Nuclei:
% OrigBlue_OrigGreen    Correlation:    0.49955        -0.07395
% OrigBlue_OrigRed      Correlation:    0.59886        -0.02752
% OrigGreen_OrigRed     Correlation:    0.83605         0.68489
%
% Features measured:      Feature Number:
% Correlation          |         1
% Slope                |         2
%
% See also MeasureObjectIntensity, MeasureImageIntensity.

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

% MBray 2009_03_20: Comments on variables for pyCP upgrade
%
% Recommended variable order (setting, followed by current variable in MATLAB CP)
% (1) What images would you like use to measure pixel correlations? You must choose
%   at least two. (ImageName{1}, ImageName{2})
% A button should be added that lets the user add/substract images for (1),
% but must have no less than 2 images.
%
% (2) If you would like to measure pixel correlations within an object, what object 
%   would you like to use? (ObjectName{1})
%
% (i) A button should be added that lets the user add/substract objects for (2)
%   for the images in (1)
% (ii) Setting (2) should default to a "Do not use" option.
% (iii) A button should be added after (2) that lets the user add/substract
% images (and more asociated objects)

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%

drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = Choose at least two image types to measure correlations between:
%choiceVAR01 = Do not use
%infotypeVAR01 = imagegroup
ImageName{1} = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = (All pairwise correlations will be measured)
%choiceVAR02 = Do not use
%infotypeVAR02 = imagegroup
ImageName{2} = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 =
%choiceVAR03 = Do not use
%infotypeVAR03 = imagegroup
ImageName{3} = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 =
%choiceVAR04 = Do not use
%infotypeVAR04 = imagegroup
ImageName{4} = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%textVAR05 = Choose objects within which to measure the correlations (Choosing Image will measure correlations across the entire images)
%choiceVAR05 = Do not use
%choiceVAR05 = Image
%infotypeVAR05 = objectgroup
ObjectName{1} = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu

%textVAR06 =
%choiceVAR06 = Do not use
%choiceVAR06 = Image
%infotypeVAR06 = objectgroup
ObjectName{2} = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%inputtypeVAR06 = popupmenu

%textVAR07 =
%choiceVAR07 = Do not use
%choiceVAR07 = Image
%infotypeVAR07 = objectgroup
ObjectName{3} = char(handles.Settings.VariableValues{CurrentModuleNum,7});
%inputtypeVAR07 = popupmenu

%textVAR08 =
%choiceVAR08 = Do not use
%choiceVAR09 = Image
%infotypeVAR08 = objectgroup
ObjectName{4} = char(handles.Settings.VariableValues{CurrentModuleNum,8});
%inputtypeVAR08 = popupmenu

%textVAR09 =
%choiceVAR09 = Do not use
%choiceVAR09 = Image
%infotypeVAR09 = objectgroup
ObjectName{5} = char(handles.Settings.VariableValues{CurrentModuleNum,9});
%inputtypeVAR09 = popupmenu

%textVAR10 =
%choiceVAR10 = Do not use
%choiceVAR10 = Image
%infotypeVAR10 = objectgroup
ObjectName{6} = char(handles.Settings.VariableValues{CurrentModuleNum,10});
%inputtypeVAR10 = popupmenu

%%%%%%%%%%%%%%%%%
%%% FEATURES  %%%
%%%%%%%%%%%%%%%%%
if nargin > 1 
    switch varargin{1}
%feature:categories
        case 'categories'
            if nargin == 1 || ismember(varargin{2},ObjectName)
                result = { 'Correlation' };
            else
                result = {};
            end
%feature:measurements
        case 'measurements'
            result = {};
            if strcmp(varargin{3},'Correlation') &&...
                ismember(varargin{2},ObjectName)
                imgnames = { ImageName{~strcmp(ImageName,'Do not use')}};
                %%% Take all combinations of images in pairs of two
                %%% and join them.
                combos = nchoosek(1:length(imgnames),2);
                result = arrayfun(@(z) CPjoinstrings(...
                    imgnames{combos(z,1)},...
                    imgnames{combos(z,2)}),1:size(combos,1),'UniformOutput',false);
                if strcmp(varargin{2}, 'Image')
                    result = cat(2,result,arrayfun(@(z) CPjoinstrings(...
                        'Slope',...
                        imgnames{combos(z,1)},...
                        imgnames{combos(z,2)}),1:size(combos,1),'UniformOutput',false));
                end
            end
        otherwise
            error(['Unhandled category: ',varargin{1}]);
    end
    handles=result;
    return;
end
%%%VariableRevisionNumber = 3

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Get the images
ImageCount = 0;
for ImageNbr = 1:4
    if ~strcmp(ImageName{ImageNbr},'Do not use')
        ImageCount = ImageCount + 1;
        try
            %%% Checks whether image has been loaded.
            Image{ImageCount} = CPretrieveimage(handles,ImageName{ImageNbr},ModuleName,'MustBeGray','DontCheckScale'); %#ok Ignore MLint
            tmpImageName{ImageCount} = ImageName{ImageNbr}; %#ok Ignore MLint
        catch
            error(['Image processing was canceled in the ', ModuleName, ' module because there was a problem loading the image you called ', ImageName{ImageNbr}, '.'])
        end
    end
end

%%% Check so that at least two images have been entered
if ImageCount < 2
    error(['Image processing was canceled in the ', ModuleName, ' module because at least two image types must be chosen.'])
end

%%% Get rid of 'Do not use' in the ImageName cell array so we don't have to care about them later.
ImageName = tmpImageName;

%%% Get the masks of segmented objects
ObjectNameCount = 0;
for ObjectNameNbr = 1:6
    if ~strcmp(ObjectName{ObjectNameNbr},'Do not use')
        ObjectNameCount = ObjectNameCount + 1;
        tmpObjectName{ObjectNameCount} = ObjectName{ObjectNameNbr}; %#ok Ignore MLint
        if ~strcmp(ObjectName{ObjectNameNbr},'Image')
            %%% Retrieves the label matrix image that contains the
            %%% segmented objects which will be used as a mask.
            LabelMatrixImage{ObjectNameCount} = CPretrieveimage(handles,['Segmented', ObjectName{ObjectNameNbr}],ModuleName,'MustBeGray','DontCheckScale'); %#ok Ignore MLint
        else
            LabelMatrixImage{ObjectNameCount} = ones(size(Image{1}));        % Use mask of ones to indicate that the correlation should be calcualted for the entire image
        end
    end
end
%%% Get rid of 'Do not use' in the ObjectName cell array so we don't have to care about them later.
if exist('tmpObjectName','var')
    ObjectName = tmpObjectName;
else
    error(['There are no objects or images defined.  Please choose at least one object or image in the settings for ' ModuleName '.'])
end

%%% Check so that at least one object type have been entered
if ObjectNameCount < 1
    error(['At least one object type must be entered in the ',ModuleName,' module.'])
end

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Produce feature names for all pairwise image combinations
CorrelationFeatures = {};
for i = 1:ImageCount-1
    for j = i+1:ImageCount
        CorrelationFeatures{end+1} = ['Correlation_',ImageName{i},ImageName{j}];
    end
end

%%% For each object type and for each segmented object, calculate the correlation between all combinations of images
for ObjectNameNbr = 1:ObjectNameCount

    for i=1:ImageCount
        %%% For the cases where the label matrix was produced from a cropped
        %%% image, the sizes of the images will not be equal. So, we crop the
        %%% LabelMatrix and try again to see if the matrices are then the
        %%% proper size. Removes Rows and Columns that are completely blank.
        if any(size(Image{i}) < size(LabelMatrixImage{ObjectNameNbr}))
            ColumnTotals = sum(LabelMatrixImage{ObjectNameNbr},1);
            RowTotals = sum(LabelMatrixImage{ObjectNameNbr},2)';
            warning off all
            ColumnsToDelete = ~logical(ColumnTotals);
            RowsToDelete = ~logical(RowTotals);
            warning on all
            drawnow
            CroppedLabelMatrix = LabelMatrixImage{ObjectNameNbr};
            CroppedLabelMatrix(:,ColumnsToDelete,:) = [];
            CroppedLabelMatrix(RowsToDelete,:,:) = [];
            LabelMatrixImage{ObjectNameNbr} = [];
            LabelMatrixImage{ObjectNameNbr} = CroppedLabelMatrix;
            %%% In case the entire image has been cropped away, we store a
            %%% single zero pixel for the variable.
            if isempty(LabelMatrixImage{ObjectNameNbr})
                LabelMatrixImage{ObjectNameNbr} = 0;
            end
        end

        if any(size(Image{i}) ~= size(LabelMatrixImage{ObjectNameNbr}))
            error(['Image processing was canceled in the ', ModuleName, ' module. The size of the image you want to measure is not the same as the size of the image from which the ',ObjectName{ObjectNameNbr},' objects were identified.'])
        end
    end

    %%% Calculate the correlation in all objects for all pairwise image combinations
    NbrOfObjects = max(LabelMatrixImage{ObjectNameNbr}(:));          % Get number of segmented objects
    Correlation = zeros(max([NbrOfObjects 1]),length(CorrelationFeatures));   % Pre-allocate memory
    for ObjectNbr = 1:NbrOfObjects                                   % Loop over objects
        FeatureNbr = 1;                                              % Easiest way to keep track of the feature number, i.e. which combination of images
        for i = 1:ImageCount-1                                       % Loop over all combinations of images
            for j = i+1:ImageCount
                index = find(LabelMatrixImage{ObjectNameNbr} == ObjectNbr);   % Get the indexes for the this object number
                try
                    if isempty(index) || numel(index) == 1 % If the object does not exist in the label matrix or is only one pixel, the correlation calculation will fail, so we assign the correlation to be NaN.
                        CorrelationForCurrentObject = 0;
                    else
                        c = corrcoef([Image{i}(index) Image{j}(index)]);             % Get the values for these indexes in the images and calculate the correlation
                        CorrelationForCurrentObject = c(1,2);
                        
                        % Checks for undefined values, and sets them to zero
                        if isnan(CorrelationForCurrentObject) || isinf(CorrelationForCurrentObject)
                            CorrelationForCurrentObject = 0;
                        end

                        %%% Gets slope measurement if objects are images
                        if (strcmp(ObjectName{ObjectNameNbr},'Image'))
                            if ~exist('SlopeFeatures')
                                SlopeFeatures = {};
                            end
                            xsize = size(Image{i});
                            ysize = size(Image{j});
                            if ~(xsize == ysize)        % Images must be the same size for the slope measurement to make sense
                                sizeerr = 1;
                                error(['Image processing was cancelled in the ', ModuleName, ' module because the images are not the same size.']);
                            else
                                SlopeFeatures{end+1} = ['Slope_',ImageName{i},ImageName{j}];
                                x = Image{i}(:);
                                y = Image{j}(:);
                                p = polyfit(x,y,1); % Get the values for the luminescence in these images and calculate the slope
                                SlopeForCurrentObject = p(1);
                                Slope(ObjectNbr,FeatureNbr) = SlopeForCurrentObject; % Store the slope
                                sizeerr = 0;
                            end
                        end
                    end
                    Correlation(ObjectNbr,FeatureNbr) = CorrelationForCurrentObject; % Store the correlation
                    FeatureNbr = FeatureNbr + 1;
                catch
                    if sizeerr
                        error(['Image processing was cancelled in the ', ModuleName, 'module becase images ', ImageName{i}, ' and ', ImageName{j}, ' are not the same size.'])
                    else
                        error(['Image processing was cancelled in the ', ModuleName, ' module because there was a problem calculating the correlation.'])
                    end
                end
            end
        end
    end

    %%% Store the correlation and slope measurements
    %%% Note: we end up with 'Correlation_Correclation_*', but that's OK since 
    %%% both the Category and FeatureName are both 'Correlation' here 
    for f=1:size(Correlation,2)
        handles = CPaddmeasurements(handles, ObjectName{ObjectNameNbr}, ...
            CPjoinstrings('Correlation', CorrelationFeatures{f}), ...
            Correlation(:, f));
    end
    if strcmp(ObjectName{ObjectNameNbr},'Image')
        for f = 1:size(Slope,2)
            handles = CPaddmeasurements(handles, ObjectName{ObjectNameNbr}, ...
                CPjoinstrings('Correlation', SlopeFeatures{f}), ...
                Slope(:, f));
        end
    end
    correlations{ObjectNameNbr} = Correlation;
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    % Remove uicontrols from last cycle
    delete(findobj(ThisModuleFigureNumber,'tag','TextUIControl'));
        
    %%% Activates the appropriate figure window.
    CPfigure(handles,'Text',ThisModuleFigureNumber);
    %%% Get size of window
    Position = get(ThisModuleFigureNumber,'Position');
    Height = Position(4);
    Width  = Position(3);

    %%% Displays the results.
    Displaytexthandle = uicontrol(ThisModuleFigureNumber,'style','text', 'position', [0 Height-40 Width 20],'tag','TextUIControl',...
        'fontname','Helvetica','FontSize',handles.Preferences.FontSize,'backgroundcolor',[.7 .7 .9],'horizontalalignment','center','fontweight','bold');
    TextToDisplay = ['Average correlations in cycle # ',num2str(handles.Current.SetBeingAnalyzed)];
    set(Displaytexthandle,'string',TextToDisplay)

    for ObjectNameNbr = 0:ObjectNameCount
        row = 1;
        %%% Write object names
        %%% Don't write any object type name in the first colum
        if ObjectNameNbr > 0
            h = uicontrol(ThisModuleFigureNumber,'style','text','position',[110+60*ObjectNameNbr Height-110 70 25],'tag','TextUIControl',...
                'fontname','Helvetica','FontSize',handles.Preferences.FontSize,'backgroundcolor',[.7 .7 .9],'horizontalalignment','center',...
                'fontweight','bold');
            set(h,'string',ObjectName{ObjectNameNbr});
        end
        %%% Write image names or correlation measurements
        FeatureNbr = 1;
        for i = 1:ImageCount-1
            for j = i+1:ImageCount
                %%% First column, write image names
                if ObjectNameNbr == 0
                    h = uicontrol(ThisModuleFigureNumber,'style','text','position',[20 Height-120-40*row 120 40],'tag','TextUIControl',...
                        'fontname','Helvetica','FontSize',handles.Preferences.FontSize,'backgroundcolor',[.7 .7 .9],'horizontalalignment','left',...
                        'fontweight','bold');
                    TextToDisplay = sprintf('%s and \n%s',ImageName{i},ImageName{j});
                    set(h,'string',TextToDisplay);
                else
                    %%% Calculate the average correlation over the objects
                    c = mean(correlations{ObjectNameNbr}(:,FeatureNbr));
                    uicontrol(ThisModuleFigureNumber,'style','text','position',[110+60*ObjectNameNbr Height-125-40*row 70 40],'tag','TextUIControl',...
                        'fontname','Helvetica','FontSize',handles.Preferences.FontSize,'backgroundcolor',[.7 .7 .9],'horizontalalignment','center',...
                        'string',sprintf('%0.2f',c));
                    FeatureNbr = FeatureNbr + 1;
                end
                row = row + 1;
            end
        end
    end
end