function handles = MeasureCorrelation(handles)

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
%                                        Image overall:    In Nuclei:
% OrigBlue  /OrigGreen      Correlation:    0.49955        -0.07395
% OrigBlue  /OrigRed        Correlation:    0.59886        -0.02752
% OrigGreen /OrigRed        Correlation:    0.83605         0.68489
%
% See also MeasureObjectIntensity, MeasureImageIntensity.

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
        catch error(['Image processing was canceled in the ', ModuleName, ' module because there was a problem loading the image you called ', ImageName{ImageNbr}, '.'])
        end
    end
end
%%% Get rid of 'Do not use' in the ImageName cell array so we don't have to care about them later.
ImageName = tmpImageName;   

%%% Check so that at least two images have been entered
if ImageCount < 2
    error(['Image processing was canceled in the ', ModuleName, ' module because at least two image types must be chosen.'])
end

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
ObjectName = tmpObjectName;

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
        CorrelationFeatures{end+1} = [ImageName{i},'_',ImageName{j}];
    end
end

%%% For each object type and for each segmented object, calculate the correlation between all combinations of images
for ObjectNameNbr = 1:ObjectNameCount
    
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
    end

    if any(size(Image{i}) ~= size(LabelMatrixImage{ObjectNameNbr}))
        error(['Image processing was canceled in the ', ModuleName, ' module. The size of the image you want to measure is not the same as the size of the image from which the ',ObjectName{ObjectNameNbr},' objects were identified.'])
    end
    
    %%% Calculate the correlation in all objects for all pairwise image combinations
    NbrOfObjects = max(LabelMatrixImage{ObjectNameNbr}(:));          % Get number of segmented objects
    Correlation = zeros(NbrOfObjects,length(CorrelationFeatures));   % Pre-allocate memory
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

                        %%% Gets slope measurement if objects are images
                        if (strcmp(ObjectName{ObjectNameNbr},'Image'))
                            if ~exist('SlopeFeatures')
                                SlopeFeatures = {};
                            end
                            xsize = size(Image{i});
                            ysize = size(Image{j});
                            if ~(xsize == ysize)        % Images must be the same size for the slope measurement to make sense
                                sizeerr = 1;
                                error;
                            else
                            SlopeFeatures{end+1} = ['Slope_',ImageName{i},'_',ImageName{j}];
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
    if strcmp (ObjectName{ObjectNameNbr},'Image')
        handles.Measurements.(ObjectName{ObjectNameNbr}).CorrelationFeatures = [CorrelationFeatures SlopeFeatures];
        handles.Measurements.(ObjectName{ObjectNameNbr}).Correlation(handles.Current.SetBeingAnalyzed) = {[Correlation Slope]};
    else
        handles.Measurements.(ObjectName{ObjectNameNbr}).CorrelationFeatures = CorrelationFeatures;
        handles.Measurements.(ObjectName{ObjectNameNbr}).Correlation(handles.Current.SetBeingAnalyzed) = {Correlation};
    end
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    %%% Activates the appropriate figure window.
    CPfigure(handles,'Text',ThisModuleFigureNumber);
    %%% Get size of window
    Position = get(ThisModuleFigureNumber,'Position');
    Height = Position(4);
    Width  = Position(3);

    %%% Displays the results.
    Displaytexthandle = uicontrol(ThisModuleFigureNumber,'style','text', 'position', [0 Height-40 Width 20],...
        'fontname','Helvetica','FontSize',handles.Preferences.FontSize,'backgroundcolor',[.7 .7 .9],'horizontalalignment','center','fontweight','bold');
    TextToDisplay = ['Average correlations in cycle # ',num2str(handles.Current.SetBeingAnalyzed)];
    set(Displaytexthandle,'string',TextToDisplay)

    for ObjectNameNbr = 0:ObjectNameCount
        row = 1;
        %%% Write object names
        %%% Don't write any object type name in the first colum
        if ObjectNameNbr > 0         
            h = uicontrol(ThisModuleFigureNumber,'style','text','position',[110+60*ObjectNameNbr Height-110 70 25],...
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
                    h = uicontrol(ThisModuleFigureNumber,'style','text','position',[20 Height-120-40*row 120 40],...
                        'fontname','Helvetica','FontSize',handles.Preferences.FontSize,'backgroundcolor',[.7 .7 .9],'horizontalalignment','left',...
                        'fontweight','bold');
                    TextToDisplay = sprintf('%s and \n%s',ImageName{i},ImageName{j});
                    set(h,'string',TextToDisplay);
                else
                    %%% Calculate the average correlation over the objects
                    c = mean(handles.Measurements.(ObjectName{ObjectNameNbr}).Correlation{handles.Current.SetBeingAnalyzed}(:,FeatureNbr));
                    uicontrol(ThisModuleFigureNumber,'style','text','position',[110+60*ObjectNameNbr Height-125-40*row 70 40],...
                        'fontname','Helvetica','FontSize',handles.Preferences.FontSize,'backgroundcolor',[.7 .7 .9],'horizontalalignment','center',...
                        'string',sprintf('%0.2f',c));
                    FeatureNbr = FeatureNbr + 1;
                end
                row = row + 1;
            end
        end
    end
end