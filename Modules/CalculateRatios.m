function handles = CalculateRatios(handles)

% Help for the Calculate Ratios module:
% Category: Measurement
%
% SHORT DESCRIPTION:
% Calculates the ratio between any measurements already measured (e.g.
% Intensity of green staining in cytoplasm/Area of cells)
% *************************************************************************
%
% This module can take any measurements produced by previous modules and
% calculate a ratio. Resulting ratios can also be used to calculate other
% ratios and be used in Classify Objects.
%
% This module can work on an object-by-object basis (calculating the ratio
% for each object), on an image-by-image basis, or it can also calculate
% ratios for object measurements by whole image measurements (to allow
% normalization). Be careful with your denominator data. Any 0's found in
% it will be changed to the average of the rest of the data. If all
% denominator data is 0, all ratios will be set to 0 too.
%
% The ratios will be stored under the numerator object data. If the
% numerator is an object, data will be under the name Ratio. If the
% numerator is an image, data will be under the name SingleRatio or
% MultipleRatio depending on whether the denominator is another image or an
% object, respectively.
%
% Feature Number:
% The feature number specifies which features from the Measure module(s)
% will be used for the ratio. See each Measure module's help for the
% numbered list of the features measured by that module.
%
% See also all Measure modules.

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
%
% Website: http://www.cellprofiler.org
%
% $Revision: 1843 $

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = Which object would you like to use for the numerator?
%choiceVAR01 = Image
%infotypeVAR01 = objectgroup
%inputtypeVAR01 = popupmenu
ObjectName{1} = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = Which category of measurements would you like to use?
%choiceVAR02 = AreaShape
%choiceVAR02 = Correlation
%choiceVAR02 = Intensity
%choiceVAR02 = Neighbors
%choiceVAR02 = Texture
%inputtypeVAR02 = popupmenu custom
Measure{1} = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Which feature do you want to use? (Enter the feature number - see help for details)
%defaultVAR03 = 1
FeatureNumber{1} = str2double(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = For INTENSITY or TEXTURE features, which image's measurements would you like to use?
%infotypeVAR04 = imagegroup
%inputtypeVAR04 = popupmenu
Image{1} = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = Which object would you like to use for the denominator?
%choiceVAR05 = Image
%infotypeVAR05 = objectgroup
%inputtypeVAR05 = popupmenu
ObjectName{2} = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = Which category of measurements would you like to use?
%choiceVAR06 = AreaShape
%choiceVAR06 = Correlation
%choiceVAR06 = Intensity
%choiceVAR06 = Neighbors
%choiceVAR06 = Texture
%inputtypeVAR06 = popupmenu custom
Measure{2} = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = Which feature do you want to use? (Enter the feature number - see help for details)
%defaultVAR07 = 1
FeatureNumber{2} = str2double(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = For INTENSITY or TEXTURE features, which image's measurements would you like to use?
%infotypeVAR08 = imagegroup
%inputtypeVAR08 = popupmenu
Image{2} = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%textVAR09 = Do you want the log (base 10) of the ratio?
%choiceVAR09 = No
%choiceVAR09 = Yes
%inputtypeVAR09 = popupmenu
LogChoice = char(handles.Settings.VariableValues{CurrentModuleNum,9});

%%%VariableRevisionNumber = 4

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% MAKE MEASUREMENTS & SAVE TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;

% Get the correct fieldname where measurements are located
OrigMeasure = Measure;
CellFlg = 0;
for i=1:2
    CurrentMeasure = Measure{i};
    CurrentObjectName = ObjectName{i};
    CurrentImage = Image{i};
    switch CurrentMeasure
        case 'AreaShape'
            if strcmp(CurrentObjectName,'Image')
                CurrentMeasure = '^AreaOccupied_.*Features$';
                Fields = fieldnames(handles.Measurements.Image);
                TextComp = regexp(Fields,CurrentMeasure);
                A = cellfun('isempty',TextComp);
                try
                    CurrentMeasure = Fields{find(A==0)+1};
                catch
                    error(['Image processing was canceled in the ', ModuleName, ' module because the category of measurement you chose, ', Measure{i}, ', was not available for ', ObjectName{i}]);
                end
                CellFlg = 1;
            end
        case 'Intensity'
            CurrentMeasure = ['Intensity_' CurrentImage];
        case 'Neighbors'
            CurrentMeasure = 'NumberNeighbors';
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

% Get measurements
try
    NumeratorMeasurements = handles.Measurements.(ObjectName{1}).(Measure{1}){SetBeingAnalyzed};
    NumeratorMeasurements = NumeratorMeasurements(:,FeatureNumber{1});
    if CellFlg
        NumeratorMeasurements = NumeratorMeasurements{1};
    end
catch
    error(['Image processing was canceled in the ', ModuleName, ' module because an error ocurred when retrieving the numerator data. Either the category of measurement you chose, ', Measure{1},', was not available for ', ObjectName{1},', or the feature number, ', num2str(FeatureNumber{1}), ', exceeded the amount of measurements.']);
end
try
    DenominatorMeasurements = handles.Measurements.(ObjectName{2}).(Measure{2}){SetBeingAnalyzed};
    DenominatorMeasurements = DenominatorMeasurements(:,FeatureNumber{2});
catch
    error(['Image processing was canceled in the ', ModuleName, ' module because an error ocurred when retrieving the denominator data. Either the category of measurement you chose, ', Measure{2},', was not available for ', ObjectName{2},', or the feature number, ', num2str(FeatureNumber{2}), ', exceeded the amount of measurements.']);
end

% Check size of data
if length(NumeratorMeasurements) ~= length(DenominatorMeasurements)
    try
        if strcmp(ObjectName{1},'Image')
            NumeratorMeasurements = NumeratorMeasurements*ones(size(DenominatorMeasurements));
        elseif strcmp(ObjectName{2},'Image')
            DenominatorMeasurements = DenominatorMeasurements*ones(size(NumeratorMeasurements));
        else
            error('');
        end
    catch
        error(['Image processing was canceled in the ', ModuleName, ' module because the specified object names ',ObjectName{1},' and ',ObjectName{2},' do not have the same amount of measurements.']);
    end
end

% Make measurements and store in handle structure
DenominatorMeasurements(DenominatorMeasurements==0) = NaN;
DenominatorMeasurements(isnan(DenominatorMeasurements)) = nanmean(DenominatorMeasurements);
FinalMeasurements = NumeratorMeasurements./DenominatorMeasurements;
if strcmp(LogChoice,'Yes')
    FinalMeasurements = log10(FinalMeasurements);
end
if isnan(FinalMeasurements)
    FinalMeasurements(:)=0;
end
NewFieldName = [ObjectName{1},'_',Measure{1}(1),'_',num2str(FeatureNumber{1}),'_dividedby_',ObjectName{2},'_',Measure{2}(1),'_',num2str(FeatureNumber{2})];
if strcmp(ObjectName{1},'Image')
    if length(FinalMeasurements)==1
        handles = CPaddmeasurements(handles,ObjectName{1},'SingleRatio',NewFieldName,FinalMeasurements);
    else
        handles = CPaddmeasurements(handles,ObjectName{1},'MultipleRatio',NewFieldName,FinalMeasurements);
    end
else
    handles = CPaddmeasurements(handles,ObjectName{1},'Ratio',NewFieldName,FinalMeasurements);
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

FontSize = handles.Preferences.FontSize;
ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule', CurrentModule]);
if any(findobj == ThisModuleFigureNumber)

    % Activates display window
    CPfigure(handles,'Text',ThisModuleFigureNumber);

    % Title
    uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0 0.95 1 0.04],...
        'HorizontalAlignment','center','Backgroundcolor',[.7 .7 .9],'fontname','Helvetica',...
        'fontsize',FontSize,'fontweight','bold','string',sprintf('Average ratio features, cycle #%d',SetBeingAnalyzed),'UserData',SetBeingAnalyzed);
    
    if SetBeingAnalyzed == handles.Current.StartingImageSet
        
        % Text for Name of measurement
        if strcmp(OrigMeasure{1},'Intensity') || strcmp(OrigMeasure{1},'Texture')
            DisplayName1 = [ObjectName{1} ' ' handles.Measurements.(ObjectName{1}).([Measure{1} 'Features']){FeatureNumber{1}} ' in ' Image{1}];
        else
            DisplayName1 = [ObjectName{1} ' ' handles.Measurements.(ObjectName{1}).([Measure{1} 'Features']){FeatureNumber{1}}];
        end
        if strcmp(OrigMeasure{2},'Intensity') || strcmp(OrigMeasure{2},'Texture')
            DisplayName2 = [ObjectName{2} ' ' handles.Measurements.(ObjectName{2}).([Measure{2} 'Features']){FeatureNumber{2}} ' in ' Image{2}];
        else
            DisplayName2 = [ObjectName{2} ' ' handles.Measurements.(ObjectName{2}).([Measure{2} 'Features']){FeatureNumber{2}}];
        end
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0 0.9 1 0.04],...
            'HorizontalAlignment','center','Backgroundcolor',[.7 .7 .9],'fontname','Helvetica',...
            'fontsize',FontSize,'fontweight','bold','string',[DisplayName1 ' divided by ' DisplayName2],'UserData',SetBeingAnalyzed);

        % Text for Number of objects
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.05 0.8 0.25 0.03],...
            'HorizontalAlignment','left','Backgroundcolor',[.7 .7 .9],'fontname','Helvetica',...
            'fontsize',FontSize,'fontweight','bold','string','Number of objects:','UserData',SetBeingAnalyzed);

        % Text for Average Ratio
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.05 0.75 0.25 0.03],...
            'HorizontalAlignment','left','BackgroundColor',[.7 .7 .9],'fontname','Helvetica',...
            'fontsize',FontSize,'fontweight','bold','string','Average Ratio:','UserData',SetBeingAnalyzed);
    end
    
    % Number of objects
    uicontrol(ThisModuleFigureNumber, 'style', 'text', 'units','normalized', 'position', [0.3 0.8 0.1 0.03],...
        'HorizontalAlignment', 'center', 'Background', [.7 .7 .9], 'fontname', 'Helvetica', ...
        'fontsize',FontSize,'string',num2str(length(FinalMeasurements)),'UserData',SetBeingAnalyzed);

    % Average Ratio
    uicontrol(ThisModuleFigureNumber, 'style', 'text', 'units','normalized', 'position', [0.3 0.75 0.1 0.03],...
        'HorizontalAlignment', 'center', 'Background', [.7 .7 .9], 'fontname', 'Helvetica', ...
        'fontsize',FontSize,'string',sprintf('%4.2f',mean(FinalMeasurements)),'UserData',SetBeingAnalyzed);
end