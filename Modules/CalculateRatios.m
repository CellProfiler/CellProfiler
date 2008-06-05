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
% denominator data is 0, all ratios will be set to 0 too. Also, if you are
% choosing to log-transform your ratios, any ratios that are equal to
% zero will also be changed to the average of the rest of the data, because
% you cannot take the log of zero.
%
% The ratios will be stored along with the numerator object's data. If the
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
% See also CalculateRatiosDataTool, all Measure modules.

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

%textVAR01 = What do you want to call the ratio calculated by this module?
%defaultVAR01 = Automatic
RatioName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = Which object would you like to use for the numerator?
%choiceVAR02 = Image
%infotypeVAR02 = objectgroup
%inputtypeVAR02 = popupmenu
ObjectName{1} = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Which category of measurements would you like to use?
%choiceVAR03 = AreaShape
%choiceVAR03 = Children
%choiceVAR03 = Correlation
%choiceVAR03 = Intensity
%choiceVAR03 = Neighbors
%choiceVAR03 = Texture
%inputtypeVAR03 = popupmenu custom
Category{1} = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = Which feature do you want to use? (Enter the feature number - see help for details)
%defaultVAR04 = 1
FeatureNumber{1} = str2double(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = For INTENSITY or TEXTURE features, which image's measurements would you like to use?
%infotypeVAR05 = imagegroup
%inputtypeVAR05 = popupmenu
Image{1} = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = For TEXTURE features, what previously measured texture scale do you want to use?
%defaultVAR06 = 1
TextureScale{1} = str2double(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = Which object would you like to use for the denominator?
%choiceVAR07 = Image
%infotypeVAR07 = objectgroup
%inputtypeVAR07 = popupmenu
ObjectName{2} = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = Which category of measurements would you like to use?
%choiceVAR08 = AreaShape
%choiceVAR08 = Correlation
%choiceVAR08 = Intensity
%choiceVAR08 = Neighbors
%choiceVAR08 = Texture
%inputtypeVAR08 = popupmenu custom
Category{2} = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%textVAR09 = Which feature do you want to use? (Enter the feature number - see help for details)
%defaultVAR09 = 1
FeatureNumber{2} = str2double(handles.Settings.VariableValues{CurrentModuleNum,9});

%textVAR10 = For INTENSITY or TEXTURE features, which image's measurements would you like to use?
%infotypeVAR10 = imagegroup
%inputtypeVAR10 = popupmenu
Image{2} = char(handles.Settings.VariableValues{CurrentModuleNum,10});

%textVAR11 = For TEXTURE features, what previously measured texture scale do you want to use?
%defaultVAR11 = 1
TextureScale{2} = str2double(handles.Settings.VariableValues{CurrentModuleNum,11});

%textVAR12 = Do you want the log (base 10) of the ratio?
%choiceVAR12 = No
%choiceVAR12 = Yes
%inputtypeVAR12 = popupmenu
LogChoice = char(handles.Settings.VariableValues{CurrentModuleNum,12});

%%%VariableRevisionNumber = 6

%%%%%%%%%%%%%%%%%%%%%%%%%
%%% MAKE MEASUREMENTS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow
SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;

%% Check FeatureNumber
for NumerDenom = 1:2
    if isempty(FeatureNumber{NumerDenom}) || isnan(FeatureNumber{NumerDenom})
        error(['Image processing was canceled in the ', ModuleName, ' module because your entry for feature number is not valid.']);
    end
end

%% Get the correct fieldname where measurements are located
OrigCategory = Category;
for NumerDenom=1:2
    FeatureName{NumerDenom} = CPgetfeaturenamesfromnumbers(handles,ObjectName{NumerDenom},...
        Category{NumerDenom},FeatureNumber{NumerDenom},Image{NumerDenom},TextureScale{NumerDenom});
    % end

    %% Get measurements
    try
        %% NOTE: Numerator is NumerDenomMeasurements{1} and 
        %% NOTE: Denominator is NumerDenomMeasurements{2} 
        NumerDenomMeasurements{NumerDenom} = ...
            handles.Measurements.(ObjectName{NumerDenom}).(FeatureName{NumerDenom}){SetBeingAnalyzed};
    catch
        error(['Image processing was canceled in the ', ModuleName, ...
            ' module because an error ocurred when retrieving the numerator data.' ...
            ' Either the category of measurement you chose, ', FeatureName{NumerDenom},...
            ', was not available for ', ObjectName{NumerDenom},', or the feature number, ', ...
            num2str(FeatureNumber{NumerDenom}), ', exceeded the amount of measurements.']);
    end
end
% Check size of data, and make them the same size
if length(NumerDenomMeasurements{1}) ~= length(NumerDenomMeasurements{2})
    try
        if strcmp(ObjectName{1},'Image')
            NumerDenomMeasurements{1} = NumerDenomMeasurements{1}*ones(size(NumerDenomMeasurements{2}));
        elseif strcmp(ObjectName{2},'Image')
            NumerDenomMeasurements{2} = NumerDenomMeasurements{2}*ones(size(NumerDenomMeasurements{2}));
        else
            error('');
        end
    catch
        error(['Image processing was canceled in the ', ModuleName, ' module because the specified object names ',ObjectName{1},' and ',ObjectName{2},' do not have the same amount of measurements.']);
    end
end

%%%%%% Make measurements and store in handle structure
%% Replace NaNs and zeros (since we cannot divide by zero) with the mean
%% of the remaining values.
NumerDenomMeasurements{2}(NumerDenomMeasurements{2}==0) = NaN;
NumerDenomMeasurements{2}(isnan(NumerDenomMeasurements{2})) = CPnanmean(NumerDenomMeasurements{2});
FinalMeasurements = NumerDenomMeasurements{1}./NumerDenomMeasurements{2};
if strcmp(LogChoice,'Yes')
    %%% We cannot take the log of zero, so replace zeros with the mean of the remaining values.
    %FinalMeasurements(FinalMeasurements==0)=CPnanmean(FinalMeasurements);
    FinalMeasurements = log10(FinalMeasurements);
end
FinalMeasurements(isnan(FinalMeasurements))=0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~isvarname(RatioName)
    RatioName = 'Automatic';
    CPwarndlg(['The ratio name you entered was invalid, and has been replaced with ',RatioName,'.']);
elseif strcmpi(RatioName, 'Automatic')

    %% TODO
%     %% Since Matlab's max name length is 63, we need to truncate the fieldname
%     MinStrLen = 5;
%     FirstFeatureNameSubstrings = textscan(FirstFeatureName,'%s','delimiter','_');
%     for idxStr = 1:length(FirstFeatureNameSubstrings{1})
%         Str = FirstFeatureNameSubstrings{1}{idxStr};
%         FirstTruncatedName{idxStr} = Str(1:min(length(Str),MinStrLen));
%     end
%     SecondFeatureNameSubstrings = textscan(SecondFeatureName,'%s','delimiter','_');
%     for idxStr = 1:length(SecondFeatureNameSubstrings{1})
%         Str = SecondFeatureNameSubstrings{1}{idxStr};
%         SecondTruncatedName{idxStr} = Str(1:min(length(Str),MinStrLen));
%     end
%     NewFirstFeatureName = CPjoinstrings(FirstTruncatedName{:});
%     NewSecondFeatureName = CPjoinstrings(SecondTruncatedName{:});


    RatioName = CPjoinstrings('Ratio',ObjectName{1},FeatureName{1},'DividedBy',...
                                ObjectName{2},FeatureName{2});
else
    RatioName = CPjoinstrings('Ratio', RatioName);
end

if strcmp(ObjectName{1}, 'Image') and strcmp(ObjectName{2}, 'Image'),
    handles = CPaddmeasurements(handles,'Image',CPjoinstrings('Ratio', RatioName),FinalMeasurements);
else
    if ~ strcmp(ObjectName{1}, 'Image'),
        handles = CPaddmeasurements(handles,ObjectName{1},RatioName,FinalMeasurements);
    end
    if ~ strcmp(ObjectName{2}, 'Image'),
        handles = CPaddmeasurements(handles,ObjectName{2},RatioName,FinalMeasurements);
    end
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
        if strcmp(OrigCategory{1},'Intensity') || strcmp(OrigCategory{1},'Texture')
            DisplayName1 = [ObjectName{1} ' ' FeatureName{1} ' in ' Image{1}];
        else
            DisplayName1 = [ObjectName{1} ' ' FeatureName{1}];
        end
        if strcmp(OrigCategory{2},'Intensity') || strcmp(OrigCategory{2},'Texture')
            DisplayName2 = [ObjectName{2} ' ' FeatureName{2} ' in ' Image{2}];
        else
            DisplayName2 = [ObjectName{2} ' ' FeatureName{2}];
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