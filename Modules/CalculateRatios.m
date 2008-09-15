function handles = CalculateRatios(handles,varargin)

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
% Saving:
% The ratio measurements are stored as 'Ratio_...'. If both measures are 
% image-based, then a single ratio (per cycle) will be stored as 'Image' data.  
% If one measure is object-based and one image-based, then the ratios will
% be stored associated with the object, one ratio per object.  If both are 
% objects, then the ratios are stored with both objects.
%
% Feature Number:
% The feature number specifies which features from the Measure module(s)
% will be used for the ratio. See each Measure module's help for the
% numbered list of the features measured by that module.
%
% Features measured:                         Feature Number:
% (As named in module's first setting)     |       1
%
% Note: If you want to use the output of this module in a subsequesnt
% calculation, we suggest you specify the output name rather than use
% Automatic naming.
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

%textVAR01 = What do you want to call the ratio calculated by this module?  The prefix 'Ratio_' will be applied to your entry, or simply leave as 'Automatic' and a sensible name will be generated
%defaultVAR01 = Automatic
RatioName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = Which object would you like to use for the numerator?
%choiceVAR02 = Image
%infotypeVAR02 = objectgroup
%inputtypeVAR02 = popupmenu
ObjectName{1} = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Which category of measurements would you like to use?
%inputtypeVAR03 = popupmenu category
Category{1} = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = Which feature do you want to use? 
%defaultVAR04 = 1
%inputtypeVAR04 = popupmenu measurement
FeatureNumber{1} = handles.Settings.VariableValues{CurrentModuleNum,4};

%textVAR05 = For INTENSITY, AREAOCCUPIED or TEXTURE features, which image's measurements would you like to use?
%infotypeVAR05 = imagegroup
%inputtypeVAR05 = popupmenu
ImageName{1} = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = For TEXTURE, RADIAL DISTRIBUTION, OR NEIGHBORS features, what previously measured size scale (TEXTURE OR NEIGHBORS) or previously used number of bins (RADIALDISTRIBUTION) do you want to use?
%defaultVAR06 = 1
%inputtypeVAR06 = popupmenu scale
SizeScale{1} = str2double(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = Which object would you like to use for the denominator?
%choiceVAR07 = Image
%infotypeVAR07 = objectgroup
%inputtypeVAR07 = popupmenu
ObjectName{2} = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = Which category of measurements would you like to use?
%inputtypeVAR08 = popupmenu category
Category{2} = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%textVAR09 = Which feature do you want to use?
%defaultVAR09 = 1
%inputtypeVAR09 = popupmenu measurement
FeatureNumber{2} = handles.Settings.VariableValues{CurrentModuleNum,9};

%textVAR10 = For INTENSITY, AREAOCCUPIED or TEXTURE features, which image's measurements would you like to use?
%infotypeVAR10 = imagegroup
%inputtypeVAR10 = popupmenu
ImageName{2} = char(handles.Settings.VariableValues{CurrentModuleNum,10});

%textVAR11 = For TEXTURE, RADIAL DISTRIBUTION, OR NEIGHBORS features, what previously measured size scale (TEXTURE OR NEIGHBORS) or previously used number of bins (RADIALDISTRIBUTION) do you want to use?
%defaultVAR11 = 1
%inputtypeVAR11 = popupmenu scale

SizeScale{2} = str2double(handles.Settings.VariableValues{CurrentModuleNum,11});

%textVAR12 = Do you want the log (base 10) of the ratio?
%choiceVAR12 = No
%choiceVAR12 = Yes
%inputtypeVAR12 = popupmenu
LogChoice = char(handles.Settings.VariableValues{CurrentModuleNum,12});

%%%VariableRevisionNumber = 6

%%%%%%%%%%%%%%%%%%%%%%%%%
%%% FEATURES          %%%
%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin > 1
    switch(varargin{1})
%feature:categories
        case 'categories'
            result = {};
            if nargin == 1 || ismember(varargin{2},ObjectName)
                result = { 'Ratio' };
            end
%feature:measurements
        case 'measurements'
            result = {};
            if all(isnan(str2double(FeatureNumber))) % is numerator or denominator a legacy feature number?
                if ismember(varargin{2},ObjectName) &&...
                    strcmp(varargin{3},'Ratio')
                    result = { getRatioName(RatioName,ObjectName,FeatureNumber) };
                end
            end
        otherwise
            error(['Unhandled category: ',varargin{1}]);
    end
    handles=result;
    return;
end

%%%%%%%%%%%%%%%%%%%%%%%%%
%%% MAKE MEASUREMENTS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow
SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;

%% Check FeatureNumber
for NumerDenom = 1:2
    if isempty(FeatureNumber{NumerDenom})
        error(['Image processing was canceled in the ', ModuleName, ' module because your entry for feature number is not valid.']);
    end
end

%% Get the correct fieldname where measurements are located
for NumerDenom=1:2

    %% Get measurements
    try
        FeatureName{NumerDenom} = CPgetfeaturenamesfromnumbers(handles,ObjectName{NumerDenom},...
            Category{NumerDenom},FeatureNumber{NumerDenom},ImageName{NumerDenom},SizeScale{NumerDenom});
        
        %% NOTE:    Numerator data will be NumerDenomMeasurements{1} and 
        %%          Denominator data will be NumerDenomMeasurements{2} 
        NumerDenomMeasurements{NumerDenom} = ...
            handles.Measurements.(ObjectName{NumerDenom}).(FeatureName{NumerDenom}){SetBeingAnalyzed};
    catch
        error([ lasterr '  Module Image processing was canceled in the ', ModuleName, ...
            ' module (#' num2str(CurrentModuleNum) ...
            ') because an error ocurred when retrieving the data.  '...
            ' Likely the category of measurement you chose, ',...
            Category{NumerDenom}, ', was not available for ', ...
            ObjectName{NumerDenom},' with feature number ' num2str(FeatureNumber{NumerDenom}) ...
            ', possibly specific to image ''' ImageName{NumerDenom} ''' and/or ' ...
            'Texture Scale = ' num2str(SizeScale{NumerDenom}) '.']);
    end
end

% Check size of data, and make them the same size
if length(NumerDenomMeasurements{1}) ~= length(NumerDenomMeasurements{2})
    if strcmp(ObjectName{1},'Image')
        NumerDenomMeasurements{1} = NumerDenomMeasurements{1}*ones(size(NumerDenomMeasurements{2}));
    elseif strcmp(ObjectName{2},'Image')
        NumerDenomMeasurements{2} = NumerDenomMeasurements{2}*ones(size(NumerDenomMeasurements{1}));
    else
        error(['Image processing was canceled in the ', ModuleName, ...
            ' module because the specified object names ''',ObjectName{1},...
            ''' and ''',ObjectName{2},''' do not have the same number of measurements.']);
    end
end

%%%%%% Make measurements and store in handle structure
%% Replace NaNs and zeros (since we cannot divide by zero) with the mean
%% of the remaining values.
NumerDenomMeasurements{2}(NumerDenomMeasurements{2}==0) = NaN;
NumerDenomMeasurements{2}(isnan(NumerDenomMeasurements{2})) = CPnanmean(NumerDenomMeasurements{2});
if ~all(isnan(NumerDenomMeasurements{2}))
    FinalMeasurements = NumerDenomMeasurements{1}./NumerDenomMeasurements{2};
else
    FinalMeasurements = NaN;
    CPwarndlg(['A ratio of ' NumerDenomMeasurements{1} ' and ' NumerDenomMeasurements{2} ...
        ' within ' ModuleName ' on cycle '  str2double(SetBeingAnalyzed) ...
        ' resulted in all NaNs.  You may want to check your settings.'])
end
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
    CPwarndlg(['The ratio name you entered was invalid, and has been replaced '...
        'with an automatically generated name based on your inputs (equivalent to ''Automatic'' setting).']);
end
RatioName = getRatioName(RatioName,ObjectName,FeatureName);
RatioName = CPtruncatefeaturename(CPjoinstrings(RatioName));

%% Save, depending on type of measurement (ObjectName)
%% Note that Image measurements are scalars, while Objects are potentially vectors
if strcmp(ObjectName{1}, 'Image') && strcmp(ObjectName{2}, 'Image'),
    handles = CPaddmeasurements(handles,'Image',RatioName,FinalMeasurements);
else
    if ~strcmp(ObjectName{1}, 'Image'),
        handles = CPaddmeasurements(handles,ObjectName{1},RatioName,FinalMeasurements);
    end
    %% Add to second object (if different from the first)
    if ~strcmp(ObjectName{1}, ObjectName{2}) && ~strcmp(ObjectName{2}, 'Image'),
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
        if strcmp(Category{1},'Intensity') || strcmp(Category{1},'Texture')
            DisplayName1 = [ObjectName{1} ' ' FeatureName{1} ' in ' ImageName{1}];
        else
            DisplayName1 = [ObjectName{1} ' ' FeatureName{1}];
        end
        if strcmp(Category{2},'Intensity') || strcmp(Category{2},'Texture')
            DisplayName2 = [ObjectName{2} ' ' FeatureName{2} ' in ' ImageName{2}];
        else
            DisplayName2 = [ObjectName{2} ' ' FeatureName{2}];
        end
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0 0.9 1 0.06],...
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
end
function RatioName = getRatioName(RatioName,ObjectName, FeatureName)
if strcmpi(RatioName, 'Automatic')
    RatioName = CPjoinstrings(ObjectName{1},FeatureName{1},...
                          'DividedBy',ObjectName{2},FeatureName{2});
end

end