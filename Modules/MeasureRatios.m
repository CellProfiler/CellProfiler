function handles = MeasureRatios(handles)

% Help for the Measure Ratios module:
% Category: Measurement
%
% This module has not yet been documented.  You can enter 'Area' or
% 'IntegratedIntensity' or 'MeanIntensity' for example. Sometimes I
% get divide by zero errors; We should add error checking to be sure
% the proper number of measurements exist.  It has some hard coded
% lines at the moment for CorrRed, CorrGreen, CorrBlue.
%
% How it works:
%
% See also MEASUREIMAGEAREAOCCUPIED,
% MEASUREOBJECTINTENSITYTEXTURE, MEASUREOBJECTAREASHAPE,
% MEASURECORRELATION,
% MEASUREIMAGEINTENSITY.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne Carpenter <carpenter@wi.mit.edu>
%   Thouis Jones   <thouis@csail.mit.edu>
%   In Han Kang    <inthek@mit.edu>
%
% $Revision$

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow

%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

%textVAR01 = Enter the measurement name
%defaultVAR01 = Area
MeasurementName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = For which objects do you want to measure ratios (numerator)?
%infotypeVAR02 = objectgroup
%inputtypeVAR02 = popupmenu
NumeratorObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = denominator
%infotypeVAR03 = objectgroup
DenominatorObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% MAKE MEASUREMENTS & SAVE TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;

% Get index for the given Measurement name by searching
% all extracted features
fn = fieldnames(handles.Measurements.(NumeratorObjectName));
FeatureNo = [];
MeasurementType =[];                                         % E.g. Shape, Intensity, Texture....
for k = 1:length(fn)
    if strfind(fn{k},'CorrBlueFeatures')
        features = handles.Measurements.(NumeratorObjectName).(fn{k});             % Cell array with feature names
        for j = 1:length(features)
            if strcmp(features{j},MeasurementName)
                MeasurementType = fn{k}(1:end-8);
                FeatureNo = j;
            end
        end
    end
end

% Didn't find a matching feature -> error message
if isempty(FeatureNo)
    errordlg(sprintf('Did not find the specified measurement %s in the MeasureRatios module',MeasurementName));
end

% Get measurements
NumeratorMeasurements = handles.Measurements.(NumeratorObjectName).(MeasurementType){SetBeingAnalyzed};
NumeratorMeasurements = NumeratorMeasurements(:,FeatureNo);
DenominatorMeasurements = handles.Measurements.(DenominatorObjectName).(MeasurementType){SetBeingAnalyzed};
DenominatorMeasurements = DenominatorMeasurements(:,FeatureNo);

if length(NumeratorMeasurements) ~= length(DenominatorMeasurements)
    errordlg(sprintf('The specified object names %s and %s in the MeasureRatios do not have the same object count.',NumeratorObjectName,DenominatorObjectName));
end
try
NewFieldName = [MeasurementName,'CorrBlue',NumeratorObjectName,'_dividedby_',DenominatorObjectName];
NewFieldNameFeatures = [MeasurementName,'CorrBlue',NumeratorObjectName,'_dividedby_',DenominatorObjectName,'Features'];
handles.Measurements.UserDefined.(NewFieldName)(SetBeingAnalyzed) = {NumeratorMeasurements./DenominatorMeasurements};
handles.Measurements.UserDefined.(NewFieldNameFeatures) = {NewFieldName};
end


% REPEAT FOR OTHER INTENSITY MEASURES ----------------
% Get index for the given Measurement name by searching
% all extracted features
fn = fieldnames(handles.Measurements.(NumeratorObjectName));
FeatureNo = [];
MeasurementType =[];                                         % E.g. Shape, Intensity, Texture....
for k = 1:length(fn)
    if strfind(fn{k},'CorrGreenFeatures')
        features = handles.Measurements.(NumeratorObjectName).(fn{k});             % Cell array with feature names
        for j = 1:length(features)
            if strcmp(features{j},MeasurementName)
                MeasurementType = fn{k}(1:end-8);
                FeatureNo = j;
            end
        end
    end
end

% Didn't find a matching feature -> error message
if isempty(FeatureNo)
    errordlg(sprintf('Did not find the specified measurement %s in the MeasureRatios module',MeasurementName));
end

% Get measurements
NumeratorMeasurements = handles.Measurements.(NumeratorObjectName).(MeasurementType){SetBeingAnalyzed};
NumeratorMeasurements = NumeratorMeasurements(:,FeatureNo);
DenominatorMeasurements = handles.Measurements.(DenominatorObjectName).(MeasurementType){SetBeingAnalyzed};
DenominatorMeasurements = DenominatorMeasurements(:,FeatureNo);

if length(NumeratorMeasurements) ~= length(DenominatorMeasurements)
    errordlg(sprintf('The specified object names %s and %s in the MeasureRatios do not have the same object count.',NumeratorObjectName,DenominatorObjectName));
end
try
NewFieldName = [MeasurementName,'CorrGreen',NumeratorObjectName,'_dividedby_',DenominatorObjectName];
NewFieldNameFeatures = [MeasurementName,'CorrGreen',NumeratorObjectName,'_dividedby_',DenominatorObjectName,'Features'];
handles.Measurements.UserDefined.(NewFieldName)(SetBeingAnalyzed) = {NumeratorMeasurements./DenominatorMeasurements};
handles.Measurements.UserDefined.(NewFieldNameFeatures) = {NewFieldName};
end

% REPEAT FOR OTHER INTENSITY MEASURES ----------------
% Get index for the given Measurement name by searching
% all extracted features
fn = fieldnames(handles.Measurements.(NumeratorObjectName));
FeatureNo = [];
MeasurementType =[];                                         % E.g. Shape, Intensity, Texture....
for k = 1:length(fn)
    if strfind(fn{k},'CorrRedFeatures')
        features = handles.Measurements.(NumeratorObjectName).(fn{k});             % Cell array with feature names
        for j = 1:length(features)
            if strcmp(features{j},MeasurementName)
                MeasurementType = fn{k}(1:end-8);
                FeatureNo = j;
            end
        end
    end
end

% Didn't find a matching feature -> error message
if isempty(FeatureNo)
    errordlg(sprintf('Did not find the specified measurement %s in the MeasureRatios module',MeasurementName));
end

% Get measurements
NumeratorMeasurements = handles.Measurements.(NumeratorObjectName).(MeasurementType){SetBeingAnalyzed};
NumeratorMeasurements = NumeratorMeasurements(:,FeatureNo);
DenominatorMeasurements = handles.Measurements.(DenominatorObjectName).(MeasurementType){SetBeingAnalyzed};
DenominatorMeasurements = DenominatorMeasurements(:,FeatureNo);

if length(NumeratorMeasurements) ~= length(DenominatorMeasurements)
    errordlg(sprintf('The specified object names %s and %s in the MeasureRatios do not have the same object count.',NumeratorObjectName,DenominatorObjectName));
end
try
NewFieldName = [MeasurementName,'CorrRed',NumeratorObjectName,'_dividedby_',DenominatorObjectName];
NewFieldNameFeatures = [MeasurementName,'CorrRed',NumeratorObjectName,'_dividedby_',DenominatorObjectName,'Features'];
handles.Measurements.UserDefined.(NewFieldName)(SetBeingAnalyzed) = {NumeratorMeasurements./DenominatorMeasurements};
handles.Measurements.UserDefined.(NewFieldNameFeatures) = {NewFieldName};
end

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow
if SetBeingAnalyzed == handles.Current.StartingImageSet
    %%% The figure window display is unnecessary for this module, so the figure
    %%% window is closed the first time through the module.
    fieldname = ['FigureNumberForModule',CurrentModule];
    ThisModuleFigureNumber = handles.Current.(fieldname);
    if any(findobj == ThisModuleFigureNumber) == 1;
        close(ThisModuleFigureNumber);
    end
end
