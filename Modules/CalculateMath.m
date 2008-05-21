function handles = CalculateMath(handles)

% Help for the Calculate Math module:
% Category: Measurement
%
% SHORT DESCRIPTION:
% This module can take any measurements produced by previous modules and
% can manipulate the numbers using basic arithmetic operations.
% *************************************************************************
%
% The arithmetic operations available in this module include addition,
% subtraction, multiplication and division. The operation can be chosen
% by adjusting the operations setting. The resulting data can also be
% logged or raised to a power. This data can then be used in other
% calculations and can be used in Classify Objects.
%
% This module currently works on an object-by-object basis (it calculates
% the requested operation for each object) but can also apply the operation
% for measurements made for entire images (but only for measurements
% produced by the Correlation module).
%
% Feature Number:
% The feature number specifies which features from the Measure module(s)
% will be used for the operation. See each Measure module's help for the
% numbered list of the features measured by that module.
%
% The calculations are stored along with the *first* object's data.
%
% See also CalculateRatios, all Measure modules.

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

%textVAR01 = Which object would you like to use as the first measurement (The option IMAGE currently only works with Correlation measurements)?
%choiceVAR01 = Image
%infotypeVAR01 = objectgroup
%inputtypeVAR01 = popupmenu
FirstObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = Which category of measurements would you like to use?
%choiceVAR02 = AreaShape
%choiceVAR02 = Correlation
%choiceVAR02 = Intensity
%choiceVAR02 = Neighbors
%choiceVAR02 = Texture
%inputtypeVAR02 = popupmenu custom
FirstCategory = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Which feature do you want to use? (Enter the feature number - see help for details)
%defaultVAR03 = 1
FirstFeatureNumber = str2double(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = For INTENSITY or TEXTURE features, which image's measurements would you like to use?
%infotypeVAR04 = imagegroup
%inputtypeVAR04 = popupmenu
FirstImage = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = For TEXTURE features, what previously measured texture scale do you want to use?
%defaultVAR05 = 1
FirstTextureScale = str2double(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = Which object would you like to use as the second measurement? (The option IMAGE currently only works with Correlation measurements)?
%choiceVAR06 = Image
%infotypeVAR06 = objectgroup
%inputtypeVAR06 = popupmenu
SecondObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = Which category of measurements would you like to use?
%choiceVAR07 = AreaShape
%choiceVAR07 = Correlation
%choiceVAR07 = Intensity
%choiceVAR07 = Neighbors
%choiceVAR07 = Texture
%inputtypeVAR07 = popupmenu custom
SecondCategory = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = Which feature do you want to use? (Enter the feature number - see help for details)
%defaultVAR08 = 1
SecondFeatureNumber = str2double(handles.Settings.VariableValues{CurrentModuleNum,8});

%textVAR09 = For INTENSITY or TEXTURE features, which image's measurements would you like to use?
%infotypeVAR09 = imagegroup
%inputtypeVAR09 = popupmenu
SecondImage = char(handles.Settings.VariableValues{CurrentModuleNum,9});

%textVAR10 = For TEXTURE features, what previously measured texture scale do you want to use?
%defaultVAR10 = 1
SecondTextureScale = str2double(handles.Settings.VariableValues{CurrentModuleNum,10});

%textVAR11 = Do you want the log (base 10) of the ratio?
%choiceVAR11 = No
%choiceVAR11 = Yes
%inputtypeVAR11 = popupmenu
LogChoice = char(handles.Settings.VariableValues{CurrentModuleNum,11});

%textVAR12 = Raise to what power (*after* chosen operation below)?
%defaultVAR12 = 1
Power = str2double(handles.Settings.VariableValues{CurrentModuleNum,12});

%textVAR13 = Operation?
%choiceVAR13 = Multiply
%choiceVAR13 = Divide
%choiceVAR13 = Add
%choiceVAR13 = Subtract
%inputtypeVAR13 = popupmenu
Operation = char(handles.Settings.VariableValues(CurrentModuleNum, 13));

%%%VariableRevisionNumber = 5

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% MAKE MEASUREMENTS & SAVE TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;

FirstFeatureName = CPgetfeaturenamesfromnumbers(handles, FirstObjectName, ...
    FirstCategory, FirstFeatureNumber, FirstImage,FirstTextureScale);

SecondFeatureName = CPgetfeaturenamesfromnumbers(handles, SecondObjectName, ...
    SecondCategory, SecondFeatureNumber, SecondImage,SecondTextureScale);

% Get measurements
FirstMeasurements = handles.Measurements.(FirstObjectName).(FirstFeatureName){SetBeingAnalyzed};
SecondMeasurements = handles.Measurements.(SecondObjectName).(SecondFeatureName){SetBeingAnalyzed};

%% Check sizes (note, 'Image' measurements have length=1)
if length(FirstMeasurements) ~= length(SecondMeasurements) && ...
        ~(length(FirstMeasurements) ==1 || length(SecondMeasurements) == 1)
    error(['Image processing was canceled in the ', ModuleName, ' module because the specified object names ',FirstObjectName,' and ',SecondObjectName,' do not have the same object count.']);
end

%% Since Matlab's max name length is 63, we need to truncate the fieldname
MinStrLen = 5;
FirstFeatureNameSubstrings = textscan(FirstFeatureName,'%s','delimiter','_');
for idxStr = 1:length(FirstFeatureNameSubstrings{1})
    Str = FirstFeatureNameSubstrings{1}{idxStr};
    FirstTruncatedName{idxStr} = Str(1:min(length(Str),MinStrLen));
end
SecondFeatureNameSubstrings = textscan(SecondFeatureName,'%s','delimiter','_');
for idxStr = 1:length(SecondFeatureNameSubstrings{1})
    Str = SecondFeatureNameSubstrings{1}{idxStr};
    SecondTruncatedName{idxStr} = Str(1:min(length(Str),MinStrLen));
end
NewFirstFeatureName = CPjoinstrings(FirstTruncatedName{:});
NewSecondFeatureName = CPjoinstrings(SecondTruncatedName{:});

%% Construct field name
%% Note that we are not including FirstObjectName, since Math measurements 
%%  are stored under the first object's structure
MathFieldName = CPjoinstrings('Math',NewFirstFeatureName,Operation(1:4),...
                            SecondObjectName,NewSecondFeatureName);

%% Do Math
if( strcmpi(Operation, 'Multiply') )
    FinalMeasurements = FirstMeasurements .* SecondMeasurements;
elseif( strcmpi(Operation, 'Divide') )
    FinalMeasurements = FirstMeasurements ./ SecondMeasurements;
elseif( strcmpi(Operation, 'Add') )
    FinalMeasurements = FirstMeasurements + SecondMeasurements;
elseif( strcmpi(Operation, 'Subtract') )
    FinalMeasurements = FirstMeasurements - SecondMeasurements;
end
    
if strcmp(LogChoice,'Yes')
    FinalMeasurements = log10(FinalMeasurements);
end

if ~isnan(Power)
    FinalMeasurements = FinalMeasurements .^ Power;
end

handles = CPaddmeasurements(handles,FirstObjectName,MathFieldName,FinalMeasurements);

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% The figure window display is unnecessary for this module, so it is
%%% closed during the starting image cycle.
CPclosefigure(handles,CurrentModule)