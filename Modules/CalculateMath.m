function handles = CalculateMath(handles)

% Help for the Calculate Math module:
% Category: Measurement
%
% SHORT DESCRIPTION:
% This module can take measurements produced by previous modules and
% performs basic arithmetic operations.
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
% for measurements made for entire images.
%
% Feature Number:
% The feature number specifies which features from the Measure module(s)
% will be used for the operation. See each Measure module's help for the
% numbered list of the features measured by that module.
%
% Saving:
% The math measurements are stored as 'Math_...'. If both measures are 
% image-based, then a single calculation (per cycle) will be stored as 'Image' data.  
% If one measure is object-based and one image-based, then the calculations will
% be stored associated with the object, one calculation per object.  If both are 
% objects, then the calculations are stored with both objects.
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

%textVAR01 = Which object would you like to use as the first measurement?
%choiceVAR01 = Image
%infotypeVAR01 = objectgroup
%inputtypeVAR01 = popupmenu
ObjectName{1} = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = Which category of measurements would you like to use?
%choiceVAR02 = AreaOccupied
%choiceVAR02 = AreaShape
%choiceVAR02 = Children
%choiceVAR02 = Parent
%choiceVAR02 = Correlation
%choiceVAR02 = Intensity
%choiceVAR02 = Neighbors
%choiceVAR02 = Ratio
%choiceVAR02 = Texture
%choiceVAR02 = RadialDistribution
%choiceVAR02 = Granularity
%inputtypeVAR02 = popupmenu custom
Category{1} = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Which feature do you want to use? (Enter the feature number or name - see help for details)
%defaultVAR03 = 1
FeatureNumber{1} = handles.Settings.VariableValues{CurrentModuleNum,3};

%textVAR04 = For INTENSITY or TEXTURE features, which image's measurements would you like to use?
%infotypeVAR04 = imagegroup
%inputtypeVAR04 = popupmenu
ImageName{1} = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = For TEXTURE features, what previously measured texture scale do you want to use?
%defaultVAR05 = 1
SizeScale{1} = str2double(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = Which object would you like to use as the second measurement?
%choiceVAR06 = Image
%infotypeVAR06 = objectgroup
%inputtypeVAR06 = popupmenu
ObjectName{2} = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = Which category of measurements would you like to use?
%choiceVAR07 = AreaOccupied
%choiceVAR07 = AreaShape
%choiceVAR07 = Children
%choiceVAR07 = Parent
%choiceVAR07 = Correlation
%choiceVAR07 = Intensity
%choiceVAR07 = Neighbors
%choiceVAR07 = Ratio
%choiceVAR07 = Texture
%choiceVAR07 = RadialDistribution
%choiceVAR07 = Granularity
%inputtypeVAR07 = popupmenu custom
Category{2} = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = Which feature do you want to use? (Enter the feature number or name - see help for details)
%defaultVAR08 = 1
FeatureNumber{2} = handles.Settings.VariableValues{CurrentModuleNum,8};

%textVAR09 = For INTENSITY or TEXTURE features, which image's measurements would you like to use?
%infotypeVAR09 = imagegroup
%inputtypeVAR09 = popupmenu
ImageName{2} = char(handles.Settings.VariableValues{CurrentModuleNum,9});

%textVAR10 = For TEXTURE features, what previously measured texture scale do you want to use?
%defaultVAR10 = 1
SizeScale{2} = str2double(handles.Settings.VariableValues{CurrentModuleNum,10});

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

%textVAR14 = What do you want to call the output calculated by this module? The prefix, "Math_" will be applied to your entry or simply leave as "Automatic" and a sensible name will be generated.'
%defaultVAR14 = Automatic
OutputFeatureName = char(handles.Settings.VariableValues(CurrentModuleNum,14));
%%%VariableRevisionNumber = 6

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% MAKE MEASUREMENTS & SAVE TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;

for idx = 1:2
    try
        FeatureName{idx} = CPgetfeaturenamesfromnumbers(handles, ObjectName{idx}, ...
            Category{idx}, FeatureNumber{idx}, ImageName{idx},SizeScale{idx});

        Measurements{idx} = handles.Measurements.(ObjectName{idx}).(FeatureName{idx}){SetBeingAnalyzed};
    catch
        error(['Image processing was canceled in the ', ModuleName, ...
            ' module (#' num2str(CurrentModuleNum) ...
            ') because an error ocurred when retrieving the data.  '...
            'Likely the category of measurement you chose, ',...
            Category{idx}, ', was not available for ', ...
            ObjectName{idx},' with feature number ' FeatureNumber{idx} ...
            ', possibly specific to image ''' ImageName{idx} ''' and/or ' ...
            'Texture Scale = ' num2str(SizeScale{idx}) '.']);
    end
end
    
%% Check sizes (note, 'Image' measurements have length=1)
if length(Measurements{1}) ~= length(Measurements{2}) && ...
        ~(length(Measurements{1}) ==1 || length(Measurements{2}) == 1)
    error(['Image processing was canceled in the ', ModuleName, ' module because the specified object names ',ObjectName{1},' and ',ObjectName{2},' do not have the same object count.']);
end

%% Construct field name
if isempty(OutputFeatureName) || strcmp(OutputFeatureName,'Automatic') == 1
    FullFeatureName = CPjoinstrings('Math',ObjectName{1},FeatureName{1},...
                                    Operation,ObjectName{2},FeatureName{2});

    %% Since Matlab's max name length is 63, we need to truncate the fieldname
    MinStrLen = 5;
    TruncFeatureName = CPtruncatefeaturename(FullFeatureName,MinStrLen);
else
    TruncFeatureName = CPjoinstrings('Math',OutputFeatureName);
end
%% Do Math
if( strcmpi(Operation, 'Multiply') )
    FinalMeasurements = Measurements{1} .* Measurements{2};
elseif( strcmpi(Operation, 'Divide') )
    FinalMeasurements = Measurements{1} ./ Measurements{2};
elseif( strcmpi(Operation, 'Add') )
    FinalMeasurements = Measurements{1} + Measurements{2};
elseif( strcmpi(Operation, 'Subtract') )
    FinalMeasurements = Measurements{1} - Measurements{2};
end
    
if strcmp(LogChoice,'Yes')
    FinalMeasurements = log10(FinalMeasurements);
end

if ~isnan(Power)
    FinalMeasurements = FinalMeasurements .^ Power;
end

%% Save, depending on type of measurement (ObjectName)
%% Note that Image measurements are scalars, while Objects are potentially vectors
if strcmp(ObjectName{1}, 'Image') && strcmp(ObjectName{2}, 'Image'),
    handles = CPaddmeasurements(handles,'Image',TruncFeatureName,FinalMeasurements);
else
    if ~strcmp(ObjectName{1}, 'Image'),
        handles = CPaddmeasurements(handles,ObjectName{1},TruncFeatureName,FinalMeasurements);
    end
    if ~strcmp(ObjectName{1}, ObjectName{2}) && ~ strcmp(ObjectName{2},'Image')
        handles = CPaddmeasurements(handles,ObjectName{2},TruncFeatureName,FinalMeasurements);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%% TODO add display of numbers, as in CalculateRatios, and remove CPclosefigure

%%% The figure window display is unnecessary for this module, so it is
%%% closed during the starting image cycle.
CPclosefigure(handles,CurrentModule)