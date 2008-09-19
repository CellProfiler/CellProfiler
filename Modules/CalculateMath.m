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
% Category: 'Math'
% Features measured:                         Feature Number:
% (As named in module's last setting)     |       1
%
% Note: If you want to use the output of this module in a subsequesnt
% calculation, we suggest you specify the output name rather than use
% Automatic naming.
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
%inputtypeVAR02 = popupmenu category
Category{1} = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Which feature do you want to use? (Enter the feature number or name - see help for details)
%defaultVAR03 = 1
%inputtypeVAR03 = popupmenu measurement
FeatureNumber{1} = handles.Settings.VariableValues{CurrentModuleNum,3};

%textVAR04 = For INTENSITY, AREAOCCUPIED or TEXTURE features, which image's measurements would you like to use?
%infotypeVAR04 = imagegroup
%inputtypeVAR04 = popupmenu
ImageName{1} = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = For TEXTURE, RADIAL DISTRIBUTION, OR NEIGHBORS features, what previously measured size scale (TEXTURE OR NEIGHBORS) or previously used number of bins (RADIALDISTRIBUTION) do you want to use?
%defaultVAR05 = 1
%inputtypeVAR05 = popupmenu scale
SizeScale{1} = str2double(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = Which object would you like to use as the second measurement?
%choiceVAR06 = Image
%infotypeVAR06 = objectgroup
%inputtypeVAR06 = popupmenu
ObjectName{2} = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = Which category of measurements would you like to use?
%inputtypeVAR07 = popupmenu category
Category{2} = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = Which feature do you want to use? (Enter the feature number or name - see help for details)
%defaultVAR08 = 1
%inputtypeVAR08 = popupmenu measurement
FeatureNumber{2} = handles.Settings.VariableValues{CurrentModuleNum,8};

%textVAR09 = For INTENSITY, AREAOCCUPIED or TEXTURE features, which image's measurements would you like to use?
%infotypeVAR09 = imagegroup
%inputtypeVAR09 = popupmenu
ImageName{2} = char(handles.Settings.VariableValues{CurrentModuleNum,9});

%textVAR10 = For TEXTURE, RADIAL DISTRIBUTION, OR NEIGHBORS features, what previously measured size scale (TEXTURE OR NEIGHBORS) or previously used number of bins (RADIALDISTRIBUTION) do you want to use?
%defaultVAR10 = 1
%inputtypeVAR10 = popupmenu scale
SizeScale{2} = str2double(handles.Settings.VariableValues{CurrentModuleNum,10});

%textVAR11 = Do you want the log (base 10) of the ratio?
%choiceVAR11 = No
%choiceVAR11 = Yes
%inputtypeVAR11 = popupmenu
LogChoice = char(handles.Settings.VariableValues{CurrentModuleNum,11});

%textVAR12 = Enter a factor to multiply the first feature by (before other operations):
%defaultVAR12 = 1
MultiplyFactor1 = str2double(handles.Settings.VariableValues{CurrentModuleNum,12});

%textVAR13 = Enter a factor to multiply the second feature by (before other operation):
%defaultVAR13 = 1
MultiplyFactor2 = str2double(handles.Settings.VariableValues{CurrentModuleNum,13});

%textVAR14 = Enter an exponent to raise the result to (*after* chosen operation below)?
%defaultVAR14 = 1
Power = str2double(handles.Settings.VariableValues{CurrentModuleNum,14});

%textVAR15 = Enter a factor to multiply the result by (*after* chosen operation below)?
%defaultVAR15 = 1
MultiplyFactor3 = str2double(handles.Settings.VariableValues{CurrentModuleNum,15});

%textVAR16 = Operation?
%choiceVAR16 = Multiply
%choiceVAR16 = Divide
%choiceVAR16 = Add
%choiceVAR16 = Subtract
%inputtypeVAR16 = popupmenu
Operation = char(handles.Settings.VariableValues(CurrentModuleNum, 16));

%textVAR17 = What do you want to call the output calculated by this module? The prefix, "Math_" will be applied to your entry or simply leave as "Automatic" and a sensible name will be generated.'
%defaultVAR17 = Automatic
OutputFeatureName = char(handles.Settings.VariableValues(CurrentModuleNum,17));
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
        error([lasterr '  Image processing was canceled in the ', ModuleName, ...
            ' module (#' num2str(CurrentModuleNum) ...
            ') because an error ocurred when retrieving the data.  '...
            'Likely the category of measurement you chose, ',...
            Category{idx}, ', was not available for ', ...
            ObjectName{idx},' with feature number ' FeatureNumber{idx} ...
            ', possibly specific to image ''' ImageName{idx} ''' and/or ' ...
            'Texture Scale = ' num2str(SizeScale{idx}) '.']);
    end
end

%%% Check to make sure multiply factors are valid entries. If not change to
%%% default and warn user.
if isnan(MultiplyFactor1)
    if isempty(findobj('Tag',['Msgbox_' ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': First multiply factor invalid']))
        CPwarndlg(['The first image multiply factor you have entered in the ', ModuleName, ' module is invalid, it is being reset to 1.'],[ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': First multiply factor invalid'],'replace');
    end
    MultiplyFactor1 = 1;
end
if isnan(MultiplyFactor2)
    if isempty(findobj('Tag',['Msgbox_' ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Second multiply factor invalid']))
        CPwarndlg(['The second image multiply factor you have entered in the ', ModuleName, ' module is invalid, it is being reset to 1.'],[ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Second multiply factor invalid'],'replace');
    end
    MultiplyFactor2 = 1;
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
    TruncFeatureName = CPtruncatefeaturename(FullFeatureName);
else
    TruncFeatureName = CPjoinstrings('Math',OutputFeatureName);
end
%% Do Math
if( strcmpi(Operation, 'Multiply') )
    FinalMeasurements = (MultiplyFactor1.*Measurements{1}) .* (MultiplyFactor2.*Measurements{2};
elseif( strcmpi(Operation, 'Divide') )
    FinalMeasurements = (MultiplyFactor1.*Measurements{1}) ./ (MultiplyFactor2.*Measurements{2});
elseif( strcmpi(Operation, 'Add') )
    FinalMeasurements = (MultiplyFactor1.*Measurements{1}) + (MultiplyFactor2.*Measurements{2});
elseif( strcmpi(Operation, 'Subtract') )
    FinalMeasurements = (MultiplyFactor1.*Measurements{1}) - (MultiplyFactor2.*Measurements{2});
end
    
if strcmp(LogChoice,'Yes')
    FinalMeasurements = log10(FinalMeasurements);
end

if ~isnan(Power)
    FinalMeasurements = FinalMeasurements .^ Power;
end

if ~isnan(MultiplyFactor3)
    FinalMeasurements = FinalMeasurements.*MultiplyFactor3;
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