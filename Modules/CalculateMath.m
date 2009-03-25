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

% MBray 2009_03_20: Comments on variables for pyCP upgrade
%
% Recommended variable order (setting, followed by current variable in MATLAB CP)
% (1) What do you want to call the measurement calculated by this module? (OutputFeatureName)
% (2) What operation would you like to perform? (Operation)
% (do we currently allow 'raise to a power' here?)
% Perhaps show settings for the operands in two columns (eg, numerator settings on
% left, denominator settings on right). Maybe show mathematical symbol (based on answer
% to (2)) between the columns. Or, if this is too time-consuming, at least
% show, based on the selection in (2), which operand is which. In other
% words, if they choose Division, then show: Operand 1/operand 2. If they
% choose subtraction, show: operand 1 - operand 2.
%
% (2.5) What type of number is the first operand? 
% CHOICES: (a) measurement of an individual object, (b) measurement from a whole
% image, (c) number
% If they choose (a) then provide the list of objects, if (b) then list of
% images, if (c) then provide an edit box for them to enter something. This
% new ability to calculate per-image features or enter numbers then affects
% the rest of the questions below (they are currently written for
% per-object measurements, but hopefully it's clear how to adjust them for
% per-image measurements or for a number the user specifies):
%
% (3a) Which object's measurement would you like to use as the first operand? (ObjectName{1})
% (3b) What is the measurement category? (Category{1})
% (3c) What is the measurement feature? (FeatureNumber{1})
% (3d) (If the answer to (3c) involves a scale) What scale was used to 
%      calculate the feature? (SizeScale{1}) 
%      (If the answer to (3c) involves an image) What image was used to
%      calculate the feature? (ImageName{1})
% (3e) What number would you like to multiply the operand by?  (MultiplyFactor1)
% (3f) What power would you like to raise the operand to? (this feature
% currently doesn't exist; should we ask the user what operation they want
% to perform on the first operand prior to calculating with the second
% operand, then offer multiply, raise to a power, and perhaps other
% options? or is this just getting way too confusing?)
%
%
% (3.5) Again, insert "what type of number is the second operand?"
%
% (4a) Which object's measurement would you like to use as the second operand? (ObjectName{2})
% (4b) What is the measurement category? (Category{2})
% (4c) What is the measurement feature? (FeatureNumber{2})
% (4d) (If the answer to (4c) involves a scale) What scale was used to 
%      calculate the feature? (SizeScale{2}) 
%      (If the answer to (4c) involves an image) What image was used to
%      calculate the feature? (ImageName{2})
% (4e) What number would you like to multiply the operand by? (MultiplyFactor2)
% (4f) What power would you like to raise the operand to? (this feature
% currently doesn't exist)
%
% question: should we allow the user to choose additional operands (perhaps just for
% addition and multiplication) or just let them use an additional
% CalculateMath module in the pipeline for that sort of thing? I think the
% latter would be fine, especially if it keeps this module from ending up
% being confusing.
%
% Note: we could eliminate the next two questions because the user could
% just put an additional CalculateMath module in the pipeline, right?
% (5a) What power would you like to raise the result to? (Power)
% (5b) What number would you like to multiply the above result by? (MultiplyFactor3)
%
% NOTES:
% (i) Setting (1) should not have an automatic option; the user should
% specify the name explicitly. (Anne says: why not, Mark? Do we tend to run
% into trouble with the name being too long? If so, then this makes sense.
% We should make sure then, that the custom-entered feature name is
% choosable in other drop down menus downstream in the pipeline - I think
% we have a 'custom' measurement type.)
% (ii) Measurement category/feature/image/scale settings should only be shown if
% the measurement hierarchy requires it.
% (iii) Some clarification will be needed to show the order of operations to
% the user

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
    
% Check sizes (note, 'Image' measurements have length=1)
if length(Measurements{1}) ~= length(Measurements{2}) && ...
        ~(length(Measurements{1}) ==1 || length(Measurements{2}) == 1)
    error(['Image processing was canceled in the ', ModuleName, ' module because the specified object names ',ObjectName{1},' and ',ObjectName{2},' do not have the same object count.']);
end

% Construct field name
if isempty(OutputFeatureName) || strcmp(OutputFeatureName,'Automatic') == 1
    FullFeatureName = CPjoinstrings('Math',ObjectName{1},FeatureName{1},...
                                    Operation,ObjectName{2},FeatureName{2});

    % Since Matlab's max name length is 63, we need to truncate the fieldname
    TruncFeatureName = CPtruncatefeaturename(FullFeatureName);
else
    TruncFeatureName = CPjoinstrings('Math',OutputFeatureName);
end
% Do Math
if( strcmpi(Operation, 'Multiply') )
    FinalMeasurements = (MultiplyFactor1.*Measurements{1}) .* (MultiplyFactor2.*Measurements{2});
elseif( strcmpi(Operation, 'Divide') )
    Measurements{2}(Measurements{2}==0) = NaN;
    Measurements{2}(isnan(Measurements{2})) = CPnanmean(Measurements{2});
    
    if ~all(isnan(Measurements{2}))
    FinalMeasurements = (MultiplyFactor1.*Measurements{1}) ./ (MultiplyFactor2.*Measurements{2});
    else
    FinalMeasurements = NaN;
    CPwarndlg(['A ratio of ' NumerDenomMeasurements{1} ' and ' NumerDenomMeasurements{2} ...
        ' within ' ModuleName ' on cycle '  str2double(SetBeingAnalyzed) ...
        ' resulted in all NaNs.  You may want to check your settings.'])
    end

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

% Save, depending on type of measurement (ObjectName)
% Note that Image measurements are scalars, while Objects are potentially vectors
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

% TODO add display of numbers, as in CalculateRatios, and remove CPclosefigure

%%% The figure window display is unnecessary for this module, so it is
%%% closed during the starting image cycle.
CPclosefigure(handles,CurrentModule)