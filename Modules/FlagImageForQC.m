function handles = FlagImageForQC(handles)

% Help for the Flag Image for QC (quality control) module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% This module allows you to flag an image if it fails some quality control
% measurement you specify. 
% *************************************************************************
%
% This module adds a measurement in the handles structure under
% 'Experiment'.  The measurement is a flag that a user can assign if the
% image fails some quality control measurement he or she specifies.  The
% vale of the measurement is '1' if the image has failed QC, and '0' if it
% has passed. The flag can be used in post-processing to filter out images
% the user does not want to analyze in CP Analyst, for example, or in
% creating an illumination function (currently, this is only possible using LoadImageDirectory).
%
% To flag an image by more than one measurement, you can use multiple
% 'FlagImageForQC' modules and select, 'Append an existing flag' and enter
% the name of the flag you want to append.
%
% By default, the measurements you are using to flag an image are
% measurements from that image.
% 
% This module requires the measurements of all prior modules & cycles, so
% it must be the last in the pipeline, and it will not work if run on a
% cluster of computers, unless you run all cycles as one batch.
% 
% See also
%
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
% $Revision: 6876 $

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = Which image would you like to flag for quality control?
%infotypeVAR01 = imagegroup
%inputtypeVAR01 = popupmenu
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = Which category of measurements would you like to use?
%inputtypeVAR02 = popupmenu category
Category = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Which feature do you want to use? (Enter the feature number or name - see help for details)
%defaultVAR03 = 1
%inputtypeVAR03 = popupmenu measurement
FeatureNumOrName = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = Images with a measurement below this value will be flagged:
%choiceVAR04 = No minimum
%inputtypeVAR04 = popupmenu custom
MinValue1 = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = Images with a measurement above this value will be flagged:
%choiceVAR05 = No maximum
%inputtypeVAR05 = popupmenu custom
MaxValue1 = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = Did you want to append an existing flag, or create a new one? ** Warning, if you choose to append a flag, you will be overwriting it with the appended flag **
%choiceVAR06 = Create a new flag
%choiceVAR06 = Append existing flag
%inputtypeVAR06 = popupmenu
NewOrAppend = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = If you're creating a new flag, what do you want to call it?
%defaultVAR07 = NewFlag
%inputtypeVAR07 = popupmenu custom
NewName = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = If you're appending an existing flag, what did you call it?
%defaultVAR08 = Flag
%inputtypeVAR08 = popupmenu custom
OldName = char(handles.Settings.VariableValues{CurrentModuleNum,8});


%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;

if isempty(FeatureNumOrName)
    error(['Image processing was canceled in the ', ModuleName, ' module because your entry for feature number is not valid.']);
end
try
    FeatureName = CPgetfeaturenamesfromnumbers(handles, 'Image', ...
        Category, FeatureNumOrName, ImageName);

catch
    error([lasterr '  Image processing was canceled in the ', ModuleName, ...
        ' module (#' num2str(CurrentModuleNum) ...
        ') because an error ocurred when retrieving the data.  '...
        'Likely the category of measurement you chose, ',...
        Category, ', was not available with feature ' num2str(FeatureNumOrName) ...
        ', possibly specific to image ''' ImageName '''']);
end
MeasureInfo = handles.Measurements.Image.(FeatureName){SetBeingAnalyzed};

if strcmpi(MinValue1, 'No minimum')
    MinValue1 = -Inf;
else
    MinValue1 = str2double(MinValue1);
end

if strcmpi(MaxValue1, 'No maximum')
    MaxValue1 = Inf;
else
    MaxValue1 = str2double(MaxValue1);
end

if strcmpi(MinValue1, 'No minimum') && strcmpi(MaxValue1, 'No maximum')
    CPwarndlg(['You are not flagging any images with the default settings ' ...
        ModuleName ' (module #' num2str(CurrentModuleNum) ')'])
end



if strcmpi(NewOrAppend,'Append existing flag')
    try
        OldName = CPjoinstrings('QCFlag',OldName);
        FlagToAppend = handles.Measurements.Image.(OldName){SetBeingAnalyzed};
    catch
        error([lasterr ' Image processing was cancelled in the ', ModuleName, ...
            ' module (#' num2str(CurrentModuleNum)...
            ') because an error ocurred when retrieving the data.  '...
            'Likely the name of the QCFlag you specified to append,',...
            OldName,',did not exist']);
    end
end


%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow
%% Do Filtering

if MeasureInfo < MinValue1 || MeasureInfo > MaxValue1
    Flag = 1;
else 
    Flag = 0;
end

if strcmpi(NewOrAppend,'Append existing flag')
    try
        Flag = Flag + FlagToAppend;
    catch
        error([lasterr 'Image processing was cancelled in the ', ModuleName, ...
            ' module (#' num2str(CurrentModuleNum)...
            ') because an error while saving the data.  '...
            'Likely the name of the QCFlag you specified to append,',...
            OldName,',was not the same length as the one you were appending to it']);
    end
    if Flag == 2
        Flag =1;
    end
    
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

if strcmpi(NewOrAppend,'Append existing flag')
    handles.Measurements.Image.(OldName){SetBeingAnalyzed} = Flag;
end
if strcmpi(NewOrAppend,'Create a new flag')
    NewName = CPjoinstrings('QCFlag',NewName);
    handles = CPaddmeasurements(handles,'Image',NewName,Flag,SetBeingAnalyzed);
end
    

%% display %%
ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    %%% Activates the appropriate figure window.
    CPfigure(handles,'Text',ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure('','NarrowText',ThisModuleFigureNumber)
    end
    
    if isempty(findobj('Parent',ThisModuleFigureNumber,'tag','TextUIControl'))
        displaytexthandle = uicontrol(ThisModuleFigureNumber,'tag','TextUIControl','style','text','units','normalized','position', [0.1 0.1 0.8 0.8],'fontname','helvetica','backgroundcolor',[.7 .7 .9],'horizontalalignment','left','FontSize',handles.Preferences.FontSize);
    else
        displaytexthandle = findobj('Parent',ThisModuleFigureNumber,'tag','TextUIControl');
    end

    DisplayText = strvcat(['QCFlag = ',num2str(Flag)]);
    set(displaytexthandle,'string',DisplayText)
end



