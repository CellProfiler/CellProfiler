function handles = DisplayDataOnImage(handles)

% Help for the Display Data on Image module:
% Category: Other
%
% SHORT DESCRIPTION:
% Produce image with measured data on top of measured objects.
% *************************************************************************
%
% The resulting images with data on top can be saved using the Save Images
% module.
%
% Feature Number:
% The feature number specifies which feature from the Measure module will
% be used for display. See each Measure module's help for the numbered
% list of the features measured by that module.
%
% See also MeasureObjectAreaShape, MeasureObjectIntensity,
% MeasureObjectTexture, MeasureCorrelation, MeasureNeighbors.

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
%   Susan Ma
%   Wyman Li
%
% Website: http://www.cellprofiler.org
%
% $Revision: 2614 $

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = Which object would you like to use for the data (The option IMAGE currently only works with Correlation measurements)?
%choiceVAR01 = Image
%infotypeVAR01 = objectgroup
%inputtypeVAR01 = popupmenu
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = Which category of measurements would you like to use?
%choiceVAR02 = AreaShape
%choiceVAR02 = Intensity
%choiceVAR02 = Texture
%choiceVAR02 = Correlation
%choiceVAR02 = Neighbors
%inputtypeVAR02 = popupmenu custom
Measure = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Which feature do you want to use? (Enter the feature number - see HELP for explanation)
%defaultVAR03 = 1
FeatureNo = str2double(handles.Settings.VariableValues{CurrentModuleNum,3});

if isempty(FeatureNo)
    error(['Image processing was canceled in the ', ModuleName, ' module because your entry for the Feature Number is invalid.']);
end

%textVAR04 = For INTENSITY or TEXTURE features, which image was used to make the measurements?
%infotypeVAR04 = imagegroup
%inputtypeVAR04 = popupmenu
Image = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = Which image do you want to display the data on?
%infotypeVAR05 = imagegroup
%inputtypeVAR05 = popupmenu
DisplayImage = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = What do you want to call the generated image with data?
%defaultVAR06 = OrigDataDisp
%infotypeVAR06 = imagegroup indep
DataImage = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines which cycle is being analyzed.
SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;

if strcmp(Measure,'Intensity') || strcmp(Measure,'Texture')
    Measure = [Measure, '_',Image];
end

%%% Checks whether the image to be analyzed exists in the handles structure.
if ~isfield(handles.Pipeline, DisplayImage)
    %%% If the image is not there, an error message is produced.  The error
    %%% is not displayed: The error function halts the current function and
    %%% returns control to the calling function (the analyze all images
    %%% button callback.)  That callback recognizes that an error was
    %%% produced because of its try/catch loop and breaks out of the image
    %%% analysis loop without attempting further modules.
    error(['Image processing was canceled in the ', ModuleName, ' module because it could not find the input image.  It was supposed to be named ', ImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
%%% Reads the image.
OrigImage = handles.Pipeline.(DisplayImage);
if max(OrigImage(:)) > 1 || min(OrigImage(:)) < 0
    CPwarndlg(['The images you have loaded in the ', ModuleName, ' module are outside the 0-1 range, and you may be losing data.'],'Outside 0-1 Range','replace');
end

%%%%%%%%%%%%%%%%%%%%%
%%% DATA ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow

ErrorFlag = 0;
try
    tmp = handles.Measurements.(ObjectName).(Measure){SetBeingAnalyzed};
catch
    ErrorFlag = 1;
end

if ErrorFlag
    ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
    %%% Creates the display window.
    DataHandle = CPfigure(handles,ThisModuleFigureNumber);
    title('No objects identified.');
    CPwarndlg(['No objects were identified. This could mean that the measurements you have specified in the ',ModuleName,' are not being processed. Please verify that the Measure module precedes this module.']); 
else
    ListOfMeasurements = tmp(:,FeatureNo);
    StringListOfMeasurements = cellstr(num2str(ListOfMeasurements));

    %%% Extracts the XY locations. This is temporarily hard-coded
    Xlocations = handles.Measurements.(ObjectName).Location{SetBeingAnalyzed}(:,1);
    Ylocations = handles.Measurements.(ObjectName).Location{SetBeingAnalyzed}(:,2);

    %%%%%%%%%%%%%%%
    %%% DISPLAY %%%
    %%%%%%%%%%%%%%%
    drawnow
    
    ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
    drawnow
    %%% Activates the appropriate figure window.
    DataHandle = CPfigure(handles,ThisModuleFigureNumber);
    
    CPimagesc(OrigImage);
    colormap(gray);
    FeatureDisp = handles.Measurements.(ObjectName).([Measure,'Features']){FeatureNo};
    title([ObjectName,', ',FeatureDisp,' on ',Image])

    %%% Overlays the values in the proper location in the image.
    TextHandles = text(Xlocations , Ylocations , StringListOfMeasurements,...
        'HorizontalAlignment','center', 'color', [1 1 0],'fontsize',handles.Preferences.FontSize);

    %%% Create structure and save it to the UserData property of the window
    Info = get(DataHandle,'UserData');
    Info.ListOfMeasurements = ListOfMeasurements;
    Info.TextHandles = TextHandles;
    set(DataHandle,'UserData',Info);
end

OneFrame = getframe(DataHandle);
handles.Pipeline.(DataImage)=OneFrame.cdata;