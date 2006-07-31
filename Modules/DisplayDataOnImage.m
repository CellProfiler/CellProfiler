function handles = DisplayDataOnImage(handles)

% Help for the Display Data on Image module:
% Category: Other
%
% SHORT DESCRIPTION:
% Produces an image with measured data on top of identified objects.
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
% See also MeasureObjectAreaShape, MeasureImageAreaOccupied,
% MeasureObjectIntensity, MeasureImageIntensity, MeasureTexture,
% MeasureCorrelation, MeasureObjectNeighbors, CalculateRatios.

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
% $Revision: 2614 $

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = Which object would you like to use for the data, or if using a Ratio, what is the numerator object?
%choiceVAR01 = Image
%infotypeVAR01 = objectgroup
%inputtypeVAR01 = popupmenu
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = Which category of measurements would you like to use?
%choiceVAR02 = AreaShape
%choiceVAR02 = Correlation
%choiceVAR02 = Intensity
%choiceVAR02 = Neighbors
%choiceVAR02 = Ratio
%choiceVAR02 = Texture
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

%%% Get the correct fieldname where measurements are located
CellFlg = 0;
switch Measure
    case 'AreaShape'
        if strcmp(ObjectName,'Image')
            Measure = '^AreaOccupied_.*Features$';
            Fields = fieldnames(handles.Measurements.Image);
            TextComp = regexp(Fields,Measure);
            A = cellfun('isempty',TextComp);
            try
                Measure = Fields{find(A==0)+1};
            catch
                error(['Image processing was canceled in the ', ModuleName, ' module because the category of measurement you chose, ', Measure, ', was not available for ', ObjectName]);
            end
            CellFlg = 1;
        end
    case 'Intensity'
        Measure = ['Intensity_' Image];
    case 'Neighbors'
        Measure = 'NumberNeighbors';
    case 'Texture'
        Measure = ['Texture_[0-9]*[_]?' Image '$'];
        Fields = fieldnames(handles.Measurements.(ObjectName));
        TextComp = regexp(Fields,Measure);
        A = cellfun('isempty',TextComp);
        try
            Measure = Fields{A==0};
        catch
            error(['Image processing was canceled in the ', ModuleName, ' module because the category of measurement you chose, ', Measure, ', was not available for ', ObjectName]);
        end
    case 'Ratio'
        Measure = '.*Ratio$';
        Fields = fieldnames(handles.Measurements.(ObjectName));
        TextComp = regexp(Fields,Measure);
        A = cellfun('isempty',TextComp);
        try
            Measure = Fields{A==0};
        catch
            error(['Image processing was canceled in the ', ModuleName, ' module because the category of measurement you chose, ', Measure, ', was not available for ', ObjectName]);
        end
end

%%% Reads the image.
OrigImage = CPretrieveimage(handles,DisplayImage,ModuleName,'DontCheckColor','CheckScale');

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
    DataHandle = CPfigure(handles,'Image',ThisModuleFigureNumber);
    title('No objects identified.');
    CPwarndlg(['No objects were identified. This could mean that the measurements you have specified in the ',ModuleName,' are not being processed. Please verify that the Measure module precedes this module.']); 
else
    ListOfMeasurements = tmp(:,FeatureNo);
    if CellFlg
        ListOfMeasurements = ListOfMeasurements{1};
    end
    StringListOfMeasurements = cellstr(num2str(ListOfMeasurements));

    %%% Extracts the XY locations. This is temporarily hard-coded
    if ~strcmp(ObjectName,'Image')
        Xlocations = handles.Measurements.(ObjectName).Location{SetBeingAnalyzed}(:,1);
        Ylocations = handles.Measurements.(ObjectName).Location{SetBeingAnalyzed}(:,2);
    else
        Xlocations = size(OrigImage,2)/2;
        Ylocations = size(OrigImage,1)/2;
    end

    %%%%%%%%%%%%%%%
    %%% DISPLAY %%%
    %%%%%%%%%%%%%%%
    drawnow
    
    ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
    %%% Activates the appropriate figure window.
    DataHandle = CPfigure(handles,'Image',ThisModuleFigureNumber);
    
    CPimagesc(OrigImage,handles);
    colormap(gray);
    FeatureDisp = handles.Measurements.(ObjectName).([Measure,'Features']){FeatureNo};
    Title = [ObjectName,', ',FeatureDisp,' on ',Image];
    Title = strrep(Title,'_','\_');
    title(Title);

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