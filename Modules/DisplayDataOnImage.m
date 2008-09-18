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
% MeasureCorrelation, MeasureObjectNeighbors, and CalculateRatios modules.

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

%textVAR01 = Which object would you like to use for the data, or if using a Ratio, what is the numerator object?
%choiceVAR01 = Image
%infotypeVAR01 = objectgroup
%inputtypeVAR01 = popupmenu
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = Which category of measurements would you like to use?
%inputtypeVAR02 = popupmenu category
Category = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Which feature do you want to use? (Enter the feature number or name - see help for details)
%defaultVAR03 = 1
%inputtypeVAR03 = popupmenu measurement
FeatureNbr = handles.Settings.VariableValues{CurrentModuleNum,3};

if isempty(FeatureNbr) 
    error(['Image processing was canceled in the ', ModuleName, ' module because your entry for the Feature Number is invalid.']);
end

%textVAR04 = For INTENSITY, AREAOCCUPIED or TEXTURE features, which image was used to make the measurements?
%infotypeVAR04 = imagegroup
%inputtypeVAR04 = popupmenu
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = For TEXTURE, RADIAL DISTRIBUTION, OR NEIGHBORS features, what previously measured size scale (TEXTURE OR NEIGHBORS) or previously used number of bins (RADIALDISTRIBUTION) do you want to use?
%defaultVAR05 = 1
%inputtypeVAR05 = popupmenu scale
SizeScale = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = Which image do you want to display the data on?
%infotypeVAR06 = imagegroup
%inputtypeVAR06 = popupmenu
DisplayImage = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = What do you want to call the generated image with data?
%defaultVAR07 = OrigDataDisp
%infotypeVAR07 = imagegroup indep
DataImage = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = What resolution do you want the generated image captured at (units of DPI)?
%choiceVAR08 = 96
%choiceVAR08 = 150
%choiceVAR08 = 300
%inputtypeVAR08 = popupmenu custom
DPIToSave = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%textVAR09 = What elements from the figure do you want to save?
%choiceVAR09 = Figure
%choiceVAR09 = Axes
%choiceVAR09 = Image
%inputtypeVAR09 = popupmenu
SavedImageContents = lower(char(handles.Settings.VariableValues{CurrentModuleNum,9}));

%%%VariableRevisionNumber = 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines which cycle is being analyzed.
SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;

try
    FeatureName = CPgetfeaturenamesfromnumbers(handles,ObjectName,Category,...
        FeatureNbr,ImageName,SizeScale);
catch
    error([lasterr '  Image processing was canceled in the ', ModuleName, ...
        ' module (#' num2str(CurrentModuleNum) ...
        ') because an error ocurred when retrieving the data.  '...
        'Likely the category of measurement you chose, ',...
        Category, ', was not available for ', ...
        ObjectName,' with feature number ' num2str(FeatureNbr) ...
        ', possibly specific to image ''' ImageName ''' and/or ' ...
        'Texture Scale = ' num2str(SizeScale) '.']);
end

%%% Reads the image.
OrigImage = CPretrieveimage(handles,DisplayImage,ModuleName,'DontCheckColor','CheckScale');

%%%%%%%%%%%%%%%%%%%%%
%%% DATA ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow

ErrorFlag = 0;
try
    ListOfMeasurements = handles.Measurements.(ObjectName).(FeatureName){SetBeingAnalyzed};
catch
    ErrorFlag = 1;
end

if ErrorFlag
    ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
    %%% Creates the display window.
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    title('No objects identified.');
    CPwarndlg(['No objects were identified. This could mean that the measurements you have specified in the ',ModuleName,' are not being processed. Please verify that the Measure module precedes this module.']); 
else

    StringListOfMeasurements = cellstr(num2str(ListOfMeasurements));

    %%% Extracts the XY locations. This is temporarily hard-coded
    if ~strcmp(ObjectName,'Image')
        Xlocations = handles.Measurements.(ObjectName).Location_Center_X{SetBeingAnalyzed};
        Ylocations = handles.Measurements.(ObjectName).Location_Center_Y{SetBeingAnalyzed};
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
    FigHandle = CPfigure(handles,'Image',ThisModuleFigureNumber);

    %%% Modules won't create a figure window if the user has requested not
    %%% to. But in this case, we need a figure window in order to get a 
    %%% screenshot.
    %%% Here, if the user doesn't want the figure displayed, we make it
    %%% invisible. The image capture will still work anyway.
    userdoesntwantwindow = ~handles.Preferences.DisplayWindows(str2num(handles.Current.CurrentModuleNumber));
    if userdoesntwantwindow,
        set(FigHandle,'visible','off');
    end
    
    %% Put outlines of objects on image
    LabelMatrixObjectImage = CPretrieveimage(handles,['Segmented' ObjectName],ModuleName,'MustBeGray','DontCheckScale');
    Outlines = bwperim(LabelMatrixObjectImage);
    OutlinesOnImage = OrigImage;
    OutlinesOnImage(logical(Outlines)) = 1;
    
    [hImage,hAx] = CPimagesc(OutlinesOnImage,handles,ThisModuleFigureNumber);
    colormap(hAx,gray);
    
    uicontrol(FigHandle,'units','normalized','position',[.01 .5 .06 .04],'string','off',...
        'UserData',{OrigImage OutlinesOnImage},'backgroundcolor',[.7 .7 .9],...
        'Callback',@CP_OrigNewImage_Callback);
    
    % Now extract the FeatureName itself for the Title
    % (1) Strip Category prefix + slash
    justtheFeatureName = FeatureName(length([Category,'_'])+1:end);
    % (2) Find position of next slash (if there is one)
    slash_index = regexp(justtheFeatureName,'_');
    % (3) FeatureName is the string prior to the slash
    if ~isempty(slash_index)
        slash_index = slash_index(1);
        justtheFeatureName = justtheFeatureName(1:slash_index-1);
    end
    Title = [ObjectName,', ',justtheFeatureName,' on ',ImageName];
    title(hAx,Title);

    %%% Overlays the values in the proper location in the image.
    TextHandles = text(Xlocations , Ylocations , StringListOfMeasurements,...
        'HorizontalAlignment','center', 'color', [1 1 0],'fontsize',handles.Preferences.FontSize,'Parent',hAx);

    %%% Create structure and save it to the UserData property of the window
    Info = get(FigHandle,'UserData');
    Info.ListOfMeasurements = ListOfMeasurements;
    Info.TextHandles = TextHandles;
    set(FigHandle,'UserData',Info);

    dpi = str2double(DPIToSave);
    if isnan(dpi),
        error(['Image processing was canceled in the ', ModuleName, ' module because the value entered for the DPI (',DPIToSave,')was not numeric']);
    end
    switch SavedImageContents 
        case 'image',   opt = 'img';
        case 'axes',    opt = 'imgAx';
        case 'figure',  opt = 'all';
    end
    drawnow;    % Need this here for the CPimcapture to work correctly
    handles.Pipeline.(DataImage)= CPimcapture(FigHandle,opt,dpi);

    % if the user didn't want to display the window in the first place,
    % destroy this one so we don't have an invisible window hanging
    % around
    if userdoesntwantwindow && ishandle(FigHandle),
        CPclosefigure(handles,CurrentModule);
    end
end

