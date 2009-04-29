function handles = TrackObjects(handles,varargin)

% Help for the Track Objects module:
% Category: Object Processing
%
% SHORT DESCRIPTION:
% Allows tracking objects throughout sequential frames of a movie, so that
% each object maintains a unique identity in the output measurements.
% *************************************************************************
% This module must be run after the object to be tracked has been 
% identified using an Identification module (e.g., IdentifyPrimAutomatic).
%
% Settings:
%
% Tracking method:
% Choose between the methods based on which is most consistent from frame
% to frame of your movie. For each, the maximum search distance that a 
% tracked object will looked for is specified with the Neighborhood setting
% below:
%
%   Overlap - Compare the amount of overlaps between identified objects in 
%   the previous frame with those in the current frame. The object with the
%   greatest amount of overlap will be assigned the same label. Recommended
%   for movies with high frame rates as compared to object motion.
%       
%   Distance - Compare the distance between the centroid of each identified
%   object in the previous frame with that of the current frame. The 
%   closest objects to each other will be assigned the same label.
%   Distances are measured from the perimeter of each object. Recommended
%   for movies with lower frame rates as compared to object motion, but
%   the objects are clearly separable.
%
%   Measurement - Compare the specified measurement of each object in the 
%   current frame with that of objects in the previous frame. The object 
%   with the closest measurement will be selected as a match and will be 
%   assigned the same label. This selection requires that you run the 
%   specified Measurement module previous to this module in the pipeline so
%   that the measurement values can be used to track the objects. 
%
% Catagory/Feature Name or Number/Image/Scale:
% Specifies which type of measurement (catagory) and which feature from the
% Measure module will be used for tracking. Select the feature name from 
% the popup box or see each Measure module's help for the numbered list of
% the features measured by that module. Additional details such as the 
% image that the measurements originated from and the scale used as
% specified below if neccesary.
%
% Neighborhood:
% This indicates the region (in pixels) within which objects in the
% next frame are to be compared. To determine pixel distances, you can look
% at the markings on the side of each image (shown in pixel units) or
% using the ShowOrHidePixelData Image tool (under the Image Tools menu of 
% any CellProfiler figure window)
%
% How do you want to display the tracked objects?
% The objects can be displayed as a color image, in which an object with a 
% unique label is assigned a unique color. This same color is maintained
% throughout the object's lifetime. If desired, a number identifiying the 
% object is superimposed on the object.
%
% What number do you want displayed?
% The displayed number is the unique label assigned to the object or the
% progeny identifier.
%
% Do you want to calculate statistics:
% Select whether you want statistics on the tracked objects to be added to
% the measurements for that object. The current statistics are collected:
%
% Features measured:       Feature Number:
% TrajectoryX           |       1
% TrajectoryY           |       2
% DistanceTraveled      |       3
% IntegratedDistance    |       4
% Linearity             |       5
%
% In addition to these, the following features are also recorded: Label, 
% Lifetime.
%
% Desscription of each feature:
%   Label: Each tracked object is assigned a unique identifier (label). 
%   Results of splits or merges are seen as new objects and assigned a new
%   label.
%
%   Trajectory: The direction of motion (in x and y coordinates) of the 
%   object from the previous frame to the curent frame.
%
%   Distance traveled: The distance traveled by the object from the 
%   previous frame to the curent frame (calculated as the magnititude of 
%   the distance traveled vector).
%
%   Lifetime: The duration (in frames) of the object. The lifetime begins 
%   at the frame when an object appears and is ouput as a measurement when
%   the object disappears. At the final frame of the image set/movie, the 
%   lifetimes of all remaining objects are ouput.
%
%   Integrated distance: The total distance traveled by the object during
%   the lifetime of the object
%   
%   Linearity: A measure of how linear the object trajectity is during the
%   object lifetime. Calculated as (distance from initial to final 
%   location)/(integrated object distance). Value is in range of [0,1].
%
% What do you want to call the image with the tracked objects?
% Specify a name to give the image showing the tracked objects. This image
% can be saved with a SaveImages module placed after this module.
%
% Additional notes:
%
% In the figure window, a popupmenu allows you to display the objects as a
% solid color or as an outline with the current objects in color and the
% previous objects in white.
%
% Since the movie is processed sequentially by frame, it cannot be broken
% up into batches for execution on a distributed cluster.
%
% If running on a cluster and saving the colored image with text labels,
% the labels will not show up in the final result. This is a limitation of
% using MATLAB's hardcopy command.
%
% See also: Any of the Measure* modules, IdentifyPrimAutomatic

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003--2008.
%
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% $Revision$

% MBray 2009_03_25: Comments on variables for pyCP upgrade
%
% Recommended variable order (setting, followed by current variable in MATLAB CP)
% (1) What tracking method would you like to use? (TrackingMethod)
% (2) What did you call the objects you want to track? (ObjectName)
% (3a) What category of measurement do you want to use (MeasurementCategory)
% (3b) What feature do you want to use? (MeasurementFeature)
% (3c) (If the answer to (3b) involves a scale) What scale was used to 
%      calculate the feature? (SizeScale) 
%      (If the answer to (3b) involves an image) What image was used to
%      calculate the feature? (ImageName)
% (4) Within what pixel distance will objects be considered to find a
%   potential match? (PixelRadius)
% (5) How do you want to display the tracked objects? (DisplayType)
% (6) What do you want to call the resulting image with tracked,
%   color-coded objects? Type "Do not use" to ignore. (DataImage)
%
% (i) Option (3) should appear only if the user selects "Measuremement" in (2)
% (ii) The Measurement category/feature/image/scale settings in (3a,b,c) should only be shown if
% the measurement hierarchy requires it.
% (iii) The setting to collect statistics (CollectStatistics) should be
% removed as it should be collected by default.

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = Choose a tracking method
%choiceVAR01 = Overlap
%choiceVAR01 = Distance
%choiceVAR01 = Measurements
%inputtypeVAR01 = popupmenu
TrackingMethod = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = What did you call the objects you want to track?
%infotypeVAR02 = objectgroup
%inputtypeVAR02 = popupmenu
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = (Tracking by Measurements only) What category of measurement you want to use?
%inputtypeVAR03 = popupmenu category
MeasurementCategory = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = (Tracking by Measurements only) What feature you want to use? (Enter the feature number or name - see help for details)
%defaultVAR04 = Do not use
%inputtypeVAR04 = popupmenu measurement
MeasurementFeature = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = (Tracking by Measurements only) For INTENSITY, AREAOCCUPIED or TEXTURE features, which image's measurements do you want to use?
%infotypeVAR05 = imagegroup
%inputtypeVAR05 = popupmenu
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = (Tracking by Measurements only) For TEXTURE, RADIAL DISTRIBUTION, OR NEIGHBORS features, what previously measured size scale (TEXTURE OR NEIGHBORS) or previously used number of bins (RADIALDISTRIBUTION) do you want to use?
%defaultVAR06 = 1
%inputtypeVAR06 = popupmenu scale
SizeScale = str2double(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = Choose the neighborhood (in pixels) within which objects will be evaluated to find a potential match.
%defaultVAR07 = 50
PixelRadius = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,7}));

%textVAR08 = How do you want to display the tracked objects?
%choiceVAR08 = Color and Number
%choiceVAR08 = Color Only
%inputtypeVAR08 = popupmenu
DisplayType = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%textVAR09 = If you chose an option with "Number" above, select the number you want displayed (ProgenyID is not yet working)
%choiceVAR09 = Object ID
%choiceVAR09 = Progeny ID
%inputtypeVAR09 = popupmenu
LabelMethod = char(handles.Settings.VariableValues{CurrentModuleNum,9});

%textVAR10 = Do you want to calculate statistics?
%choiceVAR10 = No
%choiceVAR10 = Yes
%inputtypeVAR10 = popupmenu
CollectStatistics = char(handles.Settings.VariableValues{CurrentModuleNum,10});

%textVAR11 = What do you want to call the resulting image with tracked, color-coded objects? Type "Do not use" to ignore.
%defaultVAR11 = Do not use
%infotypeVAR11 = imagegroup indep
DataImage = char(handles.Settings.VariableValues{CurrentModuleNum,11});

%%%%%%%%%%%%%%%%%
%%% FEATURES  %%%
%%%%%%%%%%%%%%%%%

if nargin > 1 
    switch varargin{1}
%feature:categories
        case 'categories'
            if nargin == 1 || ismember(varargin{2},{ObjectName})
                result = { 'TrackObjects' };
            else
                result = {};
            end

%feature:measurements
        case 'measurements'
            if ismember(varargin{2},{ObjectName}) && strcmp(varargin{3},'TrackObjects')
                result = {...
                    'TrajectoryX','TrajectoryY','DistanceTraveled','IntegratedDistance','Linearity' };
            else
                result = {};
            end
        otherwise
            error(['Unhandled category: ',varargin{1}]);
    end
    handles = result;
    return;
end

%%%VariableRevisionNumber = 3

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

% Initialize a few variables
SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;
NumberOfImageSets = handles.Current.NumberOfImageSets;
CollectStatistics = strncmpi(CollectStatistics,'y',1);

% Start the analysis
if SetBeingAnalyzed == handles.Current.StartingImageSet
    % Initialize data structures
    
    % An additional structure is added to handles.Pipeline in order to keep
    % track of frame-to-frame changes
    % (1) Locations
    TrackObjInfo.Current.Locations{SetBeingAnalyzed} = ...
    cat(2,  handles.Measurements.(ObjectName).Location_Center_X{SetBeingAnalyzed},...
            handles.Measurements.(ObjectName).Location_Center_Y{SetBeingAnalyzed});
        
    CurrentLocations = TrackObjInfo.Current.Locations{SetBeingAnalyzed};
    PreviousLocations = NaN(size(CurrentLocations));
    
    % (2) Segmented, labeled image
    TrackObjInfo.Current.SegmentedImage = handles.Pipeline.(['Segmented' ObjectName]);
    
    % (3) Labels
    InitialNumObjs = length(TrackObjInfo.Current.Locations{SetBeingAnalyzed});
    CurrentLabels = (1:InitialNumObjs)';
    PreviousLabels = CurrentLabels;
    for i = 1:InitialNumObjs,
        CurrHeaders{i} = '';
    end
    % (4) Colormap
    TrackObjInfo.Colormap = [];
    % (5) Lookup table for object to color
    TrackObjInfo.ObjToColorMapping = [];
    
    if CollectStatistics
        [TrackObjInfo.Current.AgeOfObjects,TrackObjInfo.Current.SumDistance] = deal(zeros(size(CurrentLabels)));
        TrackObjInfo.Current.InitialObjectLocation = CurrentLocations;
        AgeOfObjects = TrackObjInfo.Current.AgeOfObjects;
        InitialObjectLocation = TrackObjInfo.Current.InitialObjectLocation;
        SumDistance = TrackObjInfo.Current.SumDistance;
        [CentroidTrajectory,DistanceTraveled,SumDistance,AgeOfObjects,InitialObjectLocation] = ComputeTrackingStatistics(CurrentLocations,PreviousLocations,CurrentLabels,PreviousLabels,SumDistance,AgeOfObjects,InitialObjectLocation);
        TrackObjInfo.Current.AgeOfObjects = AgeOfObjects;
        TrackObjInfo.Current.SumDistance = SumDistance;
        TrackObjInfo.Current.InitialObjectLocation = InitialObjectLocation;
    end
else
    % Extracts data from the handles structure
    TrackObjInfo = handles.Pipeline.TrackObjects.(ObjectName);
    
    % Create the new 'previous' state from the former 'current' state
    TrackObjInfo.Previous = TrackObjInfo.Current;
    
    % Get the needed variables from the 'previous' state
    PreviousLocations = TrackObjInfo.Previous.Locations{SetBeingAnalyzed-1};
    PreviousLabels = TrackObjInfo.Previous.Labels;
    PreviousSegmentedImage = TrackObjInfo.Previous.SegmentedImage;
    PrevHeaders = TrackObjInfo.Previous.Headers;

    % Get the needed variables from the 'current' state
    TrackObjInfo.Current.Locations{SetBeingAnalyzed} = ...
    cat(2,  handles.Measurements.(ObjectName).Location_Center_X{SetBeingAnalyzed},...
            handles.Measurements.(ObjectName).Location_Center_Y{SetBeingAnalyzed});
    TrackObjInfo.Current.SegmentedImage = handles.Pipeline.(['Segmented' ObjectName]);
    CurrentLocations = TrackObjInfo.Current.Locations{SetBeingAnalyzed};
    CurrentSegmentedImage = TrackObjInfo.Current.SegmentedImage;

    switch lower(TrackingMethod)
        case 'distance'
            % Create a distance map image, threshold it by search radius 
            % and relabel appropriately
            [CurrentObjNhood,CurrentObjLabels] = bwdist(CurrentSegmentedImage); 
            CurrentObjNhood = (CurrentObjNhood < PixelRadius).*CurrentSegmentedImage(CurrentObjLabels);
            
            [PreviousObjNhood,previous_obj_labels] = bwdist(PreviousSegmentedImage); 
            PreviousObjNhood = (PreviousObjNhood < PixelRadius).*PreviousSegmentedImage(previous_obj_labels);
            
            % Compute overlap of distance-thresholded objects
            MeasuredValues = ones(size(CurrentObjNhood));
            [CurrentLabels, CurrHeaders] = EvaluateObjectOverlap(CurrentObjNhood,PreviousObjNhood,PreviousLabels,PrevHeaders,MeasuredValues);
            
        case 'overlap'  % Compute object area overlap
            MeasuredValues = ones(size(CurrentSegmentedImage));
            [CurrentLabels, CurrHeaders] = EvaluateObjectOverlap(CurrentSegmentedImage,PreviousSegmentedImage,PreviousLabels,PrevHeaders,MeasuredValues);
            
        otherwise
            % Get the specified featurename
            try
                FeatureName = CPgetfeaturenamesfromnumbers(handles, ObjectName, MeasurementCategory, MeasurementFeature, ImageName, SizeScale);
            catch
                error(['Image processing was canceled in the ', ModuleName, ' module because an error ocurred when retrieving the ' MeasurementFeature ' set of data. Either the category of measurement you chose, ', MeasurementCategory,', was not available for ', ObjectName,', or the feature number, ', num2str(MeasurementFeature), ', exceeded the amount of measurements.']);
            end
            
            % The idea here is to take advantage to MATLAB's sparse/full
            % trick used in EvaluateObjectOverlap by modifying the input
            % label matrices appropriately.
            % The big problem with steps (1-3) is that bwdist limits the distance
            % according to the obj neighbors; I want the distance threshold
            % to be neighbor-independent
            
            % (1) Expand the current objects by the threshold pixel radius
            [CurrentObjNhood,CurrentObjLabels] = bwdist(CurrentSegmentedImage); 
            CurrentObjNhood = (CurrentObjNhood < PixelRadius).*CurrentSegmentedImage(CurrentObjLabels);
            
            % (2) Find those previous objects which fall within this range
            PreviousObjNhood = (CurrentObjNhood > 0).*PreviousSegmentedImage;
            
            % (3) Shrink them to points so the accumulation in sparse will
            % evaluate only a single number per previous object, the value
            % of which is assigned in the next step
            PreviousObjNhood = bwmorph(PreviousObjNhood,'shrink',inf).*PreviousSegmentedImage;
            
            % (4) Produce a labeled image for the previous objects in which the
            % labels are the specified measurements. The nice thing here is
            % that I can extend this to whatever measurements I want
            PreviousStatistics = handles.Measurements.(ObjectName).(FeatureName){SetBeingAnalyzed-1};
            PreviousStatisticsImage = (PreviousObjNhood > 0).*LabelByColor(PreviousSegmentedImage, PreviousStatistics);
            
            % (4) Ditto for the current objects 
            CurrentStatistics = handles.Measurements.(ObjectName).(FeatureName){SetBeingAnalyzed};
            CurrentStatisticsImage = LabelByColor(CurrentObjNhood, CurrentStatistics);
            
            % (5) The values that are input into EvaluateObjectOverlap are
            % the normalized measured per-object values, ie, CurrentStatistics/PreviousStatistics
            warning('off','MATLAB:divideByZero');
            MeasuredValues = PreviousStatisticsImage./CurrentStatisticsImage;
            MeasuredValues(isnan(MeasuredValues)) = 0;
            % Since the EvaluateObjectOverlap is performed by looking at
            % the max, if the metric is > 1, take the reciprocal so the
            % result is on the range of [0,1]
            MeasuredValues(MeasuredValues > 1) = CurrentStatisticsImage(MeasuredValues > 1)./PreviousStatisticsImage(MeasuredValues > 1);
            warning('on','MATLAB:divideByZero');
            
            [CurrentLabels, CurrHeaders] = EvaluateObjectOverlap(CurrentObjNhood,PreviousObjNhood,PreviousLabels,PrevHeaders,MeasuredValues);
    end

    % Compute measurements
    % At this point, the following measurements are calculated: CentroidTrajectory
    % in <x,y>, distance traveled, AgeOfObjects
    % Other measurements that were previously included in prior versions of
    % TrackObjects were the following: CellsEnteredCount, CellsExitedCount,
    % ObjectSizeChange. These were all computed at the end of the analysis,
    % not on a per-cycle basis
    % TODO: Determine whether these measurements are useful and put them
    % back if they are
    if CollectStatistics
        AgeOfObjects = TrackObjInfo.Current.AgeOfObjects;
        InitialObjectLocation = TrackObjInfo.Current.InitialObjectLocation;
        SumDistance = TrackObjInfo.Current.SumDistance;
        [CentroidTrajectory,DistanceTraveled,SumDistance,AgeOfObjects,InitialObjectLocation] = ComputeTrackingStatistics(CurrentLocations,PreviousLocations,CurrentLabels,PreviousLabels,SumDistance,AgeOfObjects,InitialObjectLocation);
        TrackObjInfo.Current.AgeOfObjects = AgeOfObjects;
        TrackObjInfo.Current.SumDistance = SumDistance;
        TrackObjInfo.Current.InitialObjectLocation = InitialObjectLocation;
    end
end

% Create colored image
CurrentIndexedLabelImage = LabelByColor(TrackObjInfo.Current.SegmentedImage, CurrentLabels);
[LabelMatrixColormap, ObjToColorMapping] = UpdateTrackObjectsDisplayImage(CurrentIndexedLabelImage, ...
                                                                          CurrentLabels, PreviousLabels, ...
                                                                          TrackObjInfo.Colormap, ...
                                                                          TrackObjInfo.ObjToColorMapping, ...
                                                                          handles.Preferences.LabelColorMap);                                                                
TrackObjInfo.Colormap = LabelMatrixColormap;
TrackObjInfo.ObjToColorMapping = ObjToColorMapping;

if isfield(TrackObjInfo,'Previous'),
    PreviousIndexedLabelImage = LabelByColor(TrackObjInfo.Previous.SegmentedImage, PreviousLabels);
else
    PreviousIndexedLabelImage = zeros(size(CurrentIndexedLabelImage));
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

LookUpTable = [0 ObjToColorMapping];
lengthCmap = size(LabelMatrixColormap,1);
CurrentColoredLabelImage = ind2rgb(gray2ind(LookUpTable(CurrentIndexedLabelImage+1)/lengthCmap,lengthCmap),LabelMatrixColormap);

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber) || (~any(findobj == ThisModuleFigureNumber) && ~strcmp(DataImage,'Do not use') )
    % Create colored images
    % (1) Colored label image of current objects
    CurrentColoredPerimImage = ind2rgb(gray2ind(LookUpTable(CPlabelperim(CurrentIndexedLabelImage)+1)/lengthCmap,lengthCmap),LabelMatrixColormap);
    % (2) Colored perimeter image of current and previous objects
    ColoredPerimeterImage = cast(repmat(255*double(CPlabelperim(PreviousIndexedLabelImage) > 0),[1 1 3]),class(CurrentColoredPerimImage));
    idx = find(CurrentColoredPerimImage(:));
    ColoredPerimeterImage(idx) = CurrentColoredPerimImage(idx);

    if any(findobj == ThisModuleFigureNumber)
        % Activates the appropriate figure window
        CPfigure(handles,'Image',ThisModuleFigureNumber);
        [ignore,hAx] = CPimagesc(CurrentColoredLabelImage,handles,ThisModuleFigureNumber);
        title(hAx,['Tracked ',ObjectName]);
    elseif ~any(findobj == ThisModuleFigureNumber) && ~strcmp(DataImage,'Do not use')
        % If no figure windows exists, then the user doesn't want any windows 
        % to be open. We need a window for the capture to work, so create one
        % but make it invisible
        ThisModuleFigureNumber = CPfigurehandle(handles);
        CPfigure(handles,'Image',ThisModuleFigureNumber);
        set(ThisModuleFigureNumber,'visible','off');
        [ignore,hAx] = CPimagesc(CurrentColoredLabelImage,handles,ThisModuleFigureNumber);
    end

    % Construct uicontrol which holds images and figure titles 
    if isempty(findobj(ThisModuleFigureNumber,'tag','PopupImage')),
        ud(1).img = CurrentColoredLabelImage;
        ud(2).img = ColoredPerimeterImage;
        ud(1).title = ['Tracked ',ObjectName,' cycle # ',num2str(SetBeingAnalyzed)];
        ud(2).title = ['Current (colored) and previous (white)', ObjectName,', cycle # ',num2str(SetBeingAnalyzed)];
        uicontrol(ThisModuleFigureNumber, 'Style', 'popup',...
                    'String', ['Tracked ',ObjectName,'|Current (colored) and previous (white) ', ObjectName],...
                    'UserData',ud,...
                    'units','normalized',...
                    'position',[.025 .95 .25 .04],...
                    'backgroundcolor',[.7 .7 .9],...
                    'tag','PopupImage',...
                    'Callback', @CP_ImagePopupmenu_Callback);
    else
        ud = get(findobj(ThisModuleFigureNumber,'tag','PopupImage'),'userdata');
        ud(1).img = CurrentColoredLabelImage;
        ud(2).img = ColoredPerimeterImage;
        set(findobj(ThisModuleFigureNumber,'tag','PopupImage'),'userdata',ud); 
    end
    if ~isempty(strfind(DisplayType, 'Number'))
        switch LabelMethod
            case 'Object ID', text_string = cellstr(num2str(CurrentLabels(:)));
            case 'Progeny ID',text_string = strcat(CurrHeaders', cellstr(num2str((CurrentLabels)')));
        end
        % There is a Mathworks page which describes how to make text become part of the image:
        % http://www.mathworks.com/support/solutions/data/1-1BALJ.html?solution=1-1BALJ
        % However, the resultant text resolution is poor
        % TODO: If a better way is found, insert it here
        text(CurrentLocations(:,1) , CurrentLocations(:,2) , text_string,...
            'HorizontalAlignment','center', 'color', [.6 .6 .6],'fontsize',10,...%handles.Preferences.FontSize,...
            'fontweight','bold','Parent',hAx);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

TrackObjInfo.Current.Labels = CurrentLabels;
TrackObjInfo.Current.Headers = CurrHeaders;

% Save the image of the tracked objects (if desired)
if ~strcmp(DataImage,'Do not use')   
    % Do the screen capture at high-res and resize to original image size.
    % This will get the image plus any text
    warning('off','MATLAB:Text:DrawStringIntoBitmap');
    CapturedImage = CPimcapture(ThisModuleFigureNumber,'img',150);
    warning('on','MATLAB:Text:DrawStringIntoBitmap');
    OrigImSize = size(CurrentColoredLabelImage(:,:,1));
    ResizedCapturedImage = cat(3,   imresize(double(CapturedImage(:,:,1)),OrigImSize),...
                                    imresize(double(CapturedImage(:,:,2)),OrigImSize),...
                                    imresize(double(CapturedImage(:,:,3)),OrigImSize))/255; 
    
    % Correct for over/under-shoot from interpolation
    ResizedCapturedImage(ResizedCapturedImage > 1) = 1; 
    ResizedCapturedImage(ResizedCapturedImage < 0) = 0;
    
    % Save to handles
    handles.Pipeline.(DataImage) = ResizedCapturedImage;
    
    if ~any(findobj == ThisModuleFigureNumber)
        % Destroy the invisible figure created earlier
        close(ThisModuleFigureNumber);
    end
end

% Saves the measurements of each tracked object
if CollectStatistics
    TrackingMeasurementPrefix = 'TrackObjects';
    handles = CPaddmeasurements(handles, ObjectName, CPjoinstrings(TrackingMeasurementPrefix,'ObjectID',num2str(PixelRadius)), ...
                    CurrentLabels(:));
    handles = CPaddmeasurements(handles, ObjectName, CPjoinstrings(TrackingMeasurementPrefix,'ProgenyID',num2str(PixelRadius)), ...
                    str2double(strcat(CurrHeaders', cellstr(num2str((CurrentLabels)')))));
    handles = CPaddmeasurements(handles, ObjectName, CPjoinstrings(TrackingMeasurementPrefix,'TrajectoryX',num2str(PixelRadius)), ...
                    CentroidTrajectory(:,1));
    handles = CPaddmeasurements(handles, ObjectName, CPjoinstrings(TrackingMeasurementPrefix,'TrajectoryY',num2str(PixelRadius)), ...
                    CentroidTrajectory(:,2));
    handles = CPaddmeasurements(handles, ObjectName, CPjoinstrings(TrackingMeasurementPrefix,'DistanceTraveled',num2str(PixelRadius)), ...
                    DistanceTraveled(:));
    
    % Record the object lifetime, integrated distance and linearity once it disappears...
    if SetBeingAnalyzed ~= NumberOfImageSets,
        [Lifetime,Linearity,IntegratedDistance] = deal(NaN(size(PreviousLabels)));
        [AbsentObjectsLabel,idx] = setdiff(PreviousLabels,CurrentLabels);
        Lifetime(idx) = AgeOfObjects(AbsentObjectsLabel);
        IntegratedDistance(idx) = SumDistance(AbsentObjectsLabel);
        % Linearity: In range of [0,1]. Defined as abs[(x,y)_final -
        % (x,y)_initial]/(IntegratedDistance).
        warning('off','MATLAB:divideByZero');
        Linearity(idx) = sqrt(sum((InitialObjectLocation(AbsentObjectsLabel,:) - PreviousLocations(idx,:)).^2,2))./SumDistance(AbsentObjectsLabel);
        warning('on','MATLAB:divideByZero');
    else %... or we reach the end of the analysis
        Lifetime = AgeOfObjects(CurrentLabels);
        IntegratedDistance = SumDistance(CurrentLabels);
        warning('off','MATLAB:divideByZero');
        Linearity = sqrt(sum((InitialObjectLocation(CurrentLabels,:) - CurrentLocations).^2,2))./SumDistance(CurrentLabels);
        warning('on','MATLAB:divideByZero');
    end
        
    IntegratedDistanceMeasurementName = CPjoinstrings(TrackingMeasurementPrefix,'IntegratedDistance',num2str(PixelRadius));
    handles = CPaddmeasurements(handles, ObjectName, IntegratedDistanceMeasurementName, ...
                    IntegratedDistance(:));
    LinearityMeasurementName = CPjoinstrings(TrackingMeasurementPrefix,'Linearity',num2str(PixelRadius));
    handles = CPaddmeasurements(handles, ObjectName, LinearityMeasurementName, ...
                    Linearity(:));
    LifetimeMeasurementName = CPjoinstrings(TrackingMeasurementPrefix,'Lifetime',num2str(PixelRadius));
    handles = CPaddmeasurements(handles, ObjectName, LifetimeMeasurementName, ...
                    Lifetime(:));
    % This is a special case: The lifetime for an object is known only
    % after the cycle where it disappeared, so I need to transfer the
    % lifetime-related measurements back one cycle, unless we're at the end
    % NOTE: If the measurement names are changed here, they need to be
    % altered in Relate since they need to be excluded from per-parent
    % measurements
    if SetBeingAnalyzed > 1 && SetBeingAnalyzed ~= NumberOfImageSets, 
        handles.Measurements.(ObjectName).(IntegratedDistanceMeasurementName){SetBeingAnalyzed-1} = ...
            handles.Measurements.(ObjectName).(IntegratedDistanceMeasurementName){SetBeingAnalyzed}; 
        handles.Measurements.(ObjectName).(LifetimeMeasurementName){SetBeingAnalyzed-1} = ...
            handles.Measurements.(ObjectName).(LifetimeMeasurementName){SetBeingAnalyzed}; 
        handles.Measurements.(ObjectName).(LinearityMeasurementName){SetBeingAnalyzed-1} = ...
            handles.Measurements.(ObjectName).(LinearityMeasurementName){SetBeingAnalyzed}; 
    end
end

% Save the structure back to handles.Pipeline 
handles.Pipeline.TrackObjects.(ObjectName) = TrackObjInfo;

%%%%%%%%%%%%%%%%%%%%%%
%%%% SUBFUNCTIONS %%%%
%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTION - EvaluateObjectOverlap
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [CurrentLabels, CurrentHeaders] = EvaluateObjectOverlap(CurrentLabelMatrix, PreviousLabelMatrix, PreviousLabels, PreviousHeaders, ChosenMetric)

% Much of the following code is adapted from CPrelateobjects

% We want to choose a previous objects's progeny based on the most overlapping
% current object.  We first find all pixels that are in both a previous and a
% current object, as we wish to ignore pixels that are background in either
% labelmatrix
ForegroundMask = (CurrentLabelMatrix > 0) & (PreviousLabelMatrix > 0);
NumberOfCurrentObj = length(unique(CurrentLabelMatrix(CurrentLabelMatrix > 0)));
NumberOfPreviousObj = length(unique(PreviousLabelMatrix(PreviousLabelMatrix > 0)));

% Use the Matlab full(sparse()) trick to create a 2D histogram of
% object overlap counts
CurrentPreviousLabelHistogram = full(sparse(CurrentLabelMatrix(ForegroundMask), PreviousLabelMatrix(ForegroundMask), ChosenMetric(ForegroundMask), NumberOfCurrentObj, NumberOfPreviousObj));

% Make sure there are overlapping current and previous objects
if any(CurrentPreviousLabelHistogram(:)),
    % For each current obj, we must choose a single previous obj parent. We will choose
    % this by maximum overlap, which in this case is maximum value in
    % the child's column in the histogram.  sort() will give us the
    % necessary parent (row) index as its second return argument.
    [OverlapCounts, CurrentObjIndexes] = sort(CurrentPreviousLabelHistogram,1);
    
    % Get the parent list.
    CurrentObjList = CurrentObjIndexes(end, :);
    
    % Nandle the case of a zero overlap -> no current obj
    CurrentObjList(OverlapCounts(end, :) == 0) = 0;
    
    % Transpose to a column vector
    CurrentObjList = CurrentObjList';
else
    % No overlapping objects
    CurrentObjList = zeros(NumberOfPreviousObj, 1);
end

% Update the label and header vectors: The following accounts for objects
% that have disappeared or newly appeared

% Disspeared: Obj in CurrentObjList set to 0, so drop label from list
CurrentLabels = zeros(1,NumberOfCurrentObj);
CurrentLabels(CurrentObjList(CurrentObjList > 0)) = PreviousLabels(CurrentObjList > 0); 

% Newly appeared: Missing index in CurrentObjList, so add new label to list
idx = setdiff(1:NumberOfCurrentObj,CurrentObjList);
CurrentLabels(idx) = max(PreviousLabels) + (1:length(idx));

CurrentHeaders(CurrentObjList(CurrentObjList > 0)) = PreviousHeaders(CurrentObjList > 0);
CurrentHeaders(idx) = {''};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTION - LabelByColor
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ColoredImage = LabelByColor(LabelMatrix, CurrentLabel)
% Relabel the label matrix so that the labels in the matrix are consistent
% with the text labels

LookupTable = [0; CurrentLabel(:)];
ColoredImage = LookupTable(LabelMatrix+1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTION - UpdateTrackObjectsDisplayImage
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [LabelMatrixColormap, ObjToColorMapping] = UpdateTrackObjectsDisplayImage(LabelMatrix, CurrentLabels, PreviousLabels, LabelMatrixColormap, ObjToColorMapping, DefaultLabelColorMap)

NumberOfColors = 256;

if isempty(LabelMatrixColormap),
    % If just starting, create a 256-element colormap
    colormap_fxnhdl = str2func(DefaultLabelColorMap);
    NumOfRegions = double(max(LabelMatrix(:)));
    cmap = [0 0 0; colormap_fxnhdl(NumberOfColors-1)];
    rand('twister', rand('twister'));
    index = rand(1,NumOfRegions)*NumberOfColors;
    
    % Save the colormap and indices into the handles
    LabelMatrixColormap = cmap;
    ObjToColorMapping = index;
else
    % See if new labels have appeared and assign them a random color
    NewLabels = setdiff(CurrentLabels,PreviousLabels);
    ObjToColorMapping(NewLabels) = rand(1,length(NewLabels))*NumberOfColors;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTION - ComputeTrackingStatistics
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [CentroidTrajectory,DistanceTraveled,SumDistance,AgeOfObjects,InitialObjectLocation] = ComputeTrackingStatistics(CurrentLocations,PreviousLocations,CurrentLabels,PreviousLabels,SumDistance,AgeOfObjects,InitialObjectLocation)
   
CentroidTrajectory = zeros(size(CurrentLocations));
[OldLabels, idx_previous, idx_current] = intersect(PreviousLabels,CurrentLabels);
CentroidTrajectory(idx_current,:) = CurrentLocations(idx_current,:) - PreviousLocations(idx_previous,:);
DistanceTraveled = sqrt(sum(CentroidTrajectory.^2,2));
DistanceTraveled(isnan(DistanceTraveled)) = 0;

AgeOfObjects(OldLabels) = AgeOfObjects(OldLabels) + 1;
[NewLabels,idx_new] = setdiff(CurrentLabels,PreviousLabels);
AgeOfObjects(NewLabels) = 1;

SumDistance(OldLabels) = SumDistance(OldLabels) + DistanceTraveled(idx_current);
SumDistance(NewLabels) = 0;

InitialObjectLocation(NewLabels,:) = CurrentLocations(idx_new,:);