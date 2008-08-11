function handles = HoughTransformCircular(handles)

% Help for the HoughTransformCircular module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Performs a Hough Transform to find circular regions in a grayscale image.
% *************************************************************************
%
% Settings:
%
% Adapted from code on the Matlab File Exchange, contributed by Tao Peng at
% http://www.mathworks.com/matlabcentral/fileexchange/loadFile.do?objectId=9168&objectType=file
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
% $Revision: 5025 $

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the image to be transformed?
%infotypeVAR01 = imagegroup
OrigImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the objects identified by this module?
%defaultVAR02 = Centers
%infotypeVAR02 = objectgroup indep
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = What are the minimum and maximum radii of the circles in pixel units (Min,Max):
%defaultVAR03 = 10,40
SizeRange = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = What is the gradient threshold? Must be non-negative, and pixels with gradient magnitudes smaller than this value are NOT considered in the computation.
%defaultVAR04 = .05
GrdThres = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = What is the radius of the filter for local maxima in the accumulation array? To detect circles whose shapes are less perfect, the radius of the filter needs to be set larger. (Minimum=3)
%defaultVAR05 = 8
FilterRadius = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = What is the tolerance for multiple concentric radii?  It ranges from 0.1 to 1, where 0.1 corresponds to the largest tolerance, meaning more radii values will be detected, and 1 corresponds to the smallest tolerance, in which case only the "principal" radius will be picked up.
%defaultVAR06 = 0.5
MultConcentricTolerance = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%%%VariableRevisionNumber = 1

%% TODO - allow user to change AccumFilter characteristics

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a
%%% variable.
try
    OrigImage = CPretrieveimage(handles,OrigImageName,ModuleName,'MustBeGray','CheckScale');
catch
    ErrorMessage = lasterr;
    error(['Image processing was canceled in the ' ModuleName ' module because: ' ErrorMessage(33:end)]);
end

%%% Checks that the Min and Max diameter parameters have valid values
%%% (This check copied from IDPrimAuto)
index = strfind(SizeRange,',');
if isempty(index),
    error(['Image processing was canceled in the ', ModuleName, ' module because the Min and Max size entry is invalid.'])
end
MinDiameter = SizeRange(1:index-1);
MaxDiameter = SizeRange(index+1:end);

MinDiameter = str2double(MinDiameter);
if isnan(MinDiameter) | MinDiameter < 0 %#ok Ignore MLint
    error(['Image processing was canceled in the ', ModuleName, ' module because the Min diameter entry is invalid.'])
end
if strcmpi(MaxDiameter,'Inf')
    MaxDiameter = Inf;
else
    MaxDiameter = str2double(MaxDiameter);
    if isnan(MaxDiameter) | MaxDiameter < 0 %#ok Ignore MLint
        error(['Image processing was canceled in the ', ModuleName, ' module because the Max diameter entry is invalid.'])
    end
end
if MinDiameter > MaxDiameter
    error(['Image processing was canceled in the ', ModuleName, ' module because the Min Diameter is larger than the Max Diameter.'])
end

%% TODO: perform checks on these values
GrdThres = str2double(GrdThres);
FilterRadius = str2double(FilterRadius);
MultConcentricTolerance = str2double(MultConcentricTolerance);

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

try
    [AccumImage, circen, cirrad] = CircularHough_Grd(OrigImage, [MinDiameter,MaxDiameter], ...
        GrdThres, FilterRadius, MultConcentricTolerance);
catch
    ErrorMessage = lasterr;
    error(['Image processing was canceled in the ' ModuleName ' module because: ' ErrorMessage(26:end)]);
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    %%% Activates the appropriate figure window.
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure(OrigImage,'OneByOne',ThisModuleFigureNumber)
    end
    CPimagesc(OrigImage,handles);
    title(['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    
    hold on
    plot(circen(:,1), circen(:,2), 'r+');
    for k = 1:size(circen, 1),
        DrawCircle(circen(k,1), circen(k,2), cirrad(k), 32, 'b-');
    end
% 
%     %%% A subplot of the figure window is set to display the Thresholded
%     %%% image.
%     subplot(2,1,2);
%     CPimagesc(AccumImage,handles);
%     title('Accumulation Array');

    %% Create circular objects
    
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%% TODO
%%% Saves the processed image to the handles structure so it can be used by
%%% subsequent modules.
% handles.Pipeline.(SmoothedImageName) = SmoothedImage;

% handles = CPaddmeasurements(handles,Object,Measure,Feature,Data) %% USAGE
% handles.Measurements.(Object).(FeaturesField) = {Feature};
% handles.Measurements.(Object).(Measure){handles.Current.SetBeingAnalyzed} = Data;
%
% Examples
% handles = CPaddmeasurements(handles,ObjectName{1},'SingleRatio',RatioName,FinalMeasurements); 
% handles = CPaddmeasurements(handles,'Image','OrigThreshold',['Edged_',ImageName],ThresholdUsed(1));
% handles = CPaddmeasurements(handles,'Image',ObjectName,'HoughCenter',circen);

% handles = CPaddmeasurements(handles,'Image',[ObjectName CurrentModule],'HoughCenterX',circen(:,1));
% handles = CPaddmeasurements(handles,'Image',[ObjectName CurrentModule],'HoughCenterY',circen(:,2));
% handles = CPaddmeasurements(handles,'Image',[ObjectName CurrentModule],'HoughRadii',cirrad);

%% Does not work, since h.Meas.Nuclei not created yet
% handles = CPaddmeasurements(handles,ObjectName,'HoughLocation','CenterX',circen(:,1));
% handles = CPaddmeasurements(handles,ObjectName,'HoughLocation','CenterY',circen(:,2));
% handles = CPaddmeasurements(handles,ObjectName,'HoughLocation','Radii',cirrad);

ObjectNameThisModule = [ObjectName CurrentModule];
handles.Measurements.(ObjectNameThisModule).LocationsFeatures = {'CenterX','CenterY','Radii'};
handles.Measurements.(ObjectNameThisModule).Locations{handles.Current.SetBeingAnalyzed} = [circen(:,1) circen(:,2) cirrad];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function DrawCircle(x, y, r, nseg, S)
% Draw a circle on the current figure using ploylines
%
%  DrawCircle(x, y, r, nseg, S)
%  A simple function for drawing a circle on graph.
%
%  INPUT: (x, y, r, nseg, S)
%  x, y:    Center of the circle
%  r:       Radius of the circle
%  nseg:    Number of segments for the circle
%  S:       Colors, plot symbols and line types
%
%  OUTPUT: None
%
%  BUG REPORT:
%  Please send your bug reports, comments and suggestions to
%  pengtao@glue.umd.edu . Thanks.

%  Author:  Tao Peng
%           Department of Mechanical Engineering
%           University of Maryland, College Park, Maryland 20742, USA
%           pengtao@glue.umd.edu
%  Version: alpha       Revision: Jan. 10, 2006


theta = 0 : (2 * pi / nseg) : (2 * pi);
pline_x = r * cos(theta) + x;
pline_y = r * sin(theta) + y;

plot(pline_x, pline_y, S);