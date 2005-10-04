function handles = IdentifyPrimManual(handles)

% Help for the Identify Primary Manually module:
% Category: Object Processing
%
% This module allows the user to identify an single object by manually
% outlining it by using the mouse to click at multiple points around
% the object.
%
% SAVING IMAGES: In addition to the object outlines and the
% pseudo-colored object images that can be saved using the
% instructions in the main CellProfiler window for this module,
% this module produces several additional images which can be
% easily saved using the Save Images module. These will be grayscale
% images where each object is a different intensity. (1) The
% preliminary segmented image, which includes objects on the edge of
% the image and objects that are outside the size range can be saved
% using the name: UneditedSegmented + whatever you called the objects
% (e.g. UneditedSegmentedNuclei). (2) The preliminary segmented image
% which excludes objects smaller than your selected size range can be
% saved using the name: SmallRemovedSegmented + whatever you called the
% objects (e.g. SmallRemovedSegmented Nuclei) (3) The final segmented
% image which excludes objects on the edge of the image and excludes
% objects outside the size range can be saved using the name:
% Segmented + whatever you called the objects (e.g. SegmentedNuclei)
%
%    Additional image(s) are calculated by this module and can be 
% saved by altering the code for the module (see the SaveImages module
% help for instructions).
%
% See also IDENTIFYPRIMTHRESHOLD,
% IDENTIFYPRIMADAPTTHRESHOLDA,
% IDENTIFYPRIMADAPTTHRESHOLDB,
% IDENTIFYPRIMADAPTTHRESHOLDC,
% IDENTIFYPRIMADAPTTHRESHOLDD,
% IDENTIFYPRIMSHAPEDIST,
% IDENTIFYPRIMSHAPEINTENS,
% IDENTIFYPRIMINTENSINTENS.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne Carpenter <carpenter@wi.mit.edu>
%   Thouis Jones   <thouis@csail.mit.edu>
%   In Han Kang    <inthek@mit.edu>
%
% $Revision$

drawnow

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%

%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

%textVAR01 = What did you call the images you want to use to manually identify an object?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the objects identified by this module?
%defaultVAR02 = Cells
%infotypeVAR02 = objectgroup indep
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Resize the image to this size before manual identification (pixels)
%defaultVAR03 = 512
MaxResolution = str2num(char(handles.Settings.VariableValues{CurrentModuleNum,3}));

%textVAR04 = What do you want to call the image of the outlines of the objects?
%defaultVAR04 = OutlinedNuclei
%infotypeVAR04 = outlinegroup indep
SaveOutlined = char(handles.Settings.VariableValues{CurrentModuleNum,4}); 

%textVAR05 =  What do you want to call the labeled matrix image?
%choiceVAR05 = Do not save
%choiceVAR05 = LabeledNuclei
%infotypeVAR05 = imagegroup indep
SaveColored = char(handles.Settings.VariableValues{CurrentModuleNum,5}); 
%inputtypeVAR05 = popupmenu custom

%textVAR06 = Do you want to save the labeled matrix image in RGB or grayscale?
%choiceVAR06 = RGB
%choiceVAR06 = Grayscale
SaveMode = char(handles.Settings.VariableValues{CurrentModuleNum,6}); 
%inputtypeVAR06 = popupmenu

%%%VariableRevisionNumber = 02

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a variable,
%%% "OrigImage".
%%% Checks whether the image exists in the handles structure.
if isfield(handles.Pipeline, ImageName) == 0
    error(['Image processing has been canceled. Prior to running the Identify Primary Manually module, you must have previously run a module to load an image. You specified in the Identify Primary Manually module that this image was called ', ImageName, ' which should have produced a field in the handles structure called ', ImageName, '. The Identify Primary Manually module cannot find this image.']);
end
OrigImage = handles.Pipeline.(ImageName);

if isempty(MaxResolution)
    errordlg('Invalid specification of the image size in the Identify Primary Manually module.')
end

% Use a low resolution image for outlining the primary region
MaxSize = max(size(OrigImage));
if MaxSize > MaxResolution
    LowResOrigImage = imresize(OrigImage,MaxResolution/MaxSize,'bicubic');
else
    LowResOrigImage = OrigImage;
end

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow



%%% Displays the image in a new figure window.
FigureHandle = figure;
ImageHandle = imagesc(LowResOrigImage);axis off,axis image
[nrows,ncols,ncolors] = size(LowResOrigImage);
if ncolors == 1
    
end
AxisHandle = gca;
set(gca,'fontsize',handles.Current.FontSize)
title([{['Image Set #',num2str(handles.Current.SetBeingAnalyzed),'. Click on consecutive points to outline the region of interest.']},...
    {'Press enter when finished, the first and last points will be connected automatically.'},...
    {'The backspace key or right mouse button will erase the last clicked point.'}]);


%%% Manual outline of the object, see local function 'getpoints' below.
%%% Continue until user has drawn a valid shape
[x,y] = getpoints(AxisHandle);
close(FigureHandle)
[X,Y] = meshgrid(1:ncols,1:nrows);
LowResInterior = inpolygon(X,Y, x,y);
FinalLabelMatrixImage = double(imresize(LowResInterior,size(OrigImage)) > 0.5);

FinalOutline = logical(zeros(size(FinalLabelMatrixImage,1),size(FinalLabelMatrixImage,2)));
FinalOutline = bwperim(FinalLabelMatrixImage > 0);


%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow



fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);

ColoredLabelMatrixImage = CPlabel2rgb(handles,FinalLabelMatrixImage);

drawnow
CPfigure(handles,ThisModuleFigureNumber);

subplot(2,2,1);imagesc(LowResOrigImage); title(['Original Image, Image Set # ', num2str(handles.Current.SetBeingAnalyzed)]); 

subplot(2,2,2); imagesc(LowResInterior); title(['Manually Identified ',ObjectName]);

subplot(2,2,3); imagesc(FinalOutline); title([ObjectName, ' Outline']);
hold on, plot(x,y,'r'),hold off

subplot(2,2,4); imagesc(ColoredLabelMatrixImage); title(['Segmented ' ObjectName]);

CPFixAspectRatio(LowResOrigImage);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow


%%% Saves the final segmented label matrix image to the handles structure.
fieldname = ['Segmented',ObjectName];
handles.Pipeline.(fieldname) = FinalLabelMatrixImage;

%%% Saves the ObjectCount, i.e. the number of segmented objects.
if ~isfield(handles.Measurements.Image,'ObjectCountFeatures')
    handles.Measurements.Image.ObjectCountFeatures = {};
    handles.Measurements.Image.ObjectCount = {};
end
column = find(~cellfun('isempty',strfind(handles.Measurements.Image.ObjectCountFeatures,ObjectName)));
if isempty(column)
    handles.Measurements.Image.ObjectCountFeatures(end+1) = {['ObjectCount ' ObjectName]};
    column = length(handles.Measurements.Image.ObjectCountFeatures);
end
handles.Measurements.Image.ObjectCount{handles.Current.SetBeingAnalyzed}(1,column) = max(FinalLabelMatrixImage(:));

%%% Saves the location of each segmented object
handles.Measurements.(ObjectName).LocationFeatures = {'CenterX','CenterY'};
tmp = regionprops(FinalLabelMatrixImage,'Centroid');
Centroid = cat(1,tmp.Centroid);
handles.Measurements.(ObjectName).Location(handles.Current.SetBeingAnalyzed) = {Centroid};


%%% Saves images to the handles structure so they can be saved to the hard
%%% drive, if the user requested.
try
    if ~strcmp(SaveColored,'Do not save')
        if strcmp(SaveMode,'RGB')
            handles.Pipeline.(SaveColored) = ColoredLabelMatrixImage;
        else
           handles.Pipeline.(SaveColored) = FinalLabelMatrixImage;
       end
    end
    if ~strcmp(SaveOutlined,'Do not save')
        handles.Pipeline.(SaveOutlined) = FinalOutline;
    end
catch
    error('The object outlines or colored objects were not calculated by an identify module (possibly because the window is closed) so these images were not saved to the handles structure. The Save Images module will therefore not function on these images. This is just for your information - image processing is still in progress, but the Save Images module will fail if you attempted to save these images.')
end

%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTION %%%
%%%%%%%%%%%%%%%%%%

function [xpts_spline,ypts_spline] = getpoints(AxisHandle)

Position = get(gca,'Position');
FigureHandle = (get(AxisHandle, 'Parent'));
PointHandles = [];
xpts = [];
ypts = [];
NbrOfPoints = 0;
done = 0;

hold on
while ~done;

    UserInput = waitforbuttonpress;                            % Wait for user input
    SelectionType = get(FigureHandle,'SelectionType');         % Get information about the last button press
    CharacterType = get(FigureHandle,'CurrentCharacter');      % Get information about the character entered

    % Left mouse button was pressed, add a point
    if UserInput == 0 & strcmp(SelectionType,'normal')

        % Get the new point and store it
        CurrentPoint  = get(AxisHandle, 'CurrentPoint');
        xpts = [xpts CurrentPoint(2,1)];
        ypts = [ypts CurrentPoint(2,2)];
        NbrOfPoints = NbrOfPoints + 1;

        % Plot the new point
        h = plot(CurrentPoint(2,1),CurrentPoint(2,2),'r.');
        set(AxisHandle,'Position',Position)                   % For some reason, Matlab moves the Title text when the first point is plotted, which in turn resizes the image slightly. This line restores the original size of the image
        PointHandles = [PointHandles h];

        % If there are any points, and the right mousebutton or the backspace key was pressed, remove a points
    elseif NbrOfPoints > 0 & ((UserInput == 0 & strcmp(SelectionType,'alt')) | (UserInput == 1 & CharacterType == char(8)))   % The ASCII code for backspace is 8

        NbrOfPoints = NbrOfPoints - 1;
        xpts = xpts(1:end-1);
        ypts = ypts(1:end-1);
        delete(PointHandles(end));
        PointHandles = PointHandles(1:end-1);

        % Enter key was pressed, manual outlining done, and the number of points are at least 3
    elseif NbrOfPoints >= 3 & UserInput == 1 & CharacterType == char(13)

        % Indicate that we are done
        done = 1;

        % Close the curve by making the first and last points the same
        xpts = [xpts xpts(1)];
        ypts = [ypts ypts(1)];

        % Remove plotted points
        if ~isempty(PointHandles)
            delete(PointHandles)
        end

    end

    % Remove old spline and draw new
    if exist('SplineCurve','var')
        delete(SplineCurve)                                % Delete the graphics object
        clear SplineCurve                                  % Clear the variable
    end
    if NbrOfPoints > 1
        q = 0:length(xpts)-1;
        qq = 0:0.1:length(xpts)-1;                          % Increase the number of points 10 times using spline interpolation
        xpts_spline = spline(q,xpts,qq);
        ypts_spline = spline(q,ypts,qq);
        SplineCurve = plot(xpts_spline,ypts_spline,'r');
        drawnow
    else
        xpts_spline = xpts;
        ypts_spline = ypts;
    end
end
hold off
