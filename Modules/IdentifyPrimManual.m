function handles = IdentifyPrimManual(handles)

% Help for the Identify Primary Manual module:
% Category: Object Processing
%
% SHORT DESCRIPTION:
% Identifies an object based on manual intervention (clicking) by the user.
% *************************************************************************
%
% This module allows the user to identify objects by manually outlining
% them. This is done by using the mouse to click at multiple points around
% the object. Multiple objects can be outlined using this module.
%
% Special note on saving images: Using the settings in this module, object
% outlines can be passed along to the module OverlayOutlines and then saved
% with the SaveImages module. Objects themselves can be passed along to the
% object processing module ConvertToImage and then saved with the
% SaveImages module. This module produces several additional types of
% objects with names that are automatically passed along with the following
% naming structure: (1) The unedited segmented image, which includes
% objects on the edge of the image and objects that are outside the size
% range, can be saved using the name: UneditedSegmented + whatever you
% called the objects (e.g. UneditedSegmentedNuclei). (2) The segmented
% image which excludes objects smaller than your selected size range can be
% saved using the name: SmallRemovedSegmented + whatever you called the
% objects (e.g. SmallRemovedSegmented Nuclei).
%
% See also IdentifyPrimAutomatic.

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
%   Peter Swire
%   Rodrigo Ipince
%   Vicky Lay
%   Jun Liu
%   Chris Gang
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the images you want to use to manually identify an object?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the objects identified by this module?
%defaultVAR02 = Cells
%infotypeVAR02 = objectgroup indep
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Enter the maximum image height or width (in pixels) to display for the manual identification. Very large images will be resized to this maximum dimension for the manual identification step. Enter "Do not resize" to display the unaltered image.
%defaultVAR03 = Do not resize
MaxResolution = char(handles.Settings.VariableValues{CurrentModuleNum,3}); %#ok

%textVAR04 = What do you want to call the outlines of the identified objects (optional)?
%defaultVAR04 = Do not save
%infotypeVAR04 = outlinegroup indep
SaveOutlines = char(handles.Settings.VariableValues{CurrentModuleNum,4});


%%%VariableRevisionNumber = 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a variable,
%%% "OrigImage".
OrigImage = CPretrieveimage(handles,ImageName,ModuleName,'DontCheckColor','CheckScale');

if strcmpi(MaxResolution,'Do not resize')
    MaxResolution = Inf;
else
    MaxResolution = str2double(MaxResolution);
    if isempty(MaxResolution)
        error('You have entered an invalid input for Max Resolution. It must be ''Do not resize'' or a number.');
    end
end

%%% Use a low resolution image for outlining the primary region, if
%%% requested.
MaxSize = max(size(OrigImage));

if MaxSize > MaxResolution
    LowResOrigImage = imresize(OrigImage,MaxResolution/MaxSize,'bicubic');
else
    LowResOrigImage = OrigImage;
end

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Displays the image in a new figure window.
FigureHandle = CPfigure;
% ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
%%% We are making it large so it's easier to outline the object of
%%% interest.
CPresizefigure(LowResOrigImage,'TwoByTwo',FigureHandle)
%%% We cannot use CPimagesc here because it interferes with the custom
%%% getpoints function (whereas the Matlab getpts function would be fine).
%%% Or maybe we can...
CPimagesc(LowResOrigImage,handles);

AxisHandle = gca;
title([{['Cycle #',num2str(handles.Current.SetBeingAnalyzed),'. Click on consecutive points to outline the region of interest.']},...
    {'The backspace key or right mouse button will erase the last clicked point.'},...
    {'Use Edit > Colormap to adjust the contrast of the image if needed.'},...
    {'Press enter when finished, the first and last points will be connected automatically.'},...
    {'Then be patient while waiting for processing to complete.'}],'fontsize',handles.Preferences.FontSize);

NewImage = zeros(size(LowResOrigImage));

loopControl = 1; 
i = 1;
%%%% For some reason, the button is unable to change the value of
%%%% loopControl that the while loop sees.
% DoneButton = uicontrol('Style', 'pushbutton', 'String', 'Done',...
%     'Position', [10 10 60 20], 'Callback', 'ButtonName=questdlg(''Continue outlining objects?'', ''IdentifyPrimManual'',''Yes'', ''No'', ''Yes'');if(strcmp(ButtonName, ''No''))loopControl = 0;end');
% uicontrol(DoneButton);

while loopControl == 1
    %%% Manual outline of the object, see local function 'getpoints' below.
    %%% Continue until user has drawn a valid shape
    [x,y] = getpoints(AxisHandle);
    [nrows,ncols] = size(LowResOrigImage);
    [X,Y] = meshgrid(1:ncols,1:nrows);
    LowResInterior = inpolygon(X,Y, x,y);
    FinalLabelMatrixImage{i} = double(imresize(LowResInterior,size(OrigImage)) > 0.5);
    FinalOutline{i} = bwperim(FinalLabelMatrixImage{i} > 0);
    NewImage(find(LowResInterior==1))=i;
    % combine the matrices
    if i ~= 1
        FinalOutline{1} = FinalOutline{1} | FinalOutline{i};
    end
    i = i+1;
    ButtonName=questdlg('Continue outlining objects?', ...
                       'IdentifyPrimManual', ...
                       'Yes', 'No', 'Yes');
    if(strcmp(ButtonName, 'No'))
        loopControl = 0;
    end
end
close(FigureHandle)

FinalOutline = FinalOutline{1};
FinalLabelMatrixImage = NewImage;

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);

if any(findobj == ThisModuleFigureNumber)
    ColoredLabelMatrixImage = CPlabel2rgb(handles,FinalLabelMatrixImage);

    %%% Activates the appropriate figure window.
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure(LowResOrigImage,'TwoByTwo',ThisModuleFigureNumber);
    end
    subplot(2,2,1); 
    CPimagesc(LowResOrigImage,handles); 
    title(['Original Image, cycle # ', num2str(handles.Current.SetBeingAnalyzed)]);
    subplot(2,2,2); 
    CPimagesc(FinalLabelMatrixImage,handles); 
    title(['Manually Identified ',ObjectName]);
    FinalOutlineOnOrigImage = OrigImage;
    FinalOutlineOnOrigImage(FinalOutline) = max(max(OrigImage));
    subplot(2,2,3); 
    CPimagesc(FinalOutlineOnOrigImage,handles); 
    title([ObjectName, ' Outline']);
    subplot(2,2,4); 
    CPimagesc(ColoredLabelMatrixImage,handles); 
    title(['Identified ' ObjectName]);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the final segmented label matrix image to the handles structure.
handles.Pipeline.(['Segmented',ObjectName]) = FinalLabelMatrixImage;
handles.Pipeline.(['UneditedSegmented',ObjectName]) = FinalLabelMatrixImage;
handles.Pipeline.(['SmallRemovedSegmented',ObjectName]) = FinalLabelMatrixImage;

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
    if ~strcmpi(SaveOutlines,'Do not save')
        handles.Pipeline.(SaveOutlines) = FinalOutline;
    end
catch error(['The object outlines were not calculated by the ', ModuleName, ' module, so these images were not saved to the handles structure. The Save Images module will therefore not function on these images. This is just for your information - image processing is still in progress, but the Save Images module will fail if you attempted to save these images.'])
end

%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTION %%%
%%%%%%%%%%%%%%%%%%%

function [xpts_spline,ypts_spline] = getpoints(AxisHandle)

Position = get(AxisHandle,'Position');
FigureHandle = (get(AxisHandle, 'Parent'));
PointHandles = [];
xpts = [];
ypts = [];
NbrOfPoints = 0;
done = 0;
%%% Turns off the CPimagetool function because it interferes with getting
%%% points.
ImageHandle = get(AxisHandle,'children');
set(ImageHandle,'ButtonDownFcn','');

hold on
while ~done;

    UserInput = waitforbuttonpress;                            % Wait for user input
    SelectionType = get(FigureHandle,'SelectionType');         % Get information about the last button press
    CharacterType = get(FigureHandle,'CurrentCharacter');      % Get information about the character entered

    % Left mouse button was pressed, add a point
    if UserInput == 0 && strcmp(SelectionType,'normal')

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
    elseif NbrOfPoints > 0 && ((UserInput == 0 && strcmp(SelectionType,'alt')) || (UserInput == 1 && CharacterType == char(8)))   % The ASCII code for backspace is 8

        NbrOfPoints = NbrOfPoints - 1;
        xpts = xpts(1:end-1);
        ypts = ypts(1:end-1);
        delete(PointHandles(end));
        PointHandles = PointHandles(1:end-1);

        % Enter key was pressed, manual outlining done, and the number of points are at least 3
    elseif NbrOfPoints >= 3 && UserInput == 1 && CharacterType == char(13)

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
set(ImageHandle,'ButtonDownFcn','CPimagetool');