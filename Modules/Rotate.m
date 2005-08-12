function handles = Rotate(handles)

% Help for the Crop module:
% Category: Image Processing
%
% Sorry, there is no help right now.

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%

%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

%textVAR01 = What did you call the image to be rotated?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = Choose rotation method.
%choiceVAR02 = Coordinates
%choiceVAR02 = Mouse
%choiceVAR02 = Angle
RotateMethod = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 = What do you want to call the rotated image?
%infotypeVAR03 = imagegroup indep
%defaultVAR03 = RotatedImage
RotatedImageName = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = Align the spots horizontally, or vertically?
%choiceVAR04 = horizontally
%choiceVAR04 = vertically
HorizOrVert = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%textVAR05 = Woud you want the rotation for the first image to be applied to all image sets or do you want to determine the rotation for each image individually as you cycle through?
%choiceVAR05 = Only Once
%choiceVAR05 = Individually
IndividualOrOnce = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu

%textVAR06 = Would you like to crop away the rotated edges?
%choiceVAR06 = Yes
%choiceVAR06 = No
CropEdges = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%inputtypeVAR06 = popupmenu

%textVAR07 = What are the coordinates of the first pixel? (only if you are using 'Coordinates' and 'Only Once').
%defaultVAR07 = 1,1
Pixel1 = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = What are the coordinates of the second pixel? (only if you are using 'Cooridinates' and 'Only Once').
%defaultVAR08 = 100,5
Pixel2 = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%textVAR09 = By what angle would you like to rotate the image? (only if you are uisng 'Angle' and 'Only Once').
%defaultVAR09 = 5
Angle = char(handles.Settings.VariableValues{CurrentModuleNum,9});




if ~isfield(handles.Pipeline, ImageName)
    error(['Image processing has been canceled. Prior to running the Spot Identifier module, you must have previously run a module to load an image. You specified in the Spot Identifier module that this image was called ', ImageName, ' which should have produced a field in the handles structure called ', ImageName, '. The Spot Identifier module cannot find this image.']);
end

OrigImage = handles.Pipeline.(ImageName);

    %%% Determines the figure number to display in.
fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
FigureHandle = CPfigure(handles,ThisModuleFigureNumber);
subplot(2,3,[1 2 4 5]);
ImageHandle = imagesc(OrigImage); 
colormap(gray), axis image, pixval off;%#ok We want to ignore MLint error checking for this line.

drawnow

if handles.Current.SetBeingAnalyzed == 1 || strcmp(IndividualOrOnce,'Individually')
    if strcmp(RotateMethod, 'Mouse')
        Answer2 = CPquestdlg('After closing this window by clicking OK, click on points in the image that are supposed to be aligned horizontally (e.g. a marker spot at the top left and the top right of the image). Then press the Enter key. If you make an error, the Delete or Backspace key will delete the previously selected point. You can use the zoom tools of matlab before clicking on this point by selecting Tools > Zoom in, click to zoom as desired, then select Tools > Zoom in again to deselect the tool. This will return you to the regular cursor so you can click the marker points.','Rotate image using the mouse','OK','Cancel','OK');
        waitfor(Answer2)
        if strcmp(Answer2, 'Cancel')
            error('Image processing was canceled during the Spot Identifier module.')
        end
        [x,y] = getpts(FigureHandle);
        if length(x) < 2
            error('The Spot Identifier was canceled because you must click on at least two points then press enter.')
        end
        [m b] = polyfit(x,y,1);
        if strcmp(HorizOrVert,'horizontally')
            AngleToRotateRadians = -atan(m);
        elseif strcmp(HorizOrVert,'vertically')
            AngleToRotateRadians = pi/2-atan(m);
        else
            error('The value of HoriOrVert is not recognized');
        end
        AngleToRotateDegrees = AngleToRotateRadians*180/pi;
    elseif strcmp(RotateMethod, 'Coordinates')
        %%% Rotates the image based on user-entered coordinates.
        if strcmp(IndividualOrOnce,'Individually')
            Answers = inputdlg({'What is the first pixel?' 'What is the second pixel?'}, 'Enter coordinates', 1, {'1,1' '100,5'});
            if isempty(Answers) == 1
                error('Image processing was canceled during the rotate module.')
            end
            Pixel1 = str2num(Answers{1});
            Pixel2 = str2num(Answers{2});
            LowerLeftX = Pixel1(1);
            LowerLeftY = Pixel1(2);
            LowerRightX = Pixel2(1);
            LowerRightY = Pixel2(2);
        elseif strcmp(IndividualOrOnce,'Only Once')
            Pixel1 = str2num(Pixel1);
            Pixel2 = str2num(Pixel2);
            LowerLeftX = Pixel1(1);
            LowerLeftY = Pixel1(2);
            LowerRightX = Pixel2(1);
            LowerRightY = Pixel2(2);
        else
            error('The value of IndividualOrOnce was not recognized');
        end
        HorizLeg = LowerRightX - LowerLeftX;
        VertLeg = LowerLeftY - LowerRightY;
        Hypotenuse = sqrt(HorizLeg^2 + VertLeg^2);
        if strcmp(HorizOrVert,'horizontally')
            AngleToRotateRadians = -asin(VertLeg/Hypotenuse);
        elseif strcmp(HorizOrVert,'vertically')
            AngleToRotateRadians = pi/2-asin(VertLeg/Hypotenuse);
        else
            error('The value of HorizOrVert is not recognized');
        end
        AngleToRotateDegrees = AngleToRotateRadians*180/pi;
    elseif strcmp(RotateMethod, 'Angle')
        if strcmp(IndividualOrOnce,'Individually')
            Answers = inputdlg({'Enter the angle by which you want to rotate the image'});
            if isempty(Answers) == 1
                error('Image processing was canceled during the rotate module.')
            end
            AngleToRotateDegrees = str2num(Answers{1});
        elseif strcmp(RotateMethod,'Only Once')
            AngleToRotateDegrees = str2num(Angle);
        else
            error('The value of RotateMethod was not recognized');
        end 
    else
        error('The RotateMethod was not recognized.');
    end
else
    column = find(strcmp(handles.Measurements.Image.RotationFeatures,['Rotation ' ImageName]));
    AngleToRotateDegrees = handles.Measurements.Image.Rotation{1}(column);
end

if ~isfield(handles.Measurements.Image,'RotationFeatures')
    handles.Measurements.Image.RotationFeatures = {};
    handles.Measurements.Image.Rotation = {};
end

column = find(strcmp(handles.Measurements.Image.RotationFeatures,['Rotation ' ImageName]));

if isempty(column)
    handles.Measurements.Image.RotationFeatures(end+1) = {['Rotation ' ImageName]};
    column = length(handles.Measurements.Image.RotationFeatures);
end
handles.Measurements.Image.Rotation{handles.Current.SetBeingAnalyzed}(1,column) = AngleToRotateDegrees;

PatienceHandle = CPmsgbox('Image rotation in progress');
drawnow
if strcmp(CropEdges,'No')
    RotatedImage = imrotate(OrigImage, AngleToRotateDegrees);
elseif strcmp(CropEdges,'Yes')
    BeforeCropRotatedImage = imrotate(OrigImage, AngleToRotateDegrees);
    theta = AngleToRotateDegrees * pi/180;
    [x y] = size(OrigImage);
    Ycrop = floor(abs((y*sin(theta) - x*cos(theta))*cos(theta)*sin(theta)/(sin(theta)^2-cos(theta)^2)));
    Xcrop = floor(abs((x*sin(theta) - y*cos(theta))*cos(theta)*sin(theta)/(sin(theta)^2-cos(theta)^2)));
    [lengthX lengthY] = size(BeforeCropRotatedImage);
    RotatedImage = BeforeCropRotatedImage(Ycrop:lengthY-Ycrop,Xcrop:lengthX-Xcrop);
else
    error('The value of CropEdges is not recognized');
end
CPfigure(FigureHandle); 
subplot(2,3,[1 2 4 5]);
ImageHandle = imagesc(RotatedImage); 
colormap(gray);
axis image;
title('Rotated Image');
pixval off;
try %#ok We want to ignore MLint error checking for this line.
   delete(PatienceHandle)
end

handles.Pipeline.(RotatedImageName) = RotatedImage;
    