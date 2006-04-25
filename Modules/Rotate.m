function handles = Rotate(handles)

% Help for the Rotate module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Rotates images either automatically or based on the user clicking.
% *************************************************************************
%
% Settings:
%
% Rotation method:
% *Coordinates - you can provide the X,Y pixel locations of two
% points in the image which should be aligned horizontally or vertically.
% *Mouse - you can click on two points in the image which should be aligned
% horizontally or vertically.
% *Angle - you can provide the numerical angle by which the image should be
% rotated.
%
% Would you like to crop away the rotated edges?
% When an image is rotated, there will be black space at the corners/edges
% unless you choose to crop away the incomplete rows and columns of the
% image. This cropping will produce an image that is not the exact same
% size as the original, which may affect downstream modules.
%
% See also Flip.

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
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the image to be rotated?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the rotated image?
%defaultVAR02 = RotatedImage
%infotypeVAR02 = imagegroup indep
RotatedImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Choose rotation method:
%choiceVAR03 = Coordinates
%choiceVAR03 = Mouse
%choiceVAR03 = Angle
RotateMethod = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 = Would you like to crop away the rotated edges?
%choiceVAR04 = Yes
%choiceVAR04 = No
CropEdges = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%textVAR05 = Do you want to determine the amount of rotation for each image individually as you cycle through, or do you want to define it only once (on the first image) and then apply it to all images?
%choiceVAR05 = Individually
%choiceVAR05 = Only Once
IndividualOrOnce = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu

%textVAR06 = For COORDINATES or MOUSE, do you want to click on points that are aligned horizontally or vertically?
%choiceVAR06 = horizontally
%choiceVAR06 = vertically
HorizOrVert = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%inputtypeVAR06 = popupmenu

%textVAR07 = For COORDINATES and ONLY ONCE, what are the coordinates of one point (X,Y)?
%defaultVAR07 = 1,1
Pixel1 = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = For COORDINATES and ONLY ONCE, what are the coordinates of the other point (X,Y)?
%defaultVAR08 = 100,5
Pixel2 = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%textVAR09 = For ANGLE and ONLY ONCE, by what angle would you like to rotate the image (in degrees, positive = counterclockwise and negative = clockwise)?
%defaultVAR09 = 5
Angle = char(handles.Settings.VariableValues{CurrentModuleNum,9});

%%%VariableRevisionNumber = 2

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

OrigImage = CPretrieveimage(handles,ImageName,ModuleName,'DontCheckColor','CheckScale');

%%% Determines the figure number to display in.
ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
FigureHandle = CPfigure(handles,'Image',ThisModuleFigureNumber);

subplot(3,2,[1 2 3 4])
ImageHandle = CPimagesc(OrigImage,handles);
%%% OK to use axis image here.
axis image 
drawnow



if handles.Current.SetBeingAnalyzed == 1 || strcmp(IndividualOrOnce,'Individually')
    if strcmp(RotateMethod, 'Mouse')
        displaytexthandle = uicontrol(ThisModuleFigureNumber,'style','text','position',[80 10 400 100],'fontname','helvetica','backgroundcolor',[0.7 0.7 0.9],'FontSize',handles.Preferences.FontSize);
        displaytext = 'Click on points in the image that are supposed to be aligned horizontally (e.g. a marker spot at the top left and the top right of the image). Then press the Enter key. If you make an error, the Delete or Backspace key will delete the previously selected point. You can use the zoom tools of matlab before clicking on this point by selecting Tools > Zoom in, click to zoom as desired, then select Tools > Zoom in again to deselect the tool. This will return you to the regular cursor so you can click the marker points.';
        set(displaytexthandle,'string',displaytext)
        [x,y] = getpts(FigureHandle);
        if length(x) < 2
            error(['Image processing was canceled in the ', ModuleName, ' module because you must click on at least two points then press enter.'])
        end
        delete(displaytexthandle);
        LowerLeftX = x(end-1);
        LowerLeftY = y(end-1);
        LowerRightX = x(end);
        LowerRightY = y(end);
        HorizLeg = LowerRightX - LowerLeftX;
        VertLeg = LowerLeftY - LowerRightY;
        Hypotenuse = sqrt(HorizLeg^2 + VertLeg^2);
        if strcmp(HorizOrVert,'horizontally')
            AngleToRotateRadians = -asin(VertLeg/Hypotenuse);
        elseif strcmp(HorizOrVert,'vertically')
            AngleToRotateRadians = pi/2-asin(VertLeg/Hypotenuse);
        else
            error(['Image processing was canceled in the ', ModuleName, ' module because the value of HorizOrVert is not recognized.']);
        end
        AngleToRotateDegrees = AngleToRotateRadians*180/pi;
    elseif strcmp(RotateMethod, 'Coordinates')
        %%% Rotates the image based on user-entered coordinates.
        if strcmp(IndividualOrOnce,'Individually')
            Answers = inputdlg({'What is the first pixel?' 'What is the second pixel?'}, 'Enter coordinates', 1, {'1,1' '100,5'});
            if isempty(Answers) == 1
                error(['Image processing was canceled in the ', ModuleName, ' module at your request.'])
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
            error(['Image processing was canceled in the ', ModuleName, ' module because the value of IndividualOrOnce is not recognized.']);
        end
        HorizLeg = LowerRightX - LowerLeftX;
        VertLeg = LowerLeftY - LowerRightY;
        Hypotenuse = sqrt(HorizLeg^2 + VertLeg^2);
        if strcmp(HorizOrVert,'horizontally')
            AngleToRotateRadians = -asin(VertLeg/Hypotenuse);
        elseif strcmp(HorizOrVert,'vertically')
            AngleToRotateRadians = pi/2-asin(VertLeg/Hypotenuse);
        else
            error(['Image processing was canceled in the ', ModuleName, ' module because the value of HorizOrVert is not recognized.']);
        end
        AngleToRotateDegrees = AngleToRotateRadians*180/pi;
    elseif strcmp(RotateMethod, 'Angle')
        if strcmp(IndividualOrOnce,'Individually')
            Answers = inputdlg({'Enter the angle by which you want to rotate the image'});
            if isempty(Answers) == 1
                error(['Image processing was canceled in the ', ModuleName, ' module at your request.'])
            end
            AngleToRotateDegrees = str2num(Answers{1});
        elseif strcmp(IndividualOrOnce,'Only Once')
            AngleToRotateDegrees = str2num(Angle);
        else
            error(['Image processing was canceled in the ', ModuleName, ' module because the rotation method is not recognized.']);
        end
    else
        error(['Image processing was canceled in the ', ModuleName, ' module because the rotation method is not recognized.']);
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
    Ycrop = max(1,Ycrop);
    Xcrop = max(1,Xcrop);
    [lengthX lengthY] = size(BeforeCropRotatedImage);
    RotatedImage = BeforeCropRotatedImage(Xcrop:lengthX-Xcrop,Ycrop:lengthY-Ycrop);
else
    error(['Image processing was canceled in the ', ModuleName, ' module because the value of CropEdges is not recognized.']);
end

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Figure must be displayed for this module.
CPfigure(FigureHandle);

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    %%% Activates the appropriate figure window.
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure(OrigImage,'TwoByOne',ThisModuleFigureNumber)
    end
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,1,1); 
    CPimagesc(OrigImage,handles); 
    title(['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the rotated
    %%% Image.
    subplot(2,1,2); 
    CPimagesc(RotatedImage,handles); 
    title('Rotated Image');
end


try %#ok We want to ignore MLint error checking for this line.
    delete(PatienceHandle)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

handles.Pipeline.(RotatedImageName) = RotatedImage;