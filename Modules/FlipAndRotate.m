function handles = FlipAndRotate(handles,varargin)

% Help for the FlipAndRotate module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Flips (mirror image) and rotates an image.
% *****************************************************************************
%
% Features measured:   Feature Number:
% Rotation             |      1
% (this is the angle of rotation)
%
% Settings:
%
% Rotation method:
% *Coordinates - you can provide the X,Y pixel locations of two
% points in the image which should be aligned horizontally or vertically.
% *Mouse - you can click on points in the image which should be aligned
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
% See also Crop.
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
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the image you want to transform?
%infotypeVAR01 = imagegroup
%inputtypeVAR01 = popupmenu
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = What do you want to call the transformed image?
%defaultVAR02 = FlippedOrigBlue
%infotypeVAR02 = imagegroup indep
OutputName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Do you want to flip from left to right?
%choiceVAR03 = Yes
%choiceVAR03 = No
%inputtypeVAR03 = popupmenu
LeftToRight = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = Do you want to flip from top to bottom?
%choiceVAR04 = Yes
%choiceVAR04 = No
%inputtypeVAR04 = popupmenu
TopToBottom = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = Choose rotation method:
%choiceVAR05 = None
%choiceVAR05 = Coordinates
%choiceVAR05 = Mouse
%choiceVAR05 = Angle
RotateMethod = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu

%textVAR06 = Would you like to crop away the rotated edges?
%choiceVAR06 = Yes
%choiceVAR06 = No
CropEdges = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%inputtypeVAR06 = popupmenu

%textVAR07 = Do you want to determine the amount of rotation for each image individually as you cycle through, or do you want to define it only once (on the first image) and then apply it to all images?
%choiceVAR07 = Individually
%choiceVAR07 = Only Once
IndividualOrOnce = char(handles.Settings.VariableValues{CurrentModuleNum,7});
%inputtypeVAR07 = popupmenu

%textVAR08 = For COORDINATES or MOUSE, do you want to click on points that are aligned horizontally or vertically?
%choiceVAR08 = horizontally
%choiceVAR08 = vertically
HorizOrVert = char(handles.Settings.VariableValues{CurrentModuleNum,8});
%inputtypeVAR08 = popupmenu

%textVAR09 = For COORDINATES and ONLY ONCE, what are the coordinates of one point (X,Y)?
%defaultVAR09 = 1,1
Pixel1 = char(handles.Settings.VariableValues{CurrentModuleNum,9});

%textVAR10 = For COORDINATES and ONLY ONCE, what are the coordinates of the other point (X,Y)?
%defaultVAR10 = 100,5
Pixel2 = char(handles.Settings.VariableValues{CurrentModuleNum,10});

%textVAR11 = For ANGLE and ONLY ONCE, by what angle would you like to rotate the image (in degrees, positive = counterclockwise and negative = clockwise)?
%defaultVAR11 = 5
Angle = char(handles.Settings.VariableValues{CurrentModuleNum,11});

%%%%%%%%%%%%%%%%%
%%% Features  %%%
%%%%%%%%%%%%%%%%%

if nargin > 1 
    switch varargin{1}
%feature:categories
        case 'categories'
            if nargin == 1 || strcmp(varargin{2},'Image')
                result = { 'Rotation' };
            else
                result = {};
            end
%feature:measurements
        case 'measurements'
            result = {};
            if nargin >= 3 &&...
                strcmp(varargin{3},'Rotation') &&...
                strcmp(varargin{2},'Image')
                result = {'Rotation' };
            end
        otherwise
            error(['Unhandled category: ',varargin{1}]);
    end
    handles=result;
    return;
end

%%%VariableRevisionNumber = 1

CPValidfieldname(OutputName);
drawnow

OrigImage = CPretrieveimage(handles,ImageName,ModuleName,'DontCheckColor','CheckScale');

%%%%%%%%%%%%%%%%%%%%%%%%
%%% FLIPPING         %%%
%%%%%%%%%%%%%%%%%%%%%%%%
FlippedImage = OrigImage;
[Rows,Columns,Dimensions] = size(FlippedImage); %#ok

if strcmp(LeftToRight,'Yes')
    if Dimensions == 1
        FlippedImage = fliplr(FlippedImage);
    else
        FlippedImage(:,:,1) = fliplr(FlippedImage(:,:,1));
        FlippedImage(:,:,2) = fliplr(FlippedImage(:,:,2));
        FlippedImage(:,:,3) = fliplr(FlippedImage(:,:,3));
    end
end

if strcmp(TopToBottom,'Yes')
    if Dimensions == 1
        FlippedImage = flipud(FlippedImage);
    else
        FlippedImage(:,:,1) = flipud(FlippedImage(:,:,1));
        FlippedImage(:,:,2) = flipud(FlippedImage(:,:,2));
        FlippedImage(:,:,3) = flipud(FlippedImage(:,:,3));
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% ROTATION IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Determines the figure number to display in.
ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
FigureHandle = CPfigure(handles,'Image',ThisModuleFigureNumber);

hAx=subplot(3,2,[1 2 3 4],'Parent',ThisModuleFigureNumber);
CPimagesc(OrigImage,handles,hAx);
%%% OK to use axis image here.
axis(hAx,'image');
drawnow

if ~strcmp(RotateMethod,'None')
    % If this is the first image set, get the angle to rotate from the
    % user.  If not, get the value that was used for the first set.
    if handles.Current.SetBeingAnalyzed == 1 || strcmp(IndividualOrOnce,'Individually')
        if strcmp(RotateMethod, 'Mouse')
            displaytexthandle = uicontrol(ThisModuleFigureNumber,'style','text','position',[80 10 400 100],'fontname','helvetica','backgroundcolor',[0.7 0.7 0.9],'FontSize',handles.Preferences.FontSize);
            displaytext = 'Click on points in the image that are supposed to be aligned horizontally (e.g. a marker spot at the top left and the top right of the image). Then press the Enter key. If you make an error, the Delete or Backspace key will delete the previously selected point. You can use the zoom tools of matlab before clicking on this point by selecting Tools > Zoom in, click to zoom as desired, then select Tools > Zoom in again to deselect the tool. This will return you to the regular cursor so you can click the marker points. Use Edit > Colormap to adjust the contrast of the image if needed.';
            set(displaytexthandle,'string',displaytext)
            [x,y] = getpts(FigureHandle);
            while length(x) < 2
                warnfig=CPwarndlg(['In the ', ModuleName, ' module while in Mouse rotation method, you must click on at least two points in the image and then press enter. Please try again.'], 'Warning');
                uiwait(warnfig);
                [x,y] = getpts(FigureHandle);
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
                Pixel1 = str2num(Answers{1}); %#ok Ignore MLint
                Pixel2 = str2num(Answers{2}); %#ok Ignore MLint
                LowerLeftX = Pixel1(1);
                LowerLeftY = Pixel1(2);
                LowerRightX = Pixel2(1);
                LowerRightY = Pixel2(2);
            elseif strcmp(IndividualOrOnce,'Only Once')
                Pixel1 = str2num(Pixel1); %#ok Ignore MLint
                %% Check to make sure that Pixel values were enetered correctly
                if isempty(Pixel1)
                    error(['The coordinates you entered for point one in the ', ModuleName, ' module are invalid.'],[ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Point one invalid'],'replace');
                end
                Pixel2 = str2num(Pixel2); %#ok Ignore MLint
                if isempty(Pixel2)
                    error(['The coordinates you entered for the other point in the ', ModuleName, ' module are invalid.'],[ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Point two invalid'],'replace');
                end
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
                AngleToRotateDegrees = str2double(Answers{1});
                if isempty(AngleToRotateDegrees)
                    error(['The angle you entered in the ', ModuleName, ' module are invalid.'],[ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Angle invalid'],'replace');
                end
            elseif strcmp(IndividualOrOnce,'Only Once')
                AngleToRotateDegrees = str2double(Angle);
                if isempty(AngleToRotateDegrees)
                    error(['The angle you entered in the ', ModuleName, ' module are invalid.'],[ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Angle invalid'],'replace');
                end
            else
                error(['Image processing was canceled in the ', ModuleName, ' module because the rotation method is not recognized.']);
            end
        else
            error(['Image processing was canceled in the ', ModuleName, ' module because the rotation method is not recognized.']);
        end
    else
        fieldname = ['Rotation_', ImageName];
        AngleToRotateDegrees = handles.Measurements.Image.(fieldname){1}; %#ok Ignore MLint
    end

    PatienceHandle = CPmsgbox('Image rotation in progress');
    drawnow
    if strcmp(CropEdges,'No')
        RotatedImage = imrotate(FlippedImage, AngleToRotateDegrees);
    elseif strcmp(CropEdges,'Yes')
        BeforeCropRotatedImage = imrotate(FlippedImage, AngleToRotateDegrees);
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
    try %#ok We want to ignore MLint error checking for this line.
        delete(PatienceHandle)
    end
else
    RotatedImage=FlippedImage;
    AngleToRotateDegrees = 0;
end

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Figure must be displayed for this module.
CPfigure(FigureHandle);

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    ImageIndex = 1;
    PosImages = [1,2,4];

    DisplayFlip = strcmp(LeftToRight,'Yes') || strcmp(TopToBottom,'Yes');
    DisplayRotate = ~ strcmp(RotateMethod,'None');
    if DisplayFlip && DisplayRotate
        Format='TwoByTwo';
        NColumns = 2;
    elseif DisplayFlip || DisplayRotate
        Format='TwoByOne';
        NColumns = 1;
    else
        Format='OneByOne';
    end
    %%% Activates the appropriate figure window.
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure(OrigImage,Format,ThisModuleFigureNumber)
    end
    if strcmp(Format,'OneByOne')
        [hImage, hAx] = CPimagesc(OrigImage,handles,ThisModuleFigureNumber);
    else
        hAx = subplot(2,NColumns,PosImages(ImageIndex),'Parent',ThisModuleFigureNumber);
        CPimagesc(OrigImage,handles,hAx);
        ImageIndex = ImageIndex+1;
    end
    title(hAx,['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    if DisplayFlip
        hAx = subplot(2,NColumns,PosImages(ImageIndex),'Parent',ThisModuleFigureNumber);
        CPimagesc(FlippedImage,handles,hAx);
        ImageIndex = ImageIndex+1;
        title(hAx,'Flipped image');
    end
    if DisplayRotate
        hAx = subplot(2,NColumns,PosImages(ImageIndex),'Parent',ThisModuleFigureNumber);
        CPimagesc(RotatedImage,handles,hAx);
        if DisplayFlip
            title(hAx,'Flipped and rotated image');
        else
            title(hAx,'Rotated image');
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%
%%% ADD MEASUREMENTS %%%
%%%%%%%%%%%%%%%%%%%%%%%%
handles = CPaddmeasurements(handles, 'Image', ['Rotation_', ImageName], ...
			    AngleToRotateDegrees);

handles.Pipeline.(OutputName) = RotatedImage;
