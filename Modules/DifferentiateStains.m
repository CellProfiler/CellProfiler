function handles = DifferentiateStains(handles)

% Help for the Differentiate Stains module
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Outputs two intensity images for each of two stains
% *****************************************************************************
%
% This module produces two intensity images representing the amount of
% each of two different stains minus the background staining. It models
% the intensity for each color X as:
%
% (1 - S1(x)*Q1 - S2(x)*Q2) = I(x)
%
% where x is the red, green and blue channel
%       S1(x) is related to the absorbance of stain 1 for color x
%       Q1 is the quantity of stain 1 at the pixel
% and similarly for the second stain.
%
% The module asks the user to pick two cells stained with each of the
% two stains (or one cell with one stain and the other with both stains)
% and a point that represents the background staining. The colors of these
% three points give the color values used to determine intensities.
%
% There are two modes of color differentiation, cooperative and competitive.
% Cooperative:
% In cooperative mode, the intensity of the image for one stain is
% decreased by the amount that the color is like the other stain and vice-
% versa. A particular pixel can have quantities of both stains.
%
% Competitive:
% In competitive mode, the module computes a vector in colorspace between 
% the colors of stains 1 and 2, finds background values for the two stains
% and then for each pixel, computes the magnitude of the pixel's color
% in the direction of the vector, subtracting the background value
% for the stain to normalize. The mode is competitive in that the measured
% amount of stain # 1 has the opposite sign (before subtracting background)
% from stain # 2 and a pixel is generally either assigned stain # 1's or
% stain # 2's color.
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

%textVAR02 = What do you want to name the first image?
%defaultVAR02 = Stain1
%infotypeVAR02 = imagegroup indep
Stain1Name = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = What do you want to name the second image?
%defaultVAR03 = Stain2
%infotypeVAR03 = imagegroup indep
Stain2Name = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = Do you want to pick representative cells on the first pass, all passes or never?
%choiceVAR04 = First
%choiceVAR04 = All
%choiceVAR04 = Never
%inputtypeVAR04 = popupmenu

Pick = char(handles.Settings.VariableValues{CurrentModuleNum,4});
if strcmp(Pick,'All') || (strcmp(Pick,'First') && handles.Current.SetBeingAnalyzed == 1)
    Pick = 1;
else
    Pick = 0;
end

%textVAR05 = Do you want to differentiate between colors competitively or cooperatively?
%choiceVAR05 = Cooperatively
%choiceVAR05 = Competitively
%inputtypeVAR05 = popupmenu
Cooperative = strcmp(char(handles.Settings.VariableValues{CurrentModuleNum,5}),'Cooperatively');

%textVAR06 = What is the color of the cell stained with stain 1? Enter it in the format, "R,G,B"
%defaultVAR06 = 0,1,1

Stain1Color =sscanf(char(handles.Settings.VariableValues{CurrentModuleNum,6}),'%f,%f,%f');
if length(Stain1Color) ~= 3
    error('The color of stain 1 must have three values')
end

%textVAR07 = What is the color of the cell stained with stain 2? Enter it in the format, "R,G,B"
%defaultVAR07 = 1,0,1

Stain2Color=sscanf(char(handles.Settings.VariableValues{CurrentModuleNum,7}),'%f,%f,%f');
if length(Stain2Color) ~= 3
    error('The color of stain 2 must have three values')
end

%textVAR08 = What is the color of the background? Enter it in the format, "R,G,B"
%defaultVAR08 = 1,1,1

BackgroundColor =sscanf(char(handles.Settings.VariableValues{CurrentModuleNum,8}),'%f,%f,%f');
if length(BackgroundColor) ~= 3
    error('The color of stain 2 must have three values')
end

%%%%%%%%%%%%%%%%%%%%%
%%% Picking cells %%%
%%%%%%%%%%%%%%%%%%%%%

InputImage = CPretrieveimage(handles,ImageName,ModuleName,'DontCheckColor','CheckScale');
%%%
%%% This is Ray's enhanced image - squaring the channel intensities and dividing
%%% by the other channels to enhance the color separation and normalize by intensity
%%%
AlteredImage = InputImage .* InputImage;
AlteredImage = AlteredImage ./ InputImage(1:end,1:end,[2,3,1]);
AlteredImage = AlteredImage ./ InputImage(1:end,1:end,[3,1,2]);
AlteredImage = AlteredImage / max(AlteredImage(:));
data = struct('DisplayImage',InputImage,...
              'InputImage',AlteredImage,...
              'Stain1Color',Stain1Color,...
              'Stain2Color',Stain2Color,...
              'Cooperative',Cooperative,...
              'BackgroundColor',BackgroundColor,...
              'handles',handles,...
              'Continue',Pick);

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
FigureHandle = CPfigure(handles,'Image',ThisModuleFigureNumber);

if any(findobj == ThisModuleFigureNumber) && Pick
    old_guidata = guidata(FigureHandle);
    try
        guidata(FigureHandle,data);
        clf(ThisModuleFigureNumber);
        SelectionType = uicontrol('Style','popup',...
                                  'Parent',FigureHandle,...
                                  'Position',[10,10,150,20],...
                                  'String','Stain 1|Stain 2|Background');
        OKButton = uicontrol('Style','pushbutton',...
                             'Parent',FigureHandle,...
                             'Position',[200,10,50,20],...
                             'String','OK',...
                             'Callback',@OkPressed);
        while data.Continue
           ShowData(data,ThisModuleFigureNumber);
           drawnow;
           uiwait(FigureHandle);
           if ~ ishandle(FigureHandle)
               break;
           end
           data = guidata(FigureHandle);
           if ~ data.Continue
               break;
           end
           color = AlteredImage(data.position(2),data.position(1),1:3);
           color = reshape(color,3,1);
           switch(get(SelectionType,'Value'))
               case 1
                   data.Stain1Color=color;
                   set(SelectionType,'Value',2);
               case 2
                   data.Stain2Color=color;
                   set(SelectionType,'Value',3);
               case 3
                   data.BackgroundColor = color;
                   set(SelectionType,'Value',1);
           end
           guidata(FigureHandle,data)
        end
        delete(SelectionType);
        delete(OKButton);
        data = guidata(FigureHandle);
        handles.Settings.VariableValues{CurrentModuleNum,5} = sprintf('%.3f,%.3f,%.3f',data.Stain1Color);
        handles.Settings.VariableValues{CurrentModuleNum,6} = sprintf('%.3f,%.3f,%.3f',data.Stain2Color);
        handles.Settings.VariableValues{CurrentModuleNum,7} = sprintf('%.3f,%.3f,%.3f',data.BackgroundColor);
    catch ME1
        guidata(FigureHandle,old_guidata)
        rethrow(ME1)
    end
end

%%%%%%%%%%%%%%%
%%% DISPLAY %%%
%%%%%%%%%%%%%%%
drawnow;
if any(findobj == ThisModuleFigureNumber)
    ShowData(data,ThisModuleFigureNumber);
end

%%%%%%%%%%%%%%%%%%%
%%% SAVE IMAGES %%%
%%%%%%%%%%%%%%%%%%%

[handles.Pipeline.(Stain1Name),handles.Pipeline.(Stain2Name)] = split_images(data);

function OkPressed(hObject, event)
data = guidata(hObject);
data.Continue=0;
guidata(hObject,data);
uiresume(gcf);
    
function ImageClickCallback(hObject, event)
data = guidata(hObject);
pos=get(get(gcbo,'Parent'),'CurrentPoint');
pos=floor(pos(1,1:2));
data.position = pos;
guidata(hObject,data);
uiresume(gcf);
    
    
function ShowData(data,ThisModuleFigureNumber)
handles=data.handles;
hAx = subplot(2,2,1,'Parent',ThisModuleFigureNumber);
image1 = CPimagesc(data.DisplayImage,handles,hAx);
title(hAx,sprintf('Original image (background=%.2f,%.2f,%.2f',data.BackgroundColor));
[Stain1Image, Stain2Image] =...
    split_images(data);
hAx = subplot(2,2,2,'Parent',ThisModuleFigureNumber);
image2 = CPimagesc(data.InputImage,handles, hAx);
title(hAx,'Color-adjusted image');
hAx = subplot(2,2,3,'Parent',ThisModuleFigureNumber);
image3 = CPimagesc(Stain1Image,handles, hAx);
title(hAx,sprintf('Stain 1 image (color=%.2f,%.2f,%.2f',data.Stain1Color));
hAx = subplot(2,2,4,'Parent',ThisModuleFigureNumber);
image4 = CPimagesc(Stain2Image, handles, hAx);
title(hAx,sprintf('Stain 2 image (color=%.2f,%.2f,%.2f',data.Stain2Color));
if data.Continue
    set(image1,'ButtonDownFcn',@ImageClickCallback);
    set(image2,'ButtonDownFcn',@ImageClickCallback);
    set(image3,'ButtonDownFcn',@ImageClickCallback);
    set(image4,'ButtonDownFcn',@ImageClickCallback);
end

function [Stain1Image, Stain2Image] = split_images(data)
Stain1Color = data.Stain1Color;
Stain2Color = data.Stain2Color;
BackColor = data.BackgroundColor;
if data.Cooperative
    %%%
    %%% Pick the two most-different channels
    %%%
    Diff = abs(Stain1Color/sum(Stain1Color) - Stain2Color/sum(Stain2Color))*3;
    DiffMin = min(Diff(:));
    Channel = find(Diff ~= DiffMin);
    %%%
    %%% Find the amount of blue in the background color
    %%%
    Stain1Background =...
        BackColor(Channel(1)) * Stain2Color(Channel(2))+...
        BackColor(Channel(2)) * Stain2Color(Channel(1));
    Stain2Background =...
        BackColor(Channel(1)) * Stain1Color(Channel(2))+...
        BackColor(Channel(2)) * Stain1Color(Channel(1));
    Stain1Image = zeros(size(data.InputImage,1),size(data.InputImage,2));
    Stain2Image = zeros(size(data.InputImage,1),size(data.InputImage,2));
    for i =1:2
        i1=abs(i-2)+1; % The other channel
        InputImage = data.InputImage(:,:,Channel(i));
        Stain1Image = Stain1Image + (InputImage .* Stain2Color(Channel(i1)));
        Stain2Image = Stain2Image + (InputImage .* Stain1Color(Channel(i1)));
    end
    Stain1Image = Stain1Image - Stain1Background;
    Stain2Image = Stain2Image - Stain2Background;
    Stain1Image(Stain1Image<0) = 0;
    Stain2Image(Stain2Image<0) = 0;
    MaxStain1Image = max(Stain1Image(:));
    MaxStain2Image = max(Stain2Image(:));
    if MaxStain1Image > 0
        Stain1Image = Stain1Image ./ max(Stain1Image(:));
    end
    if MaxStain2Image > 0
        Stain2Image = Stain2Image ./ max(Stain2Image(:));
    end
else
    Diff1 = 3*(Stain1Color/sum(Stain1Color) - Stain2Color/sum(Stain2Color));
    Diff2 = -Diff1;
    Stain1Background = dot(BackColor,Diff1);
    Stain2Background = dot(BackColor,Diff2);
    Stain1Image = zeros(size(data.InputImage,1),size(data.InputImage,2));
    Stain2Image = zeros(size(data.InputImage,1),size(data.InputImage,2));
    for i=1:3
        InputImage = data.InputImage(:,:,i);
        Stain1Image = Stain1Image + (InputImage .* Diff1(i));
        Stain2Image = Stain2Image + (InputImage .* Diff2(i));
    end
    Stain1Image = Stain1Image-Stain1Background;
    Stain2Image = Stain2Image-Stain2Background;
    Stain1Image(Stain1Image<0) = 0;
    Stain2Image(Stain2Image<0) = 0;
    Stain1Image = Stain1Image ./ max(Stain1Image(:));
    Stain2Image = Stain2Image ./ max(Stain2Image(:));
end