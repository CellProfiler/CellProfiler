function handles = ConvertToImage(handles)

% Help for the Convert To Image module:
% Category: Object Processing
%
% SHORT DESCRIPTION:
% Converts objects you have identified into an image so that it can be
% saved with the Save Images module.
% *************************************************************************
%
% Settings:
%
% Binary (black & white), grayscale, or color: Choose how you would like
% the objects to appear. Color allows you to choose a colormap which will
% produce jumbled colors for your objects. Grayscale will give each object
% a graylevel pixel intensity value corresponding to its number (also
% called label), so it usually results in objects on the left side of the
% image being very dark, and progressing towards white on the right side of
% the image. You can choose "Color" with a "Gray" colormap to produce jumbled
% gray objects.
%
% Colormap:
% Affect how the objects are colored. You can look up your default colormap
% under File > Set Preferences.  Look in matlab help online (try Google) to
% see what the following available colormaps look like:
% Jet
% HSV
% Hot
% Cool
% Spring
% Summer
% Autumn
% Winter
% Gray
% Bone
% Copper
% Pink
% Lines
% Colorcube
% Flag
% Prism
% White

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
%   Wyman Li
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the objects you want to convert to an image?
%infotypeVAR01 = objectgroup
%inputtypeVAR01 = popupmenu
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = What do you want to call the resulting image?
%defaultVAR02 = CellImage
%infotypeVAR02 = imagegroup indep
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = What colors should the resulting image use?
%choiceVAR03 = Color
%choiceVAR03 = Binary (black & white)
%choiceVAR03 = Grayscale
%inputtypeVAR03 = popupmenu
ImageMode = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = For COLOR, what do you want the colormap to be?
%defaultVAR04 = Default
ColorMap = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

LabelMatrixImage = handles.Pipeline.(['Segmented' ObjectName]);
if strcmp(ImageMode,'Binary (black & white)')
    Image = logical(LabelMatrixImage ~= 0);
elseif strcmp(ImageMode,'Grayscale')
    Image = double(LabelMatrixImage / max(max(LabelMatrixImage)));
elseif strcmp(ImageMode,'Color')
    if strcmp(ColorMap,'Default')
        Image = CPlabel2rgb(handles,LabelMatrixImage);
    else
        try
            cmap = eval([ColorMap '(max(64,max(LabelMatrixImage(:))))']);
        catch
            error(['Image processing was canceled in the ', ModuleName, ' module because the ColorMap, ' ColorMap ', that you entered, is not valid.']);
        end
        Image = label2rgb(LabelMatrixImage,cmap,'k');
    end
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    ColoredLabelMatrixImage = CPlabel2rgb(handles,LabelMatrixImage);

    CPfigure(handles,'Image',ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure(LabelMatrixImage,'TwoByOne',ThisModuleFigureNumber)
    end
    subplot(2,1,1);
    CPimagesc(ColoredLabelMatrixImage,handles);
    title('Original Identified Objects');
    subplot(2,1,2);
    CPimagesc(Image,handles);
    title('New Image');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

handles.Pipeline.(ImageName) = Image;