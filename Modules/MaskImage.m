function handles = MaskImage(handles)

% Help for the Mask Image module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Masks image and saves it for future use.
% *************************************************************************
%
% This module masks an image and saves it in the handles structure for
% future use. The masked image is based on the original image and the
% object selected. 
% 
% Note that the image saved for further processing downstream is grayscale.
% If a binary mask is desired in subsequent modules, you might be able to 
% access ['CropMask',MaskedImageName] (e.g. 'CropMaskMaskBlue'), or simply
% use the ApplyThreshold module instead of MaskImage.

% See also IdentifyPrimAutomatic, IdentifyPrimManual.

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

%textVAR01 = From which object would you like to make a mask?
%choiceVAR01 = Image
%infotypeVAR01 = objectgroup
%inputtypeVAR01 = popupmenu
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = Which image do you want to mask?
%infotypeVAR02 = imagegroup
%inputtypeVAR02 = popupmenu
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = What do you want to call the masked image?
%defaultVAR03 = MaskBlue
%infotypeVAR03 = imagegroup indep
MaskedImageName = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%%%VariableRevisionNumber = 2

%%%%%%%%%%%%%%%%
%%% ANALYSIS %%%
%%%%%%%%%%%%%%%%

OrigImage = CPretrieveimage(handles,ImageName,ModuleName,'MustBeGray','CheckScale');

ObjectLabelMatrix = handles.Pipeline.(['Segmented',ObjectName]);

ObjectLabelMatrix(ObjectLabelMatrix>0)=1;

handles.Pipeline.(MaskedImageName)=OrigImage;
handles.Pipeline.(['CropMask',MaskedImageName])=ObjectLabelMatrix;


%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow


ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
%%% Check whether that figure is open. This checks all the figure handles
%%% for one whose handle is equal to the figure number for this module.
if any(findobj == ThisModuleFigureNumber)
    %%% Activates the appropriate figure window.
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure(OrigImage,'TwoByOne',ThisModuleFigureNumber)
    end
    
    %%% A subplot of the Original image.
    subplot(2,1,1)
    CPimagesc(OrigImage,handles);
    title(['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);

    %%% A subplot of the Masked image.
    subplot(2,1,2)
    CPimagesc(ObjectLabelMatrix,handles);
    title('Image Mask');
end