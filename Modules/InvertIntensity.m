function handles = InvertIntensity(handles)

% Help for the Invert Intensity module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Converts the intensities of an image so that black becomes white.
% *************************************************************************
%
% This module takes an image and inverts it, i.e, makes a negative. In the 
% inversion of a binary image, black becomes white and white becomes
% black.  In the case of a grayscale or truecolor image, dark areas become 
% lighter and light areas become darker.

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

%textVAR01 = What did you call the image to be inverted (made negative)?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the inverted image?
%defaultVAR02 = InvertedBlue
%infotypeVAR02 = imagegroup indep
InvertedImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a
%%% variable.
OrigImage = CPretrieveimage(handles,ImageName,ModuleName,'MustBeGray','CheckScale');

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Inverts the image.
InvertedImage = imcomplement(OrigImage);

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    %%% Activates the appropriate figure window.
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure(OrigImage,'TwoByOne',ThisModuleFigureNumber)
    end
    %%% A subplot of the figure window is set to display the original image.
    hAx=subplot(2,1,1,'Parent',ThisModuleFigureNumber); 
    CPimagesc(OrigImage,handles,hAx); 
    title(hAx,['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the Inverted
    %%% Image.
    hAx=subplot(2,1,2,'Parent',ThisModuleFigureNumber); 
    CPimagesc(InvertedImage,handles,hAx); 
    title(hAx,'Inverted Image');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the Inverted image to the
%%% handles structure so it can be used by subsequent modules.
handles.Pipeline.(InvertedImageName) = InvertedImage;