function handles = Multiply(handles)

% Help for the Multiply module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% This module multiplies two images.  
%
%
%
% See also SubtractBackground, RescaleIntensity.
% 
% *************************************************************************
%
% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% This module was contributed to CellProfiler by Silvia Fiorentini, 
% University of Milan (silviafiorentini@dti.unimi.it).
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

%textVAR01 = What is the name of the first image you would like to use? 
%infotypeVAR01 = imagegroup
ImageName1 = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What is the name of the second image you would like to use?
%infotypeVAR02 = imagegroup
ImageName2 = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 = What do you want to call the resulting image? 
%defaultVAR03 = MultipliedImage
%infotypeVAR03 = imagegroup indep
MultiplyImageName = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%%% Reads the images.
OrigImage1 = CPretrieveimage(handles,ImageName1,ModuleName,'DontCheckColor','CheckScale');
OrigImage2 = CPretrieveimage(handles,ImageName2,ModuleName,'DontCheckColor','CheckScale');

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow
FinalImage = immultiply(OrigImage1,OrigImage2);

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines the figure number to display in.
ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    %%% Activates the appropriate figure window.
    CPfigure(handles,'Image',ThisModuleFigureNumber);

    %%% A subplot of the figure window is set to display the original image.
    hAx=subplot(2,2,1,'Parent',ThisModuleFigureNumber);
    CPimagesc(OrigImage1,handles,hAx);
    title(hAx,['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    hAx=subplot(2,2,2,'Parent',ThisModuleFigureNumber);
    CPimagesc(OrigImage2,handles,hAx);
    title(hAx,['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the adjusted
    %%%  image.
    hAx=subplot(2,2,3,'Parent',ThisModuleFigureNumber);
    CPimagesc(FinalImage,handles,hAx);
    title(hAx,'Multiplied Image');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

handles.Pipeline.(MultiplyImageName) = FinalImage;
