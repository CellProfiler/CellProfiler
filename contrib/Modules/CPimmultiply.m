function handles = CPimmultiply(handles)

% Help for the CPimmultiply module:
% Category: Contributed
%
% SHORT DESCRIPTION:
% 
% *************************************************************************
%
% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
%
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% DATA:

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = First group of images 
%infotypeVAR01 = imagegroup
ImageName1 = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = Second group of images 
%infotypeVAR02 = imagegroup
ImageName2 = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 = What do you want to call the resulting image? 
%defaultVAR03 = MultiplyImageName
%infotypeVAR03 = imagegroup indep
MultiplyImageName = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%%% Reads the images.
OrigImage1 = CPretrieveimage(handles,ImageName1,ModuleName,'DontCheckColor','CheckScale');
OrigImage2 = CPretrieveimage(handles,ImageName2,ModuleName,'DontCheckColor','CheckScale');

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
%     if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
%         CPresizefigure(OrigImage1,'TwoByOne',ThisModuleFigureNumber)
%     end
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,2,1);
    CPimagesc(OrigImage1,handles);
    title(['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    subplot(2,2,2);
    CPimagesc(OrigImage2,handles);
    title(['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the adjusted
    %%%  image.
    subplot(2,2,3);
    CPimagesc(FinalImage,handles);
    title('Multiply Image');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

handles.Pipeline.(MultiplyImageName) = FinalImage;
