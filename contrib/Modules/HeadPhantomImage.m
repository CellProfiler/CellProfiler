function handles = HeadPhantomImage(handles)

% Help for the HeadPhantomImage module:
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
% Authors:
%   Paolo Arcaini
%   Massimo Manara
%
% Website: http://www.cellprofiler.org
%
% DATA:

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the image to be corrected? 
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the corrected image? 
%defaultVAR02 = HeadPhantomImage
%infotypeVAR02 = imagegroup indep
CorrectedImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Type of head phantom to generate: 

%textVAR04 = What type do you want to perform? 
%choiceVAR04 = Shepp-Logan
%choiceVAR04 = Modified Shepp-Logan
%inputtypeVAR04 = popupmenu
Functions{1} = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = n: 
%defaultVAR05 = 256
n = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,5}));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a
%%% variable.
OrigImage = CPretrieveimage(handles,ImageName,ModuleName,'DontCheckColor','CheckScale');

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow
if strcmp(Functions{1}, 'Shepp-Logan')
    CorrectedImage = phantom('Shepp-Logan',n);
else
     CorrectedImage = phantom('Modified Shepp-Logan',n);
end


    %%%%%%%%%%%%%%%%%%%%%%%
    %%% DISPLAY RESULTS %%%
    %%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines the figure number to display in.
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
    %%% A subplot of the figure window is set to display the adjusted
    %%%  image.
    subplot(2,1,2);
    CPimagesc(CorrectedImage,handles);
    title('Phantom Image');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the corrected image to the handles structure so it can be used by
%%% subsequent modules.
handles.Pipeline.(CorrectedImageName) = CorrectedImage;
