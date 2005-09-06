function handles = ConvertToImage(handles)

% Help for the Quick Fix Convert Objects To Image module:
% Category: Object Processing
%
% This module hasn't really been written yet, much less documented.
% 
% See also <nothing relevant>

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne Carpenter
%   Thouis Jones
%   In Han Kang
%
% $Revision$

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%

%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

%textVAR01 = What did you call the objects you want to process?
%infotypeVAR01 = objectgroup
%inputtypeVAR01 = popupmenu
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = What do you want to call the resulting image?
%defaultVAR02 = CellImage
%infotypeVAR02 = imagegroup indep
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Do you want the image to be (Binary is the only option that works right now)
%choiceVAR03 = Binary (black or white)
%choiceVAR03 = Grayscale
%choiceVAR03 = RGB
%inputtypeVAR03 = popupmenu
ImageMode = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%%%VariableRevisionNumber = 1

%%%%%%%%

LabelMatrixImage = handles.Pipeline.(['Segmented' ObjectName]);
handles.Pipeline.(ImageName) = double(LabelMatrixImage > 0);

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%

if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
    %%% The figure window display is unnecessary for this module, so the figure
    %%% window is closed the first time through the module.
    %%% Determines the figure number.
    fieldname = ['FigureNumberForModule',CurrentModule];
    ThisModuleFigureNumber = handles.Current.(fieldname);
    %%% Closes the window if it is open.
    if any(findobj == ThisModuleFigureNumber) == 1;
        close(ThisModuleFigureNumber)
    end
    drawnow
end