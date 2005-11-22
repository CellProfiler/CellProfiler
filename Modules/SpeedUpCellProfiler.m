function handles = SpeedUpCellProfiler(handles)

% Help for the Speed Up CellProfiler module:
% Category: Other
%
% SHORT DESCRIPTION:
% Removes images from memory and/or overrides saving partial output files
% after every image cycle.
% *************************************************************************
%
% Allows faster image processing by refraining from saving the output
% file after every cycle is processed. Instead, the output file is
% saved after every Nth cycle (and always after the first and last
% cycles). For large output files, this can result in substantial
% time savings. The only disadvantage is that if processing is
% canceled prematurely, the output file will contain only data up to
% the last multiple of N, even if several cycles have been
% processed since then.
%
% See also <nothing relevant>.

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

%textVAR01 = Output files should be saved every Nth cycle (1,2,3,...  Default = 1). Note: the output file is always saved after the first or last cycle is processed, no matter what is entered here.
%defaultVAR01 = 1
SaveWhen = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = Which images would you like to save?
%choiceVAR02 = Do not use
%infotypeVAR02 = imagegroup
ImageNameList{1} = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 =
%choiceVAR03 = Do not use
%infotypeVAR03 = imagegroup
ImageNameList{2} = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 =
%choiceVAR04 = Do not use
%infotypeVAR04 = imagegroup
ImageNameList{3} = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%%%VariableRevisionNumber = 1

ListOfFields = fieldnames(handles.Pipeline);
tempPipe = handles.Pipeline;
for i = 1:length(ListOfFields)
    if all(size(tempPipe.(ListOfFields{i}))~=1) && ~any(strcmp(ImageNameList,ListOfFields{i}))
        tempPipe = rmfield(tempPipe,ListOfFields(i));
    end
end
handles.Pipeline = tempPipe;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

try SaveWhen = str2double(SaveWhen);
catch error(['Image processing was canceled in the ', ModuleName, ' module because the number of cycles must be entered as a number.'])
end
handles.Current.SaveOutputHowOften = SaveWhen;

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
    %%% The figure window display is unnecessary for this module, so the figure
    %%% window is closed the first time through the module.
    %%% Determines the figure number.
    fieldname = ['FigureNumberForModule',CurrentModule];
    ThisModuleFigureNumber = handles.Current.(fieldname);
    %%% Closes the window if it is open.
    if any(findobj == ThisModuleFigureNumber)
        close(ThisModuleFigureNumber)
    end
    drawnow
end