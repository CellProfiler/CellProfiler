function handles = SpeedUpCellProfiler(handles)

% Help for the Speed Up CellProfiler module:
% Category: Other
%
% SHORT DESCRIPTION:
% Speeds up CellProfiler processing and conserves memory.
% *************************************************************************
%
% Speeds up CellProfiler processing and conserves memory by reducing the
% frequency of saving partial output files and/or clearing the memory.
%
% Settings:
%
% Output files should be saved every Nth cycle?
% To save the output file after every cycle, as usual, leave this set to 1.
% Entering a larger integer allows faster image processing by refraining
% from saving the output file after every cycle is processed. Instead, the
% output file is saved after every Nth cycle (and always after the first
% and last cycles). For large output files, this can result in substantial
% time savings. The only disadvantage is that if processing is canceled
% prematurely, the output file will contain only data up to the last
% multiple of N, even if several cycles have been processed since then.
% Another hint: be sure you are not in Diagnostic mode (see File > Set
% Preferences) to avoid saving very large output files with intermediate
% images, because this slows down CellProfiler as well.
%
% Do you want to clear the memory?
% If yes, everything in temporary memory will be removed except for the
% images you specify. Therefore, only the images you specify will be
% accessible to modules downstream in the pipeline. This module can
% therefore be used to clear space in the memory. Note: currently, this
% option will remove everything in the memory, which may not be compatible
% with some modules, which often store non-image information in memory to
% be re-used during every cycle.

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

%textVAR01 = Output files should be saved every Nth cycle (1,2,3,...). Note: the output file is always saved after the first or last cycle is processed, no matter what is entered here.
%defaultVAR01 = 1
SaveWhen = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = Do you want to clear the memory (may not be compatible with some modules)?
%choiceVAR02 = Yes
%choiceVAR02 = No
ClearMemory = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 = If yes, which images would you like to remain in memory?
%choiceVAR03 = Do not use
%infotypeVAR03 = imagegroup
ImageNameList{1} = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 =
%choiceVAR04 = Do not use
%infotypeVAR04 = imagegroup
ImageNameList{2} = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%textVAR05 =
%choiceVAR05 = Do not use
%infotypeVAR05 = imagegroup
ImageNameList{3} = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu

%textVAR06 =
%choiceVAR06 = Do not use
%infotypeVAR06 = imagegroup
ImageNameList{4} = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%inputtypeVAR06 = popupmenu

%textVAR07 =
%choiceVAR07 = Do not use
%infotypeVAR07 = imagegroup
ImageNameList{5} = char(handles.Settings.VariableValues{CurrentModuleNum,7});
%inputtypeVAR07 = popupmenu

%textVAR08 =
%choiceVAR08 = Do not use
%infotypeVAR08 = imagegroup
ImageNameList{6} = char(handles.Settings.VariableValues{CurrentModuleNum,8});
%inputtypeVAR08 = popupmenu

%%%VariableRevisionNumber = 2

if strcmpi(ClearMemory,'Yes')
    ListOfFields = fieldnames(handles.Pipeline);
    tempPipe = handles.Pipeline;
    for i = 1:length(ListOfFields)
        if all(size(tempPipe.(ListOfFields{i}))~=1) && ~any(strcmp(ImageNameList,ListOfFields{i})) && ~iscell(tempPipe.(ListOfFields{i}))
            tempPipe = rmfield(tempPipe,ListOfFields(i));
        end
    end
    handles.Pipeline = tempPipe;
end

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

%%% The figure window display is unnecessary for this module, so it is
%%% closed during the starting image cycle.
if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
    ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
    if any(findobj == ThisModuleFigureNumber)
        close(ThisModuleFigureNumber)
    end
end