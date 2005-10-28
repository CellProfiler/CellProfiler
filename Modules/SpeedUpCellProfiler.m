function handles = SpeedUpCellProfiler(handles)

% Help for the Speed Up CellProfiler module:
% Category: Other
%
% Allows faster image processing by refraining from saving the output
% file after every image set is processed. Instead, the output file is
% saved after every Nth image set (and always after the first and last
% image sets). For large output files, this can result in substantial
% time savings. The only disadvantage is that if processing is
% canceled prematurely, the output file will contain only data up to
% the last multiple of N, even if several image sets have been
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
%   Anne Carpenter <carpenter@wi.mit.edu>
%   Thouis Jones   <thouis@csail.mit.edu>
%   In Han Kang    <inthek@mit.edu>
%   Ola Friman     <friman@bwh.harvard.edu>
%   Steve Lowe     <stevelowe@alum.mit.edu>
%   Joo Han Chang  <joohan.chang@gmail.com>
%   Colin Clarke   <colinc@mit.edu>
%   Mike Lamprecht <mrl@wi.mit.edu>
%   Susan Ma       <xuefang_ma@wi.mit.edu>
%
% $Revision$

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow

%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);
ModuleName = char(handles.Settings.ModuleNames(CurrentModuleNum));

%textVAR01 = Output files should be saved every Nth image set (1,2,3,...  Default = 1). Note: the output file is always saved after the first or last image set is processed, no matter what is entered here.
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

try SaveWhen = str2num(SaveWhen);
catch error('The number of image sets must be entered as a number in the ', ModuleName, ' module.')
end
handles.Current.SaveOutputHowOften = SaveWhen;

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

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