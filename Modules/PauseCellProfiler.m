function handles = PauseCellProfiler(handles)
% Help for the PauseCP module:
% Category: File Processing
%
% SHORT DESCRIPTION:
% Pauses CellProfiler interactively.
% *************************************************************************

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
% $Revision: 4436 $

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = This module pauses CellProfiler until the dialog box is clicked.

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%% ButtonName = questdlg(Question, Title, Btn1, Btn2,..., DEFAULT);
ButtonName = CPquestdlg('Continue previous module?','PauseCP','Continue','Cancel','Continue');
%% TODO - add Modify
% ButtonName = CPquestdlg('Continue or Modilfy previous module?','PauseCP','Continue','Modify','Cancel','Continue');

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% The figure window display is unnecessary for this module, so the figure
%%% window is closed the first time through the module.
%%% Determines the figure number.
ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
%%% Closes the window if it is open.
if any(findobj == ThisModuleFigureNumber)
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        close(ThisModuleFigureNumber)
    end
end

switch ButtonName
    case 'Continue'
        return
        %% TODO - add Modify
%     case 'Modify'
%         handles.Current.CurrentModuleNumber = num2str(str2num(handles.Current.CurrentModuleNumber) - 1);
%         set(cat(2,handles.VariableBox{:}),'enable','on','foregroundcolor','black'); %% re-enable variable boxes
%         
%         XX adding this feature
        
    case 'Cancel'
        %% TODO: This should simply stop processing, not error.
        error('CellProfiler stopped by user.');
end