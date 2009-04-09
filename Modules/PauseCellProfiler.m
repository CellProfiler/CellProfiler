function handles = PauseCellProfiler(handles)
% Help for the PauseCP module:
% Category: Other
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
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% $Revision$

% MBray 2009_03_20: Comments on variables for pyCP upgrade
% If debugging/history is implemented in pyCP, this module will probably be
% removed. 
% Anne 4-9-09: Anne votes that we should keep this module. It is
% pretty simple, and the person might want to save the pausing in the
% pipeline itself (for re-loading later) whereas debugging marks are
% session-specific. It might be rare that someone really uses this module
% but it doesn't hurt to make it available because it doesn't clutter very
% much and has a clear purpose/name.


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


%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% The figure window display is unnecessary for this module, so the figure
%%% window is closed the first time through the module.
CPclosefigure(handles,CurrentModule)

drawnow

ButtonName = CPquestdlg({'Continue processing?';'';'Note: Press Ctrl+C to interact with CellProfiler windows while paused.'},...
    'PauseCP','Continue','Cancel','Continue');

switch ButtonName
    case 'Continue'
        return

    case 'Cancel'

        %%% This should cause a cancel so no further processing is done
        %%% on this machine.
        set(handles.timertexthandle,'string','Canceling after current module')
end
