function handles = SelectDirectoriesToProcess(handles)

% Help for the SelectDirectoriesAndFiles module:
% Category: File Processing
%
% SHORT DESCRIPTION:
% Given a directory (or subdirectory structure) of input images, exclude
% directories.
% *************************************************************************
%
% This module is to be used for excluding directories from
% later consideration.
%
% This module MUST be placed (1) after the final LoadImages module for the
% pipeline and (b) before any processing or measurement modules, since it 
% trims the list of files created by LoadImages and passes the trimmed list
% to the rest of the pipeline for processing.
%
% See also LoadImages.

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

%textVAR01 = This module presents the user with a dialog box showing the directory tree. The user may then select/de-select which directories to process downstream. Execution of the pipeline is paused until the dialog box is closed by clicking 'OK' or 'Cancel.'

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow;

% Determines which cycle is being analyzed.
SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;

% Trimming the list of files to be analyzed occurs only the first time
% through this module.
if SetBeingAnalyzed ~= 1, return; end

% There MUST be a LoadImages module immediately before this one
idxLoadImages = strmatch('LoadImages',handles.Settings.ModuleNames);
if isempty(idxLoadImages) || max(idxLoadImages)+1 ~= CurrentModuleNum,
    error([ModuleName, ' must be placed immediately after the last LoadImages module in the pipeline.']);
end

fn = fieldnames(handles.Pipeline);
prefix = 'filelist';
fn = fn{strncmpi(fn,prefix,length(prefix))};
if iscell(fn), fn = fn{1}; end; % There may be several FileLists, but since they are the same, we just need the first one
pathnames = cellfun(@fileparts,handles.Pipeline.(fn),'UniformOutput',false);
uniquePathsBefore = unique(pathnames);

% Call the main subfunction
handles = CPselectdirectoriestoprocess(handles);

pathnames = cellfun(@fileparts,handles.Pipeline.(fn),'UniformOutput',false);
uniquePathsAfter = unique(pathnames);

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
% If the figure window isn't open, skip everything.
if any(findobj == ThisModuleFigureNumber)
    % Create a text string that contains the output info
    TextString{1} = ['Information on directory listing:'];
    TextString{end+1} = [' Number of directories found: ',num2str(length(uniquePathsBefore))];
    TextString{end+1} = [' Number of directories excluded: ',num2str(length(uniquePathsBefore) - length(uniquePathsAfter))];
    TextString{end+1} = '';
    
    % Create figure and display list
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure('','NarrowText',ThisModuleFigureNumber)
    end    
    currentfig = CPfigure(handles,'Text',ThisModuleFigureNumber);
    for i = 1:length(TextString),
        uicontrol(currentfig,'style','text','units','normalized','fontsize',handles.Preferences.FontSize,'HorizontalAlignment','left','string',TextString{i},'position',[.05 .9-(i-1)*.04 .95 .04],'BackgroundColor',[.7 .7 .9]);
    end
end