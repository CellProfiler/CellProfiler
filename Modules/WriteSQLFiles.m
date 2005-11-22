function handles = WriteSQLFiles(handles)

% Help for the Write SQL Files module:
% Category: File Processing
%
% SHORT DESCRIPTION:
% Exports data in database readable format, including an importing file
% with column names.
% *************************************************************************
%
% This module exports measurements to a SQL compatible format.
% It creates a MySQL script and associated data files. It calls
% the ExportSQL data tool.
%
% This module must be run at the end of a pipeline, or second to last if
% you are using the CreateBatchScripts module.
%
% See also: CREATEBATCHSCRIPTS.

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

%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);
ModuleName = char(handles.Settings.ModuleNames(CurrentModuleNum));

%pathnametextVAR01 = Enter the directory where the SQL files are to be saved.  Type period (.) to use the default output folder.
DataPath = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = What is the name of the database to use?
%defaultVAR02 = DefaultDB
DatabaseName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = What prefix should be used to name the SQL files?
%defaultVAR03 = SQL_
FilePrefix = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = What prefix should be used to name the SQL Tables in the database (should be unique per experiment)?
%defaultVAR04 =
TablePrefix = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%%%VariableRevisionNumber = 3

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

if handles.Current.NumberOfModules == 1
    error(['Image processing was canceled in the ', ModuleName, ' module because there are no other modules in the pipeline. Probably you should use the ExportSQL data tool.']);
elseif handles.Current.NumberOfModules == 2
    if ~isempty((strmatch('CreateBatchScripts',handles.Settings.ModuleNames)))
        error(['Image processing was canceled in the ', ModuleName, ' module because there are no modules in the pipeline other than CreateBatchScripts. Probably you should use the ExportSQL data tool.']);
    end
end

if CurrentModuleNum ~= handles.Current.NumberOfModules
    if isempty((strmatch('CreateBatchScripts',handles.Settings.ModuleNames))) || handles.Current.NumberOfModules ~= CurrentModuleNum+1
        error([ModuleName, ' must be the last module in the pipeline, or second to last if CreateBatchScripts is in the pipeline.']);
    end
end;

if strncmp(DataPath, '.',1)
    if length(DataPath) == 1
        DataPath = handles.Current.DefaultOutputDirectory;
    else
        DataPath = fullfile(handles.Current.DefaultOutputDirectory,DataPath(2:end));
    end
end

%% Two possibilities: we're at the end of the pipeline in an
%% interactive session, or we're in the middle of batch processing.

if isfield(handles.Current, 'BatchInfo'),
    FirstSet = handles.Current.BatchInfo.Start;
    LastSet = handles.Current.BatchInfo.End;
else
    FirstSet = 1;
    LastSet = handles.Current.NumberOfImageSets;
end
DoWriteSQL = (handles.Current.SetBeingAnalyzed == LastSet);

% Special case: We're writing batch files, and this is the first cycle.
if (strcmp(handles.Settings.ModuleNames{end},'CreateBatchScripts') && (handles.Current.SetBeingAnalyzed == 1)),
    DoWriteSQL = 1;
    FirstSet = 1;
    LastSet = 1;
end

if DoWriteSQL,
    % Initial checking of variables
    if isempty(DataPath)
        error(['Image processing was canceled in the ', ModuleName, ' module because no folder was specified.']);
    elseif ~exist(DataPath,'dir')
        error(['Image processing was canceled in the ', ModuleName, ' module because the specified folder could not be found.']);
    end
    if isempty(DatabaseName)
        error(['Image processing was canceled in the ', ModuleName, ' module because no database was specified.']);
    end
    %%% This is necessary to make sure the export works with the last
    %%% set.  Otherwise, the TimeElapsed array is missing the last
    %%% element.  The corresponding 'tic' is in CellProfiler.m.
    handles.Measurements.Image.TimeElapsed{handles.Current.SetBeingAnalyzed} = toc;
    CPConvertSQL(handles, DataPath, FilePrefix, DatabaseName, TablePrefix, FirstSet, LastSet,'');
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% The figure window display is unnecessary for this module, so the figure
%%% window is closed if it was previously open.
%%% Determines the figure number.
fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
%%% If the window is open, it is closed.
if any(findobj == ThisModuleFigureNumber)
    delete(ThisModuleFigureNumber)
end