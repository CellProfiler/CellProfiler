function handles = WriteSQLFiles(handles)

% Help for the Write SQL Files module:
% Category: Other
%
% This module exports measurements to a SQL compatible format.
% It creates a MySQL script and associated data files. It calls
% the ExportSQL data tool.
%
%
% It does not make sense to run this module in conjunction with other
% modules.  It should be the only module in the pipeline.
%
% See also: CREATEBATCHSCRIPTS.

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
%
% $Revision$

%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

%textVAR01 = Enter the directory where the SQL files are to be saved? Leave a period (.) to retrieve images from the default output directory #LongBox#
%defaultVAR01 = .
DataPath = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = Enter name of the SQL script. Script lines will be appended in this file. Make sure that a file with this name does not exist in the above directory already! #LongBox#
%defaultVAR02 = SQLScript.SQL
SQLScriptFileName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = What is the name of the database to use?
%defaultVAR03 = DefaultDB
DatabaseName = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%%%VariableRevisionNumber = 01

% This module should only be executed when all image sets have been processed
if handles.Current.SetBeingAnalyzed == handles.Current.NumberOfImageSets

    % Initial checking of variables
    if isempty(DataPath)
        errordlg('No path specified in the WriteSQLFiles module.');
    elseif strcmp(DataPath,'.')
        DataPath = handles.Current.DefaultOutputDirectory;
    elseif ~exist(DataPath,'dir')
        errordlg('Cannot locate the specified directory in the WriteSQLFiles module.');
    end

    if isempty(SQLScriptFileName)
        errordlg('SQL script file name is not specified in the WriteSQLFiles module.');
    end

    if isempty(DatabaseName)
        errordlg('No database specified in the WriteSQLFiles module.');
    end

    % Store the variables in the handles structure so that they can be retrieved
    % by the ExportSQL export tool
    handles.DataPath = DataPath;
    handles.SQLScriptFileName = SQLScriptFileName;
    handles.DatabaseName = DatabaseName;

    % Call the ExportSQL function
    handles = ExportSQL2(handles);

    % Remove the variables from the handles structure
    handles = rmfield(handles,{'DataPath','SQLScriptFileName','DatabaseName'});
end

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%

%%% The figure window display is unnecessary for this module, so the figure
%%% window is closed if it was previously open.
%%% Determines the figure number.
fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
%%% If the window is open, it is closed.
if any(findobj == ThisModuleFigureNumber) == 1;
    delete(ThisModuleFigureNumber)
end

