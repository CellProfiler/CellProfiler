function handles = ExportToExcel(handles)

% Help for the ExportToExcel module:
% Category: File Processing
%
% SHORT DESCRIPTION:
% Exports measurements into a tab-delimited text file which can be opened
% in Excel or other spreadsheet programs.
% *************************************************************************
% Note: this module is beta-version and has not been thoroughly checked.
%
% The data will be converted to a tab-delimited text file which can be read
% by Excel, another spreadsheet program, or a text editor. The file is
% stored in the default output folder.
%
% This module performs the same function as the data tool, Export Data.
% Please refer to the help for ExportData.

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
% $Revision: 2989 $

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);%#ok Ignore MLint

%textVAR01 = Which objects do you want to export?
%infotypeVAR01 = objectgroup
%choiceVAR01 = Image
%choiceVAR01 = Experiment
Object{1} = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 =
%infotypeVAR02 = objectgroup
%choiceVAR02 = /
%choiceVAR02 = Image
%choiceVAR02 = Experiment
Object{2} = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 =
%infotypeVAR03 = objectgroup
%choiceVAR03 = /
%choiceVAR03 = Image
%choiceVAR03 = Experiment
Object{3} = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 =
%infotypeVAR04 = objectgroup
%choiceVAR04 = /
%choiceVAR04 = Image
%choiceVAR04 = Experiment
Object{4} = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%textVAR05 =
%infotypeVAR05 = objectgroup
%choiceVAR05 = /
%choiceVAR05 = Image
%choiceVAR05 = Experiment
Object{5} = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu

%textVAR06 =
%infotypeVAR06 = objectgroup
%choiceVAR06 = /
%choiceVAR06 = Image
%choiceVAR06 = Experiment
Object{6} = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%inputtypeVAR06 = popupmenu

%textVAR07 =
%infotypeVAR07 = objectgroup
%choiceVAR07 = /
%choiceVAR07 = Image
%choiceVAR07 = Experiment
Object{7} = char(handles.Settings.VariableValues{CurrentModuleNum,7});
%inputtypeVAR07 = popupmenu

%textVAR08 =
%infotypeVAR08 = objectgroup
%choiceVAR08 = /
%choiceVAR08 = Image
%choiceVAR08 = Experiment
Object{8} = char(handles.Settings.VariableValues{CurrentModuleNum,8});
%inputtypeVAR08 = popupmenu

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow
tmp = {};
for n = 1:8
    if ~strcmp(Object{n}, '/')
        tmp{end+1} = Object{n};
    end
end
Object = tmp;

if handles.Current.SetBeingAnalyzed == handles.Current.NumberOfImageSets
    RawPathname = handles.Current.DefaultOutputDirectory;
    ExportInfo.ObjectNames = unique(Object);
    ExportInfo.MeasurementExtension = '.xls';
    ExportInfo.MeasurementFilename = get(handles.OutputFileNameEditBox,'string');
    ExportInfo.IgnoreNaN = 1;
    ExportInfo.SwapRowsColumnInfo = 'No';
    ExportInfo.DataParameter = 'mean';
    handles.Measurements.Image.ModuleError{handles.Current.SetBeingAnalyzed}(1,CurrentModuleNum) = 0;
    CPwritemeasurements(handles,ExportInfo,RawPathname);
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines the figure number to display in.
ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    close(ThisModuleFigureNumber)
end