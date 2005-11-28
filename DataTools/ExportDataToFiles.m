function ExportDataToFiles (handles)

% Help for the ExportDataToFiles tool:
% Category: Data Tools
%
% This tool will export your data to either an excel file or a CSV file for
% SQL databases.

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
% $Revision: 2535 $

ButtonSelected = CPquestdlg('What format do you want to export to?', 'Select Export Type', 'Excel', 'SQL', 'Cancel', 'Excel');
if strcmp(ButtonSelected, 'Excel')
    CPExportExcel(handles);
elseif strcmp(ButtonSelected, 'SQL')
    CPExportSQL(handles);
else
    return
end