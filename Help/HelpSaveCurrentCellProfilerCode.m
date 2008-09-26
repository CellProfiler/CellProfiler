function HelpSaveCurrentCellProfilerCode

% File > Save Current CellProfiler code.
%
% This is only to be used in CellProfiler Developer's version.
% It allows you to save all the Modules, DataTools, ImageTools and 
% CPsubfunctions at the current revision as a ZIP file. This file is then
% placed in the default output directory.

% $Revision$

helpdlg(help('HelpSaveCurrentCellProfilerCode'))

% We have one line of actual code in these files so that the help is
% visible. We are not using CPhelpdlg because using helpdlg instead allows
% the help to be accessed from the command line of MATLAB. The one line of
% code in each help file (helpdlg) is never run from inside CP anyway.