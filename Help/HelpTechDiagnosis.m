function HelpTechDiagnosis
helpdlg(help('HelpTechDiagnosis'))

% Technical diagnosis mode is available using File > Tech Diagnosis.
%
% This is only to be used in CellProfiler Developer's version.
% It allows you to access the workspace of CellProfiler directly at the
% command line of MATLAB, including looking into the handles structure.
%
% Type "return" at the command line of MATLAB to exit this mode.

% We have one line of actual code in these files so that the help is
% visible. We are not using CPhelpdlg because using helpdlg instead allows
% the help to be accessed from the command line of MATLAB. The one line of
% code in each help file (helpdlg) is never run from inside CP anyway.