function HelpTechDiagnosis
helpdlg(help('HelpTechDiagnosis'))

% Technical diagnosis mode is available using File > Tech Diagnosis.
%
% This is only to be used in CellProfiler Developer's version.
% It allows you to access the workspace of CellProfiler directly at the
% command line of Matlab, including looking into the handles structure.
%
% Type "return" at the command line of Matlab to exit this mode.

%%% We are not using CPhelpdlg because this allows the help to be accessed
%%% from the command line of Matlab. The code of theis module (helpdlg) is
%%% never run from inside CP anyway.