function HelpSkipErrors
helpdlg(help('HelpSkipErrors'))

% Skip errors mode can be set in File > Set preferences.
%
% This option will allow you to skip modules which have produced errors. If
% a module fails, the pipeline will continue to run. To check if any
% modules have failed, use Data Tools -> ExportData and be sure to export
% the Image data. In the resulting Image file, there will be one
% ModuleError field for each module. If any of these values are above 0,
% that means the module failed at some point in the analysis.

% We have one line of actual code in these files so that the help is
% visible. We are not using CPhelpdlg because using helpdlg instead allows
% the help to be accessed from the command line of MATLAB. The one line of
% code in each help file (helpdlg) is never run from inside CP anyway.