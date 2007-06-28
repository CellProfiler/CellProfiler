function HelpMatlabCrash

% Troubleshooting frequent crashes using CellProfiler Developer's version
% on the Mac:
%
% If you're running CellProfiler on MATLAB Service Pack 2 (R14SP2) with Mac
% OS X 10.4, MATLAB often crashes with Java errors. The only fix is to
% upgrade MATLAB to Service Pack 3(R14SP3). For more information about
% this, please visit the Mathworks website at: 
% http://www.mathworks.com/support

helpdlg(help('HelpMatlabCrash'))

% We have one line of actual code in these files so that the help is
% visible. We are not using CPhelpdlg because using helpdlg instead allows
% the help to be accessed from the command line of MATLAB. The one line of
% code in each help file (helpdlg) is never run from inside CP anyway.