function HelpMatlabCrash
helpdlg(help('HelpMatlabCrash'))

% Troubleshooting frequent crashes using CellProfiler Developer's version
% on the Mac:
%
% If you're running CellProfiler on MatLab Service Pack 2 (R14SP2) with Mac
% OS X 10.4, Matlab often crashes with Java errors. The only fix is to
% upgrade Matlab to Service Pack 3(R14SP3). For more information about
% this, please visit the Mathworks website at: 
% http://www.mathworks.com/support

%%% We are not using CPhelpdlg because this allows the help to be accessed
%%% from the command line of Matlab. The code of theis module (helpdlg) is
%%% never run from inside CP anyway.