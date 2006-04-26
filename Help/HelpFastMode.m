function HelpFastMode
helpdlg(help('HelpFastMode'))

% Fast mode can be set in File > Set preferences.
%
% If you uncheck the box you will run in diagnostic mode, where all the
% intermediate images and calculations for the most recent image cycle are
% saved in the output file, which drastically increases the output file
% size. Check the box if you would instead like to run in normal (fast)
% mode, producing smaller output files.
%
% See also the SpeedUpCellProfiler module.

% We have one line of actual code in these files so that the help is
% visible. We are not using CPhelpdlg because using helpdlg instead allows
% the help to be accessed from the command line of MATLAB. The one line of
% code in each help file (helpdlg) is never run from inside CP anyway.