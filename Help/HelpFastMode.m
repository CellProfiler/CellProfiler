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

%%% We are not using CPhelpdlg because this allows the help to be accessed
%%% from the command line of Matlab. The code of theis module (helpdlg) is
%%% never run from inside CP anyway.