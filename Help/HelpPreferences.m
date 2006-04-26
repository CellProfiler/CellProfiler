function HelpPreferences
helpdlg(help('HelpPreferences'))

% File > Set Preferences: The default preferences file (CellProfilerPreferences.mat) is 
% loaded upon starting up CellProfiler and is located in the folder where the software is
% running (or the MATLAB root in CellProfiler Developer's version). If you do not have 
% permission to write files in that location, it saves the file in the current folder 
% (Developer's version), but then the preferences will only be used when CellProfiler is 
% launched from that folder. If you do not have write permission in either location, save 
% them as personal preferences and use File > Load Preferences each time you start CellProfiler.

%%% NOTE: THE TEXT IS VERY WIDE HERE SO THAT IT FITS INTO THE SET
%%% PREFERENCES WINDOW.

% We have one line of actual code in these files so that the help is
% visible. We are not using CPhelpdlg because using helpdlg instead allows
% the help to be accessed from the command line of MATLAB. The one line of
% code in each help file (helpdlg) is never run from inside CP anyway.