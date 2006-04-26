function HelpDefaultOutputFolder
helpdlg(help('HelpDefaultOutputFolder'))

% Help for the default output folder, in the main CellProfiler window:
% Select the main folder where you want CellProfiler's output to be saved.
% Use the Browse button to select the folder, or carefully type the full
% pathname in the box. You can change the folder which appears upon
% CellProfiler startup by using File > Set Preferences.
% 
% You will have the option to save output to other locations: for example,
% the output file can be saved elsewhere by typing a full pathname in the
% 'Name the output file' box, and many modules (like Save Images) allow you
% to override the default output folder by entering the pathname in the
% settings.

% We have one line of actual code in these files so that the help is
% visible. We are not using CPhelpdlg because using helpdlg instead allows
% the help to be accessed from the command line of MATLAB. The one line of
% code in each help file (helpdlg) is never run from inside CP anyway.