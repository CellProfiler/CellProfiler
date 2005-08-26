function HelpPreferences
helpdlg(help(HelpPreferences))

% SET PREFERENCES: 
% This allows you to set the default pixel size, the default font size
% for CellProfiler displays, the default image directory, the default
% output directory, and the directory containing CellProfiler modules.
% This is just for convenience, and can be reset later, if you would
% like.  This step creates a file called CellProfilerPreferences.mat
% in the directory which is loaded every time you start up
% CellProfiler. If you do not have permission to write files to the
% root directory of Matlab, it saves the file in the current
% directory, but then the defaults will only be used when
% CellProfiler is launched from that directory. If you do not have
% write permission in either location, you are out of luck.