function HelpDefaultImageDirectory
helpdlg(help('HelpDefaultImageDirectory'))

% Help for the default image directory:
% Select the main folder containing the images you want to analyze.
% You will have the option within load images modules to retrieve
% images from more than one folder, but the folder selected here will
% be the default.  Use the Browse button to select the folder, or
% carefully type the full pathname in the box to the right.
%
% Be careful that files other than your images of interest are not
% stored within the folder you have selected.  The following file
% types are ignored by CellProfiler, so these are the only types which
% can be left in the folder with the images you want to analyze:  
%    Folders, .m, .mat, .m~, .frk~, .xls, .doc, .txt, .csv, or any file
%    beginning with a dot.
% 
% If you would like to add a particular file format to this list,
% first save a copy of the main CellProfiler program (CellProfiler.m)
% in a separate location as a backup in case you make an error*, then
% go to File > Open and select CellProfiler.m.  Find the line
% that looks like the following and add any extensions:
%
% DiscardsByExtension = regexpi(FileNamesNoDir, '\.(m|mat|m~|frk~|xls|doc|txt|csv)$', 'once');
% 
% Save the file.  You do not need to relaunch Matlab or CellProfiler
% for this change to take effect.
% 
% * Note that CellProfiler will not run properly if you save
% CellProfiler.m under a different name, due to the interaction
% between the CellProfiler.m and CellProfiler.fig files.
%
% FILE LIST WINDOW:
% The window to the right displays the files in the default image
% directory.  This allows you to check file names or look at the
% order of images from within CellProfiler.  Selecting file names in
% this list does not do anything.