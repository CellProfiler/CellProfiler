function HelpDefaultImageFolder
helpdlg(help('HelpDefaultImageFolder'))

% Help for the default image folder, in the main CellProfiler window:
% Select the main folder containing the images you want to analyze.  Use
% the Browse button to select the folder, or carefully type the full
% pathname in the box. You can change the folder which is the
% default image folder upon CellProfiler startup by using File > Set
% Preferences.
%
% The contents of the folder are shown to the left, which allows you to
% check file names or look at the order of images from within CellProfiler.
% Doubleclicking image file names in this list will open them.
% Doubleclicking on PIPE or OUT files will ask if you want to load a
% pipeline from the file. To refresh the contents of this window, press
% enter in the default image directory edit box.
% 
% You will have the option within the Load Images module to retrieve images
% from other folders, but the folder selected here will be the default.
%
% Be careful that files other than your images of interest are not stored
% within the folder you have selected.  The following file extensions are
% ignored by CellProfiler, so these are the only types which can be left in
% the folder with the images you want to analyze:  
%   m, m~, frk~, xls, doc, rtf, txt, csv, or any file beginning with a dot.
% 
%
%
% CellProfiler Developer's version note: If you would like to add a
% particular file format to this list, first save a copy of the main
% CellProfiler program (CellProfiler.m) in a separate location as a backup
% in case you make an error, then go to File > Open and select
% CellProfiler.m.  Find the line that looks like the following and add any
% extensions:
%   DiscardsByExtension = regexpi(FileNamesNoDir, '\.(m|mat|m~|frk~|xls|
%                                        doc|rtf|txt|csv)$', 'once');
% Save the file.  You do not need to relaunch Matlab or CellProfiler for
% this change to take effect.

%%% We are not using CPhelpdlg because this allows the help to be accessed
%%% from the command line of Matlab. The code of theis module (helpdlg) is
%%% never run from inside CP anyway.