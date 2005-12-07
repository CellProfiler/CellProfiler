function HelpAnalyzeImages
helpdlg(help('HelpAnalyzeImages'))

% Help for NAME THE OUTPUT FILE: 
% Type in the text you want to use to name the output file, which is
% where all of the information about the analysis as well as any
% measurements are stored. 'OUT.mat' will be added automatically at
% the end of whatever you type in the box. The file will be saved in
% the default output directory unless you type a full path and file
% name into the output file name box. The path must not have spaces or
% characters disallowed by your platform.
%
% The program prevents you from entering a name which, when 'OUT.mat'
% is appended, exists already. This prevents overwriting an output
% data file by accident.  It also prevents intentionally overwriting
% an output file for the following reason: when a file is
% 'overwritten', instead of completely overwriting the output file,
% Matlab just replaces some of the old data with the new data.  So, if
% you have an output file with 12 measurements and the new set of data
% has only 4 measurements, saving the output file to the same name
% would produce a file with 12 measurements: the new 4 followed by 8
% old measurements.
%
% Help for ANALYZE IMAGES:
% All of the images in the selected directory/directories will be
% analyzed using the modules and settings you have specified.  You
% will have the option to cancel at any time.  At the end of each data
% set, the data are stored in the output file.
%
% You can test an analysis on a single image set by setting the 'Load
% Images' modules appropriately.  For example, if using
% 'LoadImagesOrder' you can set the number of images per set to equal
% the total number of images in the folder (even if it is thousands)
% so that only the first set will be analyzed.  Or, if using
% 'LoadImagesText' you can make the identifying text specific enough
% that it will recognize only one image set in the folder.

%%% We are not using CPhelpdlg because this allows the help to be accessed
%%% from the command line of Matlab. The code of theis module (helpdlg) is
%%% never run from inside CP anyway.