function DefaultImageDirectory

% Help for the default image directory:
% Select the main folder containing the images you want to analyze.
% You will have the option within load images modules to retrieve
% images from more than one folder, but the folder selected here will
% be the default folder from which images are retrieved and where the
% output file and output images will be saved.  Use the Browse button
% to select the folder, or carefully type the full pathname in the box
% to the right.
%
% Be careful that files other than your images of interest are not
% stored within the folder you have selected.  The following file
% types are ignored by CellProfiler, so these are the only types which
% can be left in the folder with the images you want to analyze:  
%           Folders, .mat, .m, .m~, .frk, .xls, .doc, .txt, or any file
%           beginning with a dot.
% 
% If you would like to add a particular file format to this list,
% first save a copy of the main CellProfiler program (CellProfiler.m)
% in a separate location as a backup in case you make an error*, then
% go to File > Open and select CellProfiler.m, then go to the lines
% that look like the following: 
% DiscardLogical2Pre = regexpi(FileNamesNoDir, '.mat$','once');
% if strcmp(class(DiscardLogical2Pre), 'cell') == 1 
% DiscardLogical2 = cellfun('prodofsize',DiscardLogical2Pre); 
% else DiscardLogical2 = []; 
% end
% 
% Copy that group of lines and rename things with a different number
% (e.g. 9). Replace “.mat” with the text you want to search for, then
% add DiscardLogical9Pre to the line: 
% DiscardLogical = DiscardLogical1 |
% DiscardLogical2 | DiscardLogical3 | DiscardLogical4 | DiscardLogical5 |
% DiscardLogical6 | DiscardLogical7 | DiscardLogical8;
% 
% Save the file.  You do not need to relaunch Matlab or CellProfilerTM
% for this change to take effect.
% 
% * Note that CellProfiler will not run properly if you save
% CellProfiler.m under a different name, due to the interaction
% between the CellProfiler.m and CellProfiler.fig files.
%
% FILE LIST WINDOW:
% There is a window which displays the files in the folder of images
% to be analyzed.  This allows you to check file names or look at the
% order of images from within CellProfiler.  Selecting file names in
% this list does not actually do anything.