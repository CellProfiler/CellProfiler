function HelpAnalyzeImages

% ANALYZE IMAGES:
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