function ManualCompiler %#ok We want to ignore MLint error checking for this line.

% Some Pseudo code to do this eventually:
% Open 
% AboutCP.m
% Open Table of Contents.
% Open CPInstallGuide.
% Help1-5.m and accompanying CellProfiler images.
% HelpExpo
% 
% CPOverviewX.tif, where X is 1 to the total number of Overview pages.
% Put each of these tifs on its own page in the manual, in order.
% Open each algorithm (alphabetically) and print its name in large bold font at the top of the page. 
% Extract the lines after "Help for ...." and before the license begins, using the Matlab 'help' function. Print this below the algorithm name.
% Extract the variables from the algorithm using the same code CP uses, and print this below the algorithm description.
% Open the corresponding tif image file (in the ExampleImages folder), if it exists, and place this at the bottom of the page.
% [somehow deal with it if there is too much stuff to fit on one page]
% Add page numbers throughout.
% Save as a pdf.
% 
% 
% TO DO:
% Wrap all comments.
% Crop module description is outdated.}