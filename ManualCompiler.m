function ManualCompiler %#ok We want to ignore MLint error checking for this line.

% 1. Cover page (made in powerpoint - would be nice if we could
% extract directly from that, but if not I saved it as TIF)

% 2. About CellProfiler page (credits) (made in powerpoint - would be
% nice if we could extract directly from that, but if not I saved it
% as TIF)

% 3. List of modules - Retrieve the list of files starting with Alg;
% each module is annotated on the line "Help for the X module: (Y)",
% so if you search for the text after the :, you will find one of four
% categories: Object Identification, Measurement, Pre-processing, File
% Handling.  I would like these four headings to be listed in that
% order, with the list of available algorithms in each category
% following its heading.  This need not be a true table of contents,
% because page numbers need not be listed. The modules are going to be
% in ABC order anyway, so I think page numbers are unnecessary.

% 4. Extract 'help' lines from CPInstallGuide.m, have the title of the
% page be "CPInstallGuide".

% 5. Screenshot of CellProfiler, with the Data button pushed so the
% Data buttons are revealed.  I have saved it as a CPscreenshot.TIF
% (within ExampleImages), but maybe we could have this automatically
% produced eventually?

% 6. Extract 'help' lines from Help1.m through Help5.m. Have the title
% of the page be "HelpX".

% 7. Extract 'help' lines from HelpZZZ, where ZZZ is anything else (I
% guess the order is not critical here).

% 8. Open each algorithm (alphabetically) and print its name in large
% bold font at the top of the page. Extract the lines after "Help for
% ...." and before the license begins, using the Matlab 'help'
% function. Print this below the algorithm name. Extract the variables
% from the algorithm using the same code CP uses, and print this below
% the algorithm description. Open the corresponding tif image file (in
% the ExampleImages folder, these always have the exact name as the
% algorithm), if it exists, and place this at the bottom of the page.
% [somehow deal with it if there is too much stuff to fit on one page,
% though I think at the moment each module should fit on one page.]

% 9. Add page numbers throughout.

% 10. Save as a pdf.