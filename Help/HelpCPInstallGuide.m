function CPInstallGuide
helpdlg(help('CPInstallGuide'))

% CellProfilerTM Installation Guide
% 
% 1. Be sure that Matlab and the Image Processing toolbox is installed on
% your computer and has a valid license. To check for the Image Processing
% toolbox, start Matlab and type into the command window:
% 		license('test','image_toolbox')
% If the Image Processing Toolbox is installed, with a valid license you will
% see: ans = 1, but if ans = 0, the toolbox and/or license is not installed.
% 
% 2. Copy the CellProfiler files to a folder named CellProfiler on your
% computer (any location is fine).  Do not move or rename this folder or any
% folders containing it, or you will have to repeat the following steps.
% 
% 3. In Matlab, go to File > Set Path.  Choose “Add with Subfolders” and
% navigate to select the CellProfiler folder you just made.  Click “OK”,
% then “Save”, then “Close”.
% 
% 4. In the command window of Matlab, type the following to start:
% 		CellProfiler
% 	CellProfiler should then launch. 
% 
% 5. (Optional) Click on the "Set preferences" button. This is just for
% convenience, and can be reset later, if you would like.  This step
% creates a file called CellProfilerPreferences.mat in the root directory
% of Matlab and will make it easier to use CellProfiler.
% 
% 
% 
% CellProfiler was designed for Release 14 of Matlab, although it is
% possible that many functions will work on Release 13.  It has been
% tested most extensively on a Mac, but also is routinely run on
% Windows and UNIX systems. If you encounter problems, check the
% command window of Matlab for error messages.