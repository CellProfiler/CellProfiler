function CPInstallGuide
helpdlg(help('CPInstallGuide'))

% CellProfilerTM Installation Guide
% 
% 1. Be sure that Matlab and the Image Processing toolbox is installed on
% your computer and has a valid license. To check for the Image Processing
% toolbox, start Matlab and type into the command window:
% 		license('test','image_toolbox')
% 	If the Image Processing Toolbox is installed and has a valid license
% 	you will see:
% 		ans = 1
% 	If you see the following, you will need to install the toolbox and a
% 	license.
% 		ans = 0
% 
% 2. Copy the CellProfiler files to a folder named CellProfiler on your
% computer (any location is fine).  Do not move or rename this folder or any
% folders containing it, or you will have to repeat the following steps.
% 
% 3. In Matlab, go to File > Set Path.  Choose “Add with Subfolders” and
% navigate to select the CellProfiler folder you just made.  Click “OK”,
% then “Save”, then “Close”.
% 
% 4. In the command window of Matlab, type:
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
% Please note that CellProfiler was designed to work with the following Mac
% system and has not been extensively tested on other platforms/versions,
% although we routinely run the software on Windows and UNIX systems. If
% you encounter problems, check the command window of Matlab for error
% messages.
% 
% MATLAB Version 7.0.0.19901 (R14)
% Operating System: Darwin 7.5.0 Darwin Kernel Version 7.5.0: Thu Aug  5
% 19:26:16 PDT 2004; root:xnu/xnu-517.7.21.obj~3/RELEASE_PPC  Power
% Macintosh
% Java VM Version: Java 1.4.2_05 with "Apple Computer
% ---------------------------------------------------------------------
% MATLAB                                  	 Version 7.0        (R14)
% Image Processing Toolbox                 	 Version 4.2        (R14)