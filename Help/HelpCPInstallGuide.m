function HelpCPInstallGuide
helpdlg(help('HelpCPInstallGuide'))

% *************************************************************************
%
% Installing CellProfiler.exe on PC's:
%
% 1. Run the MCRInstaller.exe file included with CellProfiler software and
% install the MCR files.
%
% 2. Copy CellProfiler.exe, CellProfiler.ctf, and the Modules directory to
% any folder, but they must stay in the same path (e.g. C:\CellProfiler).
%
% 3. Run CellProfiler.exe (The first time the program is run, it will take
% a bit longer to de-compress the CTF file). If text appears to be too big,
% you can change the default font size by going to File -> Set Preferences,
% then save preferences with a smaller font size.
%
% 4. (Optional) To run an example image analysis (very quick & useful!),
% download an example image set and pipeline from www.cellprofiler.org
% Then, in CellProfiler, choose Help > HelpGettingStarted for instructions
% on how to run the example pipeline.
%
% *************************************************************************
%
% CellProfiler Installation Guide: Developer's version
% 
% 1. (Mac only) Set X11 setting...
% Close Matlab & X11. Open the Mac application called 'Terminal'
% (Applications > Utilities > Terminal) and type the following to ensure
% that display windows behave properly: 
% defaults write com.apple.x11 wm_click_through -bool true
%
% 2. Start Matlab and check licenses...
% Start Matlab and type into the main window (command line):
% 		license('test','image_toolbox')
% If the Image Processing Toolbox is installed with a valid license you
% will see: ans = 1, but if ans = 0, the toolbox and/or license is not
% installed. CellProfiler was designed for Matlab Release 14, version 7.1,
% SP3 although it is possible that many functions will work on other
% versions. We have noticed lots of crashing with Mac OS10.2 + Matlab.
% 
% 3. Copy the CellProfiler files to your computer...
% Put them in a folder named CellProfiler at any location - suggested:
% On Mac: /Applications/CellProfiler
% On PC: C:\Program Files
% Once the files are copied, do not move or rename this folder or any
% folders containing it, or you will have to repeat the next step.
% 
% 4. Start CellProfiler...
% To start CellProfiler for the first time, you must call it directly.
% After the first time, you may simply type "CellProfiler" in the main
% Matlab window to start CellProfiler. To call CellProfiler directly, type
% the following in the main Matlab window (LOCATION is where the
% CellProfiler folder is located):
% On Mac:           run '/LOCATION/CellProfiler/CellProfiler.m'
%    For example:   run '/Applications/CellProfiler/CellProfiler.m'
%
% On PC:            run '\LOCATION\CellProfiler\CellProfiler.m'
%    For example: 	run 'C:\Program Files\CellProfiler\CellProfiler.m'
% 
% 5. (Optional) Click File > Set preferences. This is just for convenience,
% and can be reset later, if you would like.  This step creates a file
% called CellProfilerPreferences.mat in the root directory of Matlab and
% will make it easier to use CellProfiler.
% 
% 6. (Optional) To run an example image analysis (very quick & useful!),
% download an example image set and pipeline from www.cellprofiler.org
% Then, in CellProfiler, choose Help > HelpGettingStarted for instructions
% on how to run the example pipeline.

%%% We are not using CPhelpdlg because this allows the help to be accessed
%%% from the command line of Matlab. The code of theis module (helpdlg) is
%%% never run from inside CP anyway.