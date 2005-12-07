function HelpCPInstallGuide
helpdlg(help('HelpCPInstallGuide'))

% CellProfiler Installation Guide (see below for Developer's version):
%
%
%
%
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
% download the folder ExampleFlyImages from:
% www.cellprofiler.org/linked_files/ExampleFlyImages.zip Then, in
% CellProfiler, click 'Load' pipeline of modules, choose the
% 'ExampleFlyPIPE.mat' file and click OK.  Next browse to choose the
% ExampleFlyImages folder as the default images directory *and* as the
% default output directory. Type in the name of an output file (e.g.
% Temp1') in the appropriate box in CellProfiler, then click 'Analyze
% images'. An analysis run should begin, producing the output file
% Temp1OUT.mat, which you can explore using the Data Tools.

%%% We are not using CPhelpdlg because this allows the help to be accessed
%%% from the command line of Matlab. The code of theis module (helpdlg) is
%%% never run from inside CP anyway.