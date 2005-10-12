function HelpCPInstallGuide
helpdlg(help('HelpCPInstallGuide'))

% CellProfilerTM Installation Guide
% 
% 1. Be sure that Matlab and the Image Processing toolbox is installed on
% your computer and has a valid license. To check for the Image Processing
% toolbox, start Matlab and type into the command window:
% 		license('test','image_toolbox')
% If the Image Processing Toolbox is installed, with a valid license you
% will see: ans = 1, but if ans = 0, the toolbox and/or license is not
% installed.
% 
% 2. Copy the CellProfiler files to a folder named CellProfiler on your
% computer (any location is fine).  Do not move or rename this folder or
% any folders containing it, or you will have to repeat the following
% steps.
% 
% 3. To start CellProfiler for the first time, you must call it directly.
% After the first time, you may simply type "CellProfiler" at the command
% line.
% 
% TO CALL CELLPROFILER DIRECTLY:
% On Mac platform (LOCATION is where the CellProfiler folder is located):
%
% 		run '/LOCATION/CellProfiler/CellProfiler.m'
%
% On PC platform (LOCATION is where the CellProfiler folder is located):
%
%       run '\LOCATION\CellProfiler\CellProfiler.m'
% 
% 4. (Optional) Click File > Set preferences. This is just for convenience,
% and can be reset later, if you would like.  This step creates a file
% called CellProfilerPreferences.mat in the root directory of Matlab and
% will make it easier to use CellProfiler.
% 
%    To run an example image analysis (very quick & useful!), download the
%    folder ExampleFlyImages from the CellProfiler website (it is in a
%    zipped file separate from the source code and manual). Then, in
%    CellProfiler, click 'Load' pipeline of modules, choose the
%    'ExampleFlySettings.mat' file from within the ExampleFlyImages folder
%    and click OK.  Next browse to choose the ExampleFlyImages folder as
%    the default images directory *and* as the default output directory.
%    Type in the name of an output file (e.g. 'Temp1') in the appropriate
%    box in CellProfiler, then click 'Analyze images'. An analysis run
%    should begin, producing the output file Temp1OUT.mat, which you can
%    explore using the Data Tools.
% 
% 5. For Mac OSX, open the Mac application called 'Terminal'
% (Applications > Utilities > Terminal) and type the following at the
% command line to ensure that display windows behave properly:
% defaults write com.apple.x11 wm_click_through -bool true
% 
% CellProfiler was designed for Release 14 of Matlab, although it is
% possible that many functions will work on Release 13.  It has been tested
% most extensively on a Mac, but also is routinely run on Windows and UNIX
% systems. If you encounter problems, check the command window of Matlab
% for error messages.