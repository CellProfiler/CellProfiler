function HelpCPInstallGuide
helpdlg(help('HelpCPInstallGuide'))

% Get the latest code from www.cellprofiler.org
%
% *************************************************************************
%
% CellProfiler Installation Guide: Macintosh OSX Version
% 
% 1. Download and unzip CellProfiler to any folder, but all files must
% remain in the same folder and the folder's name must not have spaces
% (e.g. /Applications/CellProfiler).
% 
% 2. Download and install X11.
% http://www.apple.com/downloads/macosx/apple/x11formacosx.html
% 
% 3. To start CellProfiler, double click CellProfiler.command, located in
% the folder where CellProfiler was un-zipped.
% 
% 4. (Optional steps) Get up and running quickly with one of the sample
% pipelines on our Examples page, and check out Help > HelpGettingStarted
% within CellProfiler. In CellProfiler, set some basic preferences in File
% > Set preferences, which will make it easier to use CellProfiler. You can
% drag CellProfiler.command to the dock (on the side near the trash) so the
% program is easily accessible in the future.
%
% *************************************************************************
%
% CellProfiler Installation Guide: Windows PC Version
% 
% 1. Download and extract CellProfiler.exe, CellProfiler.ctf, and the
% Modules directory to any folder, but all files must remain in the same
% folder (e.g. C:\Program Files\CellProfiler).
% 
% 2. Download and run MCRInstaller.exe (see link on CellProfiler website).
% 
% 3. To start CellProfiler, run CellProfiler.exe. The first time the
% program is run, it will take a bit longer to de-compress the CTF file.
% 
% 4. (Optional steps) Get up and running quickly with one of the sample
% pipelines on our Examples page, and check out Help > HelpGettingStarted
% within CellProfiler. In CellProfiler, set some basic preferences in File
% > Set preferences, which will make it easier to use CellProfiler.
% 
% *************************************************************************
% 
% CellProfiler Installation Guide: Developer's version
% 
% 1. (Mac only) Set X11 setting... Close Matlab & X11. Open the Mac
% application called 'Terminal' (Applications > Utilities > Terminal) and
% type the following to ensure that display windows behave properly:
% defaults write com.apple.x11 wm_click_through -bool true
% 
% 2. Start Matlab and check licenses... Start Matlab and type into the main
% window (command line): license('test','image_toolbox') If the Image
% Processing Toolbox is installed with a valid license you will see: ans =
% 1, but if ans = 0, the toolbox and/or license is not installed.
% CellProfiler was designed for Matlab Release 14, version 7.1, SP3
% although it is possible that many functions will work on other versions.
% We have noticed lots of Matlab crashing with Mac OS10.2 but OS10.3 is
% fine.
% 
% 3. Copy the CellProfiler files to your computer... Put them in a folder
% named CellProfiler at any location - suggested: On Mac:
% /Applications/CellProfiler On PC: C:\Program Files Once the files are
% copied, do not move or rename this folder or any folders containing it,
% or you will have to repeat the next step.
% 
% 4. Start CellProfiler... To start CellProfiler for the first time, you
% must call it directly. After the first time, you may simply type
% "CellProfiler" in the main Matlab window to start CellProfiler. To call
% CellProfiler directly, type the following in the main Matlab window
% (LOCATION is where the CellProfiler folder is located): On Mac: run
% '/LOCATION/CellProfiler/CellProfiler.m' For example: run
% '/Applications/CellProfiler/CellProfiler.m'
% 
% On PC: run '\LOCATION\CellProfiler\CellProfiler.m' For example: run
% 'C:\Program Files\CellProfiler\CellProfiler.m'
% 
% 5. (Optional steps) Get up and running quickly with one of the sample
% pipelines on our Examples page, and check out Help > HelpGettingStarted
% within CellProfiler. In CellProfiler, set some basic preferences in File
% > Set preferences, which will make it easier to use CellProfiler.

%%% We are not using CPhelpdlg because this allows the help to be accessed
%%% from the command line of Matlab. The code of theis module (helpdlg) is
%%% never run from inside CP anyway.