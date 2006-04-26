function HelpPixelSize
helpdlg(help('HelpPixelSize'))

% The pixel size is the number of micrometers per pixel. This is based on
% the resolution and binning of the camera and the magnification of the
% objective lens. This number is used to convert measurements to
% micrometers instead of pixels if you would like the size measurements to
% be scaled for your images. If instead you want the measurements to be
% reported in pixels, enter '1' (1 pixel = 1 micrometer) and the
% measurements will be produced in units of pixels.
%
% The default pixel size can be set in File > Set preferences. Upon
% startup, the default preferences are loaded or you can load preferences
% using File > Load Preferences. Either way, the preference for pixel size
% will be shown in the main window of CellProfiler. You can change the
% pixel size for the current session by typing it into the main window of
% CellProfiler.  This value is stored in the settings if you File > Save
% Pipeline or if you create an output file and later File > Load Pipeline
% from that output file.

% We have one line of actual code in these files so that the help is
% visible. We are not using CPhelpdlg because using helpdlg instead allows
% the help to be accessed from the command line of MATLAB. The one line of
% code in each help file (helpdlg) is never run from inside CP anyway.