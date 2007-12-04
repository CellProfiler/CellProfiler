function HelpPixelSize

% What is the pixel size? The pixel size is the number of micrometers per
% pixel. This number is used to convert measurements to micrometers instead
% of pixels, if you would like the size measurements to be scaled for your
% images. By default, the pixel size is set to "1" which means that all
% distance measurements will be in units of pixel lengths.
% 
% You can let CellProfiler convert pixel lengths to absolute units of
% measure (microns (micrometers)) for you by changing the pixel size or you
% can do the conversion yourself later. The default pixel size can be set
% in File > Set preferences. Upon startup, the default preferences are
% loaded or you can load preferences using File > Load Preferences. Either
% way, the preference for pixel size will be shown in the main window of
% CellProfiler. You can change the pixel size for the current session by
% typing it into the main window of CellProfiler.  This value is stored
% along with any pipelines you save, so you can check what pixel size was
% used in an old experiment by loading the pipeline from a pipeline file or
% output file.
% 
% How do you know what value to use for the pixel size? The pixel size
% depends on the resolution and binning of the camera and the magnification
% of the objective lens of the microscope, in addition to the physical
% setup of the microscope itself. You have two options: (1) check with the
% microscope manufacturer or service person and ask them for a table of
% pixel sizes for each possible combination of
% resolution/binning/objectives for your scope, or (2) get a 'stage
% micrometer' (a glass slide with precise markings of distances) and take
% pictures of it at all possible combinations of
% resolution/binning/objectives for your scope. Once the pictures are
% acquired, open them in CellProfiler, zoom in on them and take a look at
% the markings on the slide relative to a single pixel in the image. Make a
% table for yourself of the pixel size at each microscope/camera setting
% for future reference.
% 
% Warning: some CellProfiler modules might currently ignore the pixel size
% and produce data in pixel length units no matter what pixel size is set.
% We are working to fix this.

helpdlg(help('HelpPixelSize'))

% We have one line of actual code in these files so that the help is
% visible. We are not using CPhelpdlg because using helpdlg instead allows
% the help to be accessed from the command line of MATLAB. The one line of
% code in each help file (helpdlg) is never run from inside CP anyway.