function Help2

% Help for NAME THE OUTPUT FILE: 
% Type in the text you want to use to name the output file, which is
% where all of the information about the analysis as well as any
% measurements are stored. 'OUT.mat' will be added automatically at
% the end of whatever you type in the box. The file will be saved in
% the folder of images to be analyzed that you selected in Step 1,
% unless you type a full path and file name into the output file name
% box. The path must not have spaces or weird characters.
%
% The program prevents you from entering a name which, when 'OUT.mat'
% is appended, exists already. This prevents overwriting an output
% data file by accident.  It also prevents intentionally overwriting
% an output file for the following reason: when a file is
% ''overwritten'', instead of completely overwriting the output file,
% Matlab just replaces some of the old data with the new data.  So, if
% you have an output file with 12 measurements and the new set of data
% has only 4 measurements, saving the output file to the same name
% would produce a file with 12 measurements: the new 4 followed by 8
% old measurements.
%
% PIXEL SIZE IN MICROMETERS: 
% Enter the pixel size of the images.  This is based on the resolution
% and binning of the camera and the magnification of the objective
% lens. This number is used to convert measurements to micrometers
% instead of pixels. If you do not know the pixel size or you want the
% measurements to be reported in pixels, enter '1'. A default value
% can be remembered by CellProfiler by clicking on the 'Set
% preferences' button (see below).
%
% SAMPLE INFO: 
% If you would like text information about each image to be recorded
% in the output file along with measurements (e.g. Gene names,
% accession numbers, or sample numbers), click the Load button. You
% will then be guided through the process of choosing a text file that
% contains the text data for each image. More than one set of text
% information can be entered for each image; each set of text will be
% a separate column in the output file.        
%
% SET PREFERENCES: 
% This allows you to set the default pixel size, the folder to go to
% when you load analysis modules, and the folder to go to for all
% other CellProfilerTM functions. This is just for convenience, and
% can be reset later, if you would like.  This step creates a file
% called CellProfilerPreferences.mat in the directory. If you do not
% have permission to write files to the root directory of Matlab, it
% saves the file in the current directory, but then the defaults will
% only be used when CellProfilerTM is launched from that directory.
% If you do not have write permission in either location, you are out
% of luck.
%
% TECHNICAL DIAGNOSIS: 
% Clicking here causes text to appear in the main Matlab window.  This
% text shows the 'handles structure' which is sometimes useful for
% diagnosing problems with the software.