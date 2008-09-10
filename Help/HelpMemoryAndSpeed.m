function HelpMemoryAndSpeed

% Help for memory and speed issues in CellProfiler:
%
% There are several options in CellProfiler for dealing with out-of-memory 
% errors associated with analyzing images:
%
% (1) Resize the input images
%   If the image is high-resolution, it may be helpful to determine
%   whether the features of interest can be processed (and accurate 
%   data obtained) by using a lower-resolution image. If this is the
%   case, use the Resize module (under Image Processing) to scale down
%   the image to a more manageable size, and perform the desired
%   operations on the smaller image.
%
% (2) Re-use the parameter names
%   Each image is associated with the unique name that you give it. If
%   you have many images, and many intermediate images created by the
%   modules you've added, the total space occupied by these images may cause
%   CellProfiler to run out of memory. In this case, a solution may be
%   to re-use names that you give to your parameters in later modules
%   in your pipeline.
%   For example, if you choose to resize your image and you know that you
%   don't need the original image, you can give the resized image the same
%   name as the original. This will overwrite the original with the smaller,
%   resized image, thereby saving space. 
%   Note: You must be certain that you have no use for the original image 
%   later in the pipeline, since that data will be lost by this method.
%
% (3) Running without display windows
%   When your images are being analyzed, the display windows created by
%   each module in your pipeline requires memory to create. If you are
%   not interested in seeing the intermediate output as it is produced,
%   you can deactivate the creation of display windows. Under File > Set
%   Preferences > Display Mode, you can specify which (if any) windows you
%   want displayed. 
%   Note: The status and error windows will still be shown so you can see 
%   the pipeline progress as your images are analyzed.
%
% (4) Use the SpeedUpCellProfiler module. 
%   The SpeedUpCellProfiler module permits the user to clear the images
%   stored in memory with the exception of those specified by the user.
%   Please see the help for the SpeedUpCellProfiler module for more details
%   and caveats.
%
% In addition to these, there are other options within MATLAB and within
% the operating system of your choice in order to maximize memory. See the
% MATLAB product support page "Avoiding Out of Memory Errors"
% (http://www.mathworks.com/support/tech-notes/1100/1107.html) for details.
%
% Also, there are several options for speeding up the analysis of your
% pipeline:
% (1) Running without display windows
%   By setting the display mode under File > Set Preferences > Display
%   Mode, you can turn off the module display windows which gives a bit of
%   a gain in speed. Once your pipeline is properly set up, we recommend
%   running the entire cycle without any windows displayed.
%
% (2) Use care in object identification
%   If you have a large image which contains a large number of small
%   objects, a good deal of computer time will be used in processing each
%   individual object, many of which you might not need. In this case, make
%   sure that you adjust the diameter options in IdentifyPrimAutomatic to 
%   exclude small objects you are not interested in, or use a FilterObjects 
%   module to eliminate objects that are not of interest.

% $Revision$

helpdlg(help('HelpMemoryAndSpeed'))

% We have one line of actual code in these files so that the help is
% visible. We are not using CPhelpdlg because using helpdlg instead allows
% the help to be accessed from the command line of MATLAB. The one line of
% code in each help file (helpdlg) is never run from inside CP anyway.