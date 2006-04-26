function GSGettingStarted
helpdlg(help('GSGettingStarted'))

% The best way to learn how to use CellProfiler is to load an example
% pipeline (from www.cellprofiler.org) and try it out. Or, you can build a
% pipeline from scratch. A pipeline is a sequential set of individual image
% analysis modules. See also Help (main menu of CellProfiler) and "?"
% buttons in the main window.
%
% ************************   LOADING A PIPELINE   ************************
%
% STEP 1: Put the images and pipeline into a folder on your computer.
%
% STEP 2: Set the default image and output folders (lower right of the main
% window) to be the folder where you put the images.
% 
% STEP 3: Load the pipeline using File > Load Pipeline in the main menu of
% CellProfiler.
%
% STEP 4: Click "Analyze images" to start processing.
% 
% STEP 5: Examine the measurements using Data Tools.
% Data Tools are accessible in the main menu of CellProfiler and allow you
% to plot, view, or export your measurements (e.g. to Excel).
%
% STEP 6: If you modify the modules or settings in the pipeline, you can
% save the pipeline using File > Save Pipeline. See the end of this
% document for more information on pipeline files.
%
% *****************   BUILDING A PIPELINE FROM SCRATCH   *****************
%
% STEP 1: Place modules in a new pipeline.
% Choose image analysis modules to add to your analysis routine (your
% "pipeline") by clicking '+'. Typically, the first module which must be
% run is the Load Images module, where you specify the identity of the
% images that you want to analyze. Modules are added to the end of the
% pipeline, but their order can be adjusted in the main window by selecting
% module(s) and using the Move up '^' and Move down 'v' buttons. The '-'
% button will delete selected module(s) from the pipeline.
%
% Most pipelines depend on a major step: Identifying objects. In
% CellProfiler, the objects you identify are called Primary, Secondary, or
% Tertiary. What does this mean? Identify Primary modules identify objects
% without relying on any information other than a single grayscale input
% image (e.g. nuclei are typically primary objects). Identify Secondary
% modules require a grayscale image plus an image where primary objects
% have already been identified, because the secondary objects' locations
% are determined in part based on the primary objects (e.g. cells can be
% secondary objects). Identify Tertiary modules require images where two
% sets of objects have already been identified (e.g. nuclei and cell
% regions are used to define the cytoplasm objects, which are tertiary
% objects).
%
% Saving images in your pipeline: Due to the typically high number of
% intermediate images produced during processing, images produced during
% processing are not saved to the hard drive unless you specifically
% request it, using a Save Images module.
%
% STEP 2: Adjust the settings in each module. 
% Back in the main window of CellProfiler, click a module in the pipeline
% to see its settings in the main workspace. To learn more about the
% settings for each module, select the module in the pipeline and click the
% "?" button below the pipeline.
%
% STEP 3: Set the default image folder, default output folder, pixel
% size, and output filename.
% For more help, click their nearby "?" buttons in the main window.
%
% STEP 4: Click "Analyze images" to start processing.
% All of the images in the selected folder(s) will be analyzed using the
% modules and settings you have specified.  You will have the option to
% cancel at any time.  At the end of each cycle, the data are saved in the
% output file.
%
% STEP 5: Examine your measurements using Data Tools.
% Data Tools are accessible in the main menu of CellProfiler and allow you
% to plot, view, or export your measurements (e.g. to Excel).
%
% Note: You can test an analysis on a single image cycle by setting the
% Load Images module appropriately.  For example, if loading by order, you
% can set the number of images per set to equal the total number of images
% in the folder (even if it is thousands) so that only the first cycle will
% be analyzed.  Or, if loading by text, you can make the identifying text
% specific enough that it will recognize only one group of images in the
% folder. Once the settings look good for a few test images, you can change
% the Load Images module to recognize all images in your folder.
%
% STEP 6: Save your pipeline.
% This step can be done at any time using File > Save Pipeline. 
%
%
%
%
% Note about CellProfiler "PIPE" pipeline files: A pipeline can be loaded
% from a pipeline file or from any output file created using the pipeline.
% A pipeline file is very small and is therefore more convenient for
% sharing with colleagues. It also allows you to save your work on a
% pipeline even if it's not ready to run yet. Loading/Saving Pipeline files
% will load/save these: the image analysis modules, their settings, and the
% pixel size. It will not save the default image or output folder.

% We have one line of actual code in these files so that the help is
% visible. We are not using CPhelpdlg because using helpdlg instead allows
% the help to be accessed from the command line of MATLAB. The one line of
% code in each help file (helpdlg) is never run from inside CP anyway.