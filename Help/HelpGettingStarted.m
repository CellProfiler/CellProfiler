function HelpGettingStarted
helpdlg(help('HelpGettingStarted'))

% The best way to learn how to use CellProfiler is to load an example
% pipeline (from www.cellprofiler.org) and try it out. Or, you can build a
% pipeline from scratch (below).
%
% LOADING A PIPELINE: 
% STEP 1: Put the images and pipeline into a folder on your computer.
%
% STEP 2: Load the pipeline using File > Load Pipeline in the main menu of
% CellProfiler.
%
% STEP 3: Set the default image and output folders to be the folder where
% you put the images.
% 
% STEP 4: Click "Analyze images" to start processing.
% 
% STEP 5: Examine the measurements using Data Tools.
% Data Tools are accessible in the main menu of CellProfiler and allow you
% to plot, view, or export your measurements (e.g. to Excel).
%
% BUILDING A PIPELINE FROM SCRATCH:
% STEP 1: Build an image analysis pipeline from individual modules 
% (this help info is also available in HelpIndividualModule)
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
% For more information, click the "?" buttons near these boxes in the main
% window.
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
%
%
%
% Pipeline of modules: LOAD 
% A pipeline is just a sequential set of individual image analysis
% modules. A pipeline of modules, along with settings you previously
% selected for each image analysis module within it, can be loaded all
% at once rather than adding image analysis modules individually and
% manually typing in their settings. An example saved settings file
% and a corresponding set of images is available for download from
% cellprofiler.org. A pipeline can be loaded from a previously saved
% settings file (made using the 'Save' button), or from an output file
% previously created by running a pipeline. CellProfiler automatically
% determines which type of file you have selected and extracts the
% relevant information.  In case the settings file was created with an
% outdated version of a module, some of the behavior of settings may
% have changed, so CellProfiler warns you and guides you through
% converting your old settings file to something usable.
%
% Pipeline of modules: SAVE
% Once you have loaded the desired image analysis modules and modified
% all of the settings as desired, you may save this pipeline for
% future use by clicking 'Save' and naming the file. This creates a
% small file containing all of the image analysis modules, and their
% settings, plus the pixel size.  It does not store any imported
% sample information or the folder of images to be analyzed, etc.
%
% See also Help (main menu of CellProfiler) and "?" buttons in the main
% window.
%



%%% We are not using CPhelpdlg because this allows the help to be accessed
%%% from the command line of Matlab. The code of theis module (helpdlg) is
%%% never run from inside CP anyway.