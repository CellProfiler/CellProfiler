function HelpGettingStarted
helpdlg(help('HelpGettingStarted'))

% Choose image analysis modules to add to your analysis routine by  
% clicking '+'.  Typically, the first module which must be run is a  
% LoadImages module, where you specify the identity of the images  
% that you want to analyze. Modules are added to the end of the  
% pipeline, but their order can be adjusted by selecting module(s) and  
% using the Move up '^' and Move down 'v' buttons. The '-' button  
% will delete selected module(s) from the pipeline. Clicking a module  
% in the pipeline will reveal its settings in the space to the right of  
% the selected module. Properly formatted image analysis modules for  
% CellProfiler are Matlab m-files that end with .m.
%
% SAVING IMAGES: The thresholded images produced by this module can be
% easily saved using the Save Images module, using the name you assign. If
% you want to save other intermediate images, alter the code for this
% module to save those images to the handles structure (see the SaveImages
% module help) and then use the Save Images module.
%
% What do Primary, Secondary, Tertiary objects mean?
% Identify Primary modules identify objects without relying on any
% information other than a single grayscale input image (e.g. nuclei
% are typically primary objects). Identify Secondary modules require a
% grayscale image plus an image where primary objects have already
% been identified, because the secondary objects' locations are
% determined in part based on the primary objects (e.g. cells can be
% secondary objects). Identify Tertiary modules require images where
% two sets of objects have already been identified (e.g. nuclei and
% cell regions are used to define the cytoplasm objects, which are
% tertiary objects).
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


%%% We are not using CPhelpdlg because this allows the help to be accessed
%%% from the command line of Matlab. The code of theis module (helpdlg) is
%%% never run from inside CP anyway.