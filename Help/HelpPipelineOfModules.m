function HelpPipelineOfModules
helpdlg(help('HelpPipelineOfModules'))

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