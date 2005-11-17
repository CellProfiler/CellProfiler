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