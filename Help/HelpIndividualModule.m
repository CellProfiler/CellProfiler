function HelpIndividualModule
helpdlg(help('HelpIndividualModule'))

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