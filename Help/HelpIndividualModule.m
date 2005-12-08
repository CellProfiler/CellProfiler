function HelpIndividualModule
helpdlg(help('HelpIndividualModule'))

% Choose image analysis modules to add to your analysis routine (your
% "pipeline") by clicking '+'. Typically, the first module which must be
% run is the Load Images module, where you specify the identity of the
% images that you want to analyze. Modules are added to the end of the
% pipeline, but their order can be adjusted in the main window by selecting
% module(s) and using the Move up '^' and Move down 'v' buttons. The '-'
% button will delete selected module(s) from the pipeline.

%%% *** IF YOU MODIFY THIS TEXT, PLEASE ALSO UPDATE HelpGettingStarted!! *** 

%%% We are not using CPhelpdlg because this allows the help to be accessed
%%% from the command line of Matlab. The code of theis module (helpdlg) is
%%% never run from inside CP anyway.