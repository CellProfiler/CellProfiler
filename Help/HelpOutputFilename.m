function HelpOutputFilename
helpdlg(help('HelpOutputFilename'))

% Naming the output file:
% Type in the text you want to use to name the output file, which is where
% all of the information about the analysis as well as any measurements are
% stored. 'OUT.mat' will be added automatically at the end of whatever you
% type in the box. The file will be saved in the default output directory
% unless you type a full path and file name into the output file name box.
% The path must not have spaces or characters disallowed by your platform.
%
% The program prevents you from entering a name which exists already (when
% 'OUT.mat' is appended). This prevents overwriting an output data file by
% accident, but is also disallowed for the following reason: when a file is
% 'overwritten', instead of completely overwriting the output file, Matlab
% just replaces some of the old data with the new data.  So, if you have an
% output file with 12 measurements and the new set of data has only 4
% measurements, saving the output file to the same name would produce a
% file with 12 measurements: the new 4 followed by 8 old measurements.
%

%%% We are not using CPhelpdlg because this allows the help to be accessed
%%% from the command line of Matlab. The code of theis module (helpdlg) is
%%% never run from inside CP anyway.