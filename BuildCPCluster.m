addpath(pwd);
addpath(fullfile(pwd,'ImageTools'));
addpath(fullfile(pwd,'DataTools'));
addpath(fullfile(pwd,'CPsubfunctions'));
addpath(fullfile(pwd,'Modules'));
addpath(fullfile(pwd,'Help'));
mcc -v -R -nodisplay -m CPCluster.m
quit
