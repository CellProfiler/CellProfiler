function BuildCPCluster

% Add the current directory to the path
disp('Adding directories to path...');
addpath(pwd);

% Check to see if neccesary directories are present and if they are, add
% them to the path
try
    addpath(fullfile(pwd,'ImageTools'));
catch
    error('The ImageTools directory is not present in the current directory. Please check to see if it exists and try again.');
end
try
    addpath(fullfile(pwd,'DataTools'));
catch
    error('The DataTools directory is not present in the current directory. Please check to see if it exists and try again.');
end
try
    addpath(fullfile(pwd,'CPsubfunctions'));
catch
    error('The CPsubfunctions directory is not present in the current directory. Please check to see if it exists and try again.');
end
try
    addpath(fullfile(pwd,'Modules'));
catch
    error('The Modules directory is not present in the current directory. Please check to see if it exists and try again.');
end
try
    addpath(fullfile(pwd,'Help'));
catch
    error('The Help directory is not present in the current directory. Please check to see if it exists and try again.');
end

% Attempt to build CPCluster.m
if exist('CPCluster.m','file'),
    disp('Building CPCluster.m....');
    mcc -v -R -nodisplay -m CPCluster.m -a './CPsubfunctions/CPsplash.jpg'
    disp('Finished building');
else
    error('CPCluster.m is not present in the current directory. Please check to see if it exists and try again.');
end

% Change the permissions
!chmod -R 775 *

% Exit
quit
