function BuildCPCluster

% Add the current directory to the path
disp('Adding directories to path...');
addpath(pwd);

% Check to see if neccesary directories are present and if they are, add
% them to the path

Dirs = {'ImageTools','DataTools','CPsubfunctions','Modules','Help'};

for thisDir = Dirs
    try
        addpath(fullfile(pwd,char(thisDir)));
    catch
        error(['The ' char(thisDir) ' directory is not present in the current directory. Please check to see if it exists and try again.']);
    end
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
