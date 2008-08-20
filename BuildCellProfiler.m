function BuildCellProfiler
% BuildCellProfiler Build an exectuable copy of CellProfiler
%
%   Before running this function, set the current directory to 
%   your root (or trunk) CellProfiler directory, which includes 
%   CellProfiler.m and the Modules, CPsubfunctions, etc folders.
%
%   A new folder called CompiledCellProfiler will be created at the 
%   same level as the CP root folder.  This placement is to avoid duplicate
%   function names if you run CP from the root directory after building.
%
%   Read the readme.txt file created, and set your path variables either
%   manually, try the CellProfiler*.command scripts, or use the 
%   run_CellProfiler.sh script at a terminal prompt.
%   See readme.txt file for more details.
%
%   Try this command at the terminal (Mac) or command (PC) prompt:
%   <matlabroot>/bin/matlab -nojvm -r "cd <CellProfiler trunk directory>; BuildCellProfiler; quit" 
%   where <matlabroot> is the MATLAB installation directory and
%   <CellProfiler trunk directory> is (you guessed it) the CellProfiler trunk directory

% Check that the basic files and folders are in the right place
err_txt = 'BuildCellProfiler needs to be in the trunk CellProfiler directory';
assert(exist('./CellProfiler.m','file') == 2, err_txt)
assert(exist('./Modules','dir') == 7, err_txt)
assert(exist('./DataTools','dir') == 7, err_txt)
assert(exist('./ImageTools','dir') == 7, err_txt)
assert(exist('./CPsubfunctions','dir') == 7, err_txt)
assert(exist('./Help','dir') == 7, err_txt)

CompileWizard

% CellProfiler.m gets overwritten by CompileWizard_CellProfiler.m, 
%  so we save a tmp copy that will be moved back at the end
movefile('CellProfiler.m', 'Old_CellProfiler.m');
movefile('CompileWizard_CellProfiler.m', 'CellProfiler.m');

% Compile
% Save the current search path, to be restored later
current_search_path = pathdef;
restoredefaultpath;
addpath(pwd);

% Compile
%  -I including folders manually, since they don't get added otherwise
%  -C generate separarte CTF archive
%  -a Needed to add non-matlab .jpg file
version_info = ver('matlab');
if str2double(version_info.Version) >= 7.6, %Must include -C to produce separate CTF file
    mcc -m -C CellProfiler -I ./Modules -I ./DataTools -I ./ImageTools ...
         -I ./CPsubfunctions -I ./Help -a './CPsubfunctions/CPsplash.jpg';
else
    error('You need to have MATLAB version 7.6 (2008a) or above to run this command.')
end

% Move files and cleanup
if ~exist('../CompiledCellProfiler','dir')
    mkdir('..', 'CompiledCellProfiler')
end
if ~exist('../CompiledCellProfiler/Modules','dir')
    mkdir('../CompiledCellProfiler', 'Modules')
end

movefile('CellProfiler*.*','../CompiledCellProfiler/')
movefile('./Modules/*.txt', '../CompiledCellProfiler/Modules')
movefile('readme.txt','../CompiledCellProfiler/')
movefile('Old_CellProfiler.m', 'CellProfiler.m');
movefile('mccExcludedFiles.log','../CompiledCellProfiler/')

% Copy some useful scripts and files
copyfile('../CompiledCellProfiler/CellProfilerManual.pdf','.')
copyfile('../CompiledCellProfiler/CellProfiler*.command','.')

% Set Permissions
if ismac || isunix,
    unix('chmod 775 ../CompiledCellProfiler/CellProfiler*.command');
end

% Restore paths
path(current_search_path);