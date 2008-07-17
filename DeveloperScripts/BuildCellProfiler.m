function BuildCellProfiler
% BuildCellProfiler Build an exectuable copy of CellProfiler
%
%   Before running this function:
%       (1) copy a set of CellProfiler files and folders 
%       to a new directory.  Include CellProfiler.m and the folders 
%       Modules, DataTools, ImageTools, CPsubfunctions, and Help.  
%       (2) Set the current directory to this new directory.
%
%       A new folder called CompiledCellProfiler will be created beneath 
%       the current folder with the new compiled files.
%
%       Read the readme.txt file there, and set your path variables either
%       manually, or use the run_CellProfiler.sh script at a terminal prompt.
%       See the created readme.txt file for more details.

%% Check that the basic files and folders are in the right place
assert(exist('./CellProfiler.m','file') == 2)
assert(exist('./Modules','dir') == 7)
assert(exist('./DataTools','dir') == 7)
assert(exist('./ImageTools','dir') == 7)
assert(exist('./CPsubfunctions','dir') == 7)
assert(exist('./Help','dir') == 7)

CompileWizard

%% CellProfiler.m gets overwritten by CompileWizard_CellProfiler.m, 
%%  so we save a tmp copy that will be moved back at the end
movefile( 'CellProfiler.m', 'Old_CellProfiler.m');
movefile( 'CompileWizard_CellProfiler.m', 'CellProfiler.m');

%% Compile
mcc -m CellProfiler -a './CPsubfunctions/CPsplash.jpg'

%% Move files and cleanup
mkdir('.', 'CompiledCellProfiler')
mkdir('./CompiledCellProfiler', 'Modules')
movefile('./Modules/*.txt', './CompiledCellProfiler/Modules')
movefile('CellProfiler*.*','./CompiledCellProfiler/')
movefile('readme.txt','./CompiledCellProfiler/')
movefile('run_CellProfiler.sh','./CompiledCellProfiler/')
movefile( 'Old_CellProfiler.m', 'CellProfiler.m');