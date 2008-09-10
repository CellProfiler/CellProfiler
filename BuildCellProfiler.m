function BuildCellProfiler(usage)
% BuildCellProfiler Build a self-contained executable of CellProfiler for 
%   use on a single machine or for cluster computing. 
%
%   The argument 'Usage' is a string and can either be:
%   'single': Create a CellProfiler executable appropriate for the host
%       machine it's compiled on (e.g., PC, Macs, Unix)
%   'cluster': Create a CellProfiler executable appropriate for use on a
%       cluster (assumed to be Unix-based)
%
%   The output will be an executable called CellProfiler (Usage: 'single') 
%   or CPCluster (Usage: 'cluster'), along with any associated files 
%   neccesary to set up environment variables.
%
%   Before running this function, set the current directory to 
%   your root (or trunk) CellProfiler directory, which includes 
%   CellProfiler.m and the Modules, CPsubfunctions, etc folders.
%
%   NOTES ON "SINGLE" USAGE
%   A new folder called CompiledCellProfiler will be created at the 
%   same level as the CP root folder.  This placement is to avoid duplicate
%   function names if you run CP from the root directory after building.
%
%   Read the readme.txt file created, and set your path variables either
%   manually, try the CellProfiler*.command scripts, or use the 
%   run_CellProfiler.sh script at a terminal prompt.
%   See readme.txt file for more details.
%
%   NOTES ON "CLUSTER" USAGE
%   After building, you will need to edit the file CPCluster.py and adjust
%   the variables 'cpcluster_home' and 'mcr_path' to the appropriate
%   locations.
%
%   IMPORTANT NOTE: If you have any calls to ADDPATH in your startup.m or
%   matlabrc.m files, you will need to remove them for compilation,
%   otherwise the deployed executable will fail to open.
%
%   Try this command at the terminal (Mac) or command (PC) prompt:
%   <matlabroot>/bin/matlab -nojvm -r "cd <CellProfiler trunk directory>; BuildCellProfiler(<usage>); quit" 
%   where <matlabroot> is the MATLAB installation directory, <usage> is
%   'single' or 'cluster', and <CellProfiler trunk directory> is (you 
%   guessed it) the CellProfiler trunk directory

% $Revision: 0001 $

% Check number of input arguments
if nargin < 1,
    error(['Arguments needed. Correct usage: BuildCellProfiler(<usage>) where ',...
        '<usage> is either ''cluster'' or ''single''.']);
end

% Confirm that the basic files and folders are in the right place
err_txt = [mfilename,': BuildCellProfiler needs to be in the trunk CellProfiler directory'];
directory_str = {'Modules','DataTools','ImageTools','CPsubfunctions','Help'};
for i = 1:length(directory_str),
    assert(exist(['./',directory_str{i}],'dir') == 7, err_txt);
end

switch lower(usage),
    case 'single',
        CompileWizard

        % CellProfiler.m gets overwritten by CompileWizard_CellProfiler.m, 
        %  so we save a temporary copy that will be moved back at the end
        movefile('CellProfiler.m', 'Old_CellProfiler.m');
        movefile('CompileWizard_CellProfiler.m', 'CellProfiler.m');

        % Save the current search path, to be restored later
        current_search_path = pathdef;
        restoredefaultpath;

        % Compile CellProfiler, checking for compiler version
        % Description of flags:
        %  -I: including folders manually, since they don't get added otherwise
        %  -C: generate separate CTF archive
        %  -a: Needed to add non-matlab .jpg file
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

        % Copy some useful scripts and files back into the CP root folder
        % that are in the SVN repository 
        copyfile('../CompiledCellProfiler/CellProfilerManual.pdf','.')
        copyfile('../CompiledCellProfiler/CellProfiler*.command','.')

        % Delete unneccesary files
        delete('../CompiledCellProfiler/CellProfiler_main.c');
        delete('../CompiledCellProfiler/CellProfiler_mcc_component_data.c');
        delete('../CompiledCellProfiler/mccExcludedFiles.log');
        delete('../CompiledCellProfiler/CellProfiler.prj');
        
        % Extract the CTF archive (without running the executable)
        disp('Extracting the CTF archive....');
        if ispc,    % Need quotes for PC
            [status,result] = unix(['"',matlabroot,'/toolbox/compiler/deploy/',computer('arch'),'/extractCTF" ../CompiledCellProfiler/CellProfiler.ctf']);
        elseif ismac || isunix,
            [status,result] = unix([matlabroot,'/toolbox/compiler/deploy/',computer('arch'),'/extractCTF ../CompiledCellProfiler/CellProfiler.ctf']);
        end
        if status,  % If status isn't zero, something went wrong
            error(result);
        else
            disp('Expansion successful');
        end
        
        % Set permissions on scripts (on unix and Mac systems)
        disp('Setting permissions...');
        if ismac || isunix,
            [status,result] = unix('chmod 775 ../CompiledCellProfiler/CellProfiler*.command');
        end
        disp('Done');

        % Restore pre-existing paths
        path(current_search_path);
    case 'cluster',
        % Attempt to build CPCluster.m
        if exist('CPCluster.m','file'),
            disp('Building CPCluster.m....');
            mcc -C -R -nodisplay -m CPCluster.m -I ./Modules -I ./DataTools -I ./ImageTools ...
                 -I ./CPsubfunctions -I ./Help -a './CPsubfunctions/CPsplash.jpg'
            disp('Finished building');
        else
            error('CPCluster.m is not present in the current directory. Please check to see if it exists and try again.');
        end

        % Delete unneccesary files
        delete('CPCluster_main.c');
        delete('CPCluster_mcc_component_data.c');
        delete('mccExcludedFiles.log');
        delete('CPCluster.prj');
        
        % Extract the CTF archive (without running the executable)
        disp('Extracting the CTF archive....');
        [status,result] = unix([matlabroot,'/toolbox/compiler/deploy/',computer('arch'),'/extractCTF CPCluster.ctf']);
        if status,  % If status isn't zero, something went wrong
            error(result);
        else
            disp('Expansion successful');
        end
        
        % Change the permissions
        disp('Setting permissions...');
        [status,result] = unix('chmod -R 775 *');
        disp('Done');
    otherwise
        error('Unrecognized arguments. Please use either ''cluster'' or ''single''.');
end