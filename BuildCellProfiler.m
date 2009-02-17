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
%
% $Revision$

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

% Get svn version number
current_search_path = pathdef;
addpath Modules CPsubfunctions DataTools ImageTools Help
svn_ver = CPsvnversionnumber;
% Restore pre-existing paths
path(current_search_path);

switch lower(usage),
    case 'single',
        
        output_dir = ['CompiledCellProfiler_' svn_ver];

        % Move files and cleanup
        if ~exist(['../' output_dir],'dir')
            mkdir('..', output_dir)
        end

        CompileWizard

        % CellProfiler.m is overwritten by
        % CompileWizard_CellProfiler.m, which has additional code (Help files, etc)
        % which then gets compiled.  So we save a temporary copy that
        % will be moved back after compilation
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
            if ispc  % Add icon to argument list
               mcc -m -C CellProfiler -I ./Modules -I ./DataTools -I ./ImageTools ...
                 -I ./CPsubfunctions -I ./Help -a './CPsubfunctions/CPsplash.jpg' -M './IconForWindows.res';
           else
               mcc -m -C CellProfiler -I ./Modules -I ./DataTools -I ./ImageTools ...
                 -I ./CPsubfunctions -I ./Help -a './CPsubfunctions/CPsplash.jpg';
           end
        else
            error('You need to have MATLAB version 7.6 (2008a) or above to run this command.')
        end

        if ~exist(['../' output_dir '/Modules'],'dir')
            mkdir(['../' output_dir '/Modules'])
        end

        movefile('CellProfiler*.*',['../' output_dir])
        movefile('./Modules/*.txt', ['../' output_dir '/Modules'])
        movefile('readme.txt',['../' output_dir])
        copyfile('version.txt',['../' output_dir])
        movefile('Old_CellProfiler.m', 'CellProfiler.m');
        movefile('mccExcludedFiles.log',['../' output_dir])
        if ~ ispc
            movefile('run_CellProfiler.sh',['../' output_dir])
        end

        % Copy some useful scripts and files back into the CP root folder
        % that are in the SVN repository
        copyfile(['../' output_dir '/CellProfilerManual.pdf'],'.')
        copyfile(['../' output_dir '/CellProfiler*.command'],'.')

        % Delete unneccesary files
        delete(['../' output_dir '/CellProfiler_main.c']);
        delete(['../' output_dir '/CellProfiler_mcc_component_data.c']);
        delete(['../' output_dir '/mccExcludedFiles.log']);
        delete(['../' output_dir '/CellProfiler.prj']);

        % Extract the CTF archive (without running the executable)
        disp('Extracting the CTF archive....');
        [success,msg,msgid] = rmdir(['../' output_dir '/CellProfiler_mcr'],'s');    % Remove pre-existing MCR directory
        if ~success && ~strcmpi(msgid,'MATLAB:RMDIR:NotADirectory'), warning([mfilename,': Unable to remove previous MCR directory']); end
        if ispc,    % Need quotes for PC
            [status,result] = unix(['"',matlabroot,'/toolbox/compiler/deploy/',computer('arch'),'/extractCTF" ../' output_dir '/CellProfiler.ctf']);
        elseif ismac || isunix,
            [status,result] = unix([matlabroot,'/toolbox/compiler/deploy/',computer('arch'),'/extractCTF ../' output_dir '/CellProfiler.ctf']);
        end
        if status,  % If status isn't zero, something went wrong
            error(result);
        else
            disp('Expansion successful');
        end

        % Set permissions on scripts (on unix and Mac systems)
        disp('Setting permissions...');
        if ismac || isunix,
            [status,result] = unix(['chmod 777 ../' output_dir '/CellProfiler*.command']);
        end
        disp('Done');

        % Restore pre-existing paths
        path(current_search_path);

    case 'cluster',
        
        output_dir = svn_ver;

        % Move files and cleanup
        if ~exist(['../' output_dir],'dir')
            mkdir('..', output_dir)
        end

        % Attempt to build CPCluster.m
        assert(exist('CPCluster.m','file') == 2,...
            'CPCluster.m is not present in the current directory. Please check to see if it exists and try again.')

        %%%% Repopulate functions in the #function list

        % Get list of functions
        FunctionDirectories = {'Modules','CPsubfunctions'};
        fn_filelist = ['%#function'];
        for this_dir = FunctionDirectories
            this_dir = char(this_dir);

            filelist = dir([this_dir '/*.m']);

            for i=1:length(filelist)
                ToolName = filelist(i).name;
                ToolNameNoExt = strtok(ToolName,'.');
                fn_filelist = [fn_filelist,' ',ToolNameNoExt];
            end
        end

        % read the current CPCluster.m into memory
        fid = fopen('CPCluster.m', 'r');
        CPClustercode = fread(fid, inf, '*char')';
        fclose(fid);

        % CPCluster.m is overwritten by
        % CompileWizard_CPCluster.m, which has additional code (Help files, etc)
        % which then gets compiled.  So we save a temporary copy that
        % will be moved back after compilation
        movefile('CPCluster.m', 'Old_CPCluster.m');

        % Find keyword string, and then do the insertion
        function_idx = strfind(CPClustercode, '%%% BuildCellProfiler: INSERT FUNCTIONS HERE');
        assert(length(function_idx) == 1, 'Could not find place to put %%#functions line in CPCluster.m.');
        CPClustercode = [CPClustercode(1:function_idx-1) fn_filelist CPClustercode(function_idx:end)];

        % write out the result
        fid = fopen('CPCluster.m', 'w');
        fprintf(fid, '%s', CPClustercode);
        fclose(fid);

        % Compile
        disp('Building CPCluster.m....');
        mcc -C -R -nodisplay -m CPCluster.m -I ./Modules -I ./DataTools -I ./ImageTools ...
            -I ./CPsubfunctions -I ./Help -a './CPsubfunctions/CPsplash.jpg'
        disp('Finished building');

        % Extract the CTF archive (without running the executable)
        disp('Extracting the CTF archive....');
        [success,msg,msgid] = rmdir('CPCluster_mcr','s');    % Remove pre-existing MCR directory
        if ~success && ~strcmpi(msgid,'MATLAB:RMDIR:NotADirectory'), warning([mfilename,': Unable to remove previous MCR directory']); end
        [status,result] = unix([matlabroot,'/toolbox/compiler/deploy/',computer('arch'),'/extractCTF CPCluster.ctf']);
        if status,  % If status isn't zero, something went wrong
            error(result);
        else
            disp('Expansion successful');
        end

        % Delete unneccesary files
        delete('CPCluster_main.c');
        delete('CPCluster_mcc_component_data.c');
%         delete('mccExcludedFiles.log');
        delete('CPCluster.prj');

        % Replace original CPCluster.m
        movefile('Old_CPCluster.m','CPCluster.m');

        % Move necessary files to output directory
%         movefile('readme.txt',['../' output_dir])
%         copyfile('version.txt',['../' output_dir])
%         movefile('mccExcludedFiles.log',['../' output_dir])
%         if ~ ispc
%             movefile('run_CPCluster.sh',['../' output_dir])
%         end
%         movefile('CPCluster.ctf',['../' output_dir])
%         movefile('CPCluster',['../' output_dir])
%         movefile('CPCluster_mcr',['../' output_dir '/'])

        % Change the permissions
        disp('Setting permissions...');
        [status,result] = unix('chmod -R 777 *');
        disp('Done');
    otherwise
        error('Unrecognized arguments. Please use either ''cluster'' or ''single''.');
end