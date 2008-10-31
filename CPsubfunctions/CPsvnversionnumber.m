function svn_ver_char = CPsvnversionnumber(CP_root_dir)
%% Find the current svn version number
%% First, try using the 'svn info' command, but if svn is not
%% installed, or if deployed, loop all functions and parse out

% $Revision$

current_dir = pwd;
if nargin > 0
    cd(CP_root_dir);
end

str_to_find = 'Revision: ';

try
    if ~isdeployed
        if ~ispc
            [status,info] = unix('svn info');
        else
            [status,info] = dos('svn info');
        end

        if status == 0 %% if successful
            %% Parse out svn Revision Number
            pos = findstr(info,str_to_find);
            if length(pos) == 1
                first = pos+length(str_to_find);
                svn_ver_char = strtok(info(first:end));
                cd(current_dir)
                return
            end
        end
    end
catch
    svn_ver_char = CPsvnloopfunctions;
    cd(current_dir)
    return
end

%% If you've gotten here without returning (e.g.  if not deployed)
%% then just do the loop
svn_ver_char = CPsvnloopfunctions;
cd(current_dir)
