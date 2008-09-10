function svn_ver_char = CPsvnversionnumber
%% Find the current svn version number
%% First, try using the 'svn info' command, but if svn is not
%% installed, or if deployed, loop all functions and parse out

% $Revision$

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
                %% The '+4' gets an extra space for vers < 9999, and should
                %% be good 'til 99,999!
                last = first+4; 
                svn_ver_char = info(first:last);
                return
            end
        end
    end
catch
    svn_ver_char = CPsvnloopfunctions;
    return
end

%% If you've gotten here without returning
%% then just do the loop
svn_ver_char = CPsvnloopfunctions;
