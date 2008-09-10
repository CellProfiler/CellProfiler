function max_svn_ver_num = CPsvnloopfunctions

str_to_find = '% $Revision:';
max_svn_ver_num = 0;
dirs_to_loop = {'.','./Modules','./CPsubfunctions','DataTools','ImageTools','Help'};

%% Directory loop
for current_dir = dirs_to_loop
    current_dir_char = char(current_dir);
    files = dir([current_dir_char '/*.m']);
    
    %% File loop
    for idx = 1:length(files)
        found = 0;
        current_file = files(idx).name;
        fid = fopen(current_file);
        
        %% Find first line like this: "% $Revision: 5721 $"
        while feof(fid) == 0
            current_line = fgetl(fid);
            if strncmp(current_line,str_to_find,length(str_to_find))
%                 disp(['Found in ' current_file])
                %% Grab the number
                first = length(str_to_find)+1;
                last = first+4;
                try
                    current_svn_ver_num = str2double(current_line(first:last));
                catch
                    %% In case there is a blank or non-numeric Revision #
                    break
                end
                max_svn_ver_num = max([max_svn_ver_num; current_svn_ver_num]);
                found = 1;
                break
            end
        end
        %% DEBUG
        if ~found
            disp(['Could not find a Revision # for ' current_file])
        end
    end
end