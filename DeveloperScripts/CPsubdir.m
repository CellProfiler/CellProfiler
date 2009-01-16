function file = CPsubdir(cdir)
%SUBDIR List directory and all subdirectories.
% SUBDIR recursively calls itself and uses DIR to find all files and
% directories within a specified directory and all its subdirectories.
%
% D = SUBDIR('directory_name') returns the results in an M-by-1
% structure with the fields:
% name -- filename
% dir -- directory containing file
% date -- modification date
% bytes -- number of bytes allocated to the file
% isdir -- 1 if name is a directory and 0 if not

if nargin == 0 || isempty(cdir)
    cdir = cd; % Current directory is default
end
if cdir(end)== filesep
    cdir(end) = ''; % Remove any trailing \ from directory
end
file = CPdir(cdir); % Read current directory
for n = 1:length(file)
    file(n).dir = cdir; % Assign dir field
    if file(n).isdir && file(n).name(1)~='.'
        % Element is a directory -> recursively search this one
        tfile = subdir([cdir '\' file(n).name]); % Recursive call
        file = [file; tfile]; % Append to result to current directory structure
    end
end