%CPuigetfile Standard open file dialog box.
%    [FILENAME, PATHNAME, FILTERINDEX] = CPuigetfile(FILTERSPEC, TITLE)
%    is the exact same as the standard Matlab UIGETFILE function.
%
%    [FILENAME, PATHNAME, FILTERINDEX] = CPuigetfile(FILTERSPEC, TITLE, PATH)
%    initializes the dialog box to the directory PATH if PATH is a
%    directory and exists.  (The PATH argument is ignored if
%    FILTERSPEC is a cell array.)
function [filename, pathname, filterindex] = CPuigetfile(filterspec, title, ...
						path)
                    
origFilterspec = filterspec;                    
if nargin == 3 && ~iscell(path) && exist(path, 'dir')
  filterspec = fullfile(path, filterspec);
end

%% Corrects for matlab bug, in case a '.' occurs in the path, which screws
%% up the filterspec pulldown selection
if findstr(path, '.')
    [filename, pathname, filterindex] = uigetfile(origFilterspec, title, path);
else
    [filename, pathname, filterindex] = uigetfile(filterspec, title);
end


  