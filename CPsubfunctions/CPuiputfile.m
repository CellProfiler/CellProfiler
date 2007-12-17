%CPuiputfile Standard open file dialog box.
%    [FILENAME, PATHNAME, FILTERINDEX] = CPuiputfile(FILTERSPEC, TITLE)
%    is the exact same as the standard Matlab UIPUTFILE function.
%
%    [FILENAME, PATHNAME, FILTERINDEX] = CPuiputfile(FILTERSPEC, TITLE, PATH)
%    initializes the dialog box to the directory PATH if PATH is a
%    directory and exists.  (The PATH argument is ignored if
%    FILTERSPEC is a cell array.)
function [filename, pathname, filterindex] = CPuiputfile(filterspec, title, ...
						path)
                    
origFilterspec = filterspec;                    
if nargin == 3 && ~iscell(path) && exist(path, 'dir')
  filterspec = fullfile(path, filterspec);
end

%% Corrects for matlab bug, in case a '.' occurs in the path, which screws
%% up the filterspec pulldown selection
if findstr(path, '.')
    [filename, pathname, filterindex] = uiputfile(origFilterspec, title, path);
else
    [filename, pathname, filterindex] = uiputfile(filterspec, title);
end


  