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
if nargin == 3 && ~iscell(path) && exist(path, 'dir')
  filterspec = fullfile(path, filterspec);
end
[filename, pathname, filterindex] = uigetfile(filterspec, title);


  