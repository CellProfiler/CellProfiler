function [selection,value] = CPlistdlg(varargin)

% This CP function is present only so we can easily replace the
% listdlg if necessary.  See documentation for helpdlg for usage.

[selection,value] = listdlg(varargin{:});
