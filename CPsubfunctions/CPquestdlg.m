function ButtonName = CPquestdlg(varargin)

% This CP function is present only so we can easily replace the
% questdlg if necessary.  See documentation for questdlg for usage.

ButtonName = questdlg(varargin{:});
