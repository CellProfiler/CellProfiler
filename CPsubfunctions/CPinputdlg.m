function Answer=CPinputdlg(varargin)

% This CP function is present only so we can easily replace the
% inputdlg if necessary.  See documentation for helpdlg for usage.

Answer=inputdlg(varargin{:});
