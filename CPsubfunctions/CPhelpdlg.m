function varargout = CPhelpdlg(varargin)

% This CP function is present only so we can easily replace the
% helpdlg if necessary.  See documentation for helpdlg for usage.

if nargout > 0,
    varargout = helpdlg(varargin{:});
else
    helpdlg(varargin{:});
end

