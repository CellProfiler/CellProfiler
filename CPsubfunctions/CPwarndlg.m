function varargout = CPwarndlg(varargin)

% This CP function is present only so we can easily replace the
% warndlg if necessary.  See documentation for warndlg for usage.

if nargout > 0,
    varargout = {warndlg(varargin{:})};
else
    warndlg(varargin{:});
end

