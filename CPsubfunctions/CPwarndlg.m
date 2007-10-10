function varargout = CPwarndlg(varargin)

% This CP function is present only so we can easily replace the
% warndlg if necessary.  See documentation for warndlg for usage.

if nargout > 0,
    varargout = {warndlg(varargin{:})};
    h = varargout{1};
else
    h = warndlg(varargin{:});
end

%% This allows message boxes to be closed with 'Windows -> Close All'
userData.Application = 'CellProfiler';
set(h,'UserData',userData);