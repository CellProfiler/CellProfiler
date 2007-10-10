function varargout = CPhelpdlg(varargin)

% This CP function is present only so we can easily replace the
% helpdlg if necessary.  See documentation for helpdlg for usage.

if nargout > 0,
    varargout = {helpdlg(varargin{:})};
    h = varargout{1};
else
    h = helpdlg(varargin{:});
end

%% This allows message boxes to be closed with 'Windows -> Close All'
userData.Application = 'CellProfiler';
set(h,'UserData',userData);