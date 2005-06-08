function FigHandles=CPfigure(varargin);
userData.Application = 'CellProfiler';
try
    if nargin>0 && isfield(varargin{1},'Settings')
        userData.MyHandles=varargin{1};
        varargin=varargin(2:end);
    end
end
FigHandles=figure(varargin{:});
set(FigHandles,'UserData',userData);