function FigHandles=CPfigure(varargin);

userData.Application = 'CellProfiler';
FigHandles=figure(varargin{:});
set(FigHandles,'UserData',userData);