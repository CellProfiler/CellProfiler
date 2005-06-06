function FigHandles=CPfigure(handles, varargin);

userData.Application = 'CellProfiler';
userData.MyHandles = handles;
FigHandles=figure(varargin{:});
set(FigHandles,'UserData',userData);