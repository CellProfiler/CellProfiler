function FigHandles=CPfigure(varargin);

userData.Application = 'CellProfiler';
if isfield(varargin,'handles')
    userData.MyHandles=getfield(varargin,'handles');
    varargin=rmfield(varargin,'handles');
end
FigHandles=figure(varargin{:});
set(FigHandles,'UserData',userData);