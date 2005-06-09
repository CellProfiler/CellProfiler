function FigHandles=CPfigure(varargin);
userData.Application = 'CellProfiler';
if nargin>0 && isfield(varargin{1},'ImageToolsPopUpMenu')
    userData.MyHandles=varargin{1};
    FigHandles=figure(varargin{2:end});
    TempMenu = uimenu('Label','CellProfiler Image Tools');
    ListOfImageTools=get(userData.MyHandles.ImageToolsPopUpMenu,'String');
    for j=2:length(ListOfImageTools)
        uimenu(TempMenu,'Label',char(ListOfImageTools(j)),'Callback',['UserData=get(gcf,''userData'');' char(ListOfImageTools(j)) '(UserData.MyHandles); clear UserData']);
    end
    set(FigHandles,'UserData',userData);
else
    FigHandles=figure(varargin{:});
    set(FigHandles,'UserData',userData);
end