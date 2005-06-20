function FigHandles=CPfigure(varargin);
userData.Application = 'CellProfiler';
if nargin>0 && isfield(varargin{1},'ImageToolsPopUpMenu')
    userData.MyHandles=varargin{1};
    FigHandles=figure(varargin{2:end});
    if nargin~=2 %if nargin==2 then only referring to an existing figure
        TempMenu = uimenu('Label','CellProfiler Image Tools');
        ListOfImageTools=get(userData.MyHandles.ImageToolsPopUpMenu,'String');
        for j=2:length(ListOfImageTools)
            uimenu(TempMenu,'Label',char(ListOfImageTools(j)),'Callback',['UserData=get(gcf,''userData'');' char(ListOfImageTools(j)) '(UserData.MyHandles); clear UserData ans;']);
        end
    end
    set(FigHandles,'UserData',userData);
    set(FigHandles,'Color',[0.7 0.7 0.9]);
else
    FigHandles=figure(varargin{:});
    set(FigHandles,'UserData',userData);
    set(FigHandles,'Color',[0.7 0.7 0.9]);
end