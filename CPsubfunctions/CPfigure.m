function FigHandle=CPfigure(varargin);
userData.Application = 'CellProfiler';
if nargin>0 && isfield(varargin{1},'ImageToolsPopUpMenu')
    userData.MyHandles=varargin{1};
    FigHandle=figure(varargin{2:end});
    if nargin~=2 %if nargin==2 then only referring to an existing figure
        ZoomButtonCallback = 'try, CPInteractiveZoom; catch CPmsgbox(''Could not find the file called InteractiveZoomSubfunction.m which should be located in the CellProfiler folder.''); end';
        uimenu('Label','Interactive Zoom','Callback',ZoomButtonCallback);
        TempMenu = uimenu('Label','CellProfiler Image Tools');
        ListOfImageTools=get(userData.MyHandles.ImageToolsPopUpMenu,'String');
        for j=2:length(ListOfImageTools)
            uimenu(TempMenu,'Label',char(ListOfImageTools(j)),'Callback',['UserData=get(gcf,''userData'');' char(ListOfImageTools(j)) '(UserData.MyHandles); clear UserData ans;']);
        end
    end
    set(FigHandle,'UserData',userData);
    set(FigHandle,'Color',[0.7 0.7 0.9]);
else
    FigHandle=figure(varargin{:});
    set(FigHandle,'UserData',userData);
    set(FigHandle,'Color',[0.7 0.7 0.9]);
end