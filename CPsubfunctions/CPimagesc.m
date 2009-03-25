function [h,CurrentAxes,CurrentFig] = CPimagesc(Image,handles,varargin)

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% $Revision$

if nargin < 3
    warning('CPimagesc:NoAxisHandle','Deprecated use of CPimagesc - proper use is CPimagesc(Image,handles,hAxis)');
    h = imagesc(Image);
    CurrentAxes = get(h,'parent');
    CurrentFig = get(CurrentAxes,'parent');
elseif ishandle(varargin{1})
    switch(get(varargin{1},'Type'))
        case 'axes'
            CurrentAxes = varargin{1};
            CurrentFig = get(CurrentAxes,'Parent');
        case 'figure'
            CurrentFig = varargin{1};
            CurrentAxes = findobj(CurrentFig,'type','axes');
            if isempty(CurrentAxes)
                CurrentAxes = axes('Parent',CurrentFig);
            end
        otherwise
            error(['Unhandled graphics handle type: ',get(varargin{1},'Type')]);
    end
    h = imagesc(Image,'Parent',CurrentAxes,varargin{2:end});
else
    error('CPimagesc argument # 3 must be an axis or figure handle');
end

%%% Embeds the Image tool submenu so that it appears when the user clicks
%%% on the image. 
set(h,'ButtonDownFcn','CPimagetool');

%%% Link any image axis limits together so zoom/pan is reflected in all axes
if exist('linkaxes','file'),    % Make sure linkaxes exists (available in > R13)
    % Will need to change this line if the image tags are ever set
    AllAxesHandles = get(findobj(CurrentFig,'type','image','ButtonDownFcn','CPimagetool'),'parent');
    if iscell(AllAxesHandles), AllAxesHandles = cell2mat(AllAxesHandles); end
    
    %%% Make sure the axis limits are the same in all axes, otherwise
    %%% linkaxes will adjust them all to the same value (which would be bad)
    AllAxesLimits = [get(AllAxesHandles,'xlim') get(AllAxesHandles,'ylim')]; 
    if size(AllAxesLimits,1) > 1,
        AllAxesLimits = cell2mat(AllAxesLimits);
        if size(unique(AllAxesLimits,'rows'),1) == 1,
            linkaxes(AllAxesHandles,'xy');
        end
    end
end


%%% Sets the user's preference for font size, which should affect tick
%%% labels and current and future titles.
set(CurrentAxes,'fontsize',handles.Preferences.FontSize)

%%% Applies the user's choice for colormap.
if ndims(Image) == 2
    colormap(handles.Preferences.IntensityColorMap);
end

ImageHandles = findobj(CurrentFig,'Type','Image');
FigUserData = get(CurrentFig,'Userdata');

Font = handles.Preferences.FontSize;
 
if isempty(findobj(CurrentFig,'tag','ToggleColorR')),
    uicontrol('Style', 'checkbox', ...
        'Units','normalized',...
        'Position', [.93 .6 .06 .04], ...
        'Callback', @ToggleColor_Callback, 'parent',CurrentFig, ...
        'FontSize',Font,'BackgroundColor',[.7,.7,.9],'min',0,...
        'max',1,'value',1,'tag','ToggleColorR','string','R');
end

if isempty(findobj(CurrentFig,'tag','ToggleColorG')),
    uicontrol('Style', 'checkbox', ...
        'Units','normalized',...
        'Position', [.93 .55 .06 .04], ...
        'Callback', @ToggleColor_Callback, 'parent',CurrentFig, ...
        'FontSize',Font,'BackgroundColor',[.7,.7,.9],'min',0,...
        'max',1,'value',1,'tag','ToggleColorG','string','G');
end

if isempty(findobj(CurrentFig,'tag','ToggleColorB')),
    uicontrol('Style', 'checkbox', ...
        'Units','normalized',...
        'Position', [.93 .50 .06 .04], ...
        'Callback', @ToggleColor_Callback, 'parent',CurrentFig, ...
        'FontSize',Font,'BackgroundColor',[.7,.7,.9],'min',0,...
        'max',1,'value',1,'tag','ToggleColorB','string','B');
end

%%% The RBG buttons default to being drawn for each individual axes object, 
%%% but we delete RGB buttons if there are no color images in the figure
if isfield(FigUserData,'MyHandles')
    for i = length(ImageHandles):-1:1
        NDIM(i) = ndims(get(ImageHandles(i),'CData'));
    end
    if ~any(NDIM == 3)
        delete(findobj(CurrentFig,'Tag','ToggleColorR'))
        delete(findobj(CurrentFig,'Tag','ToggleColorG'))
        delete(findobj(CurrentFig,'Tag','ToggleColorB'))
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTION - ToggleColor  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function ToggleColor_Callback(hObject, eventdata)

ImageHandles = findobj(gcbf,'Type','Image');

str = get(gcbo,'tag');
ToggleColor = lower(str(end));

switch ToggleColor,
    case 'r', index = 1;
    case 'g', index = 2;
    case 'b', index = 3;
end

for i = length(ImageHandles):-1:1,
    if size(size(get(ImageHandles(i),'CData')),2)~=3,
        ImageHandles(i) = [];
    end
end
if ~isempty(ImageHandles),
    AllData = get(ImageHandles,'CData');
    ImageHandles = num2cell(ImageHandles);
    if ~iscell(AllData),
        AllData = {AllData};
    end
    for i = 1:length(AllData),
        tempdata{i} = AllData{i}(:,:,index);
    end
    for i = 1:length(AllData)
        data = AllData{i};
        if get(hObject,'value') == 0,
            set(hObject,'UserData',tempdata);
            data(:,:,index) = 0;
            set(ImageHandles{i},'CData',data);
        else
            tempdata = get(hObject,'UserData');
            if ~iscell(tempdata),
                tempdata = {tempdata};
            end
            data(:,:,index) = tempdata{i};
            AllData{i} = data;
            set(ImageHandles{i},'CData',data);
        end
    end
end
