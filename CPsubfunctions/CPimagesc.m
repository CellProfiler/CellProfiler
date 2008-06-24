function h = CPimagesc(Image,handles)

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

%%% Displays the image.
h = imagesc(Image);
%%% Embeds the Image tool submenu so that it appears when the user clicks
%%% on the image. 
set(h,'ButtonDownFcn','CPimagetool');

%%% Sets the user's preference for font size, which should affect tick
%%% labels and current and future titles.
set(gca,'fontsize',handles.Preferences.FontSize)

%%% Applies the user's choice for colormap.
if ndims(Image) == 2
    colormap(handles.Preferences.IntensityColorMap);
end

ImageHandles = findobj(gcf,'Type','Image');
FigUserData = get(gcf,'Userdata');

FigHandle = gcf;
Font = handles.Preferences.FontSize;
 
if isempty(findobj(gcf,'tag','ToggleColorR'))
    uicontrol('Style', 'checkbox', ...
        'Units','normalized',...
        'Position', [.93 .6 .06 .04], ...
        'Callback', @ToggleColor_Callback, 'parent',FigHandle, ...
        'FontSize',Font,'BackgroundColor',[.7,.7,.9],'min',0,...
        'max',1,'value',1,'tag','ToggleColorR','string','R');
end

if isempty(findobj(gcf,'tag','ToggleColorG'))
    uicontrol('Style', 'checkbox', ...
        'Units','normalized',...
        'Position', [.93 .55 .06 .04], ...
        'Callback', @ToggleColor_Callback, 'parent',FigHandle, ...
        'FontSize',Font,'BackgroundColor',[.7,.7,.9],'min',0,...
        'max',1,'value',1,'tag','ToggleColorG','string','G');
end

if isempty(findobj(gcf,'tag','ToggleColorB'))
    uicontrol('Style', 'checkbox', ...
        'Units','normalized',...
        'Position', [.93 .50 .06 .04], ...
        'Callback', @ToggleColor_Callback, 'parent',FigHandle, ...
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
        delete(findobj(gcf,'Tag','ToggleColorR'))
        delete(findobj(gcf,'Tag','ToggleColorG'))
        delete(findobj(gcf,'Tag','ToggleColorB'))
    end
end
%%
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
