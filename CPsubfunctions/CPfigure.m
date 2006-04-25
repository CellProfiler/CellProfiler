function FigHandle=CPfigure(varargin)

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne E. Carpenter
%   Thouis Ray Jones
%   In Han Kang
%   Ola Friman
%   Steve Lowe
%   Joo Han Chang
%   Colin Clarke
%   Mike Lamprecht
%   Susan Ma
%
% Website: http://www.cellprofiler.org
%
% $Revision$

drawnow
%    Note about the "drawnow" before the figure command:
% The "drawnow" function executes any pending figure window-related
% commands.  In general, Matlab does not update figure windows until
% breaks between image analysis modules, or when a few select commands
% are used. "figure" and "drawnow" are two of the commands that allow
% Matlab to pause and carry out any pending figure window- related
% commands (like zooming, or pressing timer pause or cancel buttons or
% pressing a help button.)  If the drawnow command is not used
% immediately prior to the figure(ThisModuleFigureNumber) line, then
% immediately after the figure line executes, the other commands that
% have been waiting are executed in the other windows.  Then, when
% Matlab returns to this module and goes to the subplot line, the
% figure which is active is not necessarily the correct one. This
% results in strange things like the subplots appearing in the timer
% window or in the wrong figure window, or in help dialog boxes.
userData.Application = 'CellProfiler';
userData.ImageFlag = 0;
if nargin>0 && isfield(varargin{1},'Pipeline')
    FixedHandles = varargin{1};
    FixedHandles = rmfield(FixedHandles,'Pipeline');
    if isfield(FixedHandles,'Measurements')
        FixedHandles = rmfield(FixedHandles,'Measurements');
    end
    userData.MyHandles = FixedHandles;
    FigHandle=figure(varargin{3:end});
    if nargin>=2
        %%% This is for the typical usage:
        %%% CPfigure(handles,'Image',ThisModuleFigureNumber)
        FigureType = varargin{2};
        Font = userData.MyHandles.Preferences.FontSize;
        set(FigHandle,'Toolbar','figure');
        FigUserData = get(FigHandle,'UserData');
        if strcmpi(FigureType,'Image')
            if isfield(FigUserData,'ImageFlag') && (FigUserData.ImageFlag == 1) %#ok Ignore MLint
                userData.ImageFlag = 1;
            else
                userData.ImageFlag = 1;
                TempMenu = uimenu('Label','CellProfiler Image Tools');
                ListOfImageTools=userData.MyHandles.Current.ImageToolsFilenames;
                for j=2:length(ListOfImageTools)
                    uimenu(TempMenu,'Label',char(ListOfImageTools(j)),'Callback',['UserData=get(gcf,''userData'');' char(ListOfImageTools(j)) '(UserData.MyHandles); clear UserData ans;']);
                end

                ToggleColorR = 'ImageHandles = findobj(gcf,''Type'',''Image'');for i = length(ImageHandles):-1:1,if size(size(get(ImageHandles(i),''CData'')),2)~=3,ImageHandles(i)=[];end;end;if ~isempty(ImageHandles),AllData=get(ImageHandles,''CData'');ImageHandles = num2cell(ImageHandles);if ~iscell(AllData), AllData={AllData};end;button=findobj(gcf,''tag'',''ToggleColorR'');for i = 1:length(AllData),tempdata{i}=AllData{i}(:,:,1);end;for i = 1:length(AllData), data=AllData{i}; if get(button,''value'')==0,set(button,''UserData'',tempdata);data(:,:,1)=0;set(ImageHandles{i},''CData'',data);else,tempdata=get(button,''UserData'');if ~iscell(tempdata),tempdata={tempdata};end;data(:,:,1)=tempdata{i};AllData{i}=data;set(ImageHandles{i},''CData'',data);end;end;end;clear data AllData button ImageHandles;';
                uicontrol('Style', 'checkbox', ...
                    'Units','normalized',...
                    'Position', [.93 .6 .06 .04], ...
                    'Callback', ToggleColorR, 'parent',FigHandle, ...
                    'FontSize',Font,'BackgroundColor',[.7,.7,.9],'min',0,...
                    'max',1,'value',1,'tag','ToggleColorR','string','R');

                ToggleColorG = 'ImageHandles = findobj(gcf,''Type'',''Image'');for i = length(ImageHandles):-1:1,if size(size(get(ImageHandles(i),''CData'')),2)~=3,ImageHandles(i)=[];end;end;if ~isempty(ImageHandles),AllData=get(ImageHandles,''CData'');ImageHandles = num2cell(ImageHandles);if ~iscell(AllData), AllData={AllData};end;button=findobj(gcf,''tag'',''ToggleColorG'');for i = 1:length(AllData),tempdata{i}=AllData{i}(:,:,2);end;for i = 1:length(AllData), data=AllData{i}; if get(button,''value'')==0,set(button,''UserData'',tempdata);data(:,:,2)=0;set(ImageHandles{i},''CData'',data);else,tempdata=get(button,''UserData'');if ~iscell(tempdata),tempdata={tempdata};end;data(:,:,2)=tempdata{i};AllData{i}=data;set(ImageHandles{i},''CData'',data);end;end;end;clear data AllData button ImageHandles;';
                uicontrol('Style', 'checkbox', ...
                    'Units','normalized',...
                    'Position', [.93 .55 .06 .04], ...
                    'Callback', ToggleColorG, 'parent',FigHandle, ...
                    'FontSize',Font,'BackgroundColor',[.7,.7,.9],'min',0,...
                    'max',1,'value',1,'tag','ToggleColorG','string','G');

                ToggleColorB = 'ImageHandles = findobj(gcf,''Type'',''Image'');for i = length(ImageHandles):-1:1,if size(size(get(ImageHandles(i),''CData'')),2)~=3,ImageHandles(i)=[];end;end;if ~isempty(ImageHandles),AllData=get(ImageHandles,''CData'');ImageHandles = num2cell(ImageHandles);if ~iscell(AllData), AllData={AllData};end;button=findobj(gcf,''tag'',''ToggleColorB'');for i = 1:length(AllData),tempdata{i}=AllData{i}(:,:,3);end;for i = 1:length(AllData), data=AllData{i}; if get(button,''value'')==0,set(button,''UserData'',tempdata);data(:,:,3)=0;set(ImageHandles{i},''CData'',data);else,tempdata=get(button,''UserData'');if ~iscell(tempdata),tempdata={tempdata};end;data(:,:,3)=tempdata{i};AllData{i}=data;set(ImageHandles{i},''CData'',data);end;end;end;clear data AllData button ImageHandles;';
                uicontrol('Style', 'checkbox', ...
                    'Units','normalized',...
                    'Position', [.93 .50 .06 .04], ...
                    'Callback', ToggleColorB, 'parent',FigHandle, ...
                    'FontSize',Font,'BackgroundColor',[.7,.7,.9],'min',0,...
                    'max',1,'value',1,'tag','ToggleColorB','string','B');
            end
        end
    end
    set(FigHandle,'UserData',userData);
    set(FigHandle,'Color',[0.7 0.7 0.9]);
    colormap(userData.MyHandles.Preferences.IntensityColorMap);
else
    %%% CPfigure with no arguments just sets the figure to be the
    %%% current/active one.
    FigHandle=figure(varargin{:});
    set(FigHandle,'UserData',userData);
    set(FigHandle,'Color',[0.7 0.7 0.9],'BackingStore','off');
end