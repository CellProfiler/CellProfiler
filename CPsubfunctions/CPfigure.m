function FigHandle=CPfigure(varargin)

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

% SEE HelpDeveloperInfo.m FOR MORE INFORMATION ON HOW TO USE THIS
% SUBFUNCTION.

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
    
    %% Try to correct for odd Figure placement
    idx = find(strcmp('Position',varargin), 1);
    if ~isempty(idx)
        pos = varargin{idx+1};
        pause(.05)
        set(FigHandle,'Position',pos)
    end
    
    if nargin>=2
        %%% This is for the typical usage:
        %%% CPfigure(handles,'Image',ThisModuleFigureNumber)
        %% Currently if varargin{2}='Image' it creates ImageTools toolbar, RGB
        %% buttons, and Raw/Stretched dropdown.  If a module does not need
        %% these, then choose 'Text' instead (or just '').
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

                PopupScale = ...
                    ['ImageHandles = findobj(gcf,''Type'',''Image'');' ...
                    'if ~isempty(ImageHandles),' ...
                        'popup=findobj(gcf,''tag'',''PopupScale'');' ...
                        'ImageHandles = num2cell(ImageHandles);' ...
                        'for i = 1:length(ImageHandles),' ...
                            'cax = ancestor(ImageHandles{i},''axes'');' ...
                            'if get(popup,''value'')==2,' ...
                                'set(cax,''CLimMode'',''manual'');' ...
                                'set(cax,''CLim'',[0 1]);' ...
                            'elseif get(popup,''value'')==1,' ...
                                'set(cax,''CLimMode'',''auto'');' ...
                            'end,' ...
                        'end,' ...
                    'end;' ...
                    'clear popup ImageHandles cax i;'];
                
                uicontrol('Style', 'popup',...
                    'String', 'Stretched|Raw',...
                    'Units','normalized',...
                    'Position', [0.8 0.95 .2 .04], ...
                    'Callback', PopupScale,...
                    'parent',FigHandle, ...
                    'FontSize',Font,...
                    'BackgroundColor',[.7,.7,.9],...
                    'tag','PopupScale');
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
    
    %% Do not overwrite the old UserData, if the figure was pre-existing 
    %% with the field UserData.MyHandles
    if nargin == 0 && isfield(get(FigHandle),'UserData')
        if ~isfield(get(FigHandle,'UserData'),'MyHandles')
            set(FigHandle,'UserData',userData);
        end
    end
    set(FigHandle,'Color',[0.7 0.7 0.9],'BackingStore','off');
end