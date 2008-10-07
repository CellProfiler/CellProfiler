function CPcreateCPAPropertiesFile(handles, DataPath, DatabaseName, TablePrefix, DatabaseType)

% $Revision$

% This function produces a CellProfiler Analyst properties file based on
% the input handles structure.

ExportInfo.DataPath = DataPath;

if strcmp(TablePrefix,'Do not use')
    TablePrefix = '';
else
    TablePrefix = [TablePrefix,'_'];
end
ExportInfo.TablePrefix  = TablePrefix;
% GUIhandles = guidata(gcbo); ExportInfo.FontSize = GUIhandles.Preferences.FontSize;
ExportInfo.FontSize = 8;

% Fill in the entries for which the information is already known, or 
% initialize entries for which information is unknown

% Try to extract file and pathnames from Measurements structure
names = fieldnames(handles.Measurements.Image);
idx = find(~cellfun('isempty',regexp(names,'ModuleError.*LoadImages$')),1,'last');
names(idx:end) = [];
idx_file = find(~cellfun('isempty',regexp(lower(names),'filename')));
idx_path = find(~cellfun('isempty',regexp(lower(names),'pathname')));

% Take a stab at guessing the primary object (the first Measurement object
% that's not an Image, Experiment, or Neighbor)
objs = fieldnames(handles.Measurements);
objs(strcmp(objs,'Image') | strcmp(objs,'Experiment') | strcmp(objs,'Neighbors')) = [];
supposed_primary_obj = objs{1};

ExportInfo.Entries.db_type = lower(DatabaseType);                                                  % Database Type
ExportInfo.Entries.db_host = 'imgdb01';                                                            % Database Host Name/IP Address
switch lower(DatabaseType),                                                                 % Database Port
    case 'mysql', ExportInfo.Entries.db_port = '3306';
    case 'oracle', ExportInfo.Entries.db_port = '1521';
end
ExportInfo.Entries.db_pwd = '';                                                             % Database Username/Password
ExportInfo.Entries.db_name = DatabaseName;                                                  % Database Name
ExportInfo.Entries.db_user = 'cpuser';
ExportInfo.Entries.spot_tables = [TablePrefix,'Per_Image'];                                 % Image/Object Tables
ExportInfo.Entries.cell_tables = [TablePrefix,'Per_Object'];
ExportInfo.Entries.uniqueID = 'ImageNumber';                                                % Unique Image Identifier
ExportInfo.Entries.objectCount = ['Image_Count_',supposed_primary_obj];   % Image Primary Object Count Column
ExportInfo.Entries.objectID = 'ObjectNumber';                                               % Unique Object Identifier in Image
ExportInfo.Entries.info_table = '';                                                         % Information Table
ExportInfo.Entries.info_to_spot = '';                                                       % Linker Column for Information and Image Tables
ExportInfo.Entries.geneinfo_table = '';                                                     % Treatment Information Column
ExportInfo.Entries.webinfo_col = '';                                                        % Web Information Column
ExportInfo.Entries.webinfo_url_prepend = 'http://imageweb/images/CPALinks';                 % Web Information URL Prepend
ExportInfo.Entries.image_transfer_protocol = 'http';                                        % Image Transfer Protocol/Image Access Prepend
ExportInfo.Entries.image_size_info = '';                                                    % Image Format Information
ExportInfo.Entries.red_image_path = ['Image_',names{idx_path(1)}];                          % Image Pathways and Filenames
ExportInfo.Entries.red_image_col = ['Image_',names{idx_file(1)}];
if length(idx_path) > 1,
    ExportInfo.Entries.green_image_path = ['Image_',names{idx_path(2)}];
    ExportInfo.Entries.green_image_col = ['Image_',names{idx_file(2)}];
end
if length(idx_path) > 2,
    ExportInfo.Entries.blue_image_path = ['Image_',names{idx_path(3)}];
    ExportInfo.Entries.blue_image_col = ['Image_',names{idx_file(3)}];
end
ExportInfo.Entries.cell_x_loc = [supposed_primary_obj,'_Location_Center_X'];                 % X/Y Coordinates for Primary Object Center
ExportInfo.Entries.cell_y_loc = [supposed_primary_obj,'_Location_Center_Y']; 
ExportInfo.Entries.classifier_per_object_ignore_substr = ...
                                    'ImageNumber,ObjectNumber,parent,Location';             % Columns to Ignore when Classifying
ExportInfo.Entries.classifier_group_table = '';                                             % Classify by Group Table 
ExportInfo.Entries.classifier_group_col = '';                                               % Classify by Group Column(s)
ExportInfo.Entries.cell_size = '50';                                                          % Width of Object Cropping Square

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);
if ~strcmp(ModuleName,'ExportToDatabase')
    % If we aren't being called from the ExportToDatabase module (e.g. a 
    %   datatool), open a window that lets the user choose what to export
    try ExportInfo = PropertiesFileWindow(ExportInfo);
    catch CPerrordlg(lasterr)
        return
    end
end

% Check whether have write permission in current dir
PropertiesFilename = [DatabaseName,'.properties'];
fid = fopen(fullfile(DataPath,PropertiesFilename), 'wt');
if fid == -1, 
    % I'm using a warning here instead of an error because I can imagine a
    % case where in batch processing where multiple batches attempt to
    % write to the same file. In that case, the first to open the file
    % should write it and the failure of other to do the same isn't a
    % problem
    warning(['Could not open ',PropertiesFilename,' in ',DataPath,' for writing.']); 
else
    % Write the file, then close
    fprintf(fid,'#%s\n',[datestr(now,'ddd'),' ',datestr(now,'mmmm dd HH:MM:SS yyyy')]);
    entries = fieldnames(ExportInfo.Entries);
    for i = 1:length(entries)
        fprintf(fid, '%s = %s\n', char(entries{i}),ExportInfo.Entries.(char(entries{i})));
    end
    fclose(fid);
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTION - PropertiesFileWindow
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ExportInfo = PropertiesFileWindow(ExportInfo)
% This function displays a window so that lets the user choose which
% measurements to export. If the return variable 'ObjectNames' is empty
% it means that either no measurements were found or the user pressed
% the Cancel button (or the window was closed). 'Summary' takes on the values 'yes'
% or 'no', depending if the user only wants a summary report (mean and std)
% or a full report.

% List all the properties. This will determine the order and properties of
% the uincontrols to be created. I increment a counter prior to each
% uicontrol property 'cause changing numbers each time I add/subtract
% a control is *really* annoying

i = 1;
uitext{i} = 'What is the database type?';
uitype{i} = 'popupmenu';
uitag{i} = 'db_type';
uistring{i} = {'mysql','oracle'};
uichoice{i} = ExportInfo.Entries.(uitag{i});

i = 1+1;
uitext{i} = 'What is the database host name/IP address?';
uitype{i} = 'edit';
uitag{i} = 'db_host';
uistring{i} = ExportInfo.Entries.(uitag{i});

i = 1+1;
uitext{i} = 'What is the database port?';
uitype{i} = 'popupmenu';
uitag{i} = 'db_port';
uistring{i} = {'3306','1521'};
uichoice{i} = ExportInfo.Entries.(uitag{i});

i = 1+1;
uitext{i} = 'What is the database name?';
uitype{i} = 'edit';
uitag{i} = 'db_name';
uistring{i} = ExportInfo.Entries.(uitag{i});

i = 1+1;
uitext{i} = 'What is the user name for access RDMS (relation database management system)?';
uitype{i} = 'edit';
uitag{i} = 'db_user';
uistring{i} = ExportInfo.Entries.(uitag{i});

i = 1+1;
uitext{i} = 'What is the user password for access to the RDMS?';
uitype{i} = 'edit';
uitag{i} = 'db_pwd';
uistring{i} = ExportInfo.Entries.(uitag{i});

i = 1+1;
uitext{i} = 'What is the prefix for the per image tables?';
uitype{i} = 'edit';
uitag{i} = 'spot_tables';
uistring{i} = ExportInfo.Entries.(uitag{i});

i = 1+1;
uitext{i} = 'What is the prefix for the per object tables?';
uitype{i} = 'edit';
uitag{i} = 'cell_tables';
uistring{i} = ExportInfo.Entries.(uitag{i});

i = 1+1;
uitext{i} = 'What is image identifer?';
uitype{i} = 'edit';
uitag{i} = 'uniqueID';
uistring{i} = ExportInfo.Entries.(uitag{i});

i = 1+1;
uitext{i} = 'What is the primary object count column?';
uitype{i} = 'edit';
uitag{i} = 'objectCount';
uistring{i} = ExportInfo.Entries.(uitag{i});

i = 1+1;
uitext{i} = 'What is the object identifer?';
uitype{i} = 'edit';
uitag{i} = 'objectID';
uistring{i} = ExportInfo.Entries.(uitag{i});

i = 1+1;
uitext{i} = 'What is the name of the information table?';
uitype{i} = 'edit';
uitag{i} = 'info_table';
uistring{i} = ExportInfo.Entries.(uitag{i});

i = 1+1;
uitext{i} = 'What is the Web information prepend?';
uitype{i} = 'edit';
uitag{i} = 'image_url_prepend';
uistring{i} = ExportInfo.Entries.(uitag{i});

i = 1+1;
uitext{i} = 'What is the image transfer protocol (ITP)';
uistring{i} = {'http','local','smb','ssh'};
uitype{i} = 'popupmenu';
uitag{i} = 'image_transfer_protocol';
uichoice{i} = ExportInfo.Entries.(uitag{i});

i = 1+1;
uitext{i} = 'Image format information. DIB images: Type the width of the images in pixels. 12-bit TIF images encoded as 16-bit: Type Y. All other formats: Type N.';
uitype{i} = 'edit';
uitag{i} = 'image_size_info';
uistring{i} = ExportInfo.Entries.(uitag{i});

i = 1+1;
uitext{i} = 'What is the image table column for the path to the Red image? (Note: If you have a color image or a single channel, enter data here)';
uitype{i} = 'popupmenu';
uitag{i} = 'red_image_path';
uistring{i} = ExportInfo.Entries.(uitag{i});

i = 1+1;
uitext{i} = 'What is the image table column for the filenames to the Red images? (Note: If you have a color image or a single channel, enter data here)';
uitype{i} = 'popupmenu';
uitag{i} = 'red_image_col';
uistring{i} = ExportInfo.Entries.(uitag{i});

i = 1+1;
uitext{i} = 'What is the image table column for the path to the Green image? (Note: If you have two channels, the second channel must do here)';
uitype{i} = 'popupmenu';
uitag{i} = 'green_image_path';
uistring{i} = ExportInfo.Entries.(uitag{i});

i = 1+1;
uitext{i} = 'What is the image table column for the filenames to the Green images? (Note: If you have a color image or a single channel, enter data here)';
uitype{i} = 'popupmenu';
uitag{i} = 'green_image_col';
uistring{i} = ExportInfo.Entries.(uitag{i});

i = 1+1;
uitext{i} = 'What is the image table column for the path to the Blue image?';
uitype{i} = 'popupmenu';
uitag{i} = 'blue_image_path';
uistring{i} = ExportInfo.Entries.(uitag{i});

i = 1+1;
uitext{i} = 'What is the image table column for the filenames to the Blue image?';
uitype{i} = 'popupmenu';
uitag{i} = 'blue_image_col';
uistring{i} = ExportInfo.Entries.(uitag{i});

i = 1+1;
uitext{i} = 'What is is the X coordinate for the primary object center';
uitype{i} = 'edit';
uitag{i} = 'cell_x_loc';
uichoice(11) = ExportInfo.Entries.(uitag{i});

i = 1+1;
uitext{i} = 'What is is the Y coordinate for the primary object center';
uitype{i} = 'edit';
uitag{i} = 'cell_y_loc';
uistring(12) = ExportInfo.Entries.(uitag{i});

i = 1+1;
uitext{i} = 'Width of Object Cropping Square';
uitype{i} = 'edit';
uitag{i} = 'cell_size';
uistring{i} = ExportInfo.Entries.(uitag{i});

NumberOfEntries = length(uitext);

% Create Export window
PropertiesDisplayFig = CPfigure;
set(PropertiesDisplayFig,'units','pixels','resize','on','menubar','none','toolbar','none',...
    'numbertitle','off','Name','Create Properties window','CloseRequestFcn','set(gcf,''UserData'',0);uiresume()');

% Some variables controlling the sizes of uicontrols
uiheight = 35;

% Set window size in inches, depends on the number of objects
[ScreenWidth,ScreenHeight] = CPscreensize;
Height = .75*ScreenHeight;
Width = 600;

uitextheight = 35*5/4;
borderwidth = uitextheight;
uitextwidth = Width*3/4-borderwidth;
uicontrolwidth = Width*1/4-borderwidth;
uicontrolheight = 20;

% Determine if a slider is needed
if Height > (NumberOfEntries+2)*uiheight,
    SliderRequired = 1;
else
    SliderRequired = 0;
end
% Center it in the screen
LeftPos = (ScreenWidth-Width)/2;
BottomPos = (ScreenHeight-Height)/2;

set(PropertiesDisplayFig,'Position',[LeftPos BottomPos Width Height]);

% Create a uipanel...
PropertiesDisplayPanel = uipanel('parent',PropertiesDisplayFig,'units','pixels',...
    'position',[0 borderwidth Width Height-2*borderwidth],...
    'bordertype','none','BackgroundColor',get(PropertiesDisplayFig,'color'));

% ...that slides if needed
if SliderRequired
    PanelPosition = get(PropertiesDisplayPanel,'position');
    PanelHeight = PanelPosition(4);
    NumberOfEntriesThatFit = floor(PanelHeight/uitextheight);
    SliderData.Callback = @PropertiesFileWindow_SliderCallback;
    SliderData.Panel = PropertiesDisplayPanel;
    SliderData.Borderwidth = borderwidth;
    SliderHandle = uicontrol(PropertiesDisplayFig,'style','slider','units','pixels',...
        'position',[Width-borderwidth*7/8 borderwidth borderwidth*3/4 Height-2*borderwidth],'userdata',SliderData,...
        'Callback','SliderData = get(gco,''UserData''); feval(SliderData.Callback,gco,SliderData); clear SliderData',...
        'Max',Height-borderwidth,'Min',Height-borderwidth-((NumberOfEntries-NumberOfEntriesThatFit)*uitextheight),...
        'Value',Height-borderwidth,'SliderStep',[1/(NumberOfEntries-NumberOfEntriesThatFit) 1.5/(NumberOfEntries-NumberOfEntriesThatFit)]);
    % Height to be used when creating other uicontrols
    ypos = Height - borderwidth - uitextheight;
else
    ypos = Height - borderwidth - uitextheight;
end

%Arrange fields in a two column display, keep track of the y position of the last object created
% Create uicontrols
h = zeros(NumberOfEntries,1);
for i = 1:NumberOfEntries,
    uicontrol('parent',PropertiesDisplayPanel,'units','pixels','style','text','string',uitext{i},...
        'position',[borderwidth ypos uitextwidth uitextheight],...
        'fontname','helvetica','fontsize',ExportInfo.FontSize,'fontweight','bold','horizontalalignment','left','backgroundcolor',get(PropertiesDisplayFig,'color'));
    
    h(i) = uicontrol('parent',PropertiesDisplayPanel,'style',uitype{i},'units','pixels','tag',uitag{i},...
        'position',[uitextwidth+borderwidth ypos+(uitextheight-uicontrolheight) uicontrolwidth uicontrolheight],...
        'fontname','helvetica','fontsize',ExportInfo.FontSize,'fontweight','bold','horizontalalignment','left','units','pixels');
    
    switch get(h(i),'style'),
        case 'popupmenu',   set(h(i),'string',uistring{i},'backgroundcolor','w','value',find(strcmpi(uistring{i},uichoice{i})));
        case 'edit',        set(h(i),'backgroundcolor','w','string',uistring{i});
    end
    
    ypos = ypos - uitextheight;
end

% Hide excess names/checkboxes
if SliderRequired
    PropertiesFileWindow_SliderCallback(SliderHandle,SliderData);
end

% Create OK/Load/Cancel buttons

ExportInfo.Status.Cancelled = 0;
ExportInfo.Status.OK = 0;

ButtonWidth = uicontrolwidth;
ButtonHeight = uicontrolheight;
uicontrol(PropertiesDisplayFig,...
    'style','pushbutton',...
    'String','OK',...
    'units','pixels',...
    'KeyPressFcn', @doFigureKeyPress,...
    'position',[(Width/3-ButtonWidth)/3 borderwidth/4 ButtonWidth ButtonHeight],...
    'Callback','uiresume(fig);',...
    'BackgroundColor',get(PropertiesDisplayFig,'color'));

uicontrol(PropertiesDisplayFig,...
    'style','pushbutton',...
    'String','Load properties file...',...
    'units','pixels',...
    'position',[Width/3+(Width/3-ButtonWidth)/3 borderwidth/4 ButtonWidth ButtonHeight],...
    'Callback',@PropertiesFileWindow_LoadPropertiesFile,...
    'Userdata',ExportInfo,...
    'BackgroundColor',get(PropertiesDisplayFig,'color')); 

uicontrol(PropertiesDisplayFig,...
    'style','pushbutton',...
    'String','Cancel',...
    'units','pixels',...
    'position',[Width/3+2*(Width/3-ButtonWidth)/3 borderwidth/4 ButtonWidth ButtonHeight],...
    'Callback','delete(gcbf);',...
    'BackgroundColor',get(PropertiesDisplayFig,'color')); 

uiwait(PropertiesDisplayFig);

if ishandle(PropertiesDisplayPanel)
    for i = 1:NumberOfEntries,
        switch get(h(i),'style'),
            case 'popupmenu',   str = get(h(i),'string'); ExportInfo.Entries.(uitag{i}) = str{get(h(i),'value')};
            case 'edit',        ExportInfo.Entries.(uitag{i}) = get(h(i),'string');
            case 'pushbutton',  ExportInfo.Entries.(uitag{i}) = get(h(i),'value');
        end
    end
    delete(PropertiesDisplayFig);
    ExportInfo.Status.OK = 1;
else
    ExportInfo.Status.Cancelled = 1;
    return;
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTION - PropertiesFileWindow_SliderCallback
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function PropertiesFileWindow_SliderCallback(SliderHandle,PanelInfo)

PanelHandle = PanelInfo.Panel;
borderwidth = PanelInfo.Borderwidth;

% Get new position for the panel
PanelPos = get(PanelHandle,'position');
NewPos = PanelPos(4) - get(SliderHandle,'value') + borderwidth;

% Hide children, if needed
Children = get(PanelHandle,'children');
for i = 1:length(Children)
    CurrentPos = get(Children(i),'position');
    if CurrentPos(2) + NewPos < borderwidth || ...
            CurrentPos(2) + CurrentPos(4) + NewPos > borderwidth + PanelPos(4)
        set(Children(i),'visible','off');
    else
        set(Children(i),'visible','on');
    end
end
%%% Set the new position
set(PanelHandle,'position',[PanelPos(1) NewPos PanelPos(3) PanelPos(4)]);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTION - PropertiesFileWindow_LoadPropertiesFile
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function PropertiesFileWindow_LoadPropertiesFile(hObject,eventdata)

[filename, pathname] = uigetfile('*.properties','Select the properties file you want to load.');

ExportInfo = get(hObject,'userdata');

fid = fopen(fullfile(pathname,filename),'r');
txt = textscan(fid,'%s', 'delimiter', '\n','whitespace', ''); txt = txt{1};
fclose(fid);

for i = 1:length(txt),
    idx = findstr(txt{i},'='); idx = idx(1);
    ExportInfo.Entries.(strtrim(txt{i}(1:idx-1))) = strtrim(txt{i}(idx+1:end));
end

h = cat(1,findobj(findobj(gcbf,'type','uipanel'),'type','edit'),...
          findobj(findobj(gcbf,'type','uipanel'),'type','popupmenu'));
for i = 1:length(h),
    switch get(h(i),'style'),
        case 'popupmenu',   set(h(i),'value',find(strcmpi(get(h(i),'string'),ExportInfo.Entries.(get(h(i),'tag')))));
        case 'edit',        set(h(i),'string',ExportInfo.Entries.(get(h(i),'tag')));
    end
end

set(hObject,'userdata',ExportInfo);
