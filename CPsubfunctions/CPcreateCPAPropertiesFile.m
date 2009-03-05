function CPcreateCPAPropertiesFile(handles, DataPath, DatabaseName, TablePrefix, DatabaseType,version)

% $Revision$

% This function produces a CellProfiler Analyst properties file based on
% the input handles structure.

if version ~= 1 && version ~= 2
    error(['Property file version ' num2str(version) ' not supported'])
end
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

%% Get nice channel names
channel_names = names(idx_file);
[foo,channel_names] = strtok(channel_names,'_');
channel_names = cellfun(@(x) x(2:end),channel_names,'UniformOutput',0);

% Take a stab at guessing the primary object (the first Measurement object
% that's not an Image, Experiment, or Neighbor)
objs = fieldnames(handles.Measurements);
objs(strcmp(objs,'Image') | strcmp(objs,'Experiment') | strcmp(objs,'Neighbors')) = [];
if ~isempty(objs)
    supposed_primary_obj = objs{end};
else
    supposed_primary_obj = '';
end

switch lower(DatabaseType),                                                                 % Database Port
    case 'mysql', db_port = '3306';
    case 'oracle', db_port = '1521';
end
ExportInfo.Entries.db_type = lower(DatabaseType);                                                  % Database Type
ExportInfo.Entries.db_host = 'imgdb01';                                                            % Database Host Name/IP Address
ExportInfo.Entries.db_port = db_port;
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
ExportInfo.Entries.webinfo_url_prepend = '';                                                % Web Information URL Prepend
ExportInfo.Entries.image_url_prepend = 'http://imageweb/images/CPALinks';                   % Image URL Prepend
ExportInfo.Entries.image_transfer_protocol = 'http';                                        % Image Transfer Protocol
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
    catch
        CPerrordlg(lasterr)
        return
    end
end

% Check whether have write permission in current dir
PropertiesFilename = [DatabaseName '_v' num2str(version) '.properties'];
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
    if version == 1
        entries = fieldnames(ExportInfo.Entries);
        for i = 1:length(entries)
            fprintf(fid, '%s = %s\n', char(entries{i}),ExportInfo.Entries.(char(entries{i})));
        end
    elseif version == 2
        if length(idx_path) == 1
            colors = {'gray'};
        else
            colors = { 'red','green','blue','cyan','magenta','yellow','gray','none','none','none' };
            colors = colors(1:length(idx_path));
        end
        properties_file = {
             '# ==============================================',...
             '#',...
             '# Classifier 2.0 properties file',...
             '#',...
             '# ==============================================',...
             [],...
             '# ==== Database Info ====',...
             sprintf('db_type      = %s',ExportInfo.Entries.db_type),...
             sprintf('db_port      = %s',ExportInfo.Entries.db_port),...
             sprintf('db_host      = %s',ExportInfo.Entries.db_host),...
             sprintf('db_name      = %s',ExportInfo.Entries.db_name),...
             sprintf('db_user      = %s',ExportInfo.Entries.db_user),...
             sprintf('db_passwd    = %s',ExportInfo.Entries.db_pwd),...
             [],...
             '# ==== Database Tables ====',...
             sprintf('image_table   = %s',ExportInfo.Entries.spot_tables),...
             sprintf('object_table  = %s',ExportInfo.Entries.cell_tables),...
             [],...
             '# ==== Database Columns ====' ,...
             '# If multiple tables have been merged, uncomment the table_id line below.  If the merging was done with the CreateMasterTablesWizard, then you can leave uncommenting should work as-is.',...
             '#table_id      = TableNumber',...
             sprintf('image_id      = %s',ExportInfo.Entries.uniqueID),...
             sprintf('object_id     = %s',ExportInfo.Entries.objectID),...
             sprintf('cell_x_loc    = %s',ExportInfo.Entries.cell_x_loc),...
             sprintf('cell_y_loc    = %s',ExportInfo.Entries.cell_y_loc),...
             [],...
             '# ==== Image Path and File Name Columns ====',...
             '# Here you specify the DB columns from your "image_table" that specify the image paths and file names.',...
             '# NOTE: These lists must have equal length!',...
             sprintf('image_channel_paths = %s',CommaDelimitedList(cellfun(@(x)['Image_',x],names(idx_path),'UniformOutput',0))),...
             sprintf('image_channel_files = %s',CommaDelimitedList(cellfun(@(x)['Image_',x],names(idx_file),'UniformOutput',0))),...
             [],...
             '# Give short names for each of the channels (respectively)...',...
             sprintf('image_channel_names = %s',CommaDelimitedList(channel_names)),...
             [],...
             '# ==== Image Accesss Info ====',...
             'image_url_prepend = http://imageweb/images/CPALinks',...
             [],...
             '# ==== Dynamic Groups ====',...
             '# Here you can define groupings to choose from when classifier scores your experiment.  (eg: per-well)',...
             '# This is OPTIONAL, you may leave "groups = ".',...
             '# FORMAT:',...
             '#   groups     =  comma separated list of group names (MUST END IN A COMMA IF THERE IS ONLY ONE GROUP)',...
             '#   group_XXX  =  MySQL select statement that returns image-keys and group-keys.  This will be associated with the group name "XXX" from above.',...
             '# EXAMPLE GROUPS:',...
             '#   groups               =  Well, Gene, Well+Gene,',...
             '#   group_SQL_Well       =  SELECT Per_Image_Table.TableNumber, Per_Image_Table.ImageNumber, Per_Image_Table.well FROM Per_Image_Table',...
             '#   group_SQL_Gene       =  SELECT Per_Image_Table.TableNumber, Per_Image_Table.ImageNumber, Well_ID_Table.gene FROM Per_Image_Table, Well_ID_Table WHERE Per_Image_Table.well=Well_ID_Table.well',...
             '#   group_SQL_Well+Gene  =  SELECT Per_Image_Table.TableNumber, Per_Image_Table.ImageNumber, Well_ID_Table.well, Well_ID_Table.gene FROM Per_Image_Table, Well_ID_Table WHERE Per_Image_Table.well=Well_ID_Table.well',...
             [],...
             'groups  =  ',...
             [],...
             '# ==== Image Filters ====',...
             '# Here you can define image filters to let you select objects from a subset of your experiment when training the classifier.',...
             '# This is OPTIONAL, you may leave "filters = ".',...
             '# FORMAT:',...
             '#   filters         =  comma separated list of filter names (MUST END IN A COMMA IF THERE IS ONLY ONE FILTER)',...
             '#   filter_SQL_XXX  =  MySQL select statement that returns image keys you wish to filter out.  This will be associated with the filter name "XXX" from above.',...
             '# EXAMPLE FILTERS:',...
             '#   filters           =  EMPTY, CDKs,',...
             '#   filter_SQL_EMPTY  =  SELECT TableNumber, ImageNumber FROM CPA_per_image, Well_ID_Table WHERE CPA_per_image.well=Well_ID_Table.well AND Well_ID_Table.Gene="EMPTY"',...
             '#   filter_SQL_CDKs   =  SELECT TableNumber, ImageNumber FROM CPA_per_image, Well_ID_Table WHERE CPA_per_image.well=Well_ID_Table.well AND Well_ID_Table.Gene REGEXP ''CDK.*''',...
             [],...
             'filters  =  ',...
             [],...
             '# ==== Meta data ====',...
             '# What are your objects called?',...
             '# FORMAT:',...
             '#   object_name  =  singular object name, plural object name,',...
             'object_name  =  cell, cells,',...
             [],...
             [],...
             '# ==== Excluded Columns ====',...
             '# DB Columns the classifier should exclude:',...
             'classifier_ignore_substrings  =  table_number_key_column, image_number_key_column, object_number_key_column',...
             [],...
             '# ==== Other ====',...
             '# Specify the approximate diameter of your objects in pixels here.',...
             'image_tile_size   =  50',...
             [],...
             '# ==== Internal Cache ====',...
             '# It shouldn''t be necessary to cache your images in the application, but the cache sizes can be set here.',...
             '# (Units = 1 image. ie: "image_buffer_size = 100", will cache 100 images before it starts replacing old ones.',...
             'image_buffer_size = 1',...
             'tile_buffer_size  = 1',...
             sprintf('image_channel_colors = %s',CommaDelimitedList(colors))...
         };
        fprintf(fid,'%s\n',properties_file{:});
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

i = i+1;
uitext{i} = 'What is the database host name/IP address?';
uitype{i} = 'edit';
uitag{i} = 'db_host';
uistring{i} = ExportInfo.Entries.(uitag{i});

i = i+1;
uitext{i} = 'What is the database port?';
uitype{i} = 'popupmenu';
uitag{i} = 'db_port';
uistring{i} = {'3306','1521'};
uichoice{i} = ExportInfo.Entries.(uitag{i});

i = i+1;
uitext{i} = 'What is the database name?';
uitype{i} = 'edit';
uitag{i} = 'db_name';
uistring{i} = ExportInfo.Entries.(uitag{i});

i = i+1;
uitext{i} = 'What is the user name for access RDMS (relation database management system)?';
uitype{i} = 'edit';
uitag{i} = 'db_user';
uistring{i} = ExportInfo.Entries.(uitag{i});

i = i+1;
uitext{i} = 'What is the user password for access to the RDMS?';
uitype{i} = 'edit';
uitag{i} = 'db_pwd';
uistring{i} = ExportInfo.Entries.(uitag{i});

i = i+1;
uitext{i} = 'What is the prefix for the per image tables?';
uitype{i} = 'edit';
uitag{i} = 'spot_tables';
uistring{i} = ExportInfo.Entries.(uitag{i});

i = i+1;
uitext{i} = 'What is the prefix for the per object tables?';
uitype{i} = 'edit';
uitag{i} = 'cell_tables';
uistring{i} = ExportInfo.Entries.(uitag{i});

i = i+1;
uitext{i} = 'What is image identifer?';
uitype{i} = 'edit';
uitag{i} = 'uniqueID';
uistring{i} = ExportInfo.Entries.(uitag{i});

i = i+1;
uitext{i} = 'What is the primary object count column?';
uitype{i} = 'edit';
uitag{i} = 'objectCount';
uistring{i} = ExportInfo.Entries.(uitag{i});

i = i+1;
uitext{i} = 'What is the object identifer?';
uitype{i} = 'edit';
uitag{i} = 'objectID';
uistring{i} = ExportInfo.Entries.(uitag{i});

i = i+1;
uitext{i} = 'What is the name of the information table?';
uitype{i} = 'edit';
uitag{i} = 'info_table';
uistring{i} = ExportInfo.Entries.(uitag{i});

i = i+1;
uitext{i} = 'What is the Web information prepend?';
uitype{i} = 'edit';
uitag{i} = 'image_url_prepend';
uistring{i} = ExportInfo.Entries.(uitag{i});

i = i+1;
uitext{i} = 'What is the image transfer protocol (ITP)';
uistring{i} = {'http','local','smb','ssh'};
uitype{i} = 'popupmenu';
uitag{i} = 'image_transfer_protocol';
uichoice{i} = ExportInfo.Entries.(uitag{i});

i = i+1;
uitext{i} = 'Image format information. DIB images: Type the width of the images in pixels. 12-bit TIF images encoded as 16-bit: Type Y. All other formats: Type N.';
uitype{i} = 'edit';
uitag{i} = 'image_size_info';
uistring{i} = ExportInfo.Entries.(uitag{i});

i = i+1;
uitext{i} = 'What is the image table column for the path to the Red image? (Note: If you have a color image or a single channel, enter data here)';
uitype{i} = 'edit';
uitag{i} = 'red_image_path';
uistring{i} = ExportInfo.Entries.(uitag{i});

i = i+1;
uitext{i} = 'What is the image table column for the filenames to the Red images? (Note: If you have a color image or a single channel, enter data here)';
uitype{i} = 'edit';
uitag{i} = 'red_image_col';
uistring{i} = ExportInfo.Entries.(uitag{i});

if isfield(ExportInfo.Entries,'green_image_path')
    i = i+1;
    uitext{i} = 'What is the image table column for the path to the Green image? (Note: If you have two channels, the second channel must do here)';
    uitype{i} = 'edit';
    uitag{i} = 'green_image_path';
    uistring{i} = ExportInfo.Entries.(uitag{i});

    i = i+1;
    uitext{i} = 'What is the image table column for the filenames to the Green images? (Note: If you have a color image or a single channel, enter data here)';
    uitype{i} = 'edit';
    uitag{i} = 'green_image_col';
    uistring{i} = ExportInfo.Entries.(uitag{i});
end
if isfield(ExportInfo.Entries,'blue_image_path')
    i = i+1;
    uitext{i} = 'What is the image table column for the path to the Blue image?';
    uitype{i} = 'edit';
    uitag{i} = 'blue_image_path';
    uistring{i} = ExportInfo.Entries.(uitag{i});

    i = i+1;
    uitext{i} = 'What is the image table column for the filenames to the Blue image?';
    uitype{i} = 'edit';
    uitag{i} = 'blue_image_col';
    uistring{i} = ExportInfo.Entries.(uitag{i});
end
i = i+1;
uitext{i} = 'What is is the X coordinate for the primary object center';
uitype{i} = 'edit';
uitag{i} = 'cell_x_loc';
uistring{i} = ExportInfo.Entries.(uitag{i});

i = i+1;
uitext{i} = 'What is is the Y coordinate for the primary object center';
uitype{i} = 'edit';
uitag{i} = 'cell_y_loc';
uistring{i} = ExportInfo.Entries.(uitag{i});

i = i+1;
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

% Center it in the screen
LeftPos = (ScreenWidth-Width)/2;
BottomPos = (ScreenHeight-Height)/2;

set(PropertiesDisplayFig,'Position',[LeftPos BottomPos Width Height]);

% Create a uipanel...
PropertiesDisplayPanel = uipanel('parent',PropertiesDisplayFig,'units','pixels',...
    'position',[0 borderwidth Width Height-2*borderwidth],...
    'bordertype','none','BackgroundColor',get(PropertiesDisplayFig,'color'));

PanelPosition = get(PropertiesDisplayPanel,'position');
PanelHeight = PanelPosition(4);
NumberOfEntriesThatFit = floor(PanelHeight/uitextheight);

% Determine if a slider is needed
if NumberOfEntriesThatFit < NumberOfEntries,
    SliderRequired = 1;
else
    SliderRequired = 0;
end

% ...that slides if needed
if SliderRequired
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
    'Callback','uiresume(gcbf);',...
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTION - CommaDelimitedList
%%% x - cell array of strings
%%% returns a string composed of the strings, separated by commas
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function result=CommaDelimitedList(x)

if isempty(x)
    result=[];
    return
end 
y = cellfun(@(x)[x,','],x(:),'UniformOutput',0);
result = [ y{:}];
return
