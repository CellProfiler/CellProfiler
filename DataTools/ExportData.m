function ExportData(handles)

% Help for the Export Data tool:
% Category: Data Tools
%
% SHORT DESCRIPTION:
% Exports measurements into a tab-delimited text file which can be opened
% in Excel.
% *************************************************************************
%
% Once image analysis is complete, use this tool to select the
% output file to extract the measurements and other information about
% the analysis.  The data will be converted to a tab-delimited text file
% which can be read by for example Excel or in a text editor.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne Carpenter
%   Thouis Jones
%   In Han Kang
%   Ola Friman
%   Steve Lowe
%   Joo Han Chang
%   Colin Clarke
%   Mike Lamprecht
%   Susan Ma
%   Wyman Li
%
% Website: http://www.cellprofiler.org
%
% $Revision: 2644 $

%%% Ask the user to choose the file from which to extract measurements.
if exist(handles.Current.DefaultOutputDirectory, 'dir')
    [RawFileName, RawPathname] = uigetfile(fullfile(handles.Current.DefaultOutputDirectory,'.','*.mat'),'Select the raw measurements file');
    PathToSave = handles.Current.DefaultOutputDirectory;
else
    [RawFileName, RawPathname] = uigetfile('*.mat','Select the raw measurements file');
    PathToSave = RawPathname;
end

if RawFileName == 0
    return
end

%%% Load the specified CellProfiler output file.
Loaded = load(fullfile(RawPathname, RawFileName));

%%% Check if it seems to be a CellProfiler output file or not.
if isfield(Loaded,'handles')
    handles = Loaded.handles;
    clear Loaded
else
    CPerrordlg('The selected file does not seem to be a CellProfiler output file.')
    return
end

%%% Opens a window that lets the user chose what to export
try ExportInfo = ObjectsToExport(handles,RawFileName);
catch CPerrordlg(lasterr)
    return
end

%%% Indicates that the Cancel button was pressed
if isempty(ExportInfo.ObjectNames)
    %%% If nothing is chosen, we still want to check if the user wants to
    %%% export the process info
    %%% Export process info
    if isfield(ExportInfo,'ExportProcessInfo')
        if strcmp(ExportInfo.ExportProcessInfo,'Yes')
            try CPtextpipe(handles,ExportInfo,RawFileName,RawPathname);
            catch CPerrordlg(lasterr)
                return
            end
        end
    end
    return
end

%%% Export process info
if strcmp(ExportInfo.ExportProcessInfo,'Yes')
    try CPtextpipe(handles,ExportInfo,RawFileName,RawPathname);
    catch CPerrordlg(lasterr)
        return
    end
end

%%% Export measurements
if ~isempty(ExportInfo.MeasurementFilename)
    try CPwritemeasurements(handles,ExportInfo,RawPathname);
    catch CPerrordlg(lasterr)
        return
    end
end

%%% Done!
CPmsgbox('Exporting is completed.')

%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTIONS %%%
%%%%%%%%%%%%%%%%%%%%

function ExportInfo = ObjectsToExport(handles,RawFileName)
% This function displays a window so that lets the user choose which
% measurements to export. If the return variable 'ObjectNames' is empty
% it means that either no measurements were found or the user pressed
% the Cancel button (or the window was closed). 'Summary' takes on the values 'yes'
% or 'no', depending if the user only wants a summary report (mean and std)
% or a full report.

% Initialize output variables
ExportInfo.ObjectNames = [];
ExportInfo.MeasurementFilename = [];
ExportInfo.ProcessInfoFilename = [];

% The fontsize is stored in the 'UserData' property of the main Matlab window
GUIhandles = guidata(gcbo);
FontSize = GUIhandles.Preferences.FontSize;

% Get measurement object fields
fields = fieldnames(handles.Measurements);
if length(fields) > 20
    error('There are more than 20 different objects in the chosen file. There is probably something wrong in the handles.Measurement structure.')

end

% Create Export window
ETh = figure;
set(ETh,'units','inches','resize','on','menubar','none','toolbar','none','numbertitle','off','Name','Export window','Color',[.7 .7 .9],'CloseRequestFcn','set(gcf,''UserData'',0);uiresume()');

% Some variables controling the sizes of uicontrols
uiheight = 0.3;

% Set window size in inches, depends on the number of objects
pos = get(ETh,'position');
Height = 2.5+length(fields)*uiheight;
Width  = 4.2;
set(ETh,'position',[pos(1)+1 pos(2) Width Height]);

if ~isempty(fields)
    % Top text
    uicontrol(ETh,'style','text','String','The following measurements were found:','FontName','Times','FontSize',FontSize,...
        'HorizontalAlignment','left','units','inches','position',[0.2 Height-0.25 4 0.2],'BackgroundColor',get(ETh,'color'))

    % Radio buttons for extracted measurements
    h = [];
    for k = 1:length(fields)
        uicontrol(ETh,'style','text','String',fields{k},'FontName','Times','FontSize',FontSize,'HorizontalAlignment','left',...
            'units','inches','position',[0.6 Height-0.3-uiheight*k 3 0.18],'BackgroundColor',get(ETh,'color'))
        h(k) = uicontrol(ETh,'Style','checkbox','units','inches','position',[0.2 Height-0.35-uiheight*k uiheight uiheight],...
            'BackgroundColor',get(ETh,'color'),'Value',1);
    end

    % Filename, remove 'OUT' and '.mat' extension from filename
    basey = 1.5;
    ProposedFilename = RawFileName;
    indexOUT = strfind(ProposedFilename,'OUT');
    if ~isempty(indexOUT),ProposedFilename = [ProposedFilename(1:indexOUT(1)-1) ProposedFilename(indexOUT(1)+3:end)];end
    indexMAT = strfind(ProposedFilename,'mat');
    if ~isempty(indexMAT),ProposedFilename = [ProposedFilename(1:indexMAT(1)-2) ProposedFilename(indexMAT(1)+3:end)];end
    ProposedFilename = [ProposedFilename,'_Export'];
    uicontrol(ETh,'style','text','String','Base file name for exported files:','FontName','Times','FontSize',FontSize,...
        'HorizontalAlignment','left','units','inches','position',[0.2 basey+0.2 2.3 uiheight],'BackgroundColor',get(ETh,'color'))
    EditMeasurementFilename = uicontrol(ETh,'Style','edit','units','inches','position',[0.2 basey 2.5 uiheight],...
        'backgroundcolor',[1 1 1],'String',ProposedFilename,'FontSize',FontSize);
    uicontrol(ETh,'style','text','String','Choose extension:','FontName','Times','FontSize',FontSize,...
        'HorizontalAlignment','left','units','inches','position',[2.9 basey+0.2 1.2 uiheight],'BackgroundColor',get(ETh,'color'))
    EditMeasurementExtension = uicontrol(ETh,'Style','edit','units','inches','position',[2.9 basey 0.7 uiheight],...
        'backgroundcolor',[1 1 1],'String','.xls','FontSize',FontSize);

else  % No measurements found
    uicontrol(ETh,'style','text','String','No measurements found!','FontName','Times','FontSize',FontSize,...
        'units','inches','position',[0 Height-0.5 6 0.15],'BackgroundColor',get(ETh,'color'),'fontweight','bold')
end

%%% Process info
basey = 0.65;
% Propose a filename. Remove 'OUT' and '.mat' extension from filename
ProposedFilename = RawFileName;
indexOUT = strfind(ProposedFilename,'OUT');
if ~isempty(indexOUT),ProposedFilename = [ProposedFilename(1:indexOUT(1)-1) ProposedFilename(indexOUT(1)+3:end)];end
indexMAT = strfind(ProposedFilename,'mat');
if ~isempty(indexMAT),ProposedFilename = [ProposedFilename(1:indexMAT(1)-2) ProposedFilename(indexMAT(1)+3:end)];end
ProposedFilename = [ProposedFilename,'_ProcessInfo'];
uicontrol(ETh,'style','text','String','Export pipeline settings?','FontName','Times','FontSize',FontSize,...
    'HorizontalAlignment','left','units','inches','position',[0.2 basey+0.23 1.6 uiheight],'BackgroundColor',get(ETh,'color'));
ExportProcessInfo = uicontrol(ETh,'style','popupmenu','String',{'No','Yes'},'FontName','Times','FontSize',FontSize,...
    'HorizontalAlignment','left','units','inches','position',[2 basey+0.33 0.7 uiheight],'BackgroundColor',[1 1 1]);
uicontrol(ETh,'style','text','String','Each feature is a:','FontName','Times','FontSize',FontSize,...
    'HorizontalAlignment','left','units','inches','position',[2.7 basey+1.6 1.8 uiheight],'BackgroundColor',get(ETh,'color'));
SwapRowsColumnInfo = uicontrol(ETh,'style','popupmenu','String',{'Column','Row'},'FontName','Times','FontSize',FontSize,...
    'HorizontalAlignment','left','units','inches','position',[2.6 basey+1.4 1 uiheight],'BackgroundColor',[1 1 1]);
EditProcessInfoFilename = uicontrol(ETh,'Style','edit','units','inches','position',[0.2 basey 2.5 uiheight],...
    'backgroundcolor',[1 1 1],'String',ProposedFilename,'FontSize',FontSize);
uicontrol(ETh,'style','text','String','Choose extension:','FontName','Times','FontSize',FontSize,...
    'HorizontalAlignment','left','units','inches','position',[2.9 basey+0.2 1.2 uiheight],'BackgroundColor',get(ETh,'color'),'FontSize',FontSize)
EditProcessInfoExtension = uicontrol(ETh,'Style','edit','units','inches','position',[2.9 basey 0.7 uiheight],...
    'backgroundcolor',[1 1 1],'String','.txt','FontSize',FontSize);
uicontrol(ETh,'style','text','String','Ignore NaN''s?','FontName','Times','FontSize',FontSize,'HorizontalAlignment','left',...
    'units','inches','position',[2.9 basey+2.35 1.8 uiheight],'BackgroundColor',get(ETh,'color'))
IgnoreNaN = uicontrol(ETh,'Style','checkbox','units','inches','position',[3.4 basey+2.2 1.8 uiheight],...
    'BackgroundColor',get(ETh,'color'),'Value',1);

% Export and Cancel pushbuttons
posx = (Width - 1.7)/2;               % Centers buttons horizontally
exportbutton = uicontrol(ETh,'style','pushbutton','String','Export','FontName','Times','FontSize',FontSize,'units','inches',...
    'position',[posx 0.1 0.75 0.3],'Callback','[foo,fig] = gcbo;set(fig,''UserData'',1);uiresume(fig);clear fig foo','BackgroundColor',[.7 .7 .9]);
cancelbutton = uicontrol(ETh,'style','pushbutton','String','Cancel','FontName','Times','FontSize',FontSize,'units','inches',...
    'position',[posx+0.95 0.1 0.75 0.3],'Callback','close(gcf)','BackgroundColor',[.7 .7 .9]);

uiwait(ETh)                         % Wait until window is destroyed or uiresume() is called

ExportInfo.IgnoreNaN = get(IgnoreNaN,'Value');

if get(ETh,'Userdata') == 1     % The user pressed the Export button

    % File names
    if ~isempty(fields)
        ExportInfo.MeasurementFilename = get(EditMeasurementFilename,'String');
        ExportInfo.MeasurementExtension = get(EditMeasurementExtension,'String');
    end
    ExportInfo.ProcessInfoFilename = get(EditProcessInfoFilename,'String');
    ExportInfo.ProcessInfoExtension = get(EditProcessInfoExtension,'String');
    if get(ExportProcessInfo,'Value') == 1                                       % Indicates a 'No' (equals 2 if 'Yes')
        ExportInfo.ExportProcessInfo = 'No';
    else
        ExportInfo.ExportProcessInfo = 'Yes';
    end
    if get(SwapRowsColumnInfo,'Value') == 1
        ExportInfo.SwapRowsColumnInfo = 'No';
    else
        ExportInfo.SwapRowsColumnInfo = 'Yes';
    end

    % Get measurements to export
    if ~isempty(fields)
        buttonchoice = get(h,'Value');
        if iscell(buttonchoice)                              % buttonchoice will be a cell array if there are several objects
            buttonchoice = cat(1,buttonchoice{:});
        end
        ExportInfo.ObjectNames = fields(find(buttonchoice));  % Get the fields for which the radiobuttons are enabled
    end
    delete(ETh)
else
    delete(ETh);
    ExportInfo.ObjectNames = [];
end