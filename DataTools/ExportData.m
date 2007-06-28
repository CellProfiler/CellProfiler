function ExportData(handles)

% Help for the Export Data tool:
% Category: Data Tools
%
% SHORT DESCRIPTION:
% Exports measurements into a tab-delimited text file which can be opened
% in Excel or other spreadsheet programs.
% *************************************************************************
%
% Once image analysis is complete, use this data tool to select the output
% file to extract the measurements and other information about the
% analysis. The data will be converted to a tab-delimited text file which
% can be read by Excel, another spreadsheet program, or a text editor. You
% can add the ExportToExcel module to your pipeline if you want to
% automatically export data.
%
% See also ExportDatabase data tool, ExportToDatabase module, ExportToExcel
% module.

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
%   Peter Swire
%   Rodrigo Ipince
%   Vicky Lay
%   Jun Liu
%   Chris Gang
%
% Website: http://www.cellprofiler.org
%
% $Revision$


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

DataExists=0;

while DataExists == 0

    %%% Opens a window that lets the user choose what to export
    try ExportInfo = ObjectsToExport(handles,RawFileName);
    catch CPerrordlg(lasterr)
        return
    end

    %%% Indicates that the Cancel button was pressed
    if ~isfield(ExportInfo, 'ExportProcessInfo')
        return
    end

    if ~isempty(ExportInfo.ObjectNames)
        DataExists=1;
    else
        %%% If nothing is chosen, we still want to check if the user wants to
        %%% export the process info
        %%% Export process info
        if strcmp(ExportInfo.ExportProcessInfo,'Yes')
            try
                DataExists=1;
                CPtextpipe(handles,ExportInfo,RawFileName,RawPathname);
            catch CPerrordlg(lasterr)
                return
            end
        else
            warnfig = CPwarndlg('You must select at least one measurement to export. If you wish to only export pipeline settings and not measurements, type a settings extension. Please try again.');
            uiwait(warnfig)
        end
    end
end

if isfield(ExportInfo, 'ExportProcessInfo')
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
    if strcmp(ExportInfo.ExportProcessInfo, 'Yes') && isempty(ExportInfo.ObjectNames)
        CPmsgbox(['Exporting is complete. Your pipeline settings have been saved as ', ExportInfo.ProcessInfoFilename, ExportInfo.ProcessInfoExtension, ' in the default output folder, ', PathToSave, '.'])
    elseif strcmp(ExportInfo.ExportProcessInfo, 'Yes')
        CPmsgbox(['Exporting is complete. Your exported data has been saved as ', ExportInfo.MeasurementExtension, ' files with base name ', ExportInfo.MeasurementFilename, ' and your pipeline settings have been saved as ', ExportInfo.ProcessInfoFilename, ExportInfo.ProcessInfoExtension, ' in the default output folder, ', PathToSave, '.'])
    else
        CPmsgbox(['Exporting is complete. Your exported data has been saved as ', ExportInfo.MeasurementExtension, ' files with base name ', ExportInfo.MeasurementFilename, ' in the default output folder, ', PathToSave, '.'])
    end
end

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
ExportInfo.Cancelled = [];

% The fontsize is stored in the 'UserData' property of the main MATLAB window
GUIhandles = guidata(gcbo);
FontSize = GUIhandles.Preferences.FontSize;

% Get measurement object fields
fields = fieldnames(handles.Measurements);
if length(fields) > 20
    error('There are more than 20 different objects in the chosen file. There is probably something wrong in the handles.Measurement structure.')

end

% Create Export window
ETh = CPfigure;
set(ETh,'units','pixels','resize','on','menubar','none','toolbar','none','numbertitle','off','Name','Export window','CloseRequestFcn','set(gcf,''UserData'',0);uiresume()');
% Some variables controlling the sizes of uicontrols
uiheight = 28;
% Set window size in inches, depends on the number of objects
ScreenSize = get(0,'ScreenSize');
Height = 330+ceil(length(fields)/2)*uiheight;
Width  = 600;
LeftPos = (ScreenSize(3)-Width)/2;
BottomPos = (ScreenSize(4)-Height)/2;
set(ETh,'position',[LeftPos BottomPos Width Height]);

if ~isempty(fields)
    NumberOfImageSetsCompleted = num2str(handles.Current.SetBeingAnalyzed); 
    NumberOfImageSetsPlanned = num2str(handles.Current.NumberOfImageSets);
    % Top text
    uicontrol(ETh,'style','text','String',...
        [NumberOfImageSetsCompleted,' image cycles were completed out of ',NumberOfImageSetsPlanned,' total cycles planned.'],...
        'FontName','helvetica','FontSize',FontSize,'FontWeight', 'bold',...
        'HorizontalAlignment','left','units','pixels','position',[20 Height-30 400 20],'BackgroundColor',get(ETh,'color'))
    
    uicontrol(ETh,'style','text','String','Measurements to export:','FontName','helvetica','FontSize',FontSize,'FontWeight', 'bold',...
        'HorizontalAlignment','left','units','pixels','position',[20 Height-60 400 20],'BackgroundColor',get(ETh,'color'))

    % Radio buttons for extracted measurements
    h = [];
    %Arrange fields in a two column display, keep track of the y position
    %of the last object created
    ypos = Height - uiheight-30;
    for k = 1:length(fields)
        if rem(k,2) == 1 %when index is odd
            ypos=ypos-uiheight;
        end   %index is even
        if rem(k,2) == 1
            uicontrol(ETh,'style','text','String',fields{k},'FontName','helvetica','FontSize',FontSize,'HorizontalAlignment','left',...
                'units','pixels','position',[50 ypos 200 20],'BackgroundColor',get(ETh,'color'))
            h(k) = uicontrol(ETh,'Style','checkbox','units','pixels','position',[20 ypos uiheight uiheight],...
                'BackgroundColor',get(ETh,'color'),'Value',1);
        else
            uicontrol(ETh,'style','text','String',fields{k},'FontName','helvetica','FontSize',FontSize,'HorizontalAlignment','left',...
                'units','pixels','position',[300 ypos 200 20],'BackgroundColor',get(ETh,'color'))
            h(k) = uicontrol(ETh,'Style','checkbox','units','pixels','position',[270 ypos uiheight uiheight],...
                'BackgroundColor',get(ETh,'color'),'Value',1);
        end
    end

else  % No measurements found
    uicontrol(ETh,'style','text','String','No measurements found','FontName','helvetica','FontSize',FontSize,...
        'units','pixels','position',[0 Height-80 600 15],'BackgroundColor',get(ETh,'color'),'fontweight','bold')
end
% Propose a filename. Remove 'OUT' and '.mat' extension from filename
ProposedFilename = RawFileName;
indexOUT = strfind(ProposedFilename,'OUT');
if ~isempty(indexOUT),
    ProposedFilename = [ProposedFilename(1:indexOUT(1)-1) ProposedFilename(indexOUT(1)+3:end)];
end
indexMAT = strfind(ProposedFilename,'mat');
if ~isempty(indexMAT),
    ProposedFilename = [ProposedFilename(1:indexMAT(1)-2) ProposedFilename(indexMAT(1)+3:end)];
end

ypos=ypos-uiheight*2;

uicontrol(ETh,'style','text','String','Each measured feature should be a:','FontName','helvetica','FontSize',FontSize,'FontWeight', 'bold',...
    'HorizontalAlignment','left','units','pixels','position',[20 ypos 400 uiheight],'BackgroundColor',get(ETh,'color'));
SwapRowsColumnInfo = uicontrol(ETh,'style','popupmenu','String',{'Column','Row'},'FontName','helvetica','FontSize',FontSize,...
    'HorizontalAlignment','left','units','pixels','position',[400 ypos+5 150 uiheight],'BackgroundColor',get(ETh, 'color'));
%Help button
Help_Callback = 'CPhelpdlg(''** If you are exporting Image data, the spreadsheet will contain measured features for each image, and you can choose whether the features should be columns and the images should be rows, or vice versa.  ** If you are exporting Object data, the spreadsheet will contain measured features for each object, and you can choose whether the features should be columns and the objects should be rows, or vice versa.  ** Because some programs (e.g. Excel) have a limit of 256 columns and 65536 rows, swapping the orientation of your exported data may be critical for opening it conveniently and completely.'')';
uicontrol(ETh,'style','pushbutton','String','?','FontName','helvetica','FontSize',FontSize,...
    'HorizontalAlignment','center','units','pixels','position',[560 ypos+8 15 uiheight],...
    'BackgroundColor',get(ETh,'color'),'FontWeight', 'bold',...
    'Callback', Help_Callback);

ypos=ypos-uiheight;
uicontrol(ETh,'style','text','String','If exporting Image data, calculate and export:','FontName','helvetica','FontSize',FontSize,'FontWeight', 'bold',...
    'HorizontalAlignment','left','units','pixels','position',[20 ypos 400 uiheight],'BackgroundColor',get(ETh,'color'));
DataExportParameter = uicontrol(ETh,'style','popupmenu','String',{'Means','Medians','Standard Deviations'},'FontName','helvetica','FontSize',FontSize,...
    'HorizontalAlignment','left','units','pixels','position',[400 ypos+5 150 uiheight],'BackgroundColor',get(ETh, 'color'));
%Help button
Help_Callback = 'CPhelpdlg(''Most per-image data is calculated from per-object data. For example, you can choose to calculate the mean, median or standard deviation of the per-object data (e.g. for each image, you might want to know the MEAN cell area). To export more than one of these options (mean, median, standard deviation), simply repeat the ExportData command and export with a different file name.'')';
uicontrol(ETh,'style','pushbutton','String','?','FontName','helvetica','FontSize',FontSize,...
    'HorizontalAlignment','center','units','pixels','position',[560 ypos+8 15 uiheight],...
    'BackgroundColor',get(ETh,'color'),'FontWeight', 'bold',...
    'Callback', Help_Callback);

ypos=ypos-uiheight;
uicontrol(ETh,'style','text','String','Ignore NaN''s (Not a Numbers) in that calculation?','FontName','helvetica','FontSize',FontSize,'FontWeight', 'bold',...
    'HorizontalAlignment','left','units','pixels','position',[20 ypos 400 uiheight],'BackgroundColor',get(ETh,'color'));
IgnoreNaN = uicontrol(ETh,'style','popupmenu','String',{'Yes','No'},'FontName','helvetica','FontSize',FontSize,...
    'HorizontalAlignment','left','units','pixels','position',[400 ypos+5 150 uiheight],'BackgroundColor',get(ETh, 'color'));
%Help button
Help_Callback = 'CPhelpdlg(''Sometimes a measurement is recorded as Not a Number, which means that it could not be calculated for some reason. For example, you cannot calculate the texture (smoothness) of an object made of only one pixel. So, when you want to calculate the mean texture for all objects in that image, you have two options: ignore NaNs and calculate the mean only for those objects that have a numerical texture measurement, or, if a NaN is present, record the mean measurement also as NaN.'')';

uicontrol(ETh,'style','pushbutton','String','?','FontName','helvetica','FontSize',FontSize,...
    'HorizontalAlignment','center','units','pixels','position',[560 ypos+8 15 uiheight],...
    'BackgroundColor',get(ETh,'color'),'FontWeight', 'bold',...
    'Callback', Help_Callback);

ypos=ypos-uiheight;
uicontrol(ETh,'style','text','String','Base filename for exported files:',...
    'FontName','helvetica','FontSize',FontSize,'FontWeight', 'bold',...
    'HorizontalAlignment','left','units','pixels','position',[20 ypos 200 uiheight],...
    'BackgroundColor',get(ETh,'color'));
EditMeasurementFilename = uicontrol(ETh,'Style','edit','units','pixels','position',[300 ypos+5 250 uiheight*.8],...
    'backgroundcolor',[1 1 1],'String',ProposedFilename,'FontSize',FontSize);
%Help button
Help_Callback = 'CPhelpdlg(''Enter the base file name which will be used to name all of your exported data files. ** For example, if you export data for Image and Cells with "Default" in the basename window, ".xls" in the file extension window, and ".txt" in the pipeline settings extension window, CellProfiler will produce three files: Default_Image.xls, Default_Cells.xls, and Default.txt .'')';
uicontrol(ETh,'style','pushbutton','String','?','FontName','helvetica','FontSize',FontSize,...
    'HorizontalAlignment','center','units','pixels','position',[560 ypos 15 uiheight],...
    'BackgroundColor',get(ETh,'color'),'FontWeight', 'bold',...
    'Callback', Help_Callback);

ypos=ypos-uiheight;
uicontrol(ETh,'style','text','String','Extension for exported measurement files:','FontName','helvetica','FontSize',FontSize,'FontWeight', 'bold',...
    'HorizontalAlignment','left','units','pixels','position',[20 ypos 400 uiheight],'BackgroundColor',get(ETh,'color'));
EditMeasurementExtension = uicontrol(ETh,'Style','edit','units','pixels','position',[500 ypos+5 50 uiheight*.8],...
    'backgroundcolor',[1 1 1],'String','.xls','FontSize',FontSize);
%Help button
Help_Callback = 'CPhelpdlg(''For the data files containing measurements, enter the extension you wish to use here. The files produced are tab-delimited format, which can be opened in many spreadsheet programs and also some text editing programs. Using .xls will allow programs like Microsoft Excel to automatically recognize and open the tab-delimited format. ** For example, if you export data for Image and Cells with "Default" in the basename window, ".xls" in the file extension window, and ".txt" in the pipeline settings extension window, CellProfiler will produce three files: Default_Image.xls, Default_Cells.xls, and Default.txt .'')';
uicontrol(ETh,'style','pushbutton','String','?','FontName','helvetica','FontSize',FontSize,...
    'HorizontalAlignment','center','units','pixels','position',[560 ypos 15 uiheight],'BackgroundColor',get(ETh,'color'),'FontWeight', 'bold',...
    'Callback', Help_Callback);

ypos=ypos-uiheight;
uicontrol(ETh,'style','text','String','Extension for exported pipeline settings file (optional):','FontName','helvetica','FontSize',FontSize,'FontWeight', 'bold',...
    'HorizontalAlignment','left','units','pixels','position',[20 ypos 500 uiheight],'BackgroundColor',get(ETh,'color'),'FontSize',FontSize);
EditProcessInfoExtension = uicontrol(ETh,'Style','edit','units','pixels','position',[500 ypos+5 50 uiheight*.8],...
    'backgroundcolor',[1 1 1],'String','.txt','FontSize',FontSize);
%Help button
Help_Callback = 'CPhelpdlg(''To export the pipeline settings, type the desired file extension (e.g., .txt, .doc) so that the resulting file can be opened easily in a word processing program. The pipeline settings include the date and time of processing, module names in order, and the description and actual settings for each module. To refrain from exporting pipeline settings, make this box empty. ** For example, if you export data for Image and Cells with "Default" in the basename window, ".xls" in the file extension window, and ".txt" in the pipeline settings extension window, CellProfiler will produce three files: Default_Image.xls, Default_Cells.xls, and Default.txt .'')';
uicontrol(ETh,'style','pushbutton','String','?','FontName','helvetica','FontSize',FontSize,...
    'HorizontalAlignment','center','units','pixels','position',[560 ypos 15 uiheight],'BackgroundColor',get(ETh,'color'),'FontWeight', 'bold',...
    'Callback', Help_Callback);

% Export and Cancel pushbuttons
posx = (Width - 200)/2;               % Centers buttons horizontally
uicontrol(ETh,'style','pushbutton','String','Cancel','FontName','helvetica','FontSize',FontSize,'FontWeight', 'bold','units','pixels',...
    'position',[posx 10 75 uiheight],'Callback','close(gcf)','BackgroundColor',[.7 .7 .9]);
uicontrol(ETh,'style','pushbutton','String','Export','FontName','helvetica','FontSize',FontSize,'FontWeight', 'bold','units','pixels',...
    'position',[posx+125 10 75 uiheight],'Callback','[foo,fig] = gcbo;set(fig,''UserData'',1);uiresume(fig);clear fig foo','BackgroundColor',[.7 .7 .9]);

uiwait(ETh)                         % Wait until window is destroyed or uiresume() is called

ExportInfo.IgnoreNaN = get(IgnoreNaN,'Value');

if get(ETh,'Userdata') == 1     % The user pressed the Export button

    % File names
    if ~isempty(fields)
        ExportInfo.MeasurementFilename = get(EditMeasurementFilename,'String');
        ExportInfo.MeasurementExtension = get(EditMeasurementExtension,'String');
    end
    if isempty(get(EditProcessInfoExtension, 'String'));             % Indicates a 'No' (contains string if 'Yes')
        ExportInfo.ExportProcessInfo = 'No';
    else
        ExportInfo.ExportProcessInfo = 'Yes';
        ExportInfo.ProcessInfoFilename = get(EditMeasurementFilename,'String');
        ExportInfo.ProcessInfoExtension = get(EditProcessInfoExtension,'String');
    end
    if get(SwapRowsColumnInfo,'Value') == 1
        ExportInfo.SwapRowsColumnInfo = 'No';
    else
        ExportInfo.SwapRowsColumnInfo = 'Yes';
    end

    if get(DataExportParameter,'Value')==1
        ExportInfo.DataParameter = 'mean';
    else if get(DataExportParameter,'Value')==2
            ExportInfo.DataParameter = 'median';
        else if get(DataExportParameter,'Value')==3
                ExportInfo.DataParameter = 'std';
            end;
        end;
    end;

    % Get measurements to export
    if ~isempty(fields)
        buttonchoice = get(h,'Value');
        if iscell(buttonchoice)                              % buttonchoice will be a cell array if there are several objects
            buttonchoice = cat(1,buttonchoice{:});
        end
        ExportInfo.ObjectNames = fields(find(buttonchoice));  %#ok Get the fields for which the radiobuttons are enabled
    end

    delete(ETh)
else
    delete(ETh);
    ExportInfo.ObjectNames = [];
end