function ExportData(handles)

% Help for the Export Data tool:
% Category: Data Tools
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
%   Anne Carpenter <carpenter@wi.mit.edu>
%   Thouis Jones   <thouis@csail.mit.edu>
%   In Han Kang    <inthek@mit.edu>
%   Ola Friman     <friman@bwh.harvard.edu>
%   Steve Lowe     <stevelowe@alum.mit.edu>
%   Joo Han Chang  <joohan.chang@gmail.com>
%   Colin Clarke   <colinc@mit.edu>
%   Mike Lamprecht <mrl@wi.mit.edu>
%   Susan Ma       <xuefang_ma@wi.mit.edu>
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
    errordlg('The selected file does not seem to be a CellProfiler output file.')
    return
end

%%% Opens a window that lets the user chose what to export
try ExportInfo = ObjectsToExport(handles,RawFileName);
catch errordlg(lasterr)
    return
end

%%% Indicates that the Cancel button was pressed
if isempty(ExportInfo.ObjectNames)
    %%% If nothing is chosen, we still want to check if the user wants to
    %%% export the process info
    %%% Export process info
    if isfield(ExportInfo,ExportProcessInfo)
        if strcmp(ExportInfo.ExportProcessInfo,'Yes')
            try CPtextpipe(handles,ExportInfo,RawFileName,RawPathname);
            catch errordlg(lasterr)
                return
            end
        end
    end
    return
end

%%% Create a waitbarhandle that can be accessed from the functions below
global waitbarhandle
waitbarhandle = waitbar(0,'');

%%% Export process info
if strcmp(ExportInfo.ExportProcessInfo,'Yes')
    try CPtextpipe(handles,ExportInfo,RawFileName,RawPathname);
    catch errordlg(lasterr)
        return
    end
end

%%% Export measurements
if ~isempty(ExportInfo.MeasurementFilename)
    try WriteMeasurements(handles,ExportInfo,RawPathname);
    catch errordlg(lasterr)
        return
    end
end

%%% Done!
close(waitbarhandle)
CPmsgbox('Exporting is completed.')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function WriteMeasurements(handles,ExportInfo,RawPathname)
%%% This function exports full and summary measurement reports

%%% Get the handle to the waitbar and update the text in the waitbar
global waitbarhandle
waitbar(0,waitbarhandle,'')

%%% Step 1: Create a cell array containing matrices with all measurements for each object type
%%% concatenated.
SuperMeasurements = cell(length(ExportInfo.ObjectNames),1);
SuperFeatureNames = cell(length(ExportInfo.ObjectNames),1);
for Object = 1:length(ExportInfo.ObjectNames)
    ObjectName = ExportInfo.ObjectNames{Object};

    %%% Get fields in handles.Measurements
    fields = fieldnames(handles.Measurements.(ObjectName));

    %%% Organize numerical features and text in a format suitable for exportation. This
    %%% piece of code creates a super measurement matrix, where
    %%% all features for all objects are stored. There will be one such
    %%% matrix for each image set, and each column in such a matrix
    %%% corresponds to one feature, e.g. Area or IntegratedIntensity.
    MeasurementNames = {};
    Measurements = {};
    TextNames = {};
    Text = {};

    for k = 1:length(fields)

        % Suffix 'Features' indicates that we have found a cell array with measurements, i.e.,
        % where each cell contains a matrix of size
        % [(Nbr of objects in image) x (Number of features of this feature type)]
        if length(fields{k}) > 8 & strcmp(fields{k}(end-7:end),'Features')
            % Get the associated cell array of measurements
            try
                CellArray = handles.Measurements.(ObjectName).(fields{k}(1:end-8));
            catch
                error(['Error in handles.Measurements structure. The field ',fields{k},' does not have an associated measurement field.']);

            end
            if length(Measurements) == 0
                Measurements = CellArray;
            else
                % Loop over the image sets
                for j = 1:length(CellArray)
                    Measurements{j} = cat(2,Measurements{j},CellArray{j});
                end
            end

            % Construct informative feature names
            tmp = handles.Measurements.(ObjectName).(fields{k});     % Get the feature names
            for j = 1:length(tmp)
                tmp{j} = [tmp{j} ' (', ObjectName,', ',fields{k}(1:end-8),')'];
            end
            MeasurementNames = cat(2,MeasurementNames,tmp);

            % Suffix 'Text' indicates that we have found a cell array with text information, i.e.,
            % where each cell contains a cell array of strings
        elseif length(fields{k}) > 4 & strcmp(fields{k}(end-3:end),'Text')

            % Get the associated cell array of measurements
            try
                CellArray = handles.Measurements.(ObjectName).(fields{k}(1:end-4));
            catch
                error(['Error in handles.Measurements structure. The field ',fields{k},' does not have an associated text field.']);

            end

            %%% If this is the first measurement structure encounterered we have to initialize instead of concatenate
            if length(Text) == 0
                Text = CellArray;
                %%% else concatenate
            else
                % Loop over the image sets
                for j = 1:length(CellArray)
                    Text{j} = cat(2,Text{j},CellArray{j});
                end
            end
            TextNames = cat(2,TextNames,handles.Measurements.(ObjectName).(fields{k}));
        end
    end % end loop over the fields in the current object type

    %%% Create the super measurement structure
    SuperMeasurements{Object} = Measurements;
    SuperMeasurementNames{Object} = MeasurementNames;
    SuperText{Object} = Text;
    SuperTextNames{Object} = TextNames;
end % end loop over object types, i.e., Cells, Nuclei, Cytoplasm, Image

%%% Step 2: Write the measurements to file
for Object = 1:length(ExportInfo.ObjectNames)

    ObjectName = ExportInfo.ObjectNames{Object};

    % Open a file for exporting the measurements
    % Add dot in extension if it's not there
    if ExportInfo.MeasurementExtension(1) ~= '.';
        ExportInfo.MeasurementExtension = ['.',ExportInfo.MeasurementExtension];
    end
    filename = [ExportInfo.MeasurementFilename,'_',ObjectName,ExportInfo.MeasurementExtension];
    fid = fopen(fullfile(RawPathname,filename),'w');
    if fid == -1
        error(sprintf('Cannot create the output file %s. There might be another program using a file with the same name.',filename));
    end

    % Get the measurements and feature names to export
    Measurements     = SuperMeasurements{Object};
    MeasurementNames = SuperMeasurementNames{Object};
    Text             = SuperText{Object};
    TextNames        = SuperTextNames{Object};

    % The 'Image' object type is gets some special treatement.
    % Add the average values for all other measurements
    if strcmp(ObjectName,'Image')
        for k = 1:length(ExportInfo.ObjectNames)
            if ~strcmp('Image',ExportInfo.ObjectNames{k})
                MeasurementNames = cat(2,MeasurementNames,SuperMeasurementNames{k});
                tmpMeasurements = SuperMeasurements{k};
                if ExportInfo.IgnoreNaN == 1
                    for imageset = 1:length(Measurements)
                        Measurements{imageset} = cat(2,Measurements{imageset},CPnanmean(tmpMeasurements{imageset},1));
                    end
                else
                    for imageset = 1:length(Measurements)
                        Measurements{imageset} = cat(2,Measurements{imageset},mean(tmpMeasurements{imageset},1));
                    end
                end
            end
        end
    else     % If not the 'Image' field, add a Object Nbr feature instead
        MeasurementNames = cat(2,{'Object Nbr'},MeasurementNames);
    end


    %%% Write tab-separated file that can be imported into Excel
    % Header part
    fprintf(fid,'%s\n\n', ObjectName);

    % Write data in columns or rows?
    if strcmp(ExportInfo.SwapRowsColumnInfo,'No')

        % Write feature names in one row
        % Interleave feature names with commas and write to file
        strMeasurement = cell(2*length(MeasurementNames),1);
        strMeasurement(1:2:end) = {'\t'};
        strMeasurement(2:2:end) = MeasurementNames;
        strText = cell(2*length(TextNames),1);
        strText(1:2:end) = {'\t'};
        strText(2:2:end) = TextNames;
        fprintf(fid,sprintf('%s%s\n',char(cat(2,strText{:})),char(cat(2,strMeasurement{:}))));

        % Loop over the images sets
        for imageset = 1:max(length(Measurements),length(Text))

            % Update waitbar
            waitbar(imageset/length(ExportInfo.ObjectNames),waitbarhandle,sprintf('Exporting %s',ObjectName));

            % Write info about the image set (some unnecessary code here)
            fprintf(fid,'Set #%d, %s',imageset,handles.Measurements.Image.FileNames{imageset}{1});

            %%% Write measurements and text row by row
            %%% First, determine number of rows to write. Have to do this to protect
            %%% for the cases of no Measurements or no Text.
            if ~isempty(Measurements)
                NbrOfRows = size(Measurements{imageset},1);
            elseif ~isempty(Text)
                NbrOfRows = size(Text{imageset},1);
            else
                NbrOfRows = 0;
            end

            for row = 1:NbrOfRows    % Loop over the rows

                % If not the 'Image' field, write an object number
                if ~strcmp(ObjectName,'Image')
                    fprintf(fid,'\t%d',row);
                end

                % Write text
                strText = {};
                if ~isempty(TextNames)
                    strText = cell(2*length(TextNames),1);
                    strText(1:2:end) = {'\t'};                 % 'Text' is a cell array where each cell contains a cell array
                    tmp = Text{imageset}(row,:);               % Get the right row in the right image set
                    index = strfind(tmp,'\');                  % To use sprintf(), we need to duplicate any '\' characters
                    for k = 1:length(index)
                        for l = 1:length(index{k})
                            tmp{k} = [tmp{k}(1:index{k}(l)+l-1),'\',tmp{k}(index{k}(l)+l:end)];   % Duplicate '\':s
                        end
                    end
                    strText(2:2:end) = tmp;                    % Interleave with tabs
                end

                % Write measurements
                strMeasurement = {};
                if ~isempty(MeasurementNames)
                    tmp = cellstr(num2str(Measurements{imageset}(row,:)','%g'));  % Create cell array with measurements
                    strMeasurement = cell(2*length(tmp),1);
                    strMeasurement(1:2:end) = {'\t'};                             % Interleave with tabs
                    strMeasurement(2:2:end) = tmp;
                end
                fprintf(fid,sprintf('%s%s\n',char(cat(2,strText{:})),char(cat(2,strMeasurement{:}))));            % Write to file
            end

        end

        %%% Write each measurement as a row, with each object as a column
    else
        %%% Write first row where the image set starting points are indicated
        fprintf(fid,'\t%d',[]);
        for imageset = 1:length(Measurements)
            str = cell(size(Measurements{imageset},1)+1,1);
            str(1) = {sprintf('Set #%d, %s',imageset,handles.Measurements.Image.FileNames{imageset}{1})};
            str(2:end) = {'\t'};
            fprintf(fid,sprintf('%s',cat(2,str{:})));
        end
        fprintf(fid,'\n');

        %%% If the current object type isn't 'Image'
        %%% add the 'Object count' to the Measurement matrix
        if ~strcmp(ObjectName,'Image')
            for imageset = 1:length(Measurements)
                Measurements{imageset} = cat(2,[1:size(Measurements{imageset},1)]',Measurements{imageset});
            end
        end

        %%% Start by writing text
        %%% Loop over rows, writing one image set's text features at the time
        for row = 1:length(TextNames)
            fprintf(fid,'%s',TextNames{row});
            for imageset = 1:length(Text)
                strText = cell(2*size(Text{imageset},1),1);
                strText(1:2:end) = {'\t'};                 % 'Text' is a cell array where each cell contains a cell array
                tmp = Text{imageset}(:,row)';              % Get the right row in the right image set
                index = strfind(tmp,'\');                  % To use sprintf(), we need to duplicate any '\' characters
                for k = 1:length(index)
                    for l = 1:length(index{k})
                        tmp{k} = [tmp{k}(1:index{k}(l)+l-1),'\',tmp{k}(index{k}(l)+l:end)];   % Duplicate '\':s
                    end
                end
                strText(2:2:end) = tmp;                    % Interleave with tabs
                fprintf(fid,sprintf('%s',char(cat(2,strText{:}))));
            end
            fprintf(fid,'\n');
        end

        %%% Next, write numerical measurements
        %%% Loop over rows, writing one image set's measurements at the time
        for row = 1:length(MeasurementNames)
            fprintf(fid,'%s',MeasurementNames{row});
            for imageset = 1:length(Measurements)
                tmp = cellstr(num2str(Measurements{imageset}(:,row),'%g'));  % Create cell array with measurements
                strMeasurement = cell(2*size(Measurements{imageset},1),1);
                strMeasurement(1:2:end) = {'\t'};
                strMeasurement(2:2:end) = tmp;                    % Interleave with tabs
                fprintf(fid,sprintf('%s',char(cat(2,strMeasurement{:}))));
            end
            fprintf(fid,'\n');
        end


    end % Ends 'if' row/column flip

    fclose(fid);
end % Ends 'for'-loop over object types

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
FontSize = GUIhandles.Current.FontSize;

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
uicontrol(ETh,'style','text','String','Swap Rows/Columns?','FontName','Times','FontSize',FontSize,...
    'HorizontalAlignment','left','units','inches','position',[2.5 basey+1.7 1.8 uiheight],'BackgroundColor',get(ETh,'color'));
SwapRowsColumnInfo = uicontrol(ETh,'style','popupmenu','String',{'No','Yes'},'FontName','Times','FontSize',FontSize,...
    'HorizontalAlignment','left','units','inches','position',[2.9 basey+1.4 0.7 uiheight],'BackgroundColor',[1 1 1]);
EditProcessInfoFilename = uicontrol(ETh,'Style','edit','units','inches','position',[0.2 basey 2.5 uiheight],...
    'backgroundcolor',[1 1 1],'String',ProposedFilename,'FontSize',FontSize);
uicontrol(ETh,'style','text','String','Choose extension:','FontName','Times','FontSize',FontSize,...
    'HorizontalAlignment','left','units','inches','position',[2.9 basey+0.2 1.2 uiheight],'BackgroundColor',get(ETh,'color'),'FontSize',FontSize)
EditProcessInfoExtension = uicontrol(ETh,'Style','edit','units','inches','position',[2.9 basey 0.7 uiheight],...
    'backgroundcolor',[1 1 1],'String','.txt','FontSize',FontSize);
uicontrol(ETh,'style','text','String','Ignore NaN''s?','FontName','Times','FontSize',FontSize,'HorizontalAlignment','left',...
    'units','inches','position',[2.9 basey+2.4 1.8 uiheight],'BackgroundColor',get(ETh,'color'))
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