function handles = ExportData(handles)

% Help for the Export Mean Data tool:
% Category: Data Tools
%
% Once image analysis is complete, use this tool to select the
% output file to extract the measurements and other information about
% the analysis.  The data will be converted to a comma-delimited text file
% which can be read by for example Excel.

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

load(fullfile(RawPathname, RawFileName));

%%% Opens a window that lets the user chose what to export and if
%%% only a summary (mean and std) should be exported

[ObjectNames,Summary] = ObjectsToExport(handles);

for Object = 1:length(ObjectNames)
    ObjectName = ObjectNames{Object};

    % Open a file for exporting the measurements
    if strcmp(Summary,'no')
        ExportFileName = [RawFileName(1:end-4),'_',ObjectName,'.xls'];
    else
        ExportFileName = [RawFileName(1:end-4),'_',ObjectName,'_Summary.xls'];
    end
    fid = fopen(fullfile(RawPathname,ExportFileName),'w');

    %%% Get fields in handles.Measurements
    fields = fieldnames(handles.Measurements.(ObjectName));


    %%% Organize features in format suitable for exportation. Create
    %%% a cell array Measurements where all features are concatenated
    %%% with each column corresponding to a separate feature
    FeatureNames = {};
    Measurements = {};
    for k = 1:length(fields)
        if ~isempty(strfind(fields{k},'Features'))                          % Found a field with feature names
            % Concatenate measurement and feature name matrices
            tmp = handles.Measurements.(ObjectName).(fields{k}(1:end-8));   % Get the associated cell array of measurements
            if isempty(Measurements)                                        % Have to initialize
                Measurements = tmp;
            else
                for j = 1:length(tmp)
                    Measurements(j) = {cat(2,Measurements{j},tmp{j})};
                end
            end

            % Construct informative feature names
            tmp = handles.Measurements.(ObjectName).(fields{k});
            for j = 1:length(tmp)
                tmp{j} = [tmp{j} ' (' , fields{k}(1:end-8),')'];
            end
            FeatureNames = cat(2,FeatureNames,tmp);
        end
    end

    % Count objects
    NumObjects = zeros(length(Measurements),1);
    for k = 1:length(Measurements)
        NumObjects(k) = size(Measurements{k},1);
    end

    % If only summary report desired, replace the entries in Measurements with
    % mean, median and standard deviation of the measurements
    if strcmp(Summary,'yes')
        for k = 1:length(Measurements)
            if ~isempty(Measurements{k})       % Make sure there are some measurements
                Measurements{k} = [mean(Measurements{k});median(Measurements{k});std(Measurements{k})];
            end
        end
    end
    SummaryInfo = {'Mean','Median','Std'};

    % Get general information from handles.Measurements.GeneralInfo
    InfoFields = fieldnames(handles.Measurements.GeneralInfo);
    Filenames  = InfoFields(strmatch('Filename',InfoFields));
    Thresholds = InfoFields(strmatch('ImageThreshold',InfoFields));
    for k = 1:length(Thresholds)
        if strcmp(Thresholds{k}(15:end),ObjectName)
            Threshold = handles.Measurements.GeneralInfo.(Thresholds{k});
        end
    end
    Time = handles.Measurements.GeneralInfo.TimeElapsed;


    %%% Write comma-separated file that can be imported into Excel

    % Header part
    if strcmp(Summary,'no')
        fprintf(fid,'%s: Full report\nTotal time: %0.2f s', ObjectName,Time{end});
    else
        fprintf(fid,'%s: Summary\nTotal time: %0.2f s', ObjectName,Time{end});
    end
    fprintf(fid,'\n');

    % Write feature names in one row
    % Interleave feature names with commas and write to file
    str = cell(2*length(FeatureNames),1);
    str(1:2:end) = {','};
    str(2:2:end) = FeatureNames;
    fprintf(fid,',%s\n',cat(2,str{:}));


    % Loop over the images sets
    for k = 1:length(Measurements)

        % Write info about the image set
        fprintf(fid,'Set #%d, %d objects\n',k,NumObjects(k));
        fprintf(fid,'Filenames: ');
        for j = 1:length(Filenames)
            fprintf(fid,'"%s":%s ',Filenames{j}(9:end),handles.Measurements.GeneralInfo.(Filenames{j}){k});
        end
        fprintf(fid,',Threshold: %g\n',Threshold{k});

        % Write measurements
        if ~isempty(Measurements{k})
            for row = 1:size(Measurements{k},1)                        % Loop over the rows
                if strcmp(Summary,'yes')
                    fprintf(fid,'%s',SummaryInfo{row});
                end
                tmp = cellstr(num2str(Measurements{k}(row,:)','%g'));  % Create cell array with measurements
                str = cell(2*length(tmp),1);                           % Interleave with commas
                str(1:2:end) = {','};
                str(2:2:end) = tmp;
                fprintf(fid,',%s\n',cat(2,str{:}));                    % Write to file
            end
        end
        fprintf(fid,'\n');                                         % Separate image sets with a blank row
    end
    fclose(fid);
end



function [ObjectNames,Summary] = ObjectsToExport(handles)
% This function displays a window so that lets the user choose which
% measurements to export. If the return variable 'ObjectNames' is empty
% it means that either no measurements were found or the user pressed
% the Cancel button (or the window was closed). 'Summary' takes on the values'yes'
% or 'no', depending if the user only wants a summary report (mean and std)
% or a full report.

% The fontsize is stored in the 'UserData' property of the main Matlab window
FontSize = get(0,'UserData');

% Get measurement object fields
fields = fieldnames(handles.Measurements);

% Remove the 'GeneralInfo' field
index = setdiff(1:length(fields),strmatch('GeneralInfo',fields));
fields = fields(index);

% Create Export window
ETh = figure;
set(ETh,'units','inches','resize','on','menubar','none','toolbar','none','numbertitle','off','Name','Export window');
pos = get(ETh,'position');
Height = 1.5+length(fields)*0.25;                       % Window height in inches, depends on the number of objects
set(ETh,'position',[pos(1) pos(2) 4 Height]);

if ~isempty(fields)
% Top text
uicontrol(ETh,'style','text','String','The following measurements were found:','FontName','Times','FontSize',FontSize,...
    'units','inches','position',[0 Height-0.2 4 0.15],'BackgroundColor',get(ETh,'color'),'fontweight','bold')

% Radio buttons for extracted measurements
h = [];
for k = 1:length(fields)
    uicontrol(ETh,'style','text','String',fields{k},'FontName','Times','FontSize',FontSize,'HorizontalAlignment','left',...
        'units','inches','position',[0.6 Height-0.35-0.2*k 3 0.15],'BackgroundColor',get(ETh,'color'))
    h(k) = uicontrol(ETh,'Style','Radiobutton','units','inches','position',[0.2 Height-0.35-0.2*k 0.2 0.2],...
        'BackgroundColor',get(ETh,'color'),'Value',1);
end

% Full report or Summary report
reportbutton = uicontrol(ETh,'Style','popup','units','inches','position',[0.2 0.7 1.5 0.2],...
    'backgroundcolor',[1 1 1],'String','Full report|Summary report');
else  % No measurements found
uicontrol(ETh,'style','text','String','No measurements found!','FontName','Times','FontSize',FontSize,...
    'units','inches','position',[0 Height-0.5 4 0.15],'BackgroundColor',get(ETh,'color'),'fontweight','bold')
end

% Export and Cancel pushbuttons
exportbutton = uicontrol(ETh,'style','pushbutton','String','Export','FontName','Times','FontSize',FontSize,'units','inches',...
    'position',[1.15 0.1 0.75 0.3],'Callback','[foo,fig] = gcbo;set(fig,''UserData'',1);uiresume(fig)');
cancelbutton = uicontrol(ETh,'style','pushbutton','String','Cancel','FontName','Times','FontSize',FontSize,'units','inches',...
    'position',[2.1 0.1 0.75 0.3],'Callback','[foo,fig] = gcbo;set(fig,''UserData'',0);uiresume(fig)');

% Disable export button if there are no measurements
if isempty(fields)
    set(exportbutton,'enable','off')
end

uiwait(ETh)                         % Wait until window is destroyed or uiresume() is called

if get(ETh,'Userdata') == 0
    ObjectNames = [];                                   % The user pressed the Cancel button
    Summary = [];
    close(ETh)
elseif get(ETh,'Userdata') == 1
    
    if get(reportbutton,'value') == 1
        Summary = 'no';
    else
        Summary = 'yes';
    end

    buttonchoice = get(h,'Value');
    if iscell(buttonchoice)                              % buttonchoice will be a cell array if there are several objects
        buttonchoice = cat(1,buttonchoice{:});
    end
    ObjectNames = fields(find(buttonchoice));            % Get the fields for which the radiobuttons are enable
    close(ETh)
else
    ObjectNames = [];
end

